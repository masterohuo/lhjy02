"""
factor_system.py - lhjy02 因子系统

100+ factors across 7 categories. Pure vectorized pandas operations.
Provides FactorScreener for IC analysis and factor effectiveness filtering.

Usage:
    from data_loader import load_all_tables
    from factor_system import generate_all_factors, FactorScreener

    df = load_all_tables(start_date='20200101', end_date='20250101')
    df = generate_all_factors(df)

    screener = FactorScreener(df, factor_cols, forward_return_col='forward_ret_5d')
    ic = screener.calculate_ic()
    effective = screener.filter_effective_factors(min_abs_ic=0.01)
"""
import numpy as np
import pandas as pd
from numba import njit

EPS = 1e-8


# ============================================================
# Numba-accelerated rolling functions (same logic, 10-50x faster)
# ============================================================

@njit(nogil=True, cache=True)
def _numba_slope(values, window, min_periods):
    """Rolling linear regression slope."""
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        w = values[start:i + 1]
        m = len(w)
        x = np.arange(m, dtype=np.float64) - (m - 1) / 2.0
        denom = np.sum(x * x)
        if denom > 1e-12:
            out[i] = np.sum(x * w) / denom
    return out


@njit(nogil=True, cache=True)
def _numba_max_dd(returns, window, min_periods):
    """Rolling maximum drawdown."""
    n = len(returns)
    out = np.full(n, np.nan)
    cum = np.empty(window + 1, dtype=np.float64)
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        r = returns[start:i + 1]
        m = len(r)
        cum[0] = 1.0
        peak = 1.0
        dd_max = 0.0
        for j in range(m):
            cum[j + 1] = cum[j] * (1.0 + r[j])
            if cum[j + 1] > peak:
                peak = cum[j + 1]
            dd = cum[j + 1] / peak - 1.0
            if dd < dd_max:
                dd_max = dd
        out[i] = dd_max
    return out


@njit(nogil=True, cache=True)
def _numba_nr7(values, window, min_periods):
    """NR7: narrowest range in N days."""
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        w = values[start:i + 1]
        if values[i] == np.min(w):
            out[i] = 1.0
    return out


@njit(nogil=True, cache=True)
def _numba_ulcer(returns, window, min_periods):
    """Rolling ulcer index."""
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        r = returns[start:i + 1]
        m = len(r)
        cum = np.empty(m + 1, dtype=np.float64)
        cum[0] = 1.0
        peak = 1.0
        sse = 0.0
        for j in range(m):
            cum[j + 1] = cum[j] * (1.0 + r[j])
            if cum[j + 1] > peak:
                peak = cum[j + 1]
            dd = cum[j + 1] / peak - 1.0
            sse += dd * dd
        if sse > 0:
            out[i] = np.sqrt(sse / m)
    return out


# ============================================================
# Helpers
# ============================================================

def _roll(df, col, w, func='mean'):
    """Grouped rolling window aggregation. Returns Series aligned with df index.

    Parameters
    ----------
    df : pd.DataFrame
    col : str — column name
    w : int — window size
    func : str — 'mean', 'sum', 'std', 'min', 'max', 'skew', 'kurt'

    Returns
    -------
    pd.Series
    """
    mp = max(1, w // 2)
    r = df.groupby('ts_code')[col].rolling(w, min_periods=mp)

    if func == 'mean':
        return r.mean().reset_index(level=0, drop=True)
    elif func == 'sum':
        return r.sum().reset_index(level=0, drop=True)
    elif func == 'std':
        return r.std(ddof=0).reset_index(level=0, drop=True)
    elif func == 'min':
        return r.min().reset_index(level=0, drop=True)
    elif func == 'max':
        return r.max().reset_index(level=0, drop=True)
    elif func == 'skew':
        return r.skew().reset_index(level=0, drop=True)
    elif func == 'kurt':
        return r.kurt().reset_index(level=0, drop=True)
    else:
        raise ValueError(f"Unknown function: {func}")


def _roll_std(df, col, w):
    """Grouped rolling standard deviation. Convenience wrapper for _roll(..., 'std')."""
    return _roll(df, col, w, 'std')


def _shift(df, col, n):
    """Shift column within each stock group."""
    return df.groupby('ts_code')[col].shift(n)


def _ema(df, col, span):
    """Exponential moving average, grouped by ts_code."""
    return df.groupby('ts_code')[col].transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )


def _roll_corr(df, col1, col2, w):
    """Rolling Pearson correlation between two columns, grouped by ts_code."""
    def _f(g):
        return g[col1].rolling(w, min_periods=max(1, w // 2)).corr(g[col2])

    return df.groupby('ts_code').apply(_f).reset_index(level=0, drop=True)


def _consecutive_signed(df, col):
    """Signed consecutive-day streak count.

    Positive = consecutive positive days; negative = consecutive negative days.
    Resets on sign change.
    """
    sign = np.sign(df[col].fillna(0))
    changed = sign.groupby(df['ts_code']).transform(
        lambda x: (x != x.shift(1).fillna(x.iloc[0])).astype(int)
    )
    streak_id = changed.groupby(df['ts_code']).cumsum()
    count = df.groupby([df['ts_code'], streak_id]).cumcount() + 1
    return count * sign


# ============================================================
# A. Retail Counterparty (20 factors)
# ============================================================

def _compute_A(df):
    new_cols = {}

    sm_vol = df['buy_sm_vol'] + df['sell_sm_vol']
    md_vol = df['buy_md_vol'] + df['sell_md_vol']
    lg_vol = df['buy_lg_vol'] + df['sell_lg_vol']
    elg_vol = df['buy_elg_vol'] + df['sell_elg_vol']

    sm_net = df['buy_sm_vol'] - df['sell_sm_vol']
    md_net = df['buy_md_vol'] - df['sell_md_vol']
    lg_net = df['buy_lg_vol'] - df['sell_lg_vol']
    elg_net = df['buy_elg_vol'] - df['sell_elg_vol']
    inst_net = lg_net + elg_net

    vol = df['volume'].replace(0, np.nan)

    # A1-A5: net volumes (normalized)
    new_cols['sm_net_vol'] = sm_net / vol
    new_cols['md_net_vol'] = md_net / vol
    new_cols['lg_net_vol'] = lg_net / vol
    new_cols['elg_net_vol'] = elg_net / vol
    new_cols['inst_net_vol'] = inst_net / vol

    # A6: retail net vol
    new_cols['retail_net_vol'] = new_cols['sm_net_vol']

    # A7-A8: institutional ratios
    new_cols['inst_retail_ratio'] = inst_net / (sm_net.abs() + EPS)
    new_cols['inst_md_ratio'] = inst_net / (md_net.abs() + EPS)

    # A9-A16: buy/sell ratios
    new_cols['sm_buy_ratio'] = df['buy_sm_vol'] / (sm_vol + EPS)
    new_cols['sm_sell_ratio'] = 1.0 - new_cols['sm_buy_ratio']
    new_cols['md_buy_ratio'] = df['buy_md_vol'] / (md_vol + EPS)
    new_cols['md_sell_ratio'] = 1.0 - new_cols['md_buy_ratio']
    new_cols['lg_buy_ratio'] = df['buy_lg_vol'] / (lg_vol + EPS)
    new_cols['lg_sell_ratio'] = 1.0 - new_cols['lg_buy_ratio']
    new_cols['elg_buy_ratio'] = df['buy_elg_vol'] / (elg_vol + EPS)
    new_cols['elg_sell_ratio'] = 1.0 - new_cols['elg_buy_ratio']

    # A17-A18: imbalance
    new_cols['sm_imbalance'] = sm_net / (sm_vol + EPS)
    inst_buy = df['buy_lg_vol'] + df['buy_elg_vol']
    inst_sell = df['sell_lg_vol'] + df['sell_elg_vol']
    new_cols['inst_imbalance'] = (inst_buy - inst_sell) / (inst_buy + inst_sell + EPS)

    # A19: normalized net moneyflow
    new_cols['net_mf_amount_norm'] = df['net_mf_amount'] / df['amount'].replace(0, np.nan)

    # A20: retail participation
    new_cols['retail_participation'] = sm_vol / vol

    return new_cols


# ============================================================
# B. Fund Flow (20 factors)
# ============================================================

def _compute_B(df):
    new_cols = {}

    # B1-B9: rolling net flows
    for w in [5, 10, 20]:
        new_cols[f'lg_net_vol_{w}d'] = _roll(df, 'lg_net_vol', w, 'mean')
        new_cols[f'inst_net_vol_{w}d'] = _roll(df, 'inst_net_vol', w, 'mean')
        new_cols[f'net_mf_amount_{w}d'] = _roll(df, 'net_mf_amount_norm', w, 'mean')

    # B10: flow persistence (signed consecutive days of inst direction)
    inst_raw = df['inst_net_vol'] * df['volume']
    new_cols['flow_persistence'] = _consecutive_signed(
        df.assign(_tmp_fi=inst_raw), '_tmp_fi'
    )

    # B11: flow acceleration (change in 5d rolling inst flow)
    new_cols['flow_acceleration'] = new_cols['inst_net_vol_5d'] - _shift(
        df.assign(_inst_5d=new_cols['inst_net_vol_5d']), '_inst_5d', 5
    )

    # B12: flow-price divergence (-corr between inst flow and return over 20d)
    ret_1d = df.groupby('ts_code')['close'].pct_change()
    new_cols['flow_price_divergence'] = -_roll_corr(
        df.assign(_r=ret_1d), 'inst_net_vol', '_r', 20
    )

    # B13: lg buy/sell volume ratio (5d rolling)
    new_cols['lg_buy_sell_ratio'] = (
        _roll(df, 'buy_lg_vol', 5, 'sum') / (_roll(df, 'sell_lg_vol', 5, 'sum') + EPS)
    )

    # B14: institution accumulation momentum (10d slope / 10d std)  [numba-accelerated]
    slope_vals = df.groupby('ts_code')['inst_net_vol'].transform(
        lambda g: pd.Series(_numba_slope(g.values, 10, 5), index=g.index)
    )
    new_cols['inst_accumulation_momentum'] = slope_vals / (_roll(df, 'inst_net_vol', 10, 'std') + EPS)

    # B15: flow volatility (cv of inst_net_vol over 10d)
    mean10 = _roll(df, 'inst_net_vol', 10, 'mean')
    new_cols['flow_volatility'] = _roll(df, 'inst_net_vol', 10, 'std') / (mean10.abs() + EPS)

    # B16: lg concentration (lg share of total absolute net flow)
    total_abs = ((df['buy_sm_vol'] - df['sell_sm_vol']).abs() +
                 (df['buy_md_vol'] - df['sell_md_vol']).abs() +
                 (df['buy_lg_vol'] - df['sell_lg_vol']).abs() +
                 (df['buy_elg_vol'] - df['sell_elg_vol']).abs())
    new_cols['lg_concentration'] = (df['buy_lg_vol'] - df['sell_lg_vol']).abs() / (total_abs + EPS)

    # B17: net_mf_amount ratio to 5d MA
    ma5 = _roll(df, 'net_mf_amount_norm', 5, 'mean')
    new_cols['net_mf_amount_ma5_ratio'] = df['net_mf_amount_norm'] / (ma5.abs() + EPS) - 1.0

    # B18: institution flow stability (1 - |autocorr| over 10d)
    lag1 = _shift(df, 'inst_net_vol', 1)
    ac = _roll_corr(df.assign(_l=lag1), 'inst_net_vol', '_l', 10)
    new_cols['inst_flow_stability'] = 1.0 - ac.abs()

    # B19: smart money 5d (elg rolling)
    new_cols['smart_money_5d'] = _roll(df, 'elg_net_vol', 5, 'mean')

    # B20: fund reversal signal (deviation of 5d from 20d flow)
    new_cols['fund_reversal_signal'] = new_cols['inst_net_vol_5d'] - new_cols['inst_net_vol_20d']

    return new_cols


# ============================================================
# C. Technical (35 factors)
# ============================================================

def _compute_C(df):
    new_cols = {}

    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    ret_1d = df.groupby('ts_code')['close'].pct_change()

    ma5 = _roll(df, 'close', 5, 'mean')
    ma10 = _roll(df, 'close', 10, 'mean')
    ma20 = _roll(df, 'close', 20, 'mean')
    ma60 = _roll(df, 'close', 60, 'mean')

    # store for reuse by _compute_E
    new_cols['_ma5'] = ma5
    new_cols['_ma10'] = ma10
    new_cols['_ma20'] = ma20
    new_cols['_ma60'] = ma60

    # C1-C4: momentum
    for w, label in [(5, '5d'), (10, '10d'), (20, '20d'), (60, '60d')]:
        new_cols[f'momentum_{label}'] = c / _shift(df, 'close', w) - 1.0

    # C5-C8: MA deviation
    new_cols['ma_deviation_5'] = c / ma5 - 1.0
    new_cols['ma_deviation_10'] = c / ma10 - 1.0
    new_cols['ma_deviation_20'] = c / ma20 - 1.0
    new_cols['ma_deviation_60'] = c / ma60 - 1.0

    # C9-C10: volume ratio
    new_cols['volume_ratio_5'] = v / _roll(df, 'volume', 5, 'mean')
    new_cols['volume_ratio_20'] = v / _roll(df, 'volume', 20, 'mean')

    # C11: RSI 14
    delta = df.groupby('ts_code')['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _roll(df.assign(_gain=gain), '_gain', 14, 'mean')
    avg_loss = _roll(df.assign(_loss=loss), '_loss', 14, 'mean')
    rs = avg_gain / (avg_loss + EPS)
    new_cols['rsi_14'] = 100.0 - 100.0 / (1.0 + rs)

    # C12-C14: MACD
    ema12 = _ema(df, 'close', 12)
    ema26 = _ema(df, 'close', 26)
    new_cols['macd_dif'] = ema12 - ema26
    new_cols['macd_dea'] = _ema(df.assign(_d=new_cols['macd_dif']), '_d', 9)
    new_cols['macd_hist'] = 2.0 * (new_cols['macd_dif'] - new_cols['macd_dea'])

    # C15-C16: Bollinger
    std20 = _roll(df, 'close', 20, 'std')
    upper = ma20 + 2.0 * std20
    lower = ma20 - 2.0 * std20
    new_cols['bollinger_position'] = (c - lower) / (upper - lower + EPS)
    new_cols['bollinger_width'] = (upper - lower) / (ma20 + EPS)

    # C17-C19: KDJ
    low9 = _roll(df, 'low', 9, 'min')
    high9 = _roll(df, 'high', 9, 'max')
    rsv = (c - low9) / (high9 - low9 + EPS) * 100.0
    new_cols['kdj_k'] = rsv.groupby(df['ts_code']).transform(
        lambda x: x.ewm(alpha=1 / 3, adjust=False).mean()
    )
    new_cols['kdj_d'] = new_cols['kdj_k'].groupby(df['ts_code']).transform(
        lambda x: x.ewm(alpha=1 / 3, adjust=False).mean()
    )
    new_cols['kdj_j'] = 3.0 * new_cols['kdj_k'] - 2.0 * new_cols['kdj_d']

    # C20: OBV (normalized by 20d MA)
    close_diff_sign = np.sign(df.groupby('ts_code')['close'].diff().fillna(0))
    obv_raw = (close_diff_sign * v).groupby(df['ts_code']).cumsum()
    obv_ma20 = _roll(df.assign(_o=obv_raw), '_o', 20, 'mean')
    new_cols['obv'] = obv_raw / (obv_ma20.abs() + EPS) - 1.0

    # C21: ATR 14 (normalized by close)
    prev_c = _shift(df, 'close', 1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    new_cols['atr_14'] = _roll(df.assign(_tr=tr), '_tr', 14, 'mean') / (c + EPS)

    # C22: MFI 14
    tp = (h + l + c) / 3.0
    mf = tp * v
    tp_prev = _shift(df.assign(_t=tp), '_t', 1)
    pos = mf.where(tp > tp_prev, 0.0)
    neg = mf.where(tp < tp_prev, 0.0)
    pos_sum = _roll(df.assign(_p=pos), '_p', 14, 'sum')
    neg_sum = _roll(df.assign(_n=neg), '_n', 14, 'sum')
    mfr = pos_sum / (neg_sum + EPS)
    new_cols['mfi_14'] = 100.0 - 100.0 / (1.0 + mfr)

    # C23: Williams %R 14
    hh14 = _roll(df, 'high', 14, 'max')
    ll14 = _roll(df, 'low', 14, 'min')
    new_cols['wr_14'] = (hh14 - c) / (hh14 - ll14 + EPS) * -100.0

    # C24: CCI 20
    tp_cci = (h + l + c) / 3.0
    tp_ma20 = _roll(df.assign(_t=tp_cci), '_t', 20, 'mean')
    tp_md = _roll(df.assign(_mad=(tp_cci - tp_ma20).abs()), '_mad', 20, 'mean')
    new_cols['cci_20'] = (tp_cci - tp_ma20) / (0.015 * tp_md + EPS)

    # C25: ROC 10
    new_cols['roc_10'] = (c / _shift(df, 'close', 10) - 1.0) * 100.0

    # C26: amplitude (avg daily range over 5d)
    amp = (h - l) / c
    new_cols['amplitude_5'] = _roll(df.assign(_a=amp), '_a', 5, 'mean')

    # C27: EMA ratio
    new_cols['ema12_ema26_ratio'] = ema12 / (ema26 + EPS)

    # C28: PPO
    new_cols['ppo'] = (ema12 - ema26) / (ema26 + EPS) * 100.0

    # C29: Chaikin Money Flow 20
    mfm = ((c - l) - (h - c)) / (h - l + EPS) * v
    new_cols['cmf_20'] = _roll(df.assign(_m=mfm), '_m', 20, 'sum') / (_roll(df, 'volume', 20, 'sum') + EPS)

    # C30: MA5/MA20 ratio
    new_cols['ma5_ma20_ratio'] = ma5 / (ma20 + EPS)

    # C31: PVT (Price Volume Trend, normalized)
    pvt_raw = ret_1d * v
    pvt_cum = _roll(df.assign(_pvt_raw=pvt_raw), '_pvt_raw', 20, 'sum')
    pvt_m = _roll(df.assign(_p=pvt_cum), '_p', 20, 'mean')
    new_cols['pvt'] = pvt_cum / (pvt_m.abs() + EPS)

    # C32: Force Index 13 (normalized)
    fi = df.groupby('ts_code')['close'].diff() * v
    fi_ema = fi.groupby(df['ts_code']).transform(lambda x: x.ewm(span=13, adjust=False).mean())
    fi_std = _roll(df.assign(_fe=fi_ema), '_fe', 20, 'std')
    new_cols['force_index_13'] = fi_ema / (fi_std + EPS)

    # C33: TRIX 15
    e1 = _ema(df, 'close', 15)
    e2 = _ema(df.assign(_e1=e1), '_e1', 15)
    e3 = _ema(df.assign(_e2=e2), '_e2', 15)
    new_cols['trix_15'] = e3.groupby(df['ts_code']).pct_change()

    # C34: DPO 20
    k = 20 // 2 + 1
    dpo_raw = c - _shift(df.assign(_m=ma20), '_m', k)
    new_cols['dpo_20'] = dpo_raw / (ma20 + EPS)

    # C35: PSY 12
    up = (ret_1d > 0).astype(int)
    new_cols['psy_12'] = _roll(df.assign(_u=up), '_u', 12, 'mean') * 100.0

    return new_cols


# ============================================================
# D. Fundamental (10 factors)
# ============================================================

def _compute_D(df):
    new_cols = {}

    # D1-D2: cross-sectional percentile rank
    new_cols['pe_percentile_1y'] = df.groupby('date')['pe'].transform(lambda x: x.rank(pct=True))
    new_cols['pb_percentile_1y'] = df.groupby('date')['pb'].transform(lambda x: x.rank(pct=True))

    # D3-D4: log market cap
    new_cols['log_total_mv'] = np.log(df['total_mv'].clip(lower=EPS))
    new_cols['log_float_mv'] = np.log(df['float_mv'].clip(lower=EPS))

    # D5-D6: rolling turnover
    new_cols['turnover_5d'] = _roll(df, 'turnover_rate', 5, 'mean')
    new_cols['turnover_20d'] = _roll(df, 'turnover_rate', 20, 'mean')

    # D7: volume_ratio preserved from daily_basic (already loaded)

    # D8-D9: z-score within date
    pe_m = df.groupby('date')['pe'].transform('mean')
    pe_s = df.groupby('date')['pe'].transform('std')
    new_cols['pe_zscore'] = (df['pe'] - pe_m) / (pe_s + EPS)
    pb_m = df.groupby('date')['pb'].transform('mean')
    pb_s = df.groupby('date')['pb'].transform('std')
    new_cols['pb_zscore'] = (df['pb'] - pb_m) / (pb_s + EPS)

    # D10: market cap / average turnover amount
    amt5 = _roll(df, 'amount', 5, 'mean')
    new_cols['mv_turnover_ratio'] = df['total_mv'] / (amt5 + EPS)

    return new_cols


# ============================================================
# E. Trend / Reversal (15 factors)
# ============================================================

def _compute_E(df):
    new_cols = {}

    c, h, l = df['close'], df['high'], df['low']

    ret_1d = df.groupby('ts_code')['close'].pct_change()

    # reuse precomputed MAs from _compute_C
    ma5 = df['_ma5']
    ma10 = df['_ma10']
    ma20 = df['_ma20']
    ma60 = df['_ma60']

    # E1-E3: ADX, +DI, -DI (14)
    prev_c = _shift(df, 'close', 1)
    prev_h = _shift(df, 'high', 1)
    prev_l = _shift(df, 'low', 1)

    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = _roll(df.assign(_tr=tr), '_tr', 14, 'mean')

    up_move = h - prev_h
    down_move = prev_l - l
    pdm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    ndm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    di_p = _roll(df.assign(_pdm=pdm), '_pdm', 14, 'mean') / (atr14 + EPS) * 100.0
    di_m = _roll(df.assign(_ndm=ndm), '_ndm', 14, 'mean') / (atr14 + EPS) * 100.0

    dx = (di_p - di_m).abs() / (di_p + di_m + EPS) * 100.0
    new_cols['adx_14'] = _roll(df.assign(_dx=dx), '_dx', 14, 'mean')
    new_cols['di_plus_14'] = di_p
    new_cols['di_minus_14'] = di_m

    # E4-E5: MA crossover proximity
    new_cols['ma5_cross_ma20'] = (ma5 - ma20) / (ma20 + EPS)
    new_cols['ma10_cross_ma60'] = (ma10 - ma60) / (ma60 + EPS)

    # E6-E7: price vs 20d high/low
    high20 = _roll(df, 'high', 20, 'max')
    low20 = _roll(df, 'low', 20, 'min')
    new_cols['price_vs_20d_high'] = (c - high20) / (high20 + EPS)
    new_cols['price_vs_20d_low'] = (c - low20) / (low20 + EPS)

    # E8-E9: consecutive up/down (unsigned)
    streak = _consecutive_signed(df.assign(_r=ret_1d), '_r')
    new_cols['consecutive_up'] = streak.clip(lower=0)
    new_cols['consecutive_down'] = (-streak).clip(lower=0)

    # E10: gap ratio
    new_cols['gap_ratio'] = (df['open'] - prev_c) / (prev_c + EPS)

    # E11: MA20 slope
    new_cols['ma_slope_20'] = (ma20 - _shift(df.assign(_m=ma20), '_m', 10)) / 10.0

    # E12-E13: higher high / lower low vs 20d ago
    new_cols['higher_high_20'] = ((h > _shift(df, 'high', 20)) & (c > _shift(df, 'close', 20))).astype(float)
    new_cols['lower_low_20'] = ((l < _shift(df, 'low', 20)) & (c < _shift(df, 'close', 20))).astype(float)

    # E14: inside day
    new_cols['inside_day'] = ((h <= prev_h) & (l >= prev_l)).astype(float)

    # E15: NR7 (narrowest range in 7 days)  [numba-accelerated]
    dr = h - l
    new_cols['nr7'] = dr.groupby(df['ts_code']).transform(
        lambda x: pd.Series(_numba_nr7(x.values, 7, 7), index=x.index)
    )

    return new_cols


# ============================================================
# F. Volatility / Risk (12 factors)
# ============================================================

def _compute_F(df):
    new_cols = {}

    ret_1d = df.groupby('ts_code')['close'].pct_change()

    # F1-F4: historical volatility (annualized)
    for w in [5, 10, 20, 60]:
        new_cols[f'hist_vol_{w}'] = _roll(df.assign(_r=ret_1d), '_r', w, 'std') * np.sqrt(252)

    # F5: downside volatility 20
    neg = ret_1d.clip(upper=0)
    new_cols['downside_vol_20'] = _roll(df.assign(_n=neg), '_n', 20, 'std') * np.sqrt(252)

    # F6-F7: max drawdown  [numba-accelerated]
    for w in [20, 60]:
        new_cols[f'max_dd_{w}'] = df.groupby('ts_code')['close'].transform(
            lambda x: pd.Series(_numba_max_dd(x.pct_change().fillna(0).values, w, w // 2), index=x.index)
        )

    # F8-F9: skewness & kurtosis
    new_cols['skewness_20'] = _roll(df.assign(_r=ret_1d), '_r', 20, 'skew')
    new_cols['kurtosis_20'] = _roll(df.assign(_r=ret_1d), '_r', 20, 'kurt')

    # F10: beta 60 (vs cross-sectional mean return)
    mkt = ret_1d.groupby(df['date']).transform('mean')
    rh = _roll_corr(df.assign(_r=ret_1d, _m=mkt), '_r', '_m', 60)
    ss = _roll(df.assign(_r=ret_1d), '_r', 60, 'std')
    ms = _roll(df.assign(_m=mkt), '_m', 60, 'std')
    new_cols['beta_60'] = rh * ss / (ms + EPS)

    # F11: VaR 20 (parametric 95%)
    new_cols['var_20'] = -1.645 * _roll(df.assign(_r=ret_1d), '_r', 20, 'std')

    # F12: ulcer index 20  [numba-accelerated]
    new_cols['ulcer_index_20'] = df.groupby('ts_code')['close'].transform(
        lambda x: pd.Series(_numba_ulcer(x.pct_change().fillna(0).values, 20, 10), index=x.index)
    )

    return new_cols


# ============================================================
# G. Behavioral (8 factors)
# ============================================================

def _compute_G(df):
    new_cols = {}

    c, v = df['close'], df['volume']
    ret_1d = df.groupby('ts_code')['close'].pct_change()
    ret_5d = c / _shift(df, 'close', 5) - 1.0
    ret_20d = c / _shift(df, 'close', 20) - 1.0
    sm_vol = df['buy_sm_vol'] + df['sell_sm_vol']

    # G1: disposition effect (corr between retail sell ratio and past returns)
    sell_r = df['sell_sm_vol'] / (sm_vol + EPS)
    new_cols['disposition_effect'] = _roll_corr(
        df.assign(_s=sell_r, _r20=ret_20d), '_s', '_r20', 20
    )

    # G2: anchoring bias (deviation from 60d high)
    h60 = _roll(df, 'high', 60, 'max')
    new_cols['anchoring_bias'] = (c - h60) / (h60 + EPS)

    # G3: herding intensity (R² of stock return vs market return)
    mkt = ret_1d.groupby(df['date']).transform('mean')
    new_cols['herding_intensity'] = _roll_corr(
        df.assign(_r=ret_1d, _m=mkt), '_r', '_m', 20
    ) ** 2

    # G4: overreaction (negative of extreme 5d returns → reversal)
    rs = _roll(df.assign(_r=ret_1d), '_r', 20, 'std')
    extreme = ((ret_5d / (rs + EPS)).abs() > 2.0).astype(float)
    new_cols['overreaction'] = -ret_5d * extreme

    # G5: attention proxy (abnormal volume vs 20d MA)
    vm = _roll(df, 'volume', 20, 'mean')
    new_cols['attention_proxy'] = v / (vm + EPS) - 1.0

    # G6: information response speed (negative of return autocorrelation)
    lag = _shift(df.assign(_r=ret_1d), '_r', 1)
    ac = _roll_corr(df.assign(_r=ret_1d, _l=lag), '_r', '_l', 20)
    new_cols['info_response_speed'] = -ac

    # G7: retail panic/greed (retail net relative to volatility)
    rn = (df['buy_sm_vol'] - df['sell_sm_vol']) / (sm_vol + EPS)
    rm = _roll(df.assign(_rn=rn), '_rn', 20, 'mean')
    rs2 = _roll(df.assign(_rn=rn), '_rn', 20, 'std')
    new_cols['retail_panic_greed'] = (rn - rm) / (rs2 + EPS)

    # G8: sentiment momentum (5d change in retail net direction)
    new_cols['sentiment_momentum'] = rn - _shift(df.assign(_rn=rn), '_rn', 5)

    return new_cols


# ============================================================
# Factor registry
# ============================================================

FACTOR_CATEGORIES = {
    'A_Retail_Counterparty': [
        'sm_net_vol', 'md_net_vol', 'lg_net_vol', 'elg_net_vol',
        'inst_net_vol', 'retail_net_vol', 'inst_retail_ratio', 'inst_md_ratio',
        'sm_buy_ratio', 'sm_sell_ratio', 'md_buy_ratio', 'md_sell_ratio',
        'lg_buy_ratio', 'lg_sell_ratio', 'elg_buy_ratio', 'elg_sell_ratio',
        'sm_imbalance', 'inst_imbalance', 'net_mf_amount_norm', 'retail_participation',
    ],
    'B_Fund_Flow': [
        'lg_net_vol_5d', 'lg_net_vol_10d', 'lg_net_vol_20d',
        'inst_net_vol_5d', 'inst_net_vol_10d', 'inst_net_vol_20d',
        'net_mf_amount_5d', 'net_mf_amount_10d', 'net_mf_amount_20d',
        'flow_persistence', 'flow_acceleration', 'flow_price_divergence',
        'lg_buy_sell_ratio', 'inst_accumulation_momentum', 'flow_volatility',
        'lg_concentration', 'net_mf_amount_ma5_ratio', 'inst_flow_stability',
        'smart_money_5d', 'fund_reversal_signal',
    ],
    'C_Technical': [
        'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
        'ma_deviation_5', 'ma_deviation_10', 'ma_deviation_20', 'ma_deviation_60',
        'volume_ratio_5', 'volume_ratio_20', 'rsi_14',
        'macd_dif', 'macd_dea', 'macd_hist',
        'bollinger_position', 'bollinger_width',
        'kdj_k', 'kdj_d', 'kdj_j', 'obv',
        'atr_14', 'mfi_14', 'wr_14', 'cci_20',
        'roc_10', 'amplitude_5', 'ema12_ema26_ratio', 'ppo',
        'cmf_20', 'ma5_ma20_ratio', 'pvt', 'force_index_13',
        'trix_15', 'dpo_20', 'psy_12',
    ],
    'D_Fundamental': [
        'pe_percentile_1y', 'pb_percentile_1y', 'log_total_mv', 'log_float_mv',
        'turnover_5d', 'turnover_20d', 'volume_ratio',
        'pe_zscore', 'pb_zscore', 'mv_turnover_ratio',
    ],
    'E_Trend_Reversal': [
        'adx_14', 'di_plus_14', 'di_minus_14',
        'ma5_cross_ma20', 'ma10_cross_ma60',
        'price_vs_20d_high', 'price_vs_20d_low',
        'consecutive_up', 'consecutive_down', 'gap_ratio',
        'ma_slope_20', 'higher_high_20', 'lower_low_20',
        'inside_day', 'nr7',
    ],
    'F_Volatility_Risk': [
        'hist_vol_5', 'hist_vol_10', 'hist_vol_20', 'hist_vol_60',
        'downside_vol_20', 'max_dd_20', 'max_dd_60',
        'skewness_20', 'kurtosis_20', 'beta_60',
        'var_20', 'ulcer_index_20',
    ],
    'G_Behavioral': [
        'disposition_effect', 'anchoring_bias', 'herding_intensity',
        'overreaction', 'attention_proxy', 'info_response_speed',
        'retail_panic_greed', 'sentiment_momentum',
    ],
}

ALL_FACTOR_COLS = [f for cat in FACTOR_CATEGORIES.values() for f in cat]


# ============================================================
# Main
# ============================================================

def generate_all_factors(df):
    """Generate all 100+ factors. Returns df with factor columns appended.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame from data_loader.load_all_tables().
        Required columns: ts_code, date, open, high, low, close, volume, amount,
        buy_sm_vol, sell_sm_vol, buy_md_vol, sell_md_vol,
        buy_lg_vol, sell_lg_vol, buy_elg_vol, sell_elg_vol,
        net_mf_amount, total_mv, float_mv, pe, pb,
        turnover_rate, volume_ratio.

    Returns
    -------
    pd.DataFrame
        Original data with all factor columns appended, plus forward_ret_{1,5,10,20}d.
    """
    df = df.copy()

    if not df.index.is_monotonic_increasing:
        df = df.sort_values(['ts_code', 'date']).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Defensive: ensure metadata columns survived DataFrame operations
    if 'date' not in df.columns:
        raise KeyError(f'date column lost during factor pipeline. Columns: {list(df.columns[:10])}')

    # Build all factor columns via dicts and concat to avoid fragmentation warnings
    df = pd.concat([df, pd.DataFrame(_compute_A(df), index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(_compute_B(df), index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(_compute_C(df), index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(_compute_D(df), index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(_compute_E(df), index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(_compute_F(df), index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(_compute_G(df), index=df.index)], axis=1)

    # Forward returns for IC analysis / ML targets
    fwd_cols = {}
    for h in [1, 5, 10, 20]:
        fwd_cols[f'forward_ret_{h}d'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.shift(-h) / x - 1.0
        )
    df = pd.concat([df, pd.DataFrame(fwd_cols, index=df.index)], axis=1)

    # Clean up: replace inf, drop internal temp columns
    df = df.replace([np.inf, -np.inf], np.nan)
    tmp = [c for c in df.columns if c.startswith('_tmp_') or c.startswith('_ma')]
    if tmp:
        df = df.drop(columns=tmp)

    return df


# ============================================================
# FactorScreener
# ============================================================

class FactorScreener:
    """Evaluate factor effectiveness via IC analysis and quantile returns."""

    def __init__(self, df, factor_cols=None, forward_return_col='forward_ret_5d',
                 date_col='date'):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing factor columns, date, and forward return.
        factor_cols : list of str, optional
            Factor columns to evaluate. Defaults to ALL_FACTOR_COLS.
        forward_return_col : str
            Column name for the forward return (e.g. 'forward_ret_5d').
        date_col : str
            Column name for the date.
        """
        self.df = df
        self.factor_cols = factor_cols if factor_cols is not None else ALL_FACTOR_COLS
        self.forward_return_col = forward_return_col
        self.date_col = date_col
        self.factor_cols = [c for c in self.factor_cols if c in df.columns]

    def calculate_ic(self, method='spearman'):
        """Calculate mean cross-sectional IC for each factor (vectorized).

        Computes IC per date (correlation between factor value and forward return),
        then averages across all dates.

        Parameters
        ----------
        method : str
            'spearman' (rank IC) or 'pearson'.

        Returns
        -------
        pd.Series
            Mean IC per factor, sorted descending by absolute value.
        """
        target_col = self.forward_return_col
        factor_cols = [c for c in self.factor_cols if c in self.df.columns]
        cols = factor_cols + [target_col]

        # Drop rows where any factor or target is NaN (per-group handled separately)
        sub = self.df[cols + [self.date_col]].dropna()
        if sub.empty or len(sub) < 30:
            return pd.Series({c: np.nan for c in factor_cols})

        if method == 'spearman':
            # Rank within each date once for all factors
            ranked = sub.groupby(self.date_col)[cols].rank()
            ranked[self.date_col] = sub[self.date_col].values
        else:
            ranked = sub[cols].copy()
            ranked[self.date_col] = sub[self.date_col].values

        # Per-factor IC: correlation with target, averaged across dates
        # Uses pre-ranked data for efficiency (one rank operation for all factors)
        ic_results = {}
        for col in factor_cols:
            gb = ranked.groupby(self.date_col)[[col, target_col]]
            ic_by_date = gb.corr().loc[(slice(None), col), target_col]
            ic_results[col] = ic_by_date.mean()

        result = pd.Series(ic_results).sort_values(ascending=False)
        return result

    def calculate_quantile_returns(self, factor_col, n_quantiles=5):
        """Calculate mean forward return per factor quantile.

        Groups stocks into quantiles by factor value within each date,
        then computes the mean forward return per quantile bucket.

        Parameters
        ----------
        factor_col : str
            Factor column name.
        n_quantiles : int
            Number of quantile buckets.

        Returns
        -------
        pd.Series
            Index = quantile (0=lowest factor, N-1=highest).
            Values = mean forward return.
        """
        df = self.df.copy()
        fv = df[factor_col]
        target = df[self.forward_return_col]
        valid = fv.notna() & target.notna()
        df = df.loc[valid].copy()

        df['_q'] = df.groupby(self.date_col)[factor_col].transform(
            lambda x: pd.qcut(x.rank(method='first'), n_quantiles,
                              labels=False, duplicates='drop')
        )

        result = df.dropna(subset=['_q']).groupby('_q')[self.forward_return_col].mean()
        result.index = result.index.astype(int)
        result.index.name = 'quantile'
        return result.sort_index()

    def filter_effective_factors(self, min_abs_ic=0.01, method='spearman'):
        """Filter factors whose absolute mean IC meets the threshold.

        Parameters
        ----------
        min_abs_ic : float
            Minimum absolute IC threshold (default 0.01).
        method : str
            'spearman' or 'pearson'.

        Returns
        -------
        list of str
            Factor names with |IC| >= min_abs_ic.
        """
        ic = self.calculate_ic(method=method)
        return ic[ic.abs() >= min_abs_ic].index.tolist()

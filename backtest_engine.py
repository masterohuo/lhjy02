"""
backtest_engine.py - lhjy02 rolling-window backtest engine.

Monthly model retraining, weekly (Friday) rebalancing, daily mark-to-market.
"""
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    COMMISSION_RATE, INITIAL_CASH, MAX_POS_PCT, MAX_POSITIONS,
    RESULTS_DIR, SLIPPAGE, STAMP_TAX, TOP_N_STOCKS, TOTAL_POS_PCT,
    TRAIN_YEARS,
)
from data_loader import load_all_tables, load_index_daily
from factor_system import generate_all_factors
from model_trainer import TriModelTrainer
from stock_selector import StockSelector

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Rolling-window backtest engine for the lhjy02 stock selection strategy."""

    def __init__(self):
        self.cash = float(INITIAL_CASH)
        self.initial_cash = float(INITIAL_CASH)
        self.positions: dict[str, dict] = {}
        self.trade_history: list[dict] = []
        self.daily_values: dict[pd.Timestamp, float] = {}

    # ------------------------------------------------------------------
    # Core backtest loop
    # ------------------------------------------------------------------

    def run_rolling_backtest(self, start_date="2021-01-01", end_date=None,
                             retrain_freq="M"):
        """Run rolling-window backtest with periodic model retraining.

        For each retrain date (monthly by default):
          a. Training data: (retrain_date - TRAIN_YEARS) to retrain_date
          b. Test data: retrain_date to next retrain_date
          c. Train models on training data
          d. For each rebalance date in test period (weekly Friday):
             - Get latest factor data
             - Predict ensemble scores
             - Select stocks and construct portfolio
             - Execute simulated trades with slippage and costs
             - Track portfolio value daily

        Parameters
        ----------
        start_date : str
            Backtest start date (YYYY-MM-DD).
        end_date : str or None
            Backtest end date (defaults to today).
        retrain_freq : str
            Retrain frequency pandas offset alias ('M' for monthly).

        Returns
        -------
        dict
            daily_values (pd.Series), trade_history (pd.DataFrame), metrics (dict).
        """
        if end_date is None:
            end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        retrain_dates = pd.date_range(
            start_dt, end_dt, freq="MS" if retrain_freq == "M" else retrain_freq
        )
        if len(retrain_dates) == 0:
            retrain_dates = pd.DatetimeIndex([start_dt])

        for i, retrain_date in enumerate(retrain_dates):
            train_start = retrain_date - pd.DateOffset(years=TRAIN_YEARS)

            if i < len(retrain_dates) - 1:
                test_end = retrain_dates[i + 1] - pd.Timedelta(days=1)
            else:
                test_end = end_dt

            logger.info(
                "窗口 %d/%d: 训练 [%s, %s] 测试 [%s, %s]",
                i + 1, len(retrain_dates),
                train_start.date(), retrain_date.date(),
                retrain_date.date(), test_end.date(),
            )

            # Train models on this window's training data
            try:
                trainer = TriModelTrainer()
                models = trainer.train_all(
                    start_date=train_start.strftime("%Y%m%d"),
                    end_date=retrain_date.strftime("%Y%m%d"),
                )
            except Exception as exc:
                logger.error("窗口 %d 训练失败: %s", i + 1, exc, exc_info=True)
                continue

            # Load test-period data and generate factors once
            try:
                test_df = load_all_tables(
                    start_date=retrain_date.strftime("%Y%m%d"),
                    end_date=test_end.strftime("%Y%m%d"),
                )
            except Exception as exc:
                logger.warning("窗口 %d 数据加载失败: %s", i + 1, exc)
                continue

            if test_df.empty:
                logger.warning("窗口 %d 无测试数据, 跳过", i + 1)
                continue

            test_df = generate_all_factors(test_df)
            test_df = test_df.sort_values(["ts_code", "date"]).reset_index(drop=True)

            # Identify feature columns (all numeric, non-metadata, non-label)
            _meta_exclude = {
                "ts_code", "date", "label", "future_return", "target",
                "up_limit", "down_limit", "open", "high", "low",
            }
            feature_cols = [
                c for c in test_df.columns
                if c not in _meta_exclude
                and pd.api.types.is_numeric_dtype(test_df[c])
                and not c.startswith("forward_ret")
            ]

            selector = StockSelector(models_dict=models)
            all_dates = sorted(test_df["date"].unique())

            # Generate rebalance dates (weekly Fridays within test period)
            rebalance_candidates = pd.date_range(
                retrain_date, test_end, freq="W-FRI"
            )
            rebalance_set = {d for d in rebalance_candidates if d in all_dates}

            # Walk forward day by day
            for day in all_dates:
                day_data = test_df[test_df["date"] == day]

                if day in rebalance_set:
                    self._do_rebalance(day_data, feature_cols, selector, day)

                self._mark_to_market(day_data, day)

        return self._build_results()

    # ------------------------------------------------------------------
    # Rebalance logic
    # ------------------------------------------------------------------

    def _do_rebalance(self, day_data, feature_cols, selector, date):
        """Predict scores, select stocks, build portfolio, and execute trades."""
        # Predict ensemble scores from factor data
        scores = selector.predict_scores(day_data, feature_cols)

        # Build scored DataFrame with metadata columns
        meta_cols = ["ts_code", "close", "float_mv", "up_limit"]
        meta = [c for c in meta_cols if c in day_data.columns]
        day_scored = pd.concat([
            day_data[meta].reset_index(drop=True),
            scores.reset_index(drop=True),
        ], axis=1)
        if "ts_code" not in day_scored.columns and "ts_code" in day_data.columns:
            day_scored["ts_code"] = day_data["ts_code"].values

        # Select top stocks
        selected = selector.select_stocks(day_scored, top_n=TOP_N_STOCKS)
        if selected.empty:
            return

        # Build current-position snapshot for construct_portfolio
        price_map = dict(zip(day_data["ts_code"], day_data["close"]))
        positions_value = sum(
            p["shares"] * price_map.get(ts, p["cost_price"])
            for ts, p in self.positions.items()
        )
        total_value = self.cash + positions_value

        current_positions_fmt = {}
        for ts, p in self.positions.items():
            px = price_map.get(ts, p["cost_price"])
            val = p["shares"] * px
            current_positions_fmt[ts] = {
                "weight": val / total_value if total_value > 0 else 0.0,
                "value": val,
            }

        # Build target portfolio and execute trades
        portfolio = selector.construct_portfolio(
            selected,
            current_positions=current_positions_fmt if current_positions_fmt else None,
            total_cash=total_value,
        )

        self.simulate_trades(portfolio, self.positions, day_data, date)

    # ------------------------------------------------------------------
    # Trade simulation
    # ------------------------------------------------------------------

    def simulate_trades(self, target_portfolio, current_positions,
                        day_prices, date):
        """Execute simulated trades with slippage, commission, and stamp tax.

        Commission: 0.03% on both buy and sell.
        Stamp tax: 0.1% on sells only.
        Slippage: 0.2% (config.SLIPPAGE).

        Parameters
        ----------
        target_portfolio : dict
            Output of StockSelector.construct_portfolio() with 'orders' key.
        current_positions : dict
            {ts_code: {shares, cost_price}}.
        day_prices : pd.DataFrame
            Must contain ts_code, close columns.
        date : pd.Timestamp
            Trade date for record-keeping.
        """
        if day_prices.empty:
            return

        orders = target_portfolio.get("orders", [])
        if not orders:
            return

        price_lookup = dict(zip(day_prices["ts_code"], day_prices["close"]))
        positions_snapshot = dict(current_positions)

        for order in orders:
            ts_code, action, _target_weight, target_value = order
            close_price = price_lookup.get(ts_code)
            if not close_price or close_price <= 0:
                continue

            if action == "BUY":
                exec_price = close_price * (1.0 + SLIPPAGE)
                cost_per_share = exec_price * (1.0 + COMMISSION_RATE)

                lot = 100
                shares = int(target_value / cost_per_share / lot) * lot
                if shares <= 0:
                    continue

                total_cost = shares * cost_per_share
                if total_cost > self.cash:
                    shares = int(self.cash / cost_per_share / lot) * lot
                    if shares <= 0:
                        continue
                    total_cost = shares * cost_per_share

                self.cash -= total_cost

                if ts_code in self.positions:
                    old = self.positions[ts_code]
                    total_shares = old["shares"] + shares
                    avg_cost = (
                        (old["shares"] * old["cost_price"] + shares * exec_price)
                        / total_shares
                    )
                    self.positions[ts_code] = {"shares": total_shares, "cost_price": avg_cost}
                else:
                    self.positions[ts_code] = {"shares": shares, "cost_price": exec_price}

                self.trade_history.append({
                    "date": date, "ts_code": ts_code, "action": "BUY",
                    "shares": shares, "price": exec_price,
                    "amount": shares * exec_price,
                    "commission": shares * exec_price * COMMISSION_RATE,
                    "stamp_tax": 0.0,
                })

            elif action == "SELL":
                pos = positions_snapshot.get(ts_code)
                if pos is None:
                    continue

                exec_price = close_price * (1.0 - SLIPPAGE)

                # target_value is the desired remaining position value
                current_val = pos["shares"] * close_price
                if target_value < current_val and current_val > 0:
                    sell_fraction = (current_val - target_value) / current_val
                    shares = int(pos["shares"] * sell_fraction / 100) * 100
                else:
                    shares = pos["shares"]

                if shares <= 0:
                    continue
                shares = min(shares, pos["shares"])

                trade_amount = shares * exec_price
                commission = trade_amount * COMMISSION_RATE
                stamp_tax = trade_amount * STAMP_TAX
                proceeds = trade_amount - commission - stamp_tax

                self.cash += proceeds

                self.trade_history.append({
                    "date": date, "ts_code": ts_code, "action": "SELL",
                    "shares": shares, "price": exec_price,
                    "amount": trade_amount,
                    "commission": commission,
                    "stamp_tax": stamp_tax,
                })

                remaining = pos["shares"] - shares
                if remaining > 0:
                    self.positions[ts_code]["shares"] = remaining
                else:
                    del self.positions[ts_code]

    # ------------------------------------------------------------------
    # Daily mark-to-market
    # ------------------------------------------------------------------

    def _mark_to_market(self, day_data, date):
        """Record total portfolio value (cash + positions at close) for the day."""
        total = self.cash
        price_map = dict(zip(day_data["ts_code"], day_data["close"]))
        for ts_code, pos in self.positions.items():
            price = price_map.get(ts_code, pos["cost_price"])
            total += pos["shares"] * price
        self.daily_values[date] = total

    # ------------------------------------------------------------------
    # Results assembly
    # ------------------------------------------------------------------

    def _build_results(self):
        """Package backtest data into a structured results dict."""
        if not self.daily_values:
            return {
                "daily_values": pd.Series(dtype=float),
                "trade_history": pd.DataFrame(),
                "metrics": {},
            }

        dv = pd.Series(self.daily_values, name="portfolio_value")
        dv.index = pd.to_datetime(dv.index)
        dv = dv.sort_index()

        th = pd.DataFrame(self.trade_history)
        if not th.empty:
            th["date"] = pd.to_datetime(th["date"])

        return {
            "daily_values": dv,
            "trade_history": th,
            "metrics": self.calculate_metrics(dv, th),
        }

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def calculate_metrics(self, daily_values=None, trade_history=None):
        """Compute all performance metrics from the backtest results.

        Parameters
        ----------
        daily_values : pd.Series, optional
            Date to portfolio value. Uses self.daily_values if not provided.
        trade_history : pd.DataFrame, optional
            Trade records.

        Returns
        -------
        dict
        """
        if daily_values is None:
            daily_values = pd.Series(self.daily_values, name="portfolio_value")
            daily_values.index = pd.to_datetime(daily_values.index)
            daily_values = daily_values.sort_index()

        if trade_history is None:
            trade_history = pd.DataFrame(self.trade_history)

        if daily_values.empty:
            return {}

        dv = daily_values
        values = dv.values
        initial = float(values[0])
        final = float(values[-1])
        daily_rets = dv.pct_change().dropna()
        n_days = len(daily_rets)

        # Total return
        total_return = (final - initial) / initial

        # Annualized return (CAGR)
        years = max(n_days / 252.0, 1.0 / 252.0)
        annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0

        # Max drawdown and drawdown duration
        cummax_vals = np.maximum.accumulate(values)
        drawdowns = (values - cummax_vals) / cummax_vals
        max_dd = float(np.min(drawdowns))

        underwater = values < cummax_vals
        max_dd_duration = 0
        stretch = 0
        for flag in underwater:
            if flag:
                stretch += 1
                max_dd_duration = max(max_dd_duration, stretch)
            else:
                stretch = 0

        # Sharpe ratio (annualized, risk-free = 2%)
        rf_daily = (1.0 + 0.02) ** (1.0 / 252.0) - 1.0
        excess = daily_rets - rf_daily
        ann_vol = float(daily_rets.std()) * np.sqrt(252) if n_days > 1 else 0.0
        sharpe = (
            float(excess.mean() / excess.std() * np.sqrt(252))
            if excess.std() > 1e-10 else 0.0
        )

        # Win rate and profit/loss ratio (per-rebalance-period returns)
        win_rate, pl_ratio = self._calc_rebalance_stats(dv, trade_history)

        # Turnover rate
        turnover = self._calc_turnover(trade_history, dv)

        # Information ratio (vs CSI 300 benchmark)
        info_ratio = self._calc_information_ratio(dv)

        # Calmar ratio
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

        return {
            "total_return": total_return,
            "annualized_return": annual_return,
            "annualized_volatility": ann_vol,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": int(max_dd_duration),
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "profit_loss_ratio": pl_ratio,
            "turnover_rate": turnover,
            "information_ratio": info_ratio,
            "calmar_ratio": calmar,
            "initial_value": initial,
            "final_value": final,
            "n_trading_days": n_days,
        }

    def _calc_rebalance_stats(self, daily_values, trade_history):
        """Win rate and profit/loss ratio based on per-rebalance-period returns."""
        if trade_history.empty:
            return 0.0, 0.0

        rebalance_dates = sorted(trade_history["date"].unique())
        if len(rebalance_dates) < 2:
            return 0.0, 0.0

        period_rets = []
        for i in range(len(rebalance_dates) - 1):
            d0, d1 = rebalance_dates[i], rebalance_dates[i + 1]
            idx0 = daily_values.index.searchsorted(d0)
            idx1 = daily_values.index.searchsorted(d1)
            if idx0 < len(daily_values) and idx1 < len(daily_values):
                v0 = float(daily_values.iloc[idx0])
                v1 = float(daily_values.iloc[idx1])
                if v0 > 0:
                    period_rets.append((v1 - v0) / v0)

        if not period_rets:
            return 0.0, 0.0

        period_rets = np.array(period_rets)
        win_rate = float((period_rets > 0).mean())

        wins = period_rets[period_rets > 0]
        losses = period_rets[period_rets < 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = abs(float(losses.mean())) if len(losses) > 0 else 1e-10
        pl_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

        return win_rate, pl_ratio

    def _calc_turnover(self, trade_history, daily_values):
        """Average turnover per rebalance (total traded value / portfolio value)."""
        if trade_history.empty:
            return 0.0

        turnovers = []
        for d in sorted(trade_history["date"].unique()):
            day_trades = trade_history[trade_history["date"] == d]
            total_traded = float(day_trades["amount"].sum())

            idx = daily_values.index.searchsorted(d)
            if idx < len(daily_values):
                pv = float(daily_values.iloc[idx])
                if pv > 0:
                    turnovers.append(total_traded / pv)

        return float(np.mean(turnovers)) if turnovers else 0.0

    def _calc_information_ratio(self, daily_values):
        """Information ratio vs CSI 300 benchmark."""
        bm = self.compare_benchmark()
        if bm is None or bm.empty:
            return 0.0

        strat_rets = daily_values.pct_change().dropna()
        bm_rets = bm.set_index("date")["benchmark_cumret"].pct_change().dropna()

        common = strat_rets.index.intersection(bm_rets.index)
        if len(common) < 10:
            return 0.0

        active = strat_rets.loc[common] - bm_rets.loc[common]
        tracking_err = float(active.std()) * np.sqrt(252)
        excess_annual = float(active.mean()) * 252.0

        return excess_annual / tracking_err if tracking_err > 1e-10 else 0.0

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    def compare_benchmark(self):
        """Load CSI 300 index daily close and compute cumulative returns.

        Returns
        -------
        pd.DataFrame or None
            Columns: date, benchmark_close, benchmark_cumret.
        """
        if not self.daily_values:
            return None

        dv = pd.Series(self.daily_values)
        dv.index = pd.to_datetime(dv.index)
        dv = dv.sort_index()

        start_d = dv.index.min()
        end_d = dv.index.max()

        try:
            idx_df = load_index_daily(
                start_date=start_d.strftime("%Y%m%d"),
                end_date=end_d.strftime("%Y%m%d"),
            )
        except Exception as exc:
            logger.warning("加载指数数据失败: %s", exc)
            return None

        if idx_df.empty:
            return None

        # Prefer CSI 300 (000300.SH), fall back to first available index
        target_codes = ["000300.SH", "399300.SZ"]
        bm_data = idx_df[idx_df["ts_code"].isin(target_codes)]
        if bm_data.empty:
            bm_data = idx_df[idx_df["ts_code"] == idx_df["ts_code"].iloc[0]]

        bm_data = bm_data.sort_values("date")
        first_close = float(bm_data["close"].iloc[0])
        bm_data["benchmark_cumret"] = bm_data["close"] / first_close

        return bm_data[["date", "close", "benchmark_cumret"]].rename(
            columns={"close": "benchmark_close"}
        ).reset_index(drop=True)


# ======================================================================
# Direct run function
# ======================================================================

def run_backtest(start_date=None, end_date=None, quick=False):
    """Run the full lhjy02 rolling backtest and print/save results.

    Parameters
    ----------
    start_date : str or None
        Start date. Defaults to 1 year ago if quick, else '2021-01-01'.
    end_date : str or None
        End date. Defaults to today.
    quick : bool
        If True, use 1 year of data with monthly retrain for faster iteration.
    """
    if quick:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    else:
        if start_date is None:
            start_date = "2021-01-01"
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

    print()
    print("=" * 50)
    print("  lhjy02 Rolling Backtest")
    print("=" * 50)
    print(f"  Mode    : {'Quick (1 year)' if quick else 'Full'}")
    print(f"  Period  : {start_date} ~ {end_date}")
    print(f"  Capital : {INITIAL_CASH:,}")
    print("=" * 50)

    engine = BacktestEngine()
    results = engine.run_rolling_backtest(
        start_date=start_date, end_date=end_date, retrain_freq="M"
    )

    metrics = results.get("metrics", {})
    dv = results.get("daily_values", pd.Series(dtype=float))

    if not metrics:
        print("\nNo results generated. Check data availability and date ranges.")
        return results

    # Print formatted results table
    print()
    print("\U0001F4C8 lhjy02 Strategy Backtest Results")
    print("═" * 41)
    print(f"\U0001F4C5 Period: {start_date} ~ {end_date}")
    print(f"\U0001F4B0 Initial Capital: ¥{metrics.get('initial_value', INITIAL_CASH):,.0f}")
    print(f"\U0001F4CA Final Value:     ¥{metrics.get('final_value', 0):,.0f}")
    print(f"\U0001F4CA Total Return:    {metrics.get('total_return', 0) * 100:+.2f}%")
    print(f"\U0001F4C8 Annual Return:   {metrics.get('annualized_return', 0) * 100:+.2f}%")
    print(f"\U0001F4C9 Max Drawdown:    {metrics.get('max_drawdown', 0) * 100:+.2f}%")
    print(f"⚡ Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"\U0001F3AF Win Rate:        {metrics.get('win_rate', 0) * 100:.1f}%")
    print(f"\U0001F4C8 Profit/Loss Ratio: {metrics.get('profit_loss_ratio', 0):.2f}")
    print(f"\U0001F504 Avg Turnover:    {metrics.get('turnover_rate', 0) * 100:.1f}%")
    print(f"\U0001F4CA Info Ratio:      {metrics.get('information_ratio', 0):.3f}")
    print(f"\U0001F4AA Calmar Ratio:    {metrics.get('calmar_ratio', 0):.3f}")
    print("═" * 41)
    print()

    # Save CSV results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_results(results, start_date)

    # Plot equity curve
    _plot_equity_curve(results, engine)

    return results


# ======================================================================
# Output helpers
# ======================================================================

def _save_results(results, start_date):
    """Save daily values and trade history to timestamped CSV files."""
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    tag = start_date.replace("-", "") if start_date else "full"

    dv = results.get("daily_values")
    if isinstance(dv, pd.Series) and not dv.empty:
        dv_df = dv.reset_index()
        dv_df.columns = ["date", "portfolio_value"]
        path = RESULTS_DIR / f"backtest_equity_{tag}_{stamp}.csv"
        dv_df.to_csv(path, index=False)
        print(f"[save] Equity curve  -> {path}")

    th = results.get("trade_history")
    if isinstance(th, pd.DataFrame) and not th.empty:
        path = RESULTS_DIR / f"backtest_trades_{tag}_{stamp}.csv"
        th.to_csv(path, index=False)
        print(f"[save] Trade history -> {path}")


def _plot_equity_curve(results, engine):
    """Plot equity curve with drawdown subplot, save to RESULTS_DIR."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return

    dv = results.get("daily_values")
    if dv is None or (isinstance(dv, pd.Series) and dv.empty):
        return

    dv = dv if isinstance(dv, pd.Series) else pd.Series(dv)
    dv.index = pd.to_datetime(dv.index)
    dv = dv.sort_index()

    try:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Equity curve (normalized to 1.0)
        cumulative = dv / float(dv.iloc[0])
        ax1.plot(cumulative.index, cumulative.values,
                 label="lhjy02 Strategy", linewidth=1.2, color="#1f77b4")

        # Benchmark overlay
        try:
            bm = engine.compare_benchmark()
            if bm is not None and not bm.empty:
                bm = bm.set_index("date")
                common_idx = cumulative.index.intersection(bm.index)
                if len(common_idx) > 1:
                    ax1.plot(common_idx, bm.loc[common_idx, "benchmark_cumret"].values,
                             label="CSI 300", linewidth=1.0, color="#d62728", alpha=0.7)
        except Exception:
            pass

        ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
        ax1.set_title("lhjy02 Rolling Backtest - Equity Curve", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Cumulative Return (x initial)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

        # Drawdown
        cummax_vals = dv.values.cummax()
        drawdowns = (dv.values - cummax_vals) / cummax_vals * 100.0
        ax2.fill_between(dv.index, 0, drawdowns, color="#d62728", alpha=0.35)
        ax2.plot(dv.index, drawdowns, color="#d62728", linewidth=0.5)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

        fig.autofmt_xdate()
        fig.tight_layout()

        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"equity_curve_{stamp}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Equity curve -> {path}")
    except Exception:
        pass


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_backtest(quick=True)

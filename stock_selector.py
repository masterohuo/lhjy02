"""
stock_selector.py - lhjy02 三模型集成选股与组合构建模块
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    INITIAL_CASH, MAX_POS_PCT, MAX_POSITIONS, MODELS_DIR,
    STOP_LOSS, TOP_N_STOCKS, TOTAL_POS_PCT,
)

try:
    from model_trainer import TriModelTrainer  # noqa: F401
    _HAS_TRAINER = True
except ImportError:
    _HAS_TRAINER = False

logger = logging.getLogger(__name__)


class RiskManager:
    """风控管理器 — 止损检查、日内亏损监控、持仓验证"""

    def check_stop_loss(self, positions_df: pd.DataFrame) -> list:
        """返回触发 -8% 止损的股票列表。

        positions_df 必须包含 ts_code, cost_price, current_price 列。
        """
        if positions_df.empty:
            return []

        required = {'ts_code', 'cost_price', 'current_price'}
        missing = required - set(positions_df.columns)
        if missing:
            logger.warning(f"check_stop_loss: missing columns {missing}")
            return []

        pnl = (
            (positions_df['current_price'] - positions_df['cost_price'])
            / positions_df['cost_price']
        )
        stopped = positions_df.loc[pnl <= STOP_LOSS, 'ts_code'].tolist()
        if stopped:
            logger.warning(f"Stop loss triggered: {stopped}")
        return stopped

    def check_daily_loss(self, current_value: float, initial_value: float) -> bool:
        """日内亏损是否超过 -3%。"""
        if initial_value <= 0:
            return False
        return (current_value - initial_value) / initial_value < -0.03

    def validate_positions(self, target_positions: dict) -> tuple:
        """验证目标持仓：单票 ≤20%，总仓位 ≤80%。

        Returns (passed: bool, reason: str).
        """
        if not target_positions:
            return True, "empty positions"

        weights = list(target_positions.values())

        max_single = max(weights)
        if max_single > MAX_POS_PCT:
            return False, (
                f"Single stock {max_single:.2%} > max {MAX_POS_PCT:.2%}"
            )

        total = sum(weights)
        if total > TOTAL_POS_PCT:
            return False, (
                f"Total weight {total:.2%} > max {TOTAL_POS_PCT:.2%}"
            )

        return True, "OK"


class StockSelector:
    """三模型 (LightGBM + XGBoost + CatBoost) 集成选股器"""

    def __init__(self, models_dict: dict | None = None):
        self.risk_manager = RiskManager()
        self.models = models_dict if models_dict is not None else self._load_models()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_models(self) -> dict:
        """从 MODELS_DIR 加载 pickle 文件。"""
        models: dict[str, object] = {}
        model_files = {
            'lgb': MODELS_DIR / 'lgb_model.pkl',
            'xgb': MODELS_DIR / 'xgb_model.pkl',
            'cat': MODELS_DIR / 'cat_model.pkl',
        }
        for name, path in model_files.items():
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        models[name] = pickle.load(f)
                    logger.info(f"Loaded {name} model from {path}")
                except Exception as exc:
                    logger.warning(f"Failed to load {name}: {exc}")
            else:
                logger.warning(f"Model file not found: {path}")
        return models

    # ------------------------------------------------------------------
    # 预测评分
    # ------------------------------------------------------------------
    def predict_scores(
        self, X: pd.DataFrame | np.ndarray, feature_names: list[str]
    ) -> pd.DataFrame:
        """三模型预测 → z-score 标准化 → 等权集成。

        Returns DataFrame with columns:
          lgb_score, xgb_score, cat_score,
          lgb_zscore, xgb_zscore, cat_zscore,
          ensemble_score
        """
        # 提取特征矩阵
        if isinstance(X, pd.DataFrame):
            avail = [f for f in feature_names if f in X.columns]
            X_arr = X[avail].values if avail else X.values
        else:
            X_arr = np.asarray(X)

        n_samples = len(X_arr)
        results: dict[str, np.ndarray] = {}

        for name in ('lgb', 'xgb', 'cat'):
            model = self.models.get(name)
            col = f'{name}_score'
            if model is not None:
                try:
                    raw = np.asarray(model.predict(X_arr)).ravel()
                    results[col] = raw.astype(np.float64)
                except Exception as exc:
                    logger.warning(f"{name} predict failed: {exc}")
                    results[col] = np.zeros(n_samples, dtype=np.float64)
            else:
                results[col] = np.zeros(n_samples, dtype=np.float64)

        scores_df = pd.DataFrame(results)

        # z-score 标准化
        zscore_cols = {}
        for name in ('lgb', 'xgb', 'cat'):
            col = f'{name}_score'
            vals = scores_df[col].values
            sd = np.std(vals)
            if sd > 1e-10:
                zscore_cols[f'{name}_zscore'] = (vals - np.mean(vals)) / sd
            else:
                zscore_cols[f'{name}_zscore'] = np.zeros_like(vals)

        for col, vals in zscore_cols.items():
            scores_df[col] = vals

        # 等权集成
        zcols = ['lgb_zscore', 'xgb_zscore', 'cat_zscore']
        scores_df['ensemble_score'] = scores_df[zcols].mean(axis=1)

        return scores_df

    # ------------------------------------------------------------------
    # 选股
    # ------------------------------------------------------------------
    def select_stocks(
        self,
        df_with_scores: pd.DataFrame,
        top_n: int = TOP_N_STOCKS,
        exclude_st: bool = True,
    ) -> pd.DataFrame:
        """按 ensemble_score 排名选 top_n 只股票。

        自动过滤 ST / *ST / N 股票和涨停股；平局时优先市值更大者。
        """
        df = df_with_scores.copy()

        # 排除 ST / *ST / N
        if exclude_st and 'ts_code' in df.columns:
            st_mask = df['ts_code'].str.startswith(
                ('ST', '*ST', 'N', 'st', '*st', 'n')
            )
            n_st = st_mask.sum()
            if n_st:
                logger.info(f"Excluding {n_st} ST/N stocks")
                df = df[~st_mask]

        # 排除涨停（无法买入）
        if 'close' in df.columns and 'up_limit' in df.columns:
            limit_mask = (
                (df['close'] >= df['up_limit']) & (df['up_limit'] > 0)
            )
            n_limit = limit_mask.sum()
            if n_limit:
                logger.info(f"Excluding {n_limit} limit-up stocks")
                df = df[~limit_mask]

        # 排序（得分降序，市值降序处理平局）
        sort_cols = ['ensemble_score']
        ascending = [False]
        if 'float_mv' in df.columns:
            sort_cols.append('float_mv')
            ascending.append(False)

        df = df.sort_values(sort_cols, ascending=ascending)

        selected = df.head(top_n).reset_index(drop=True)
        logger.info(
            f"Selected {len(selected)} / {len(df)} candidates"
        )
        return selected

    # ------------------------------------------------------------------
    # 组合构建
    # ------------------------------------------------------------------
    def construct_portfolio(
        self,
        selected_stocks_df: pd.DataFrame,
        current_positions: dict[str, dict] | None = None,
        total_cash: float | None = None,
    ) -> dict:
        """等权构建组合，应用仓位上限，生成调仓指令。

        Parameters
        ----------
        selected_stocks_df : 选中的股票，需含 ts_code 列
        current_positions : {ts_code: {'weight': float, 'value': float}}
        total_cash : 总资金，默认 INITIAL_CASH

        Returns
        -------
        dict with keys:
          target_weights   — {ts_code: weight}
          orders           — [(ts_code, action, target_weight, target_value)]
          total_weight     — float
          cash_reserve     — float
        """
        if total_cash is None:
            total_cash = INITIAL_CASH
        current_positions = current_positions or {}

        n = len(selected_stocks_df)
        if n == 0:
            return {
                'target_weights': {},
                'orders': [],
                'total_weight': 0.0,
                'cash_reserve': total_cash,
            }

        # 等权 + 单票上限
        equal_w = 1.0 / n
        capped_w = min(equal_w, MAX_POS_PCT)
        total_w = capped_w * n

        # 总仓位超限则等比例缩减
        if total_w > TOTAL_POS_PCT:
            scale = TOTAL_POS_PCT / total_w
            capped_w *= scale
            total_w = TOTAL_POS_PCT

        # 目标权重
        target_weights: dict[str, float] = {}
        for _, row in selected_stocks_df.iterrows():
            target_weights[row['ts_code']] = capped_w

        # 生成订单
        orders: list[tuple] = []
        target_set = set(target_weights)
        current_set = set(current_positions)

        for ts_code in target_set - current_set:
            value = total_cash * capped_w
            orders.append((ts_code, 'BUY', capped_w, value))

        for ts_code in current_set - target_set:
            orders.append((ts_code, 'SELL', 0.0, 0.0))

        for ts_code in target_set & current_set:
            cur_w = current_positions[ts_code].get('weight', 0)
            diff = abs(capped_w - cur_w)
            if diff > 0.005:  # 0.5% 以上差异才调整
                value = total_cash * capped_w
                action = 'BUY' if capped_w > cur_w else 'SELL'
                orders.append((ts_code, action, capped_w, value))

        cash_reserve = total_cash * (1.0 - total_w)

        logger.info(
            f"Portfolio: {n} stocks, weight {total_w:.1%}, "
            f"cash ¥{cash_reserve:,.0f}, {len(orders)} orders"
        )

        return {
            'target_weights': target_weights,
            'orders': orders,
            'total_weight': total_w,
            'cash_reserve': cash_reserve,
        }


# ------------------------------------------------------------------
# 一键流水线
# ------------------------------------------------------------------
def select_and_build_portfolio(
    df: pd.DataFrame, models: dict | None = None
) -> dict:
    """完整选股流水线：因子生成 → 模型评分 → 精选 Top N → 组合构建。

    Parameters
    ----------
    df : 原始行情 DataFrame（至少包含 ts_code, close 列）
    models : 可选，预训练模型 dict

    Returns
    -------
    dict with target_weights, orders, selected_stocks, ...
    """
    # 1. 因子特征生成（如未包含）
    _has_factors = any(
        col.startswith(('sm_', 'md_', 'lg_', 'inst_', 'momentum_',
                        'rsi_', 'macd_', 'hist_vol_', 'ma_deviation_'))
        for col in df.columns
    )

    if not _has_factors:
        try:
            from factor_system import generate_all_factors  # noqa: F811
            logger.info("Generating factor features ...")
            df = generate_all_factors(df)
        except ImportError:
            logger.warning(
                "factor_system not available; using raw columns as features"
            )

    # 2. 初始化
    selector = StockSelector(models_dict=models)

    # 3. 特征列选择（排除所有原始数据库字段，仅保留衍生因子）
    exclude = {
        # Metadata / label
        'ts_code', 'date', 'label', 'future_return', 'target',
        # Raw price/volume (stock_daily)
        'pre_close', 'close', 'change', 'pct_chg', 'amount', 'volume',
        'up_limit', 'down_limit', 'open', 'high', 'low',
        # Raw fundamental/market (stock_daily_basic)
        'pe', 'pe_ttm', 'pb', 'total_mv', 'float_mv',
        'turnover_rate', 'volume_ratio',
        # Raw moneyflow volume (stock_moneyflow)
        'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
        'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',
        'net_mf_amount',
        # Non-predictive
        'adj_factor',
    }
    meta_cols = [c for c in ['ts_code', 'close', 'up_limit', 'float_mv',
                              'total_mv', 'pre_close'] if c in df.columns]

    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith('_')
    ]
    logger.info(f"Using {len(feature_cols)} features for prediction")

    # 4. 评分
    scores_df = selector.predict_scores(df, feature_cols)

    # 5. 合并
    df_scored = pd.concat(
        [df[meta_cols].reset_index(drop=True),
         scores_df.reset_index(drop=True)], axis=1
    )
    if 'ts_code' not in df_scored.columns and 'ts_code' in df.columns:
        df_scored['ts_code'] = df['ts_code'].values

    # 6. 选股
    selected = selector.select_stocks(df_scored, top_n=TOP_N_STOCKS)

    # 7. 组合
    portfolio = selector.construct_portfolio(selected, total_cash=INITIAL_CASH)
    portfolio['selected_stocks'] = selected

    return portfolio

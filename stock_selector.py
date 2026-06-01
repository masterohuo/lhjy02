"""
stock_selector.py - lhjy02 三模型集成选股与组合构建模块
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ATR_STOP_MULTIPLIER, DAILY_MAX_LOSS, INITIAL_CASH, MAX_HOLD_DAYS,
    MAX_POS_PCT, MAX_POSITIONS, MODELS_DIR, STOP_LOSS, TOP_N_STOCKS,
    TOTAL_POS_PCT, TRAILING_STOP,
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
            logger.warning(f"止损检查: 缺少列 {missing}")
            return []

        pnl = (
            (positions_df['current_price'] - positions_df['cost_price'])
            / positions_df['cost_price']
        )
        stopped = positions_df.loc[pnl <= STOP_LOSS, 'ts_code'].tolist()
        if stopped:
            logger.warning(f"触发止损: {stopped}")
        return stopped

    def check_daily_loss(self, current_value: float, initial_value: float) -> bool:
        """日内亏损是否超过 DAILY_MAX_LOSS。"""
        if initial_value <= 0:
            return False
        return (current_value - initial_value) / initial_value <= DAILY_MAX_LOSS

    def check_trailing_stop(
        self, positions: dict, position_highs: dict, price_map: dict
    ) -> list:
        """移动止损：从持仓期间最高价回落超过TRAILING_STOP → 卖出。

        Returns 触发移动止损的股票列表。
        """
        stopped = []
        for ts_code in positions:
            high = position_highs.get(ts_code)
            current = price_map.get(ts_code)
            if high and current and high > 0:
                if (current - high) / high <= TRAILING_STOP:
                    stopped.append(ts_code)
        if stopped:
            logger.warning("移动止损触发: %s", stopped)
        return stopped

    def check_time_stop(
        self, positions: dict, entry_dates: dict,
        current_date, price_map: dict
    ) -> list:
        """时间止损：持仓超过MAX_HOLD_DAYS且浮亏 → 减半仓。

        Returns 触发时间止损的股票列表。
        """
        stopped = []
        for ts_code, pos in positions.items():
            entry_date = entry_dates.get(ts_code)
            if entry_date is None:
                continue
            hold_days = (current_date - entry_date).days
            if hold_days > MAX_HOLD_DAYS:
                current_price = price_map.get(ts_code, pos["cost_price"])
                if current_price > 0 and (
                    (current_price - pos["cost_price"]) / pos["cost_price"] < 0
                ):
                    stopped.append(ts_code)
        if stopped:
            logger.warning("时间止损触发: %s", stopped)
        return stopped

    def check_atr_stop(
        self, positions: dict, day_data, price_map: dict
    ) -> list:
        """ATR波动止损：当前价 <= 成本价 - ATR_STOP_MULTIPLIER × ATR(14)。
        没有ATR数据时跳过，返回空列表。
        """
        # 查找ATR列
        atr_cols = ['atr14', 'ATR14', 'atr_14', 'atr']
        atr_col = None
        for c in atr_cols:
            if c in day_data.columns:
                atr_col = c
                break
        if atr_col is None:
            return []

        atr_map = dict(zip(day_data["ts_code"], day_data[atr_col]))
        stopped = []
        for ts_code, pos in positions.items():
            atr_val = atr_map.get(ts_code)
            if atr_val is None or (isinstance(atr_val, float) and np.isnan(atr_val)):
                continue
            if atr_val <= 0:
                continue
            stop_price = pos["cost_price"] - ATR_STOP_MULTIPLIER * atr_val
            current_price = price_map.get(ts_code, pos["cost_price"])
            if current_price <= stop_price:
                stopped.append(ts_code)
        if stopped:
            logger.warning("ATR止损触发: %s", stopped)
        return stopped

    def validate_positions(self, target_positions: dict) -> tuple:
        """验证目标持仓：单票 ≤20%，总仓位 ≤80%。

        Returns (passed: bool, reason: str).
        """
        if not target_positions:
            return True, "空仓"

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
                    logger.info(f"已加载 {name} 模型: {path}")
                except Exception as exc:
                    logger.warning(f"加载 {name} 失败: {exc}")
            else:
                logger.warning(f"模型文件不存在: {path}")
        return models

    # ------------------------------------------------------------------
    # 预测评分
    # ------------------------------------------------------------------
    def _get_model_feature_names(self) -> list[str] | None:
        """Try to extract feature names from loaded models."""
        for model in self.models.values():
            if model is not None:
                for attr in ('feature_name_', 'feature_names_in_', 'feature_names_'):
                    if hasattr(model, attr):
                        names = getattr(model, attr)
                        if names is not None and len(names) > 0:
                            return list(names)
        return None

    def predict_scores(
        self, X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """三模型预测 → z-score 标准化 → 等权集成。

        Returns DataFrame with columns:
          lgb_score, xgb_score, cat_score,
          lgb_zscore, xgb_zscore, cat_zscore,
          ensemble_score
        """
        # Resolve feature_names: explicit arg > stored > model-inferred > numeric fallback
        if feature_names is None:
            if hasattr(self, 'feature_names_') and self.feature_names_:
                feature_names = self.feature_names_
            else:
                feature_names = self._get_model_feature_names()
        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = [c for c in X.columns
                           if pd.api.types.is_numeric_dtype(X[c])
                           and c not in {'ts_code', 'date', 'label'}]

        # Extract aligned feature matrix (fill missing columns with 0)
        if isinstance(X, pd.DataFrame) and feature_names:
            avail = [f for f in feature_names if f in X.columns]
            missing = [f for f in feature_names if f not in X.columns]
            if missing:
                logger.warning("缺失 %d 个特征, 用0填充", len(missing))
            X_arr = X[avail].values if avail else X.values
            if missing:
                X_arr = np.hstack([X_arr, np.zeros((len(X_arr), len(missing)))])
        elif isinstance(X, pd.DataFrame):
            X_arr = X.values
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
                    logger.warning(f"{name} 预测失败: {exc}")
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

        # 三模型共识加权
        # 1. 每个模型独立排名（百分位，越小越好）
        for name in ('lgb', 'xgb', 'cat'):
            col = f'{name}_score'
            scores_df[f'{name}_rank'] = scores_df[col].rank(ascending=False, pct=True)

        # 2. 统计每个股票在几个模型的前20%
        top20_threshold = 0.20
        in_top20 = (
            (scores_df['lgb_rank'] <= top20_threshold).astype(int)
            + (scores_df['xgb_rank'] <= top20_threshold).astype(int)
            + (scores_df['cat_rank'] <= top20_threshold).astype(int)
        )

        # 3. 共识权重映射: 3模型→1.5, 2模型→1.2, 1模型→0.8, 0模型→0.6
        consensus_map = {3: 1.5, 2: 1.2, 1: 0.8, 0: 0.6}
        scores_df['consensus_weight'] = in_top20.map(consensus_map)

        # 4. 最终得分 = 集成得分 × 共识权重
        scores_df['final_score'] = scores_df['ensemble_score'] * scores_df['consensus_weight']

        return scores_df

    # ------------------------------------------------------------------
    # 选股
    # ------------------------------------------------------------------
    def select_stocks(
        self,
        df_with_scores: pd.DataFrame,
        top_n: int = TOP_N_STOCKS,
        exclude_st: bool = True,
        prefilter: bool = True,
    ) -> pd.DataFrame:
        """按 ensemble_score 排名选 top_n 只股票。

        自动过滤 ST / *ST / N 股票和涨停股；平局时优先市值更大者。
        """
        df = df_with_scores.copy()

        # 三层漏斗预筛选
        if prefilter:
            from universe import StockUniverse
            from config import UNIVERSE_CONFIG
            uconfig = {**UNIVERSE_CONFIG, "exclude_limit_board": True, "top_n": len(df)}
            universe = StockUniverse(**uconfig)
            df = universe.filter(df)

        # 排除 ST / *ST / N
        if exclude_st and 'ts_code' in df.columns:
            st_mask = df['ts_code'].str.startswith(
                ('ST', '*ST', 'N', 'st', '*st', 'n')
            )
            n_st = st_mask.sum()
            if n_st:
                logger.info(f"排除 {n_st} 只ST/N股票")
                df = df[~st_mask]

        # 排除涨停（无法买入）
        if 'close' in df.columns and 'up_limit' in df.columns:
            limit_mask = (
                (df['close'] >= df['up_limit']) & (df['up_limit'] > 0)
            )
            n_limit = limit_mask.sum()
            if n_limit:
                logger.info(f"排除 {n_limit} 只涨停股票")
                df = df[~limit_mask]

        # 排序（final_score优先，得分降序，市值降序处理平局）
        if 'final_score' in df.columns:
            score_col = 'final_score'
        else:
            score_col = 'ensemble_score'

        sort_cols = [score_col]
        ascending = [False]
        if 'float_mv' in df.columns:
            sort_cols.append('float_mv')
            ascending.append(False)

        df = df.sort_values(sort_cols, ascending=ascending)

        selected = df.head(top_n).reset_index(drop=True)
        logger.info(
            f"已选股 {len(selected)} / {len(df)} 只"
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

        # 等权 + 共识仓位上限
        equal_w = 1.0 / n

        # 共识仓位映射: 三模型共识→25%, 两模型→20%, 单模型→15%, 无共识→10%
        consensus_caps = {1.5: 0.25, 1.2: 0.20, 0.8: 0.15, 0.6: 0.10}
        has_consensus = 'consensus_weight' in selected_stocks_df.columns

        # 每只股票的目标权重（等权基础 + 共识上限裁剪）
        target_weights: dict[str, float] = {}
        for _, row in selected_stocks_df.iterrows():
            ts = row['ts_code']
            if has_consensus:
                cw = float(row.get('consensus_weight', 1.0))
                cap = consensus_caps.get(cw, MAX_POS_PCT)
            else:
                cap = MAX_POS_PCT
            target_weights[ts] = min(equal_w, cap)

        total_w = sum(target_weights.values())

        # 总仓位超限则等比例缩减
        if total_w > TOTAL_POS_PCT:
            scale = TOTAL_POS_PCT / total_w
            target_weights = {k: v * scale for k, v in target_weights.items()}
            total_w = TOTAL_POS_PCT

        # 生成订单
        orders: list[tuple] = []
        target_set = set(target_weights)
        current_set = set(current_positions)

        for ts_code in target_set - current_set:
            tw = target_weights[ts_code]
            orders.append((ts_code, 'BUY', tw, total_cash * tw))

        for ts_code in current_set - target_set:
            orders.append((ts_code, 'SELL', 0.0, 0.0))

        for ts_code in target_set & current_set:
            tw = target_weights[ts_code]
            cur_w = current_positions[ts_code].get('weight', 0)
            diff = abs(tw - cur_w)
            if diff > 0.005:
                action = 'BUY' if tw > cur_w else 'SELL'
                orders.append((ts_code, action, tw, total_cash * tw))

        cash_reserve = total_cash * (1.0 - total_w)

        logger.info(
            f"组合: {n}只, 仓位{total_w:.1%}, "
            f"现金¥{cash_reserve:,.0f}, {len(orders)}笔订单"
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
            logger.info("生成因子特征 ...")
            df = generate_all_factors(df)
        except ImportError:
            logger.warning(
                "因子系统不可用; 使用原始列作为特征"
            )

    # 2. 初始化
    selector = StockSelector(models_dict=models)

    # 3. 特征列选择
    # 优先使用模型存储的特征名（与训练时一致），确保IC筛选后预测特征对齐
    model_features = selector._get_model_feature_names()
    if model_features:
        feature_cols = [c for c in model_features if c in df.columns]
        logger.info("使用模型特征 %d 个进行预测 (与训练时对齐)", len(feature_cols))
    else:
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
        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and not c.startswith('_')
        ]
        logger.info(f"使用 %d 个特征进行预测", len(feature_cols))

    meta_cols = [c for c in ['ts_code', 'close', 'up_limit', 'float_mv',
                              'total_mv', 'pre_close'] if c in df.columns]

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

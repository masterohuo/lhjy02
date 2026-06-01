"""
model_trainer.py - lhjy02 3-model ensemble training system.

Trains LightGBM (regression), XGBoost (reg:squarederror), and CatBoost (RMSE)
regression models for stock return prediction and combines them into an ensemble.
"""
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

from config import (
    MODELS_DIR, LGBM_PARAMS, XGB_PARAMS, CAT_PARAMS,
    MAX_STOCKS, PREDICT_HORIZON, UNIVERSE_CONFIG,
)
from data_loader import load_all_tables
from factor_system import generate_all_factors
from universe import StockUniverse

logger = logging.getLogger(__name__)


class TriModelTrainer:
    """3-model ensemble trainer for stock return prediction (regression)."""

    def __init__(self):
        self.lgb_model = None
        self.xgb_model = None
        self.cat_model = None
        self.feature_names_ = None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def feature_names(self):
        return getattr(self, 'feature_names_', None)

    def prepare_data(self, start_date=None, end_date=None, max_stocks=2000):
        """Load data, generate factors, create labels, and clean features."""
        df = load_all_tables(start_date=start_date, end_date=end_date, include_basic_info=True)
        if df.empty:
            raise ValueError("No data loaded from database.")

        df = generate_all_factors(df)

        # Create label: future 5-day return per stock
        df = df.sort_values(["ts_code", "date"])
        df["label"] = df.groupby("ts_code")["close"].transform(
            lambda x: (x.shift(-PREDICT_HORIZON) - x) / x
        )

        # Extract metadata as numpy arrays AFTER sort but BEFORE dropna
        ts_arr = df['ts_code'].values.copy()
        date_arr = df['date'].values.copy()

        # Define feature columns (exclude meta, raw DB fields, forward_ret, internal)
        meta_set = {"ts_code", "date", "label"}

        # Raw price/volume fields from stock_daily
        price_set = {"open", "high", "low", "pre_close", "close", "change",
                     "pct_chg", "amount", "volume", "up_limit", "down_limit"}

        # Raw fundamental/market fields from stock_daily_basic
        fundamental_set = {"pe", "pe_ttm", "pb", "total_mv", "float_mv",
                           "turnover_rate", "volume_ratio"}

        # Raw moneyflow volume fields from stock_moneyflow
        moneyflow_set = {"buy_sm_vol", "sell_sm_vol", "buy_md_vol", "sell_md_vol",
                         "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol",
                         "net_mf_amount"}

        # Non-predictive raw columns (stock_basic metadata when include_basic_info=True)
        other_set = {"adj_factor", "industry", "list_date", "name", "area",
                     "symbol", "cnspell", "market", "act_name", "act_ent_type"}

        exclude_set = (meta_set | price_set | fundamental_set | moneyflow_set | other_set)

        feature_cols = [c for c in df.columns
                        if c not in exclude_set
                        and not c.startswith('forward_ret_')
                        and not c.startswith('_')
                        and pd.api.types.is_numeric_dtype(df[c])]

        self.feature_names_ = feature_cols

        # Drop NaN across features and label together so X, y stay aligned
        df = df.dropna(subset=feature_cols + ["label"]).copy()

        # Three-tier funnel stock universe filter
        if max_stocks is not None and max_stocks > 0:
            uconfig = {**UNIVERSE_CONFIG, "top_n": max_stocks}
            universe = StockUniverse(**uconfig)
            n_before = len(df)
            df = universe.filter(df)
            n_after = len(df)
            logger.info("Stock universe: %d -> %d (filtered %.1f%%)",
                       n_before, n_after, (1 - n_after / max(n_before, 1)) * 100)

        # Build meta from remaining rows' index (post-dropna)
        remaining_idx = df.index
        meta = pd.DataFrame({
            'ts_code': ts_arr[list(remaining_idx)],
            'date': pd.to_datetime(date_arr[list(remaining_idx)])
        }).reset_index(drop=True)

        y = df["label"].reset_index(drop=True)
        X = df[feature_cols].reset_index(drop=True)

        # Fill any remaining NaN in features (safety net)
        X = X.fillna(X.median())
        X = X.fillna(0.0)

        # Clip extreme values at 99.5th percentile
        for col in X.columns:
            upper = X[col].quantile(0.995)
            lower = X[col].quantile(0.005)
            X[col] = X[col].clip(lower, upper)

        return X, y, meta, feature_cols

    @staticmethod
    def _cpu_params(params, gpu_keys):
        """Return a copy of params with GPU-related keys removed."""
        return {k: v for k, v in params.items() if k not in gpu_keys}

    def train_lightgbm(self, X_train, y_train, X_val, y_val, feature_names):
        """Train LightGBM regressor with early stopping on validation set."""
        params = {k: v for k, v in LGBM_PARAMS.items() if k not in ("n_estimators",)}
        n_estimators = LGBM_PARAMS.get("n_estimators", 1000)

        # Validate objective is regression-compatible (LGBMRegressor rejects ranking)
        _ranking_objectives = {"lambdarank", "rank_xendcg", "rank_xendcg_map"}
        obj = params.get("objective", "")
        if obj in _ranking_objectives or obj.startswith("rank_"):
            logger.warning("LGBM_PARAMS objective='%s' 不兼容 LGBMRegressor，自动修正为 'regression'", obj)
            params["objective"] = "regression"

        try:
            self.lgb_model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, verbose=-1)
            self.lgb_model.fit(
                X_train[feature_names], y_train,
                eval_set=[(X_val[feature_names], y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
        except lgb.basic.LightGBMError:
            logger.info("LightGBM GPU不可用，回退到CPU...")
            cpu_params = self._cpu_params(params, {"device", "gpu_platform_id", "gpu_device_id"})
            self.lgb_model = lgb.LGBMRegressor(**cpu_params, n_estimators=n_estimators, verbose=-1)
            self.lgb_model.fit(
                X_train[feature_names], y_train,
                eval_set=[(X_val[feature_names], y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

        with open(MODELS_DIR / "lgb_model.pkl", "wb") as f:
            pickle.dump(self.lgb_model, f)

        return self.lgb_model

    def train_xgboost(self, X_train, y_train, X_val, y_val, feature_names):
        """Train XGBoost regressor with early stopping on validation set."""
        params = {k: v for k, v in XGB_PARAMS.items() if k not in ("n_estimators",)}
        n_estimators = XGB_PARAMS.get("n_estimators", 1000)

        try:
            self.xgb_model = XGBRegressor(**params, n_estimators=n_estimators,
                                           early_stopping_rounds=50, verbosity=0)
            self.xgb_model.fit(
                X_train[feature_names], y_train,
                eval_set=[(X_val[feature_names], y_val)],
                verbose=False,
            )
        except Exception:
            logger.info("XGBoost GPU不可用，回退到CPU (hist)...")
            cpu_params = self._cpu_params(params, {"tree_method", "gpu_id"})
            cpu_params["tree_method"] = "hist"
            self.xgb_model = XGBRegressor(**cpu_params, n_estimators=n_estimators,
                                           early_stopping_rounds=50, verbosity=0)
            self.xgb_model.fit(
                X_train[feature_names], y_train,
                eval_set=[(X_val[feature_names], y_val)],
                verbose=False,
            )

        with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
            pickle.dump(self.xgb_model, f)

        return self.xgb_model

    def train_catboost(self, X_train, y_train, X_val, y_val, feature_names):
        """Train CatBoost regressor with RMSE loss and early stopping."""
        params = {k: v for k, v in CAT_PARAMS.items()
                  if k not in ("iterations", "early_stopping_rounds")}
        iterations = CAT_PARAMS.get("iterations", 1000)
        early_stopping_rounds = CAT_PARAMS.get("early_stopping_rounds", 50)

        train_pool = Pool(X_train[feature_names], y_train)
        val_pool = Pool(X_val[feature_names], y_val)

        try:
            self.cat_model = CatBoostRegressor(**params, iterations=iterations, verbose=0)
            self.cat_model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        except Exception:
            logger.info("CatBoost GPU不可用，回退到CPU...")
            cpu_params = self._cpu_params(params, {"task_type", "devices"})
            self.cat_model = CatBoostRegressor(**cpu_params, iterations=iterations, verbose=0)
            self.cat_model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )

        with open(MODELS_DIR / "cat_model.pkl", "wb") as f:
            pickle.dump(self.cat_model, f)

        return self.cat_model

    def train_all(self, start_date=None, end_date=None, val_ratio=0.2):
        """完整训练流程：数据准备 → 时间切分 → 训练三模型。"""
        logger.info("开始训练三模型集成系统...")
        X, y, meta, feature_names = self.prepare_data(
            start_date=start_date, end_date=end_date
        )
        logger.info("数据准备完成: %d行, %d个特征", len(X), len(feature_names))

        # Temporal split: last val_ratio of dates for validation
        dates_sorted = meta["date"].sort_values().unique()
        split_idx = int(len(dates_sorted) * (1 - val_ratio))
        train_dates = set(dates_sorted[:split_idx])
        val_dates = set(dates_sorted[split_idx:])

        logger.info("训练集日期: %s ~ %s (%d天)",
                    dates_sorted[0].strftime('%Y-%m-%d'),
                    dates_sorted[split_idx - 1].strftime('%Y-%m-%d'),
                    len(train_dates))
        logger.info("验证集日期: %s ~ %s (%d天)",
                    dates_sorted[split_idx].strftime('%Y-%m-%d'),
                    dates_sorted[-1].strftime('%Y-%m-%d'),
                    len(val_dates))

        train_mask = meta["date"].isin(train_dates).values
        val_mask = meta["date"].isin(val_dates).values

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_val = X.loc[val_mask]
        y_val = y.loc[val_mask]

        models = {}

        logger.info("训练 LightGBM (regression)...")
        models["lgb"] = self.train_lightgbm(
            X_train, y_train, X_val, y_val, feature_names
        )
        best = getattr(models["lgb"], 'best_iteration_', None)
        lgb_iters = best + 1 if best is not None else getattr(models["lgb"], 'n_iterations_', models["lgb"].n_estimators)
        logger.info("✅ LightGBM 训练完成, 迭代次数: %d", lgb_iters)

        logger.info("训练 XGBoost (reg:squarederror)...")
        models["xgb"] = self.train_xgboost(
            X_train, y_train, X_val, y_val, feature_names
        )
        best = getattr(models["xgb"], 'best_iteration', None)
        xgb_iters = best + 1 if best is not None else models["xgb"].n_estimators
        logger.info("✅ XGBoost 训练完成, 迭代次数: %d", xgb_iters)

        logger.info("训练 CatBoost (RMSE)...")
        models["cat"] = self.train_catboost(
            X_train, y_train, X_val, y_val, feature_names
        )
        if hasattr(models["cat"], 'get_best_iteration'):
            cat_iters = models["cat"].get_best_iteration() + 1
        else:
            cat_iters = getattr(models["cat"], 'tree_count_', 0)
        logger.info("✅ CatBoost 训练完成, 迭代次数: %d", cat_iters)

        return models

    def save_models(self, models_dict):
        """Persist all models to MODELS_DIR."""
        for name, model in models_dict.items():
            with open(MODELS_DIR / f"{name}_model.pkl", "wb") as f:
                pickle.dump(model, f)

    def load_models(self):
        """Load persisted models from MODELS_DIR. Returns dict of name -> model."""
        models = {}
        for name in ("lgb", "xgb", "cat"):
            path = MODELS_DIR / f"{name}_model.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
        return models

    def analyze_factor_importance(self, models_dict, feature_names, top_n=30):
        """Compute normalized factor importance across all trained models.

        Returns DataFrame with columns:
            factor, lgb_importance, xgb_importance, cat_importance, mean_importance
        """
        records = []

        for name, model in models_dict.items():
            if name == "lgb":
                imp = model.feature_importances_
            elif name == "xgb":
                imp = model.feature_importances_
            elif name == "cat":
                imp = model.get_feature_importance()
            else:
                continue

            imp = np.array(imp, dtype=float)
            if imp.max() > imp.min():
                imp = (imp - imp.min()) / (imp.max() - imp.min())
            else:
                imp = np.zeros_like(imp)

            records.append((name, imp))

        df_imp = pd.DataFrame({"factor": feature_names})
        for name, imp in records:
            df_imp[f"{name}_importance"] = imp

        imp_cols = [f"{name}_importance" for name, _ in records]
        df_imp["mean_importance"] = df_imp[imp_cols].mean(axis=1)
        df_imp = df_imp.sort_values("mean_importance", ascending=False)
        df_imp = df_imp.head(top_n).reset_index(drop=True)

        return df_imp


# ============================================================
# 因子中文描述
# ============================================================
FACTOR_DESCRIPTIONS = {
    # A类：散户对手盘
    "sm_net_vol": "小单净流入占比",
    "md_net_vol": "中单净流入占比",
    "lg_net_vol": "大单净流入占比",
    "elg_net_vol": "特大单净流入占比",
    "inst_net_vol": "机构净流入占比",
    "retail_net_vol": "散户净流入",
    "inst_retail_ratio": "机构散户比",
    "inst_md_ratio": "机构中单比",
    "sm_buy_ratio": "小单买入比例",
    "sm_sell_ratio": "小单卖出比例",
    "md_buy_ratio": "中单买入比例",
    "md_sell_ratio": "中单卖出比例",
    "lg_buy_ratio": "大单买入比例",
    "lg_sell_ratio": "大单卖出比例",
    "elg_buy_ratio": "特大单买入比例",
    "elg_sell_ratio": "特大单卖出比例",
    "sm_imbalance": "小单不平衡度",
    "inst_imbalance": "机构不平衡度",
    "net_mf_amount_norm": "资金流归一化",
    "retail_participation": "散户参与度",
    # B类：资金流
    "lg_net_vol_5d": "大单净流入5日均值",
    "lg_net_vol_10d": "大单净流入10日均值",
    "lg_net_vol_20d": "大单净流入20日均值",
    "inst_net_vol_5d": "机构净流入5日均值",
    "inst_net_vol_10d": "机构净流入10日均值",
    "inst_net_vol_20d": "机构净流入20日均值",
    "net_mf_amount_5d": "资金流5日均值",
    "net_mf_amount_10d": "资金流10日均值",
    "net_mf_amount_20d": "资金流20日均值",
    "flow_persistence": "资金流持续性",
    "flow_acceleration": "资金流加速度",
    "flow_price_divergence": "量价背离",
    "lg_buy_sell_ratio": "大单买卖比",
    "inst_accumulation_momentum": "机构积累动量",
    "flow_volatility": "资金流波动率",
    "lg_concentration": "大单集中度",
    "net_mf_amount_ma5_ratio": "资金流MA5偏离",
    "inst_flow_stability": "机构流稳定性",
    "smart_money_5d": "聪明钱5日",
    "fund_reversal_signal": "资金反转信号",
    # C类：技术因子
    "momentum_5d": "5日动量",
    "momentum_10d": "10日动量",
    "momentum_20d": "20日动量",
    "momentum_60d": "60日动量",
    "ma_deviation_5": "MA5偏离度",
    "ma_deviation_10": "MA10偏离度",
    "ma_deviation_20": "MA20偏离度",
    "ma_deviation_60": "MA60偏离度",
    "volume_ratio_5": "5日量比",
    "volume_ratio_20": "20日量比",
    "rsi_14": "RSI(14)",
    "macd_dif": "MACD-DIF",
    "macd_dea": "MACD-DEA",
    "macd_hist": "MACD柱",
    "bollinger_position": "布林带位置",
    "bollinger_width": "布林带宽度",
    "kdj_k": "KDJ-K值",
    "kdj_d": "KDJ-D值",
    "kdj_j": "KDJ-J值",
    "obv": "能量潮",
    "atr_14": "ATR(14)",
    "mfi_14": "MFI(14)",
    "wr_14": "威廉指标(14)",
    "cci_20": "CCI(20)",
    "roc_10": "ROC(10)",
    "amplitude_5": "5日振幅均值",
    "ema12_ema26_ratio": "EMA12/EMA26",
    "ppo": "PPO",
    "cmf_20": "CMF(20)",
    "ma5_ma20_ratio": "MA5/MA20",
    "pvt": "量价趋势",
    "force_index_13": "力量指数",
    "trix_15": "TRIX(15)",
    "dpo_20": "DPO(20)",
    "psy_12": "PSY(12)",
    "prev_close": "前收盘价",
    "close": "收盘价",
    "high": "最高价",
    "low": "最低价",
    # D类：基本面
    "pe_percentile_1y": "PE年度百分位",
    "pb_percentile_1y": "PB年度百分位",
    "log_total_mv": "对数总市值",
    "log_float_mv": "对数流通市值",
    "turnover_rate_5d": "5日换手率",
    "turnover_rate_20d": "20日换手率",
    "volume_ratio": "量比",
    "pe_zscore": "PE截面Z分数",
    "pb_zscore": "PB截面Z分数",
    "mv_turnover_ratio": "市值成交额比",
    # E类：趋势/反转
    "adx_14": "ADX(14)",
    "di_plus_14": "+DI(14)",
    "di_minus_14": "-DI(14)",
    "ma5_cross_ma20": "MA5/MA20交叉",
    "ma10_cross_ma60": "MA10/MA60交叉",
    "price_vs_20d_high": "价格vs20日高",
    "price_vs_20d_low": "价格vs20日低",
    "consecutive_up": "连涨天数",
    "consecutive_down": "连跌天数",
    "gap_ratio": "缺口比率",
    "ma_slope_20": "MA20斜率",
    "higher_high_20": "突破20日高",
    "lower_low_20": "跌破20日低",
    "inside_day": "内包日",
    "hist_vol_5": "5日历史波动率",
    "hist_vol_10": "10日历史波动率",
    "hist_vol_20": "20日历史波动率",
    "hist_vol_60": "60日历史波动率",
    "volume_ratio_5": "5日量比",
    "volume_ratio_20": "20日量比",
    "kdj_d": "KDJ-D值",
    "consecutive_down": "连跌天数",
    "md_buy_ratio": "中单买入比例",
    "md_sell_ratio": "中单卖出比例",
    "nr7": "NR7窄幅",
    # F类：波动率/风险
    "hist_vol_5d": "5日历史波动率",
    "hist_vol_10d": "10日历史波动率",
    "hist_vol_20d": "20日历史波动率",
    "hist_vol_60d": "60日历史波动率",
    "downside_vol_20": "20日下行波动率",
    "max_dd_20": "20日最大回撤",
    "max_dd_60": "60日最大回撤",
    "skewness_20": "20日偏度",
    "kurtosis_20": "20日峰度",
    "beta_60": "Beta(60)",
    "var_20": "VaR(95%,20)",
    "ulcer_index_20": "溃疡指数(20)",
    "turnover_rate": "换手率",
    "total_mv": "总市值",
    "float_mv": "流通市值",
    "pe": "市盈率",
    "pb": "市净率",
    # G类：行为金融
    "disposition_effect": "处置效应",
    "anchoring_bias": "锚定偏差",
    "herding_intensity": "羊群效应",
    "overreaction": "过度反应",
    "attention_proxy": "关注度代理",
    "info_response_speed": "信息响应速度",
    "retail_panic_greed": "散户恐慌贪婪",
    "sentiment_momentum": "情绪动量",
}


def train(start_date=None, end_date=None):
    """运行完整训练流程，返回训练好的模型字典。"""
    trainer = TriModelTrainer()
    return trainer.train_all(start_date=start_date, end_date=end_date)


# ============================================================
# CLI 入口
# ============================================================
def main():
    """命令行入口：python model_trainer.py [--start-date YYYYMMDD] [--end-date YYYYMMDD]"""
    parser = argparse.ArgumentParser(
        description="lhjy02 三模型集成训练（LightGBM + XGBoost + CatBoost）"
    )
    parser.add_argument("--start-date", type=str, default=None,
                        help="训练起始日期，如 20210101（默认：5年前）")
    parser.add_argument("--end-date", type=str, default=None,
                        help="训练截止日期，如 20260501（默认：今天）")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("lhjy02 三模型集成训练")
    logger.info("=" * 60)

    models = train(start_date=args.start_date, end_date=args.end_date)

    logger.info("=" * 60)
    logger.info("训练完成！已保存模型到: %s", MODELS_DIR)
    for name, model in models.items():
        logger.info("  %s: %s", name, type(model).__name__)
    logger.info("=" * 60)

    # 输出因子重要性
    try:
        trainer = TriModelTrainer()
        X, y, meta, feature_names = trainer.prepare_data(
            start_date=args.start_date, end_date=args.end_date
        )
        importance = trainer.analyze_factor_importance(models, feature_names, top_n=20)
        logger.info("\n" + "=" * 60)
        logger.info("Top 20 因子重要性:")
        logger.info("%-4s  %-8s  %-20s  %s", "排名", "重要性", "因子", "说明")
        logger.info("-" * 60)
        for rank, (_, row) in enumerate(importance.iterrows(), 1):
            desc = FACTOR_DESCRIPTIONS.get(row['factor'], row['factor'])
            logger.info("%-4d  %.4f    %-20s  %s",
                        rank, row["mean_importance"], row["factor"], desc)
        logger.info("=" * 60)
    except Exception as e:
        logger.warning("因子重要性分析失败: %s", e)


if __name__ == "__main__":
    main()

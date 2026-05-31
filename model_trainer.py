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
    MAX_STOCKS, PREDICT_HORIZON,
)
from data_loader import load_all_tables
from factor_system import generate_all_factors

logger = logging.getLogger(__name__)


class TriModelTrainer:
    """3-model ensemble trainer for stock return prediction (regression)."""

    def __init__(self):
        self.lgb_model = None
        self.xgb_model = None
        self.cat_model = None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, start_date=None, end_date=None, max_stocks=2000):
        """Load data, generate factors, create labels, and clean features."""
        df = load_all_tables(start_date=start_date, end_date=end_date)
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

        # Define feature columns (exclude meta, price-only, forward_ret)
        meta_set = {"ts_code", "date", "label"}
        price_set = {"open", "high", "low", "up_limit", "down_limit"}
        feature_cols = [c for c in df.columns
                        if c not in meta_set and c not in price_set
                        and not c.startswith('forward_ret_')]

        # Drop NaN across features and label together so X, y stay aligned
        df = df.dropna(subset=feature_cols + ["label"]).copy()

        # Limit stocks per date
        if max_stocks is not None and max_stocks > 0:
            df = df.groupby("date", group_keys=False).apply(
                lambda g: g.head(max_stocks)
            )

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

    def train_lightgbm(self, X_train, y_train, X_val, y_val, feature_names):
        """Train LightGBM regressor with early stopping on validation set."""
        params = {k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"}
        n_estimators = LGBM_PARAMS.get("n_estimators", 1000)

        self.lgb_model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, verbose=-1)
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

        self.xgb_model = XGBRegressor(**params, n_estimators=n_estimators,
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

        self.cat_model = CatBoostRegressor(**params, iterations=iterations, verbose=0)
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
        lgb_iters = getattr(models["lgb"], 'n_iterations_', models["lgb"].n_estimators)
        logger.info("✅ LightGBM 训练完成, 迭代次数: %d", lgb_iters)

        logger.info("训练 XGBoost (reg:squarederror)...")
        models["xgb"] = self.train_xgboost(
            X_train, y_train, X_val, y_val, feature_names
        )
        xgb_iters = getattr(models["xgb"], 'best_iteration', models["xgb"].n_estimators)
        logger.info("✅ XGBoost 训练完成, 迭代次数: %d", xgb_iters)

        logger.info("训练 CatBoost (RMSE)...")
        models["cat"] = self.train_catboost(
            X_train, y_train, X_val, y_val, feature_names
        )
        cat_iters = getattr(models["cat"], 'best_iteration_', getattr(models["cat"], 'iteration_count_', 0))
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
        logger.info("\nTop 20 因子重要性:")
        for _, row in importance.iterrows():
            logger.info("  %.4f  %s", row["mean_importance"], row["factor"])
    except Exception as e:
        logger.warning("因子重要性分析失败: %s", e)


if __name__ == "__main__":
    main()

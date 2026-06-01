# Task: Fix feature alignment + exclude STAR/BSE from universe

## Fix 1: Feature column alignment between training and prediction

### Problem
Training selects `feature_cols` from training data. Prediction uses a different set of `feature_cols` from test data. Column order/names can differ, causing the model to predict with wrong features.

### Solution: Store and reuse training feature names

#### A) `model_trainer.py` — store feature_names after training

In `TriModelTrainer.__init__()`, add:
```python
self.feature_names_ = None
```

In `TriModelTrainer.prepare_data()`, after `feature_cols` is determined, store it:
```python
self.feature_names_ = feature_cols
```

In `TriModelTrainer.train_all()`, after prepare_data returns feature_names, also store it (already done via prepare_data).

#### B) `stock_selector.py` — align features during prediction

In `StockSelector.predict_scores()`, modify the feature extraction logic:

```python
def predict_scores(self, X, feature_names=None):
    # If feature_names not provided, try to get from models
    if feature_names is None:
        if hasattr(self, 'feature_names_') and self.feature_names_:
            feature_names = self.feature_names_
        elif isinstance(X, pd.DataFrame):
            # Fallback: use only numeric columns, exclude metadata
            feature_names = [c for c in X.columns 
                           if pd.api.types.is_numeric_dtype(X[c])
                           and c not in {'ts_code','date','label'}]
    
    # Extract aligned feature matrix
    if isinstance(X, pd.DataFrame):
        # Align columns to training order; fill missing with 0
        avail = [f for f in feature_names if f in X.columns]
        missing = [f for f in feature_names if f not in X.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} features, filling with 0")
        X_arr = X[avail].values if avail else X.values
        if missing:
            X_arr = np.hstack([X_arr, np.zeros((len(X_arr), len(missing)))])
    else:
        X_arr = np.asarray(X)
    
    # ... rest of method
```

#### C) `TriModelTrainer` — expose feature_names

Add a property or method so StockSelector can access it:
```python
@property
def feature_names(self):
    return getattr(self, 'feature_names_', None)
```

## Fix 2: Exclude STAR (科创板) and BSE (北交所) from stock pool

### `universe.py` — add board exclusion

In `StockUniverse.__init__()`, add:
```python
self.exclude_boards = kwargs.get('exclude_boards', [])  # e.g. ['STAR', 'BSE']
```

In `_hard_filter()`, add board exclusion step BEFORE the existing filters:
```python
# Exclude specific boards
if self.exclude_boards and 'ts_code' in df.columns:
    for board in self.exclude_boards:
        if board == 'STAR':
            df = df[~df['ts_code'].astype(str).str.startswith('688')]
        elif board == 'BSE':
            df = df[~df['ts_code'].astype(str).str.startswith(('8', '4'))]
        elif board == 'ChiNext':
            df = df[~df['ts_code'].astype(str).str.startswith(('300', '301'))]
        elif board == 'SSE':
            df = df[~df['ts_code'].astype(str).str.startswith('6')]
        elif board == 'SZSE':
            df = df[~df['ts_code'].astype(str).str.startswith('0')]
```

### `config.py` — update UNIVERSE_CONFIG

```python
UNIVERSE_CONFIG = {
    "min_daily_amount": 50_000,
    "min_list_days": 60,
    "exclude_st": True,
    "exclude_limit_board": False,
    "top_n": 2000,
    "use_stratified": False,
    "stratified_min_per_sector": 20,
    "exclude_boards": ["STAR", "BSE"],  # 排除科创板和北交所
}
```

## Fix 3: Update backtest to use training feature names

In the backtest script, when calling `predict_scores()`, pass the training feature names from the trainer.

For `backtest_engine.py`: after training, the trainer stores `feature_names_`. The selector should use these.

Actually, the simplest way: modify `StockSelector` to auto-detect feature names from the trained models. Since LightGBM stores `.feature_name_` or `.feature_names_in_`, XGBoost has `.feature_names_in_`, and CatBoost has `.feature_names_`.

In `StockSelector._load_models()` or `predict_scores()`, try to infer feature names from loaded models:
```python
def _get_model_feature_names(self):
    """Try to extract feature names from loaded models."""
    for model in self.models.values():
        if model is not None:
            for attr in ('feature_name_', 'feature_names_in_', 'feature_names_'):
                if hasattr(model, attr):
                    names = getattr(model, attr)
                    if names is not None and len(names) > 0:
                        return list(names)
    return None
```

Then in `predict_scores()`, use these names as fallback when `feature_names` is not provided.

## Constraints
- Python 3.11: /Users/maruohuo/miniforge3/envs/quant/bin/python3.11
- Don't modify `factor_system.py`, `data_loader.py`
- Keep backward compatibility - existing calls should still work
- Use pandas vectorized ops only

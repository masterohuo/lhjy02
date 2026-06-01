# Task: Adjust XGBoost params + Chinese logs

## Fix 1: XGBoost parameters (config.py)

XGBoost only did 66 iterations (LightGBM did 715, CatBoost 980). Need to adjust params so it iterates more (target 300-500).

Current params → New params:
```python
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,              # 5→6, slightly deeper trees
    "learning_rate": 0.01,       # 0.015→0.01, slower learning = more iterations
    "n_estimators": 1500,        # 1000→1500, more room to grow
    "min_child_weight": 100,     # 300→100, less restrictive splits
    "subsample": 0.7,            # 0.6→0.7, more data per tree
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,           # 2.0→1.0, less regularization
    "tree_method": "gpu_hist",
    "gpu_id": 0,
}
```

## Fix 2: Change log messages to Chinese (model_trainer.py)

All logger.info/warning messages in model_trainer.py → Chinese. Specifically:

From: `"开始训练三模型集成系统..."` → keep (already Chinese)
From: `"数据准备完成: %d行, %d个特征"` → keep
From: `"训练集日期: %s ~ %s (%d天)"` → keep
From: `"验证集日期: %s ~ %s (%d天)"` → keep
From: `"训练 LightGBM (regression)..."` → keep
From: `"Cannot perform reduction"` → N/A (not in model_trainer)
From: `"XGBoost GPU不可用，回退到CPU (hist)..."` → keep
From: `"CatBoost GPU不可用，回退到CPU..."` → keep

These are already Chinese. Let me check what else is in English...

The main English messages to fix:
1. `"Loaded %s model from %s"` → `"已加载 %s 模型: %s"`
2. `"Failed to load %s: %s"` → `"加载 %s 失败: %s"`
3. `"Model file not found: %s"` → `"模型文件不存在: %s"`
4. `"%s predict failed: %s"` → `"%s 预测失败: %s"`
5. `"Excluding %d ST/N stocks"` → `"排除 %d 只ST/N股票"`
6. `"Excluding %d limit-up stocks"` → `"排除 %d 只涨停股票"`
7. `"Selected %d / %d candidates"` → `"已选股 %d / %d 只"`
8. `"Portfolio: %d stocks, weight %.1f%%, cash ¥%.0f, %d orders"` → `"组合: %d只, 仓位%.1f%%, 现金¥%.0f, %d笔订单"`
9. `"Generating factor features ..."` → `"生成因子特征 ..."`
10. `"factor_system not available; using raw columns as features"` → `"因子系统不可用; 使用原始列作为特征"`
11. `"Using %d features for prediction"` → `"使用 %d 个特征进行预测"`
12. `"Check stop loss: missing columns"` → `"止损检查: 缺少列"`
13. `"Stop loss triggered"` → `"触发止损"`
14. `"empty positions"` → `"空仓"`

Check model_trainer.py too for any English log messages:
- `"Training failed for window %d: %s"` → `"窗口 %d 训练失败: %s"`
- `"Data load failed for window %d: %s"` → `"窗口 %d 数据加载失败: %s"`
- `"No test data for window %d, skipping."` → `"窗口 %d 无测试数据, 跳过"`
- Any others...

## Files to modify
1. `config.py` - XGB_PARAMS
2. `model_trainer.py` - Chinese logs
3. `stock_selector.py` - Chinese logs (if any English ones exist)
4. `universe.py` - Chinese logs (if any English ones exist)
5. `backtest_engine.py` - Chinese logs (if any English ones exist)

Scan all 5 files for English logger.info/warning messages and convert to Chinese.

## Constraints
- Python 3.11
- Don't break functionality, only change text
- Don't modify factor_system.py or data_loader.py

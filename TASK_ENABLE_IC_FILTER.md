# TASK: lhjy02 启用因子IC筛选

## 背景
`factor_system.py` 中有 `FactorScreener` 类，提供:
- `calculate_ic()` — 计算每个因子的 Rank IC
- `filter_effective_factors(min_abs_ic=0.01)` — 筛选 |IC| >= 0.01 的有效因子

但**从未被调用**，120+因子全量喂给模型，含大量噪声因子（IC≈0）。

## 改造要求

### 1. model_trainer.py 训练前加IC筛选

在 `TriModelTrainer.train_all()` 方法中，`generate_all_factors()` 之后、模型训练之前：

```python
# IC筛选有效因子
from factor_system import FactorScreener
screener = FactorScreener(df, forward_return_col='forward_ret_5d')
effective_factors = screener.filter_effective_factors(min_abs_ic=0.01)

# 只保留有效因子+元数据列
meta_cols = ['ts_code', 'date', 'label', 'forward_ret_5d', 'open', 'high', 'low', 'close', 'volume', 'amount', 'float_mv', 'up_limit', 'down_limit', ...]
keep_cols = [c for c in meta_cols if c in df.columns] + effective_factors
df = df[keep_cols]

# 记录因子筛选结果
logger.info(f"IC筛选: {len(effective_factors)}/{原始因子数} 个因子通过 (|IC|>=0.01)")
```

### 2. 保存筛选后的因子列表

将筛选后的因子列表保存到 models 目录:
- `models/effective_factors.json` — 有效因子名称列表
- `models/factor_ic.csv` — 所有因子的IC值（用于后续分析）

### 3. 预测时对齐特征

`stock_selector.py` 的 `predict_scores()` 已经通过 `_get_model_feature_names()` 获取模型存储的特征名，理论上应该自动对齐。但要确认：
- 模型训练时用 effective_factors，模型的 `feature_name_` 应该只包含这些因子
- 预测时 `_get_model_feature_names()` 返回的因子列表应与训练时一致

### 4. 添加IC筛选的可配置开关

在 `config.py` 中添加:
```python
# 因子IC筛选
ENABLE_IC_FILTER = True       # 是否启用IC筛选
IC_MIN_ABS = 0.01             # 最小|IC|阈值
```

### 代码约束
- 只修改 `model_trainer.py`、`config.py`、`stock_selector.py`（如需要）
- 保持接口兼容，不要破坏现有调用
- Python: `/Users/maruohuo/miniforge3/envs/quant/bin/python3.11`
- 目录: `/Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02/`

### 验证
改完后运行 quick 模式确认:
```bash
cd /Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02 && /Users/maruohuo/miniforge3/envs/quant/bin/python3.11 -c "
from backtest_engine import run_backtest
results = run_backtest(quick=True)
print('Done')
"
```

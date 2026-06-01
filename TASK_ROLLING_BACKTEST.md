# TASK: lhjy02 季度滚动回测

## 目标
改造 backtest_engine.py，支持**每季度滚动**的回测，输出详细的每季度盈亏情况。

## 具体要求

### 1. 滚动频率改为季度
- 当前 `run_rolling_backtest()` 默认 `retrain_freq="M"`（月度）
- **改为季度滚动**：每3个月重新训练一次模型
- 使用 pandas 的 `"QS"` (quarter start) 作为日期频率
- 确保季度窗口正确对齐（例如 2021-Q1, 2021-Q2...）

### 2. 每季度详细盈亏输出
在 `run_rolling_backtest()` 和 `run_backtest()` 函数中，添加**每季度（每个滚动窗口）的详细盈亏统计**：

对每个滚动窗口输出：
- 窗口编号（如 Q1, Q2...）
- 训练期起止日期
- 测试期起止日期
- 期初净值 / 期末净值
- 季度收益率 (%)
- 季度最大回撤 (%)
- 交易次数（买入/卖出）
- 换手率
- 持仓股票数量（平均）

### 3. 汇总统计
在所有窗口结束后输出：
- 总窗口数
- 盈利窗口数 / 亏损窗口数
- 平均季度收益率
- 最佳季度 / 最差季度（含日期和收益率）
- 季度胜率

### 4. 结果保存
- 每季度明细保存到 CSV：`results/quarterly_details_YYYYMMDD_HHMMSS.csv`
- 在 `run_backtest()` 的 quick 模式也支持季度滚动

### 5. 回测参数
- 训练数据：5年（config.py 中 TRAIN_YEARS=5，已设置）
- 滑点：万分之一（SLIPPAGE=0.0001，已设置）
- 佣金：万分之一（COMMISSION_RATE=0.0001，已设置）
- 印花税：保持千分之一（STAMP_TAX=0.001）
- 起始资金：50万
- 回测起始日期：2021-01-01
- 回测结束日期：今天

## 代码约束
- 所有代码在 `/Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02/` 目录下
- 使用 quant conda 环境：`/Users/maruohuo/miniforge3/envs/quant/bin/python3.11`
- 只修改 backtest_engine.py，不要改其他文件（config.py 已改好）
- 保持现有代码结构，在现有函数基础上扩展
- 添加新方法如 `_calculate_quarterly_stats()` 来计算季度统计

## 执行步骤
1. 读取 backtest_engine.py，理解现有逻辑
2. 修改 retrain_freq 默认值从 "M" 改为 "Q"
3. 在 BacktestEngine 类中添加季度统计追踪
4. 修改 `run_rolling_backtest()` 在每个窗口结束时记录季度数据
5. 添加 `_calculate_quarterly_stats()` 方法生成季度明细表
6. 修改 `run_backtest()` 输出季度明细表格
7. 保存季度明细 CSV
8. 运行回测：`python3.11 backtest_engine.py`（非 quick 模式，完整回测）

## 运行命令
```bash
cd /Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02 && /Users/maruohuo/miniforge3/envs/quant/bin/python3.11 backtest_engine.py
```

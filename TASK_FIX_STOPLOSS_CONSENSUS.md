# TASK: lhjy02 止损修复 + 三模型共识加权

## P0: 修复止损（backtest_engine.py）

### 问题
`config.py` 定义了 `STOP_LOSS = -0.08`，`stock_selector.py` 有 `RiskManager.check_stop_loss()` 方法，
但 `backtest_engine.py` 的每日循环中**完全没有调用止损**，导致股票跌-30%也不会卖出。

### 修复方案
在 `backtest_engine.py` 的 `_mark_to_market()` 方法中加入每日止损检查：

```
def _mark_to_market(self, day_data, date):
    # 1. 先做每日估值（现有逻辑）
    # 2. 检查止损：对每个持仓，如果 (当前价 - 成本价) / 成本价 <= STOP_LOSS(-0.08)
    #    则强制卖出全部持仓
    # 3. 止损卖出时：使用当天 close 价格，加滑点(SELL方向SLIPPAGE)，扣除佣金和印花税
    # 4. 记录止损交易到 trade_history，action='STOP_LOSS'
    # 5. 止损后该股票从 positions 中移除
```

具体要点：
- 从 day_data 中获取每只持仓股票的 close 价格
- 对 self.positions 中的每只股票检查是否触发止损
- 止损卖出逻辑参考 `simulate_trades()` 中的 SELL 部分
- 止损交易记录包含 action='STOP_LOSS' 字段
- 在记录 quarterly window stats 时，也要统计止损次数

## P1: 三模型共识加权选股（stock_selector.py）

### 当前逻辑
三模型各自预测 → z-score标准化 → 等权平均 → ensemble_score → 选 TOP_N 股票

### 改进方案：共识加权

```
1. 三模型各自对股票池打分 → 每个模型独立排名
2. 计算「共识度」：
   - 三模型排名都在前20%的股票 → 共识分 = 1.5
   - 两模型排名在前20%的股票 → 共识分 = 1.2
   - 只有一个模型排名在前20% → 共识分 = 0.8
   - 无模型排名在前20% → 共识分 = 0.6
3. 最终得分 = ensemble_score × 共识分
4. 按最终得分排序，选 TOP_N 只股票
5. 共识股票（三模型都选中）仓位上限提高到 25%（vs 普通15%）
```

修改 `predict_scores()` 或新增 `predict_consensus_scores()` 方法：
- 返回 DataFrame 包含: lgb_rank, xgb_rank, cat_rank, consensus_weight, final_score
- consensus_weight 根据三模型排名重叠度计算

修改 `construct_portfolio()`：
- 对共识股票（consensus_weight=1.5）给予更高仓位上限
- 对分歧股票（consensus_weight<1.0）降低仓位

### 代码约束
- 所有修改在 `/Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02/` 目录
- 使用 quant conda 环境: `/Users/maruohuo/miniforge3/envs/quant/bin/python3.11`
- **只修改 backtest_engine.py 和 stock_selector.py，不要改其他文件**
- 保持现有代码结构和接口兼容性
- 调仓频率不变（W-FRI）
- 交易成本不变（SLIPPAGE=0.0001, COMMISSION_RATE=0.0001, STAMP_TAX=0.001）

### 验证
修改完成后，运行 quick 模式回测验证不报错：
```bash
cd /Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02 && /Users/maruohuo/miniforge3/envs/quant/bin/python3.11 -c "
from backtest_engine import run_backtest
results = run_backtest(quick=True)
print('止损交易数:', sum(1 for t in results.get('trade_history', []) if t.get('action') == 'STOP_LOSS' if hasattr(t, 'get')))
"
```

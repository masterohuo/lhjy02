# TASK: lhjy02 完整风控体系

## 目标
在现有个股-8%硬止损基础上，增加四类止损，构建完整风控体系。

## 当前状态
- ✅ 个股硬止损: STOP_LOSS=-8%，已在 backtest_engine._mark_to_market() 运行
- ❌ 日内总亏损: check_daily_loss() 定义了但从未调用
- ❌ 移动止损: 没有
- ❌ 时间止损: 没有
- ❌ 波动止损(ATR): 没有

## 改造要求

### 1. config.py 新增参数
```python
# 个股硬止损
STOP_LOSS = -0.08

# 日内总亏损止损（当日净值相对昨日收盘跌超此比例→清仓所有持仓）
DAILY_MAX_LOSS = -0.05

# 移动止损（从持仓期间最高价回落超过此比例→卖出）
TRAILING_STOP = -0.10

# 时间止损（持仓超过此天数且浮动盈亏为负→减半仓）
MAX_HOLD_DAYS = 20

# ATR波动止损倍数（止损价 = 成本价 - ATR_MULTIPLIER × ATR(14)）
ATR_STOP_MULTIPLIER = 3.0
```

### 2. stock_selector.py RiskManager 新增方法

```python
def check_daily_loss(self, current_value, initial_value) -> bool:
    # 已有，检查日内亏损是否超 -5%

def check_trailing_stop(self, positions_dict, day_data) -> list:
    # 对每个持仓，计算持仓期间最高价
    # 如果 (当前价 - 最高价) / 最高价 <= TRAILING_STOP → 触发
    # 返回触发移动止损的股票列表

def check_time_stop(self, positions_dict, current_date, day_data) -> list:
    # 对每个持仓，计算持仓天数
    # 如果持仓天数 > MAX_HOLD_DAYS 且 浮动盈亏 < 0 → 触发
    # 返回触发时间止损的股票列表（减半仓，不是全卖）

def check_atr_stop(self, positions_dict, day_data) -> list:
    # 对每个持仓，计算 ATR(14)
    # 如果 当前价 <= 成本价 - ATR_STOP_MULTIPLIER × ATR → 触发
    // 返回触发ATR止损的股票列表
```

### 3. backtest_engine.py 修改

在 `_mark_to_market()` 中加入：
```
1. 先检查日内总亏损 (DAILY_MAX_LOSS)
   - 计算今日净值 / 昨日净值 - 1
   - 如果 <= -5%，清仓所有持仓，记录 action='DAILY_LOSS_STOP'
   
2. 再检查移动止损 (TRAILING_STOP)
   - 需要在 BacktestEngine 中维护每个持仓的最高价记录
   - 新增 self.position_highs: dict[str, float] = {}
   - 每次 _mark_to_market 更新最高价
   - 如果触发，卖出全部，记录 action='TRAILING_STOP'

3. 再检查时间止损 (MAX_HOLD_DAYS)
   - 需要在 BacktestEngine 中维护每个持仓的买入日期
   - 新增 self.position_entry_dates: dict[str, pd.Timestamp] = {}
   - 买入时记录日期
   - 如果触发，卖出持仓量的一半（100股取整），记录 action='TIME_STOP'

4. 再检查ATR止损 (ATR_STOP_MULTIPLIER)
   - 从 day_data 或已计算的因子中获取 ATR(14) 值
   - 如果没有ATR数据则跳过
   - 如果触发，卖出全部，记录 action='ATR_STOP'

5. 最后执行个股硬止损 (STOP_LOSS) — 已有
```

### 4. 记录持仓元数据

在 `simulate_trades()` 的 BUY 分支中：
- 记录 `self.position_highs[ts_code] = exec_price` (初始最高价=买入价)
- 记录 `self.position_entry_dates[ts_code] = date` (买入日期)

在 `_mark_to_market()` 的每日循环中：
- 更新 `self.position_highs[ts_code] = max(self.position_highs.get(ts_code, 0), close_price)`

止损卖出后清理元数据：
- `del self.position_highs[ts_code]`
- `del self.position_entry_dates[ts_code]`

### 5. 季度统计更新

在记录 quarterly window stats 时，增加各类止损计数：
```python
trailing_stops = sum(1 for t in window_trades if t["action"] == "TRAILING_STOP")
time_stops = sum(1 for t in window_trades if t["action"] == "TIME_STOP")
atr_stops = sum(1 for t in window_trades if t["action"] == "ATR_STOP")
daily_loss_stops = sum(1 for t in window_trades if t.get("action") == "DAILY_LOSS_STOP")

# 加到 quarterly_windows 记录中
```

### 6. 输出更新

在 `_print_quarterly_table()` 的 header 中增加各类止损列。

## 代码约束
- 目录: `/Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02/`
- Python: `/Users/maruohuo/miniforge3/envs/quant/bin/python3.11`
- 修改: `config.py`, `stock_selector.py`, `backtest_engine.py`
- 保持接口兼容，不破坏现有调用

## 验证
改完后编译验证:
```bash
cd /Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02 && /Users/maruohuo/miniforge3/envs/quant/bin/python3.11 -c "
import py_compile
for f in ['config.py','stock_selector.py','backtest_engine.py']:
    py_compile.compile(f, doraise=True)
print('✅ All compiled')
"
```

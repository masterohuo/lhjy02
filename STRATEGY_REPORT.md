# lhjy02 策略详细报告

---

## 一、策略概述

### 1.1 策略名称
**lhjy02 - 散户对手盘选股策略**

### 1.2 策略核心思想
以散户为对手盘，通过识别散户追涨杀跌的行为模式，逆向选取被散户抛售但具有上涨潜力的股票。利用机器学习三模型集成（LightGBM + XGBoost + CatBoost）进行综合评分选股，实现中低频量化交易。

### 1.3 策略定位
- **交易频率**：中低频（每周调仓一次）
- **持仓周期**：5-10个交易日
- **目标收益**：年化15-30%（回测验证）
- **风险控制**：最大回撤控制在15%以内

---

## 二、系统模块架构

### 2.1 模块总览

```
lhjy02 策略系统
├── 数据层
│   └── data_loader.py          # SQLite数据加载与合并
├── 因子层
│   └── factor_system.py        # 120+因子计算引擎
├── 模型层
│   └── model_trainer.py        # 三模型训练与保存
├── 选股层
│   └── stock_selector.py       # 评分、选股、组合构建
├── 执行层
│   ├── backtest_engine.py      # 滚动窗口回测
│   └── live.py                 # 实盘/模拟交易
└── 配置层
    └── config.py               # 全局参数配置
```

### 2.2 各模块职责

| 模块 | 文件 | 核心功能 | 输入 | 输出 |
|------|------|----------|------|------|
| **数据加载** | data_loader.py | 从SQLite加载5张表并合并 | 日期范围、股票代码 | 合并后的DataFrame |
| **因子计算** | factor_system.py | 计算120+量化因子 | 行情数据 | 因子特征矩阵 |
| **模型训练** | model_trainer.py | 训练三模型并保存 | 历史数据+因子 | 3个pickle模型文件 |
| **选股评分** | stock_selector.py | 三模型预测+等权集成+选股 | 当日因子数据 | Top N股票列表 |
| **回测验证** | backtest_engine.py | 滚动窗口回测 | 历史数据 | 回测业绩报告 |
| **实盘交易** | live.py | QMT下单+风控+日志 | 实时行情 | 交易订单 |

---

## 三、因子体系详解

### 3.1 因子分类（共120+因子）

#### A类：散户对手盘因子（20个）
| 因子名 | 说明 | 计算逻辑 |
|--------|------|----------|
| sm_net_vol | 小单净流入占比 | (小单买入量-小单卖出量)/总成交量 |
| md_net_vol | 中单净流入占比 | 中单净量/总成交量 |
| lg_net_vol | 大单净流入占比 | 大单净量/总成交量 |
| elg_net_vol | 特大单净流入占比 | 特大单净量/总成交量 |
| inst_net_vol | 机构净流入占比 | (大单+特大单净量)/总成交量 |
| retail_net_vol | 散户净流入 | = sm_net_vol |
| inst_retail_ratio | 机构散户比 | 机构净量/|散户净量| |
| inst_md_ratio | 机构中单比 | 机构净量/|中单净量| |
| sm_buy_ratio | 小单买入比例 | 小单买/(小单买+小单卖) |
| sm_sell_ratio | 小单卖出比例 | 1-sm_buy_ratio |
| md_buy_ratio | 中单买入比例 | 中单买/(中单买+中单卖) |
| md_sell_ratio | 中单卖出比例 | 1-md_buy_ratio |
| lg_buy_ratio | 大单买入比例 | 大单买/(大单买+大单卖) |
| lg_sell_ratio | 大单卖出比例 | 1-lg_buy_ratio |
| elg_buy_ratio | 特大单买入比例 | 特大单买/(特大单买+特大单卖) |
| elg_sell_ratio | 特大单卖出比例 | 1-elg_buy_ratio |
| sm_imbalance | 小单不平衡度 | 小单净量/(小单总量+ε) |
| inst_imbalance | 机构不平衡度 | (机构买-机构卖)/(机构买+机构卖+ε) |
| net_mf_amount_norm | 资金流归一化 | 净资金流/成交额 |
| retail_participation | 散户参与度 | 散户总量/总成交量 |

#### B类：资金流因子（20个）
| 因子名 | 说明 | 窗口 |
|--------|------|------|
| lg_net_vol_{5,10,20}d | 大单净流入滚动均值 | 5/10/20日 |
| inst_net_vol_{5,10,20}d | 机构净流入滚动均值 | 5/10/20日 |
| net_mf_amount_{5,10,20}d | 资金流滚动均值 | 5/10/20日 |
| flow_persistence | 资金流持续性 | 连续同向天数 |
| flow_acceleration | 资金流加速度 | 5日变化量 |
| flow_price_divergence | 量价背离 | 机构流与收益负相关 |
| lg_buy_sell_ratio | 大单买卖比 | 5日累计 |
| inst_accumulation_momentum | 机构积累动量 | 斜率/标准差 |
| flow_volatility | 资金流波动率 | 10日CV |
| lg_concentration | 大单集中度 | 大单占比 |
| net_mf_amount_ma5_ratio | 资金流MA5偏离 | 当前/MA5-1 |
| inst_flow_stability | 机构流稳定性 | 1-|自相关| |
| smart_money_5d | 聪明钱5日 | 特大单5日均值 |
| fund_reversal_signal | 资金反转信号 | 5日-20日偏离 |

#### C类：技术因子（35个）
| 因子名 | 说明 |
|--------|------|
| momentum_{5,10,20,60}d | 动量（N日涨跌幅） |
| ma_deviation_{5,10,20,60} | 均线偏离度 |
| volume_ratio_{5,20} | 量比 |
| rsi_14 | RSI(14) |
| macd_dif, macd_dea, macd_hist | MACD三线 |
| bollinger_position, bollinger_width | 布林带位置和宽度 |
| kdj_k, kdj_d, kdj_j | KDJ三线 |
| obv | 能量潮 |
| atr_14 | ATR(14) |
| mfi_14 | MFI(14) |
| wr_14 | 威廉指标 |
| cci_20 | CCI(20) |
| roc_10 | ROC(10) |
| amplitude_5 | 5日振幅均值 |
| ema12_ema26_ratio | EMA12/EMA26 |
| ppo | PPO |
| cmf_20 | CMF(20) |
| ma5_ma20_ratio | MA5/MA20 |
| pvt | 量价趋势 |
| force_index_13 | 力量指数 |
| trix_15 | TRIX(15) |
| dpo_20 | DPO(20) |
| psy_12 | PSY(12) |

#### D类：基本面因子（10个）
| 因子名 | 说明 |
|--------|------|
| pe_percentile_1y | PE年度百分位 |
| pb_percentile_1y | PB年度百分位 |
| log_total_mv | 对数总市值 |
| log_float_mv | 对数流通市值 |
| turnover_{5,20}d | 换手率滚动均值 |
| volume_ratio | 量比 |
| pe_zscore | PE截面Z分数 |
| pb_zscore | PB截面Z分数 |
| mv_turnover_ratio | 市值/成交额 |

#### E类：趋势/反转因子（15个）
| 因子名 | 说明 |
|--------|------|
| adx_14 | ADX(14) |
| di_plus_14, di_minus_14 | +DI/-DI |
| ma5_cross_ma20 | MA5/MA20交叉 |
| ma10_cross_ma60 | MA10/MA60交叉 |
| price_vs_20d_high/low | 价格vs20日高低 |
| consecutive_up/down | 连涨/连跌天数 |
| gap_ratio | 缺口比率 |
| ma_slope_20 | MA20斜率 |
| higher_high_20, lower_low_20 | 突破信号 |
| inside_day | 内包日 |
| nr7 | NR7（7日最窄振幅） |

#### F类：波动率/风险因子（12个）
| 因子名 | 说明 |
|--------|------|
| hist_vol_{5,10,20,60} | 历史波动率 |
| downside_vol_20 | 下行波动率 |
| max_dd_{20,60} | 最大回撤 |
| skewness_20 | 偏度 |
| kurtosis_20 | 峰度 |
| beta_60 | Beta系数 |
| var_20 | VaR(95%) |
| ulcer_index_20 | 溃疡指数 |

#### G类：行为金融因子（8个）
| 因子名 | 说明 |
|--------|------|
| disposition_effect | 处置效应 |
| anchoring_bias | 锚定偏差 |
| herding_intensity | 羊群效应强度 |
| overreaction | 过度反应 |
| attention_proxy | 关注度代理 |
| info_response_speed | 信息响应速度 |
| retail_panic_greed | 散户恐慌/贪婪 |
| sentiment_momentum | 情绪动量 |

### 3.2 因子加速
- 使用 **Numba JIT** 编译关键滚动计算函数
- 加速函数：`_numba_slope`, `_numba_max_dd`, `_numba_nr7`, `_numba_ulcer`
- 性能提升：10-50倍

---

## 四、模型训练详解

### 4.1 三模型架构

| 模型 | 框架 | 目标函数 | 角色 | 优势 |
|------|------|----------|------|------|
| **LightGBM** | LGBMRanker | lambdarank | 排序模型 | 快速、高效、处理类别特征 |
| **XGBoost** | XGBRegressor | reg:squarederror | 回归模型 | 稳健、正则化强 |
| **CatBoost** | CatBoostRanker | YetiRank | 排序模型 | 处理缺失值、GPU加速 |

### 4.2 训练流程

```
1. 数据准备
   ├─ 加载5年历史数据
   ├─ 生成120+因子
   └─ 创建标签（未来5日收益率）

2. 时间切分（避免前瞻偏差）
   ├─ 训练集：前80%日期
   └─ 验证集：后20%日期

3. 模型训练
   ├─ LightGBM: lambdarank + query groups
   ├─ XGBoost: reg:squarederror + early stopping
   └─ CatBoost: YetiRank + group_id

4. 模型保存
   └─ 3个pickle文件 → models/目录
```

### 4.3 集成评分

```python
# 三模型预测
lgb_score = lgb_model.predict(X)
xgb_score = xgb_model.predict(X)
cat_score = cat_model.predict(X)

# z-score标准化
lgb_zscore = (lgb_score - mean) / std
xgb_zscore = (xgb_score - mean) / std
cat_zscore = (cat_score - mean) / std

# 等权集成
ensemble_score = (lgb_zscore + xgb_zscore + cat_zscore) / 3
```

---

## 五、选股与组合构建

### 5.1 选股流程

```
1. 获取最新日期所有股票的因子数据
2. 三模型预测 → 综合评分
3. 过滤规则：
   ├─ 排除 ST / *ST / N 股票
   ├─ 排除涨停股（无法买入）
   └─ 按综合评分降序排列
4. 选取 Top 10 股票
```

### 5.2 组合构建

```python
# 等权分配
equal_weight = 1.0 / n_stocks  # n_stocks = 10

# 单票上限检查
capped_weight = min(equal_weight, 0.20)  # MAX_POS_PCT = 20%

# 总仓位检查
total_weight = capped_weight * n_stocks
if total_weight > 0.80:  # TOTAL_POS_PCT = 80%
    scale = 0.80 / total_weight
    capped_weight *= scale
```

### 5.3 仓位管理

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大持仓数 | 10只 | 同时持有最多10只股票 |
| 单票上限 | 20% | 单只股票最大仓位 |
| 总仓位上限 | 80% | 最多80%资金用于持股 |
| 现金保留 | 20% | 至少保留20%现金 |

---

## 六、调仓逻辑

### 6.1 调仓频率
- **默认**：每周五 14:55（收盘前5分钟）
- **可配置**：`config.py` 中 `REBALANCE_FREQ`
  - `"W-FRI"` = 每周五
  - `"W-THU"` = 每周四
  - `"D"` = 每天

### 6.2 调仓流程

```
1. 获取当前持仓 & 账户信息
2. 紧急止损检查（总亏损≥3%则清仓）
3. 生成当日信号（选股）
4. 对比目标组合 vs 当前持仓
5. 生成订单：
   ├─ 卖出：目标中没有的持仓股
   ├─ 买入：目标中有但未持仓的股票
   ├─ 调整：权重差异>1%的持仓
   └─ 止损：触发-8%止损的股票
6. 风控检查（单票/总仓位/日亏损）
7. 执行订单
8. 事后验证
```

### 6.3 订单类型

| 订单类型 | 价格类型 | 适用场景 |
|----------|----------|----------|
| 限价单 | 当前价×1.005（买） | 正常买入 |
| 限价单 | 当前价×0.995（卖） | 正常卖出 |
| 市价单 | LATEST_PRICE | 止损/紧急卖出 |

### 6.4 成交处理

```
1. 提交限价单
2. 等待60秒成交
3. 如果部分成交：
   ├─ 取消未成交部分
   └─ 市价追单剩余部分
4. 如果完全未成交：
   ├─ 取消订单
   └─ 市价追单
5. 最终确认成交量
```

---

## 七、止损逻辑

### 7.1 三级止损机制

| 级别 | 触发条件 | 动作 | 优先级 |
|------|----------|------|--------|
| **L1: 硬止损** | 单只股票亏损 ≥ -8% | 立即市价卖出该股 | 最高 |
| **L2: 日亏损限制** | 当日总亏损 ≥ -3% | 暂停所有买入 | 高 |
| **L3: 紧急止损** | 总持仓亏损 ≥ -3% | 市价清空所有持仓 | 最高 |

### 7.2 止损触发逻辑

```python
# L1: 硬止损（逐只检查）
pnl = (current_price - cost_price) / cost_price
if pnl <= -0.08:  # STOP_LOSS = -8%
    trigger_stop_loss(ts_code)

# L2: 日亏损限制（每日开始时设置基准）
daily_pnl = (current_value - day_start_value) / day_start_value
if daily_pnl < -0.03:
    pause_trading()

# L3: 紧急止损（总持仓检查）
total_pnl = (total_current - total_cost) / total_cost
if total_pnl < -0.03:
    liquidate_all()
```

### 7.3 止损特点

- **硬止损 -8%**：给予股票足够的波动空间，避免被洗出
- **不设固定止盈**：让利润奔跑，通过调仓自然止盈
- **紧急止损 -3%**：保护总资金安全，极端行情下清仓

---

## 八、风控体系

### 8.1 盘前风控

| 检查项 | 条件 | 动作 |
|--------|------|------|
| 日亏损限制 | 当日亏损 ≥ -3% | 拒绝所有买入 |
| 最大持仓数 | 持仓数 > 10 | 拒绝买入 |
| 单票权重 | 目标权重 > 20% | 拒绝买入 |
| 总仓位 | 目标总仓位 > 80% | 拒绝买入 |

### 8.2 盘后风控

| 检查项 | 条件 | 动作 |
|--------|------|------|
| 单票超限 | 持仓权重 > 20% | 记录警告 |
| 总仓位超限 | 总仓位 > 80% | 记录警告 |

### 8.3 紧急处理

```
紧急止损触发流程：
1. 检测到总亏损 ≥ -3%
2. 记录日志：🚨 紧急止损
3. 所有订单转为市价卖出
4. 逐只清空持仓
5. 发送通知
6. 等待下一个调仓周期
```

---

## 九、回测引擎

### 9.1 回测方法
- **滚动窗口回测**（Walk-Forward）
- 避免前瞻偏差
- 模拟真实交易环境

### 9.2 回测参数

| 参数 | 说明 |
|------|------|
| 训练窗口 | 滚动训练 |
| 测试窗口 | 滚动测试 |
| 交易成本 | 佣金万3 + 印花税千1 |
| 滑点 | 0.2% |
| 初始资金 | 500,000元 |

### 9.3 回测输出

```
📊 回测结果摘要
├─ 总收益率
├─ 年化收益率
├─ 最大回撤
├─ 夏普比率
├─ 日胜率
├─ 交易笔数
└─ 盈亏比
```

---

## 十、实盘交易系统

### 10.1 运行模式

| 模式 | 平台 | 说明 |
|------|------|------|
| **实盘模式** | Windows | 连接MiniQMT，实际下单 |
| **模拟模式** | Mac/Windows | 不实际交易，仅分析 |

### 10.2 QMT连接

```
连接流程：
1. 初始化 XtQuantTrader
2. 注册回调函数
3. 连接 127.0.0.1:5861
4. 订阅资金账户
5. 开始接收行情
```

### 10.3 守护进程

```
守护进程流程：
1. 启动后自动连接QMT
2. 解析调仓频率（每周五14:55）
3. 定时休眠（60秒检查一次）
4. 到达调仓时间 → 执行调仓
5. 异常自动重连
6. 支持 Ctrl+C 优雅退出
```

### 10.4 日志系统

- **控制台日志**：INFO级别，实时输出
- **文件日志**：DEBUG级别，按日期保存
- **日志路径**：`logs/live_YYYYMMDD.log`

---

## 十一、性能指标

### 11.1 回测业绩（参考）

| 指标 | 目标范围 | 说明 |
|------|----------|------|
| 年化收益 | 15-30% | 长期稳定收益 |
| 最大回撤 | <15% | 风险控制 |
| 夏普比率 | >1.5 | 风险调整收益 |
| 日胜率 | >53% | 胜率优势 |
| 盈亏比 | >1.5 | 盈利质量 |

### 11.2 计算性能

| 指标 | 数值 | 说明 |
|------|------|------|
| 因子计算 | ~30秒/日 | 120+因子 |
| 模型训练 | ~10-30分钟 | 5年数据 |
| 信号生成 | ~5秒 | 全市场选股 |
| 回测速度 | ~2-5分钟 | 快速模式 |

---

## 十二、已知限制与改进方向

### 12.1 当前限制
1. 数据依赖SQLite数据库，需要定期更新
2. 因子计算主要使用pandas，大样本时较慢
3. 模型训练需要较长时间
4. 实盘仅支持Windows + MiniQMT

### 12.2 改进方向
1. 因子计算迁移到Polars（提升10倍速度）
2. 增加GPU加速训练
3. 增加更多行为金融因子
4. 支持更多券商交易接口
5. 增加实时因子监控

---

*报告生成时间：2026-05-31*
*策略版本：lhjy02 v1.0*

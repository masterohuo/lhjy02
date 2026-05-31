# lhjy02 - 散户对手盘选股策略

以散户为对手盘的中低频选股量化策略，基于 LightGBM + XGBoost + CatBoost 三模型集成评分。

---

## 📁 项目结构

```
lhjy02/
├── config.py              # 全局配置（资金/风控/模型参数/QMT连接）
├── data_loader.py         # 数据加载（SQLite → DataFrame）
├── factor_system.py       # 因子系统（120+因子，7大类）
├── model_trainer.py       # 三模型训练（LGB/XGB/CAT集成）
├── stock_selector.py      # 综合评分选股 + 组合构建 + 风控
├── backtest_engine.py     # 滚动窗口回测引擎
├── live.py                # 实盘交易（Windows MiniQMT / Mac研究模式）
├── requirements.txt       # Python 依赖
├── models/                # 训练模型保存目录
│   ├── lgb_model.pkl      # LightGBM 模型
│   ├── xgb_model.pkl      # XGBoost 模型
│   └── cat_model.pkl      # CatBoost 模型
├── logs/                  # 运行日志目录
└── results/               # 回测结果目录
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保在 quant conda 环境中
source /Users/maruohuo/miniforge3/bin/activate quant

# 安装依赖
cd /Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02
pip install -r requirements.txt
```

### 2. 验证因子系统

```bash
# 验证120+因子是否正常计算
python factor_system.py --validate
```

### 3. 训练模型（首次必须）

```bash
# 方法1：使用包装器脚本（推荐）
cd /Users/maruohuo/.openclaw/workspace/quant_trading
./run_in_quant_env.sh lhjy02/model_trainer.py

# 方法2：手动激活环境
source /Users/maruohuo/miniforge3/bin/activate quant
cd /Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02
python model_trainer.py

# 指定日期范围训练
python model_trainer.py --start-date 20210101 --end-date 20260501
```

### 4. 运行回测

```bash
# 快速回测（约2-5分钟）
../run_in_quant_env.sh python backtest_engine.py

# 完整回测（约15-30分钟）
python backtest_engine.py --full
```

### 5. 实盘交易（仅Windows）

```bash
# Windows上直接运行
python live.py

# Mac上运行（自动进入研究/模拟模式）
python live.py
```

---

## 📊 核心特点

| 特点 | 说明 |
|------|------|
| **对手盘思维** | 识别散户追涨杀跌行为，逆向交易 |
| **120+因子** | 散户情绪、资金流、量价技术、基本面、趋势、波动率、行为金融7大类 |
| **三模型集成** | LightGBM(lambdarank) + XGBoost + CatBoost(YetiRank) 等权评分 |
| **因子筛选** | 自动计算IC/ICIR，淘汰无效因子 |
| **中低频** | 每周五调仓，降低交易成本 |
| **跨平台** | Mac研究/回测，Windows MiniQMT实盘 |
| **自动训练** | 实盘时若无模型，自动触发训练 |

---

## ⚙️ 配置说明

### config.py 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `INITIAL_CASH` | 500,000 | 初始资金 |
| `MAX_POSITIONS` | 10 | 最大持仓数 |
| `MAX_POS_PCT` | 0.20 | 单只股票最大仓位 20% |
| `TOTAL_POS_PCT` | 0.80 | 总仓位上限 80% |
| `STOP_LOSS` | -0.08 | 硬止损 -8% |
| `REBALANCE_FREQ` | "W-FRI" | 每周五调仓 |
| `PREDICT_HORIZON` | 5 | 预测未来5日收益 |

### QMT 账户配置（Windows实盘）

在 `config.py` 的 `QMT_CONFIG` 中填写账户：

```python
QMT_CONFIG = {
    "ip": "127.0.0.1",        # MiniQMT默认IP
    "port": 5861,             # MiniQMT默认端口
    "mini_mode": True,        # MiniQMT模式
    "account_id": "你的资金账户",  # ← 在这里填写
}
```

**填写说明：**
- **资金账户**：国金证券给你的资金账号（纯数字）
- 不是客户号，不是手机号
- 在MiniQMT交易端登录后，左上角可以看到

**三种配置方式：**
1. 直接修改 `config.py` 中的 `account_id`
2. 设置环境变量 `export QMT_ACCOUNT_ID="你的账户"`
3. 命令行传参（后续版本支持）

### 模型参数

| 模型 | 框架 | 目标函数 | 关键参数 |
|------|------|----------|----------|
| LightGBM | LGBMRanker | lambdarank | n_estimators=500, num_leaves=31 |
| XGBoost | XGBRegressor | reg:squarederror | max_depth=8, n_estimators=500 |
| CatBoost | CatBoostRanker | YetiRank | iterations=500, depth=8 |

---

## 🔧 常用命令

### 训练相关

```bash
# 训练三模型（默认5年数据）
python model_trainer.py

# 指定训练时间范围
python model_trainer.py --start-date 20210101 --end-date 20260501

# 仅训练起始日期
python model_trainer.py --start-date 20220101
```

### 回测相关

```bash
# 快速回测
python backtest_engine.py

# 完整回测
python backtest_engine.py --full
```

### 实盘相关

```bash
# Windows实盘（守护进程模式）
python live.py

# Mac模拟分析
python live.py

# 查看日志
tail -f logs/live_$(date +%Y%m%d).log
```

### 因子分析

```bash
# 验证因子系统
python factor_system.py --validate

# 因子IC分析（在Python中调用）
python -c "
from data_loader import load_all_tables
from factor_system import generate_all_factors, FactorScreener
df = load_all_tables(start_date='20230101', end_date='20260101')
df = generate_all_factors(df)
screener = FactorScreener(df)
ic = screener.calculate_ic()
print(ic.head(20))
"
```

---

## 📈 策略运行流程

```
┌─────────────────────────────────────────────────────────────┐
│                    策略运行全流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 数据加载 (data_loader.py)                               │
│     └─ SQLite → 5表合并 → DataFrame                        │
│                                                             │
│  2. 因子生成 (factor_system.py)                              │
│     └─ 120+因子 × 7大类 → 特征矩阵                          │
│                                                             │
│  3. 模型训练 (model_trainer.py)                              │
│     ├─ LightGBM (lambdarank)                                │
│     ├─ XGBoost (reg:squarederror)                           │
│     └─ CatBoost (YetiRank)                                  │
│                                                             │
│  4. 选股评分 (stock_selector.py)                             │
│     ├─ 三模型预测 → z-score标准化                           │
│     ├─ 等权集成 → 综合评分                                  │
│     └─ 过滤ST/涨停 → Top N                                 │
│                                                             │
│  5. 组合构建 (stock_selector.py)                             │
│     ├─ 等权分配 → 单票上限20%                               │
│     └─ 总仓位上限80%                                        │
│                                                             │
│  6. 执行交易 (live.py)                                      │
│     ├─ 对比当前持仓 → 生成订单                              │
│     ├─ 风控检查 → 下单执行                                  │
│     └─ 成交确认 → 日志记录                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 风控机制

| 风控项 | 触发条件 | 动作 |
|--------|----------|------|
| **硬止损** | 单只亏损 ≥ -8% | 立即市价卖出 |
| **日亏损限制** | 当日亏损 ≥ -3% | 暂停交易 |
| **紧急止损** | 总亏损 ≥ -3% | 清空所有持仓 |
| **单票上限** | 单只权重 > 20% | 拒绝买入 |
| **总仓位上限** | 总仓位 > 80% | 拒绝买入 |
| **最大持仓数** | 持仓数 > 10 | 拒绝买入 |

---

## 📋 操作指南

### 首次使用

1. **克隆仓库**
   ```bash
   cd /Users/maruohuo/.openclaw/workspace/quant_trading
   git clone https://github.com/masterohuo/lhjy02.git
   cd lhjy02
   ```

2. **安装依赖**
   ```bash
   source /Users/maruohuo/miniforge3/bin/activate quant
   pip install -r requirements.txt
   ```

3. **验证因子系统**
   ```bash
   python factor_system.py --validate
   ```

4. **训练模型**（首次必须，约10-30分钟）
   ```bash
   python model_trainer.py
   ```

5. **运行回测验证**
   ```bash
   python backtest_engine.py
   ```

### Windows实盘部署

1. **确保MiniQMT已安装并登录**
   - 路径：`D:\国金证券QMT交易端\`
   - 登录后左上角可以看到资金账户

2. **配置账户**
   - 编辑 `config.py`，填写 `account_id`
   - 或设置环境变量 `set QMT_ACCOUNT_ID=你的账户`

3. **运行策略**
   ```bash
   python live.py
   ```
   - 策略会在每周五 14:55 自动调仓
   - 日志保存在 `logs/` 目录

### Mac研究模式

```bash
# Mac上自动进入模拟模式，不会实际交易
python live.py

# 查看模拟选股结果
tail -f logs/live_$(date +%Y%m%d).log
```

### 模型更新

```bash
# 定期重新训练模型（建议每月一次）
python model_trainer.py --start-date 20210101
```

---

## ⚠️ 注意事项

1. **数据依赖**：策略需要A股历史数据（SQLite数据库），确保 `data/quant.db` 存在
2. **模型文件**：首次运行必须先训练模型，否则实盘模式会自动触发训练
3. **QMT连接**：Windows实盘需要MiniQMT交易端运行并登录
4. **交易成本**：已包含佣金(万3)和印花税(千1)
5. **回测偏差**：回测结果仅供参考，实盘可能有滑点和流动性差异

---

## 📞 常见问题

### Q: 运行时提示"未找到训练模型"怎么办？
A: 运行 `python model_trainer.py` 训练模型。实盘模式下会自动触发训练。

### Q: QMT连接失败怎么办？
A: 检查：
1. MiniQMT交易端是否已启动并登录
2. `config.py` 中的 `ip` 和 `port` 是否正确
3. `account_id` 是否填写了正确的资金账户

### Q: 如何查看因子重要性？
A: 训练完成后会自动输出 Top 20 因子重要性。

### Q: 如何修改调仓频率？
A: 修改 `config.py` 中的 `REBALANCE_FREQ`：
- `"W-FRI"` = 每周五
- `"W-THU"` = 每周四
- `"D"` = 每天

### Q: Mac上能实盘交易吗？
A: 不能。Mac上自动进入研究/模拟模式，不会实际下单。实盘交易必须在Windows上运行。

---

## 📊 策略报告

详细策略报告请参阅：[STRATEGY_REPORT.md](STRATEGY_REPORT.md)

---

*最后更新：2026-05-31*

# lhjy02 - 散户对手盘选股策略

以散户为对手盘的中低频选股量化策略，基于LightGBM + XGBoost + CatBoost三模型集成评分。

## 项目结构

```
lhjy02/
├── config.py          # 全局配置（资金/风控/模型参数）
├── data_loader.py     # 数据加载（SQLite → DataFrame）
├── factor_system.py   # 因子系统（150+因子，7大类）
├── model_trainer.py   # 三模型训练（LGB/XGB/CAT集成）
├── stock_selector.py  # 综合评分选股 + 组合构建 + 风控
├── backtest_engine.py # 滚动窗口回测引擎
├── live.py            # 实盘交易（Windows MiniQMT / Mac研究模式）
├── requirements.txt
├── models/            # 训练模型保存目录
├── logs/              # 日志目录
└── results/           # 回测结果目录
```

## 快速开始

```bash
# 1. 验证因子系统
cd quant_trading/lhjy02
../run_in_quant_env.sh python factor_system.py --validate

# 2. 快速回测
../run_in_quant_env.sh python backtest_engine.py

# 3. 实盘交易（仅Windows）
python live.py
```

## 核心特点

- **对手盘思维**: 识别散户追涨杀跌行为，逆向交易
- **150+因子**: 散户情绪、资金流、量价技术、基本面、趋势、波动率、行为金融7大类
- **三模型集成**: LightGBM(lambdarank) + XGBoost + CatBoost(YetiRank) 等权评分
- **因子筛选**: 自动计算IC/ICIR，淘汰无效因子
- **中低频**: 每周调仓，降低交易成本
- **跨平台**: Mac研究/回测，Windows MiniQMT实盘

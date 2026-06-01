"""
config.py - lhjy02 全局配置
以散户为对手盘的中低频选股策略
"""
import os
from pathlib import Path

_current_dir = Path(__file__).resolve().parent

# 目录配置
DB_PATH = str(_current_dir.parent / "data" / "quant.db")
MODELS_DIR = _current_dir / "models"
LOGS_DIR = _current_dir / "logs"
RESULTS_DIR = _current_dir / "results"

for _d in [MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    _d.mkdir(exist_ok=True)

# 资金配置
INITIAL_CASH = 500000
MAX_POSITIONS = 10
MAX_POS_PCT = 0.20           # 单只股票最大仓位 20%
TOTAL_POS_PCT = 0.80          # 总仓位上限 80%

# 交易参数
REBALANCE_FREQ = "W-FRI"      # 每周五调仓
SLIPPAGE = 0.002
STOP_LOSS = -0.08             # 硬止损 -8%
TAKE_PROFIT = None            # 不启用固定止盈
COMMISSION_RATE = 0.0003
STAMP_TAX = 0.001
TOTAL_TRADE_COST = COMMISSION_RATE + STAMP_TAX

# 选股参数
TOP_N_STOCKS = 10
MAX_STOCKS = 2000
PREDICT_HORIZON = 5

# 训练参数
TRAIN_YEARS = 5
VALIDATION_RATIO = 0.2

# LightGBM 参数 (回归预测收益率)
# 注意: objective 必须为 regression 类目标 (如 regression/rmse/mae/mse)，
# 不能使用 lambdarank 等 ranking 目标，否则 LGBMRegressor 会崩溃。
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.015,
    "n_estimators": 1500,
    "num_leaves": 50,
    "max_depth": 8,
    "min_child_samples": 150,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}

# XGBoost 参数 (回归预测收益率)
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.01,
    "n_estimators": 1500,
    "min_child_weight": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "tree_method": "gpu_hist",
    "gpu_id": 0,
}

# CatBoost 参数 (回归预测收益率)
CAT_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 1000,
    "depth": 5,
    "learning_rate": 0.015,
    "subsample": 0.6,
    "l2_leaf_reg": 25.0,
    "min_data_in_leaf": 500,
    "early_stopping_rounds": 50,
    "task_type": "GPU",
    "devices": "0",
}

# ============================================================
# QMT 连接配置（Windows MiniQMT 实盘交易）
# ============================================================
# 使用方法：
#   方式1: 设置环境变量  export QMT_ACCOUNT_ID="你的资金账户"
#   方式2: 直接修改下面的 account_id 字段
#   方式3: 命令行参数  python live.py --account 你的资金账户
#
# 账户类型说明：
#   - 资金账户：国金证券给你的资金账号（纯数字）
#   - 不是客户号，不是手机号
#   - 在MiniQMT交易端登录后，左上角可以看到
#
# ip/port：MiniQMT默认监听 127.0.0.1:5861，一般无需修改
# mini_mode：True=使用MiniQMT模式，False=使用完整QMT模式
# ============================================================
QMT_CONFIG = {
    "ip": "127.0.0.1",
    "port": 5861,
    "mini_mode": True,
    "account_id": os.environ.get("QMT_ACCOUNT_ID", ""),  # ← 在这里填写你的资金账户
}

# 日志
LOG_LEVEL = "INFO"

# ============================================================
# Stock Universe Configuration
# ⚠️ Tushare amount is in 千元 (thousands of CNY).
#    50_000 = 5000万CNY, 10_000 = 1000万CNY, 100_000 = 1亿CNY
# ============================================================
UNIVERSE_CONFIG = {
    "min_daily_amount": 50_000,       # 5000万日成交额 (in 千元)
    "min_list_days": 60,
    "exclude_st": True,
    "exclude_limit_board": False,     # False during training, True during live selection
    "exclude_boards": ["STAR", "BSE"],  # 排除科创板(688开头)和北交所(8/4开头)
    "top_n": 2000,
    "use_stratified": False,
    "stratified_min_per_sector": 20,
}

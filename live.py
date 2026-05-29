"""
live.py - lhjy02 live trading system for Windows MiniQMT.
Dual-mode: Live trading on Windows, Research/DryRun on macOS.
"""
import datetime
import logging
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    INITIAL_CASH,
    LOGS_DIR,
    MAX_POS_PCT,
    MAX_POSITIONS,
    MODELS_DIR,
    QMT_CONFIG,
    REBALANCE_FREQ,
    STOP_LOSS,
    TOP_N_STOCKS,
    TOTAL_POS_PCT,
)
from data_loader import load_all_tables
from factor_system import generate_all_factors
from stock_selector import StockSelector, RiskManager

# ---------------------------------------------------------------------------
# OS detection & xtquant import
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    try:
        from xtquant import xtdata, xttrader, xtconstant
        _HAS_XTQUANT = True
    except ImportError as e:
        print(f"[live] xtquant import failed: {e}")
        print("[live] Install MiniQMT and ensure xtquant is on PYTHONPATH.")
        print("[live] Falling back to paper-trading / dry-run mode.")
        _HAS_XTQUANT = False
else:
    _HAS_XTQUANT = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging(log_dir, name="live"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")
    fh = logging.FileHandler(log_dir / f"{name}_{today}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------
def send_notification(message: str):
    logger = logging.getLogger("live")
    logger.info(f"[NOTIFY] {message}")


# ===================================================================
# QMT Callback handler
# ===================================================================
if _HAS_XTQUANT:
    _CALLBACK_BASE = xttrader.XtQuantTraderCallback
else:
    _CALLBACK_BASE = object


class QmtCallback(_CALLBACK_BASE):
    """MiniQMT order/position/asset callback, delegates to LiveTrader logger."""

    def __init__(self, trader):
        super().__init__()
        self._trader = trader

    def on_stock_order(self, order):
        logger = logging.getLogger("live")
        logger.debug(
            "[QMT] Order update: %s status=%s filled=%s/%s",
            order.stock_code,
            getattr(order, "order_status", "?"),
            getattr(order, "traded_volume", 0),
            getattr(order, "order_volume", 0),
        )
        for o in self._trader.orders:
            if o.get("order_id") == getattr(order, "order_id", None):
                o["status"] = getattr(order, "order_status", o.get("status"))
                o["filled_volume"] = getattr(order, "traded_volume", o.get("filled_volume", 0))

    def on_stock_asset(self, asset):
        logger = logging.getLogger("live")
        logger.debug(
            "[QMT] Asset: total=%s cash=%s",
            getattr(asset, "total_asset", 0),
            getattr(asset, "cash", 0),
        )

    def on_stock_position(self, position):
        logger = logging.getLogger("live")
        logger.debug(
            "[QMT] Position: %s vol=%s cost=%.2f",
            position.stock_code, position.volume, position.open_price,
        )

    def on_disconnected(self):
        logger = logging.getLogger("live")
        logger.warning("[QMT] Connection lost")
        self._trader._connected = False

    def on_connected(self):
        logger = logging.getLogger("live")
        logger.info("[QMT] Connected")
        self._trader._connected = True


# ===================================================================
# Live Risk Manager
# ===================================================================
class LiveRiskManager(RiskManager):
    """Extended risk manager with pre-trade checks and emergency stop."""

    DAILY_LOSS_LIMIT = -0.03

    def __init__(self):
        super().__init__()
        self._initial_day_value: float | None = None

    def set_day_start(self, total_value: float):
        self._initial_day_value = total_value

    def pre_trade_check(self, orders: list, positions_df: pd.DataFrame,
                        account_value: float) -> tuple:
        """Pre-trade risk checks. Returns (approved: bool, reason: str)."""
        logger = logging.getLogger("live")

        # Daily loss limit
        if self._initial_day_value is not None and self._initial_day_value > 0:
            daily_pnl = (account_value - self._initial_day_value) / self._initial_day_value
            if daily_pnl < self.DAILY_LOSS_LIMIT:
                msg = f"DAILY LOSS LIMIT HIT: {daily_pnl:.2%}"
                logger.error(msg)
                return False, msg

        # Max positions check
        n_current = 0 if positions_df.empty else len(positions_df)
        n_buy = sum(1 for o in orders if o.get("action") == "BUY")
        n_sell = sum(1 for o in orders if o.get("action") == "SELL")
        n_after = n_current + n_buy - n_sell
        if n_after > MAX_POSITIONS:
            return False, f"Max positions exceeded: {n_after} > {MAX_POSITIONS}"

        return True, "OK"

    def emergency_stop(self, positions_df: pd.DataFrame) -> bool:
        """Check if total unrealized PnL triggers emergency liquidation."""
        if positions_df.empty:
            return False
        required = {"cost_price", "current_price", "volume"}
        if not required.issubset(positions_df.columns):
            return False

        cost_total = (positions_df["cost_price"] * positions_df["volume"]).sum()
        current_total = (positions_df["current_price"] * positions_df["volume"]).sum()
        if cost_total <= 0:
            return False

        total_pnl = (current_total - cost_total) / cost_total
        if total_pnl < self.DAILY_LOSS_LIMIT:
            logger = logging.getLogger("live")
            logger.critical(
                "EMERGENCY STOP: total PnL %.2f%% < %.2f%%",
                total_pnl * 100, self.DAILY_LOSS_LIMIT * 100,
            )
            return True
        return False

    def post_trade_check(self, positions_df: pd.DataFrame, total_value: float) -> tuple:
        """Verify no single position exceeds MAX_POS_PCT after trading."""
        if positions_df.empty or total_value <= 0:
            return True, "OK"

        weights = (positions_df["current_price"] * positions_df["volume"]) / total_value
        over = weights[weights > MAX_POS_PCT]
        if len(over) > 0:
            codes = positions_df.loc[over.index, "ts_code"].tolist()
            return False, f"Position limit exceeded: {codes}"
        return True, "OK"


# ===================================================================
# LiveTrader
# ===================================================================
class LiveTrader:
    """Live trading system with MiniQMT integration."""

    def __init__(self):
        self.logger = _setup_logging(LOGS_DIR, "live")

        if not IS_WINDOWS:
            self.logger.info("Running in Research/DryRun mode on macOS")
        else:
            status = "xtquant available" if _HAS_XTQUANT else "xtquant NOT available"
            self.logger.info("Running on Windows - Live mode (%s)", status)

        # Load trained models
        self.models = self._load_models()
        if self.models:
            self.logger.info("Loaded models: %s", list(self.models.keys()))
        else:
            self.logger.warning(
                "No trained models found. Run model_trainer.py first."
            )

        # Stock selector with loaded models
        self.selector = StockSelector(models_dict=self.models)

        # Risk manager
        self.risk_manager = LiveRiskManager()

        # Position & order tracking
        self.positions: dict[str, dict] = {}
        self.orders: list[dict] = []

        # QMT connection state
        self.xt_trader = None
        self._connected = False
        self._qmt_account = QMT_CONFIG.get("account_id", "")

        if IS_WINDOWS and _HAS_XTQUANT:
            self._init_qmt()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self) -> dict:
        models = {}
        for name in ("lgb", "xgb", "cat"):
            path = MODELS_DIR / f"{name}_model.pkl"
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        models[name] = pickle.load(f)
                    self.logger.info("Loaded %s model from %s", name, path)
                except Exception as e:
                    self.logger.warning("Failed to load %s: %s", name, e)
            else:
                self.logger.warning("Model file not found: %s", path)
        return models

    # ------------------------------------------------------------------
    # QMT initialization
    # ------------------------------------------------------------------
    def _init_qmt(self):
        if not _HAS_XTQUANT:
            return
        try:
            mini_qmt_path = os.environ.get(
                "MINI_QMT_PATH", r"D:\国金证券QMT交易端\userdata_mini"
            )
            session_id = int(time.time() * 1000) % 1000000

            self.xt_trader = xttrader.XtQuantTrader(mini_qmt_path, session_id)
            self.xt_trader.register_callback(QmtCallback(self))
            self.xt_trader.start()
            self.logger.info("QMT trader initialized, session=%d", session_id)

        except Exception as e:
            self.logger.error("QMT init failed: %s", e)
            self.xt_trader = None

    # ------------------------------------------------------------------
    # QMT connection
    # ------------------------------------------------------------------
    def connect_qmt(self) -> bool:
        """Connect to MiniQMT with retry (max 3 attempts)."""
        if not IS_WINDOWS:
            self.logger.info("[DryRun] Skipping QMT connection on non-Windows")
            return False

        if not _HAS_XTQUANT or self.xt_trader is None:
            self.logger.warning("xtquant not available, cannot connect to QMT")
            return False

        for attempt in range(1, 4):
            try:
                ip = QMT_CONFIG.get("ip", "127.0.0.1")
                port = QMT_CONFIG.get("port", 5861)

                result = self.xt_trader.connect(ip, port)
                if result == 0:
                    self._connected = True
                    self.logger.info("Connected to MiniQMT at %s:%d", ip, port)

                    if self._qmt_account:
                        self.xt_trader.subscribe(self._qmt_account)
                        self.logger.info("Subscribed to account %s", self._qmt_account)

                    return True
                else:
                    self.logger.warning(
                        "QMT connection attempt %d/3 returned %s", attempt, result
                    )

            except Exception as e:
                self.logger.error("QMT connection attempt %d/3 error: %s", attempt, e)

            if attempt < 3:
                time.sleep(2 ** attempt)

        self.logger.error("Failed to connect to MiniQMT after 3 attempts")
        return False

    # ------------------------------------------------------------------
    # Position & account queries
    # ------------------------------------------------------------------
    def get_current_positions(self) -> pd.DataFrame:
        """Query current holdings from xtquant.

        Returns DataFrame with ts_code, volume, cost_price, current_price, pnl.
        """
        columns = ["ts_code", "volume", "cost_price", "current_price", "pnl"]

        if not self._connected or self.xt_trader is None:
            self.logger.debug("[DryRun] get_current_positions")
            return pd.DataFrame(columns=columns)

        try:
            positions = self.xt_trader.query_stock_positions(self._qmt_account)
            if not positions:
                return pd.DataFrame(columns=columns)

            records = []
            for pos in positions:
                current_price = self._get_single_price(pos.stock_code)
                price = current_price or pos.open_price

                records.append({
                    "ts_code": pos.stock_code,
                    "volume": pos.volume,
                    "cost_price": pos.open_price,
                    "current_price": price,
                    "pnl": (
                        (price - pos.open_price) / pos.open_price
                        if pos.open_price and pos.open_price > 0
                        else 0.0
                    ),
                })

            df = pd.DataFrame(records)
            self.positions = {r["ts_code"]: r for r in records}
            return df

        except Exception as e:
            self.logger.error("get_current_positions error: %s", e)
            return pd.DataFrame(columns=columns)

    def _get_single_price(self, ts_code: str) -> float | None:
        try:
            if self._connected and _HAS_XTQUANT:
                data = xtdata.get_market_data(
                    field_list=["lastPrice"],
                    stock_list=[ts_code],
                    period="tick",
                    count=1,
                )
                if data and "lastPrice" in data:
                    vals = data["lastPrice"]
                    if vals and len(vals) > 0:
                        arr = vals[ts_code] if ts_code in vals else vals[-1]
                        if hasattr(arr, "__len__") and len(arr) > 0:
                            return float(arr[-1])
        except Exception:
            pass
        return None

    def get_account_info(self) -> dict:
        """Query account balance, available cash, total assets."""
        if not self._connected or self.xt_trader is None:
            self.logger.debug("[DryRun] get_account_info: using INITIAL_CASH")
            return {
                "total_asset": INITIAL_CASH,
                "available_cash": INITIAL_CASH,
                "market_value": 0.0,
                "frozen_cash": 0.0,
            }

        try:
            asset = self.xt_trader.query_stock_asset(self._qmt_account)
            return {
                "total_asset": asset.total_asset,
                "available_cash": asset.cash,
                "market_value": asset.market_value,
                "frozen_cash": asset.frozen_cash,
            }
        except Exception as e:
            self.logger.error("get_account_info error: %s", e)
            return {
                "total_asset": INITIAL_CASH,
                "available_cash": INITIAL_CASH,
                "market_value": 0.0,
                "frozen_cash": 0.0,
            }

    # ------------------------------------------------------------------
    # Real-time quotes
    # ------------------------------------------------------------------
    def get_realtime_quotes(self, ts_codes: list[str]) -> dict:
        """Get latest prices for given stock codes.

        Returns dict mapping ts_code to {lastPrice, open, high, low, volume, ...}
        """
        if not ts_codes:
            return {}

        quotes: dict[str, dict] = {}
        missing = list(ts_codes)

        if _HAS_XTQUANT and IS_WINDOWS:
            try:
                data = xtdata.get_market_data(
                    field_list=["lastPrice", "open", "high", "low",
                               "volume", "amount", "lastClose",
                               "bidPrice", "askPrice"],
                    stock_list=ts_codes,
                    period="tick",
                    count=1,
                )

                for ts_code in ts_codes:
                    quote = {}
                    for field in ["lastPrice", "open", "high", "low",
                                 "volume", "amount", "lastClose",
                                 "bidPrice", "askPrice"]:
                        try:
                            vals = data.get(field, {})
                            arr = vals.get(ts_code) if isinstance(vals, dict) else vals
                            if hasattr(arr, "__len__") and len(arr) > 0:
                                quote[field] = float(arr[-1])
                        except (TypeError, ValueError, IndexError):
                            quote[field] = None
                    if quote:
                        quotes[ts_code] = quote
                        missing.remove(ts_code)

            except Exception as e:
                self.logger.warning("get_realtime_quotes via xtdata failed: %s", e)

        # Fallback: use last close from DB for any still-missing codes
        if missing:
            try:
                df = load_all_tables(ts_codes=missing)
                if not df.empty:
                    latest = df.sort_values("date").groupby("ts_code").tail(1)
                    for _, row in latest.iterrows():
                        if row["ts_code"] in missing and row["ts_code"] not in quotes:
                            quotes[row["ts_code"]] = {
                                "lastPrice": row.get("close"),
                                "lastClose": row.get("pre_close"),
                            }
            except Exception as e:
                self.logger.debug("Fallback price query failed: %s", e)

        return quotes

    # ------------------------------------------------------------------
    # Daily signal generation
    # ------------------------------------------------------------------
    def run_daily_signal(self) -> dict:
        """Load latest data, generate factors, predict ensemble scores, select top stocks.

        Returns dict with keys: selected, target_portfolio, scores
        """
        self.logger.info("=" * 60)
        self.logger.info("Running daily signal generation ...")

        today = datetime.date.today()
        end_date = today.strftime("%Y%m%d")
        start_date = (today - datetime.timedelta(days=120)).strftime("%Y%m%d")

        try:
            df = load_all_tables(start_date=start_date, end_date=end_date)
            if not df.empty:
                df = generate_all_factors(df)
        except Exception as e:
            self.logger.error("Data loading failed: %s", e)
            return {"selected": pd.DataFrame(), "target_portfolio": {}, "scores": pd.DataFrame()}

        if df.empty:
            self.logger.error("No data loaded for signal generation")
            return {"selected": pd.DataFrame(), "target_portfolio": {}, "scores": pd.DataFrame()}

        latest_date = df["date"].max()
        self.logger.info("Latest data date: %s", latest_date)
        df_latest = df[df["date"] == latest_date].copy()
        self.logger.info("Stocks on latest date: %d", len(df_latest))

        # Feature preparation
        exclude = {
            "ts_code", "date", "label", "future_return", "target",
            "up_limit", "down_limit", "open", "high", "low",
        }
        meta_cols = [
            c for c in ["ts_code", "close", "up_limit", "float_mv",
                        "total_mv", "pre_close"] if c in df_latest.columns
        ]
        feature_cols = [
            c for c in df_latest.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df_latest[c])
        ]

        if not feature_cols:
            self.logger.error("No feature columns found")
            return {"selected": pd.DataFrame(), "target_portfolio": {}, "scores": pd.DataFrame()}

        self.logger.info("Using %d feature columns", len(feature_cols))

        X = df_latest[feature_cols].copy()
        X = X.fillna(X.median()).fillna(0.0)

        # Predict ensemble scores
        scores_df = self.selector.predict_scores(X, feature_cols)

        # Merge scores with metadata
        df_scored = pd.concat(
            [df_latest[meta_cols].reset_index(drop=True),
             scores_df.reset_index(drop=True)], axis=1
        )
        if "ts_code" not in df_scored.columns and "ts_code" in df_latest.columns:
            df_scored["ts_code"] = df_latest["ts_code"].values

        # Select top stocks
        selected = self.selector.select_stocks(df_scored, top_n=TOP_N_STOCKS)

        # Build target portfolio
        portfolio = self.selector.construct_portfolio(selected, total_cash=INITIAL_CASH)
        target_weights = portfolio["target_weights"]

        self.logger.info(
            "Signal complete: %d stocks selected, %d target positions",
            len(selected), len(target_weights),
        )

        if not selected.empty:
            top_picks = selected[["ts_code", "ensemble_score"]].head(5)
            for _, row in top_picks.iterrows():
                self.logger.info(
                    "  Top pick: %s  score=%.4f", row["ts_code"], row["ensemble_score"]
                )

        return {
            "selected": selected,
            "target_portfolio": target_weights,
            "scores": scores_df,
        }

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------
    def generate_orders(self, target_portfolio: dict[str, float],
                        current_positions: pd.DataFrame) -> list[dict]:
        """Compare target vs current, generate buy/sell orders with risk checks.

        Returns list of order dicts: ts_code, action (BUY/SELL), volume, price_type, reason.
        """
        orders: list[dict] = []

        if not target_portfolio:
            self.logger.info("No target portfolio, generating liquidation orders")
            for _, row in current_positions.iterrows():
                orders.append({
                    "ts_code": row["ts_code"],
                    "action": "SELL",
                    "volume": int(row["volume"]),
                    "price_type": "LIMIT",
                    "reason": "no_target",
                })
            return orders

        target_set = set(target_portfolio)
        current_set: set[str] = set()
        if not current_positions.empty and "ts_code" in current_positions.columns:
            current_set = set(current_positions["ts_code"].tolist())

        # SELL: stocks in current but not in target
        for ts_code in current_set - target_set:
            pos = current_positions[current_positions["ts_code"] == ts_code].iloc[0]
            orders.append({
                "ts_code": ts_code,
                "action": "SELL",
                "volume": int(pos["volume"]),
                "price_type": "LIMIT",
                "reason": "removed_from_target",
            })

        # Stop-loss: force SELL at market
        stopped = self.risk_manager.check_stop_loss(current_positions)
        for ts_code in stopped:
            if ts_code not in {o["ts_code"] for o in orders}:
                pos = current_positions[current_positions["ts_code"] == ts_code].iloc[0]
                orders.append({
                    "ts_code": ts_code,
                    "action": "SELL",
                    "volume": int(pos["volume"]),
                    "price_type": "MARKET",
                    "reason": "stop_loss",
                })

        # Account info for sizing
        account = self.get_account_info()
        total_value = account.get("total_asset", INITIAL_CASH)

        # BUY: stocks in target but not in current
        for ts_code in target_set - current_set:
            weight = target_portfolio[ts_code]
            target_value = total_value * weight

            quote = self.get_realtime_quotes([ts_code]).get(ts_code, {})
            price = quote.get("lastPrice", 0)
            if not price or price <= 0:
                self.logger.warning("No valid price for %s, skipping BUY", ts_code)
                continue

            volume = int(target_value / price / 100) * 100
            if volume < 100:
                self.logger.warning(
                    "%s: target volume %d < min 100 shares, skipping", ts_code, volume
                )
                continue

            orders.append({
                "ts_code": ts_code,
                "action": "BUY",
                "volume": volume,
                "price_type": "LIMIT",
                "reason": "new_position",
                "target_price": price,
            })

        # ADJUST: stocks in both target and current
        for ts_code in target_set & current_set:
            target_w = target_portfolio[ts_code]
            pos = current_positions[current_positions["ts_code"] == ts_code].iloc[0]
            cur_value = pos["current_price"] * pos["volume"]
            cur_w = cur_value / total_value if total_value > 0 else 0

            if abs(target_w - cur_w) < 0.01:
                continue

            quote = self.get_realtime_quotes([ts_code]).get(ts_code, {})
            price = quote.get("lastPrice", pos["current_price"])

            if target_w > cur_w:
                add_value = total_value * (target_w - cur_w)
                add_vol = int(add_value / price / 100) * 100
                if add_vol >= 100:
                    orders.append({
                        "ts_code": ts_code,
                        "action": "BUY",
                        "volume": add_vol,
                        "price_type": "LIMIT",
                        "reason": "rebalance_increase",
                        "target_price": price,
                    })
            else:
                reduce_value = total_value * (cur_w - target_w)
                reduce_vol = int(reduce_value / price / 100) * 100
                if reduce_vol >= 100:
                    orders.append({
                        "ts_code": ts_code,
                        "action": "SELL",
                        "volume": min(reduce_vol, int(pos["volume"])),
                        "price_type": "LIMIT",
                        "reason": "rebalance_decrease",
                    })

        # Pre-trade risk check
        approved, reason = self.risk_manager.pre_trade_check(
            orders, current_positions, total_value
        )
        if not approved:
            self.logger.error("Pre-trade check failed: %s", reason)
            # Emergency: liquidate all if triggered
            if self.risk_manager.emergency_stop(current_positions):
                self.logger.critical("EMERGENCY STOP: converting all orders to SELL")
                existing_codes = {o["ts_code"] for o in orders}
                sell_orders = [
                    o for o in orders if o["action"] == "SELL"
                ] + [
                    {
                        "ts_code": row["ts_code"],
                        "action": "SELL",
                        "volume": int(row["volume"]),
                        "price_type": "MARKET",
                        "reason": "emergency_liquidation",
                    }
                    for _, row in current_positions.iterrows()
                    if row["ts_code"] not in existing_codes
                ]
                orders = sell_orders
            else:
                # Non-emergency: only allow SELL orders
                orders = [o for o in orders if o["action"] == "SELL"]

        n_buy = sum(1 for o in orders if o["action"] == "BUY")
        n_sell = sum(1 for o in orders if o["action"] == "SELL")
        self.logger.info("Generated %d orders: %d BUY, %d SELL", len(orders), n_buy, n_sell)

        for o in orders:
            self.logger.info(
                "  Order: %s %s  vol=%d  (%s)",
                o["action"], o["ts_code"], o["volume"], o.get("reason", ""),
            )

        return orders

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def execute_orders(self, orders: list[dict]) -> list[dict]:
        """Submit orders to MiniQMT with timeout handling and market-price retry.

        Returns list of result dicts: ts_code, action, volume, status, filled_volume, price.
        """
        if not orders:
            self.logger.info("No orders to execute")
            return []

        self.logger.info("Executing %d orders ...", len(orders))

        results = [self._execute_single_order(o) for o in orders]

        filled = sum(1 for r in results if r.get("status") == "FILLED")
        partial = sum(1 for r in results if r.get("status") == "PARTIALLY_FILLED")
        rejected = sum(1 for r in results if r.get("status") == "REJECTED")
        dry = sum(1 for r in results if r.get("status") == "DRYRUN")

        self.logger.info(
            "Execution complete: %d filled, %d partial, %d rejected, %d dryrun, %d total",
            filled, partial, rejected, dry, len(results),
        )

        return results

    def _execute_single_order(self, order: dict) -> dict:
        """Execute one order: limit order + monitor + cancel/retry at market if unfilled."""
        ts_code = order["ts_code"]
        action = order["action"]
        volume = order["volume"]
        reason = order.get("reason", "")
        price_type = order.get("price_type", "LIMIT")

        # Get current price
        quotes = self.get_realtime_quotes([ts_code])
        quote = quotes.get(ts_code, {})
        current_price = quote.get("lastPrice", 0)

        if not current_price or current_price <= 0:
            self.logger.warning("No price for %s, skipping order", ts_code)
            return {
                "ts_code": ts_code, "action": action, "volume": volume,
                "status": "REJECTED", "filled_volume": 0, "price": 0,
                "reason": f"no_price ({reason})",
            }

        # Dry run / paper trading
        if not self._connected:
            self.logger.info(
                "[DryRun] %s %s  vol=%d  price=%.2f  (%s)",
                action, ts_code, volume, current_price, reason,
            )
            return {
                "ts_code": ts_code, "action": action, "volume": volume,
                "status": "DRYRUN", "filled_volume": volume,
                "price": current_price, "reason": reason,
            }

        # Live execution
        try:
            if price_type == "MARKET":
                limit_price = -1
                qmt_price_type = xtconstant.LATEST_PRICE
            elif action == "BUY":
                limit_price = round(current_price * 1.005, 2)
                qmt_price_type = xtconstant.FIX_PRICE
            else:
                limit_price = round(current_price * 0.995, 2)
                qmt_price_type = xtconstant.FIX_PRICE

            qmt_order_type = (
                xtconstant.STOCK_BUY if action == "BUY" else xtconstant.STOCK_SELL
            )

            self.logger.info(
                "[Live] %s %s  vol=%d  limit=%.2f  (%s)",
                action, ts_code, volume, limit_price if limit_price > 0 else current_price, reason,
            )

            order_id = self.xt_trader.order_stock(
                self._qmt_account, ts_code, qmt_order_type, volume,
                qmt_price_type, limit_price,
                "lhjy02_strategy", reason,
            )

            self.logger.info("Order submitted: id=%d  %s %s %d", order_id, ts_code, action, volume)

            order_record = {
                "ts_code": ts_code,
                "action": action,
                "volume": volume,
                "limit_price": limit_price,
                "order_id": order_id,
                "status": "PENDING",
                "filled_volume": 0,
                "submit_time": time.time(),
            }
            self.orders.append(order_record)

            # Wait for fill with 60-second timeout
            filled_vol = self._wait_for_fill(order_id, ts_code, volume, timeout=60)

            if filled_vol >= volume * 0.9:
                return {
                    "ts_code": ts_code, "action": action, "volume": volume,
                    "status": "FILLED", "filled_volume": filled_vol,
                    "price": limit_price if limit_price > 0 else current_price,
                    "reason": reason,
                }
            elif filled_vol > 0:
                self.logger.warning(
                    "Partial fill %s: %d/%d, retrying remaining at market", ts_code, filled_vol, volume
                )
                self._cancel_order(order_id)

                remaining = volume - filled_vol
                market_id = self.xt_trader.order_stock(
                    self._qmt_account, ts_code, qmt_order_type, remaining,
                    xtconstant.LATEST_PRICE, -1,
                    "lhjy02_strategy", f"{reason}_retry",
                )
                retry_filled = self._wait_for_fill(market_id, ts_code, remaining, timeout=30)
                total_filled = filled_vol + max(retry_filled, 0)

                return {
                    "ts_code": ts_code, "action": action, "volume": volume,
                    "status": "FILLED" if total_filled >= volume * 0.9 else "PARTIALLY_FILLED",
                    "filled_volume": total_filled,
                    "price": limit_price if limit_price > 0 else current_price,
                    "reason": f"{reason}_retried",
                }
            else:
                self._cancel_order(order_id)
                self.logger.warning("No fill for %s, retrying at market", ts_code)

                market_id = self.xt_trader.order_stock(
                    self._qmt_account, ts_code, qmt_order_type, volume,
                    xtconstant.LATEST_PRICE, -1,
                    "lhjy02_strategy", f"{reason}_retry",
                )
                retry_filled = self._wait_for_fill(market_id, ts_code, volume, timeout=30)

                return {
                    "ts_code": ts_code, "action": action, "volume": volume,
                    "status": (
                        "FILLED" if (retry_filled or 0) >= volume * 0.9 else "PARTIALLY_FILLED"
                    ),
                    "filled_volume": max(retry_filled or 0, 0),
                    "price": limit_price if limit_price > 0 else current_price,
                    "reason": f"{reason}_retried",
                }

        except Exception as e:
            self.logger.error("Order execution error for %s: %s", ts_code, e)
            return {
                "ts_code": ts_code, "action": action, "volume": volume,
                "status": "REJECTED", "filled_volume": 0,
                "price": current_price, "reason": str(e),
            }

    def _wait_for_fill(self, order_id, ts_code, target_volume, timeout=60) -> int:
        """Poll order status until filled or timeout. Returns filled volume."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                orders = self.xt_trader.query_stock_orders(self._qmt_account)
                for o in (orders or []):
                    if getattr(o, "order_id", None) == order_id:
                        if getattr(o, "traded_volume", 0) >= target_volume:
                            return o.traded_volume
                        status = getattr(o, "order_status", None)
                        if status in (54, 55, 56):  # cancelled / rejected
                            return o.traded_volume or 0
            except Exception:
                pass
            time.sleep(2)
        return 0

    def _cancel_order(self, order_id):
        try:
            self.xt_trader.cancel_order(order_id)
            self.logger.info("Order %d cancelled", order_id)
        except Exception as e:
            self.logger.warning("Cancel order %d failed: %s", order_id, e)

    # ------------------------------------------------------------------
    # Full rebalance cycle
    # ------------------------------------------------------------------
    def run_rebalance(self) -> dict:
        """Execute full rebalance: positions → signal → orders → execute → verify."""
        self.logger.info("=" * 60)
        self.logger.info("REBALANCE CYCLE START")
        self.logger.info("=" * 60)

        start_time = time.time()

        # 1. Get current positions & account info
        positions_df = self.get_current_positions()
        account = self.get_account_info()

        self.logger.info(
            "Account: total ¥%,.0f  available ¥%,.0f  positions: %d",
            account["total_asset"], account["available_cash"], len(positions_df),
        )

        # 2. Set day-start for risk tracking
        if account["total_asset"] > 0:
            self.risk_manager.set_day_start(account["total_asset"])

        # 3. Emergency stop check
        if self.risk_manager.emergency_stop(positions_df):
            self.logger.critical("EMERGENCY STOP: liquidating all positions")
            liquidate = [
                {
                    "ts_code": row["ts_code"],
                    "action": "SELL",
                    "volume": int(row["volume"]),
                    "price_type": "MARKET",
                    "reason": "emergency_liquidation",
                }
                for _, row in positions_df.iterrows()
            ]
            exec_results = self.execute_orders(liquidate)
            send_notification("lhjy02 EMERGENCY STOP: all positions liquidated")
            return {
                "status": "EMERGENCY_STOP",
                "orders": liquidate,
                "results": exec_results,
                "elapsed": time.time() - start_time,
            }

        # 4. Run daily signal
        signal = self.run_daily_signal()
        target_portfolio = signal.get("target_portfolio", {})

        if not target_portfolio:
            self.logger.warning("No target portfolio generated, skipping rebalance")
            return {
                "status": "NO_SIGNAL",
                "orders": [],
                "results": [],
                "elapsed": time.time() - start_time,
            }

        # 5. Generate orders
        orders = self.generate_orders(target_portfolio, positions_df)

        # 6. Execute orders
        results = self.execute_orders(orders)

        # 7. Post-trade verification
        updated_positions = self.get_current_positions()
        passed, reason = self.risk_manager.post_trade_check(
            updated_positions, account["total_asset"]
        )
        if not passed:
            self.logger.warning("Post-trade check failed: %s", reason)
            send_notification(f"lhjy02 post-trade WARNING: {reason}")

        # 8. Summary
        elapsed = time.time() - start_time
        filled = sum(1 for r in results if r.get("status") == "FILLED")
        self.logger.info(
            "REBALANCE COMPLETE in %.1fs: %d/%d orders filled",
            elapsed, filled, len(results),
        )

        send_notification(
            f"lhjy02 rebalance done: {filled}/{len(results)} filled, "
            f"account ¥{account['total_asset']:,.0f}"
        )

        return {
            "status": "OK",
            "target_portfolio": target_portfolio,
            "orders": orders,
            "results": results,
            "account": account,
            "elapsed": elapsed,
        }

    # ------------------------------------------------------------------
    # Daemon scheduler
    # ------------------------------------------------------------------
    def run_daemon(self):
        """Background daemon: run rebalance on schedule (daily 14:55 or weekly Friday 14:55)."""
        self.logger.info("=" * 60)
        self.logger.info("DAEMON STARTED")
        self.logger.info("Rebalance frequency: %s", REBALANCE_FREQ)
        self.logger.info("=" * 60)

        # Parse schedule
        freq = REBALANCE_FREQ.upper()
        if freq.startswith("W-"):
            day_map = {
                "MON": 0, "TUE": 1, "WED": 2, "THU": 3,
                "FRI": 4, "SAT": 5, "SUN": 6,
            }
            day_str = freq.split("-")[1]
            target_weekday = day_map.get(day_str, 4)
            self.logger.info("Weekly rebalance on %s at 14:55", day_str)
        elif freq in ("D", "DAILY", "1D"):
            target_weekday = None
            self.logger.info("Daily rebalance at 14:55")
        else:
            target_weekday = 4
            self.logger.info(
                "Unknown freq '%s', defaulting to Friday 14:55", REBALANCE_FREQ
            )

        TARGET_HOUR, TARGET_MINUTE = 14, 55

        # Connect to QMT if Windows
        if IS_WINDOWS and _HAS_XTQUANT:
            if not self.connect_qmt():
                self.logger.warning("QMT not connected, will retry each cycle")

        try:
            while True:
                now = datetime.datetime.now()
                next_run = self._next_scheduled_time(
                    now, target_weekday, TARGET_HOUR, TARGET_MINUTE
                )
                wait_sec = max(1, (next_run - now).total_seconds())

                self.logger.info(
                    "Next rebalance at %s (%.0fs from now)",
                    next_run.strftime("%Y-%m-%d %H:%M:%S"), wait_sec,
                )

                # Sleep until schedule (in 60s chunks for interrupt responsiveness)
                while time.time() < next_run.timestamp():
                    sleep_chunk = min(60, max(1, next_run.timestamp() - time.time()))
                    time.sleep(sleep_chunk)

                # Run rebalance
                try:
                    if IS_WINDOWS and _HAS_XTQUANT and not self._connected:
                        self.connect_qmt()

                    result = self.run_rebalance()
                    self.logger.info("Rebalance result: %s", result.get("status"))

                except Exception as e:
                    self.logger.error("Rebalance error: %s", e, exc_info=True)
                    send_notification(f"lhjy02 rebalance ERROR: {e}")

        except KeyboardInterrupt:
            self.logger.info("DAEMON STOPPED by user (KeyboardInterrupt)")
            send_notification("lhjy02 daemon stopped")
        except Exception as e:
            self.logger.error("DAEMON CRASHED: %s", e, exc_info=True)
            send_notification(f"lhjy02 daemon CRASHED: {e}")
            raise

    @staticmethod
    def _next_scheduled_time(now, target_weekday, target_hour, target_minute):
        """Compute the next scheduled run datetime."""
        today = now.date()
        t = datetime.datetime.combine(
            today, datetime.time(target_hour, target_minute)
        )

        if target_weekday is not None:
            days_ahead = (target_weekday - now.weekday()) % 7
            if days_ahead == 0 and now >= t:
                days_ahead = 7
            return datetime.datetime.combine(
                today + datetime.timedelta(days=days_ahead),
                datetime.time(target_hour, target_minute),
            )
        else:
            if now >= t:
                t = datetime.datetime.combine(
                    today + datetime.timedelta(days=1),
                    datetime.time(target_hour, target_minute),
                )
            return t

    # ------------------------------------------------------------------
    # Research / Paper Analysis mode (macOS)
    # ------------------------------------------------------------------
    def run_paper_analysis(self) -> dict:
        """Run signal generation and analysis without trading (research / dry-run mode)."""
        self.logger.info("=" * 60)
        self.logger.info("PAPER ANALYSIS MODE")
        self.logger.info("=" * 60)

        signal = self.run_daily_signal()

        if signal["selected"].empty:
            self.logger.warning("No stocks selected during paper analysis")
            return signal

        selected = signal["selected"]

        if "ensemble_score" in selected.columns:
            self.logger.info(
                "Score range: %.4f ~ %.4f",
                selected["ensemble_score"].min(), selected["ensemble_score"].max(),
            )

        if "close" in selected.columns:
            self.logger.info(
                "Price range: ¥%.2f ~ ¥%.2f",
                selected["close"].min(), selected["close"].max(),
            )

        self.logger.info("Target portfolio: %s", signal["target_portfolio"])
        return signal


# ---------------------------------------------------------------------------
# Module-level entry points
# ---------------------------------------------------------------------------
def create_trader() -> LiveTrader:
    return LiveTrader()


def main():
    trader = create_trader()

    if IS_WINDOWS and _HAS_XTQUANT:
        trader.run_daemon()
    else:
        print("")
        print("Running in Research/DryRun mode on macOS")
        print("")
        trader.run_paper_analysis()


if __name__ == "__main__":
    main()

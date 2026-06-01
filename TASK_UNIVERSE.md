# Task: lhjy02 Stock Pool Refactor - Three-Tier Funnel Universe

## Background
lhjy02 uses `groupby("date").apply(g.head(2000))` to select training stocks. This causes:
1. Code-alphabet bias (000xxx always first, 688/8xxxxx cut off)
2. No liquidity filter (zombie stocks included)
3. Distribution shift between training and prediction

## Goal
Create `universe.py` with `StockUniverse` class, replace `head(2000)` everywhere.

## File 1: Create `universe.py`

```python
class StockUniverse:
    """
    Three-tier funnel stock pool filter.
    
    Tier 1 - Hard filter: remove ST/N stamps, new listings (<60d), low turnover (<10M CNY)
    Tier 2 - Tradability ranking: log(float_mv) * sqrt(turnover_rate), pick top N
    Tier 3 - Stratified sampling (optional): by Shenwan industry
    
    Uses vectorized pandas ops only (no for loops).
    """
    
    def __init__(self, **kwargs):
        self.min_daily_amount = kwargs.get('min_daily_amount', 10_000_000)  # 10M CNY
        self.min_list_days = kwargs.get('min_list_days', 60)
        self.exclude_st = kwargs.get('exclude_st', True)
        self.exclude_limit_board = kwargs.get('exclude_limit_board', False)
        self.top_n = kwargs.get('top_n', 2000)
        self.use_stratified = kwargs.get('use_stratified', False)
        self.stratified_min_per_sector = kwargs.get('stratified_min_per_sector', 20)
    
    def filter(self, df) -> pd.DataFrame:
        """Main entry: hard_filter -> rank_by_tradability -> optional stratified_sample."""
        logger.info("Universe filter: %d rows input", len(df))
        df = self._hard_filter(df)
        df = self._rank_by_tradability(df)
        if self.use_stratified:
            df = self._stratified_sample(df)
        logger.info("Universe filter: %d rows output", len(df))
        return df
    
    def _hard_filter(self, df) -> pd.DataFrame:
        """
        Vectorized filters:
        - Exclude ST/*ST/PT/N stocks (ts_code starts with those prefixes)
        - Exclude stocks listed < min_list_days (if 'list_date' column exists, compute date - list_date)
        - Exclude stocks with daily amount < min_daily_amount
        - If exclude_limit_board: exclude limit-up/down stocks
        Log count removed at each step.
        """
        pass  # IMPLEMENT
    
    def _rank_by_tradability(self, df) -> pd.DataFrame:
        """
        Per date, compute score = log(float_mv) * sqrt(turnover_rate).
        Use circ_mv column for float_mv. 
        Fill NaN in float_mv/turnover_rate with date median.
        Sort descending, take top_n per date.
        Use groupby + nlargest or rank-based approach (vectorized, no for loop).
        """
        pass  # IMPLEMENT
    
    def _stratified_sample(self, df) -> pd.DataFrame:
        """
        If 'industry' column exists:
        - Group by industry
        - Keep at least stratified_min_per_sector per sector
        - Remaining slots allocated by sector market cap proportion
        If no industry column, fall back to _rank_by_tradability.
        """
        pass  # IMPLEMENT
    
    @staticmethod
    def get_board(code):
        """Identify board from stock code prefix."""
        code = str(code)
        if code.startswith('688'): return 'STAR'
        if code.startswith(('300','301')): return 'ChiNext'
        if code.startswith(('8','4')): return 'BSE'
        if code.startswith('6'): return 'SSE'
        if code.startswith('0'): return 'SZSE'
        return 'Other'
```

## File 2: Modify `config.py`

Add at bottom:
```python
# ============================================================
# Stock Universe Configuration
# ============================================================
UNIVERSE_CONFIG = {
    "min_daily_amount": 10_000_000,
    "min_list_days": 60,
    "exclude_st": True,
    "exclude_limit_board": False,   # False during training, True during live selection
    "top_n": 2000,
    "use_stratified": False,
    "stratified_min_per_sector": 20,
}
```

## File 3: Modify `data_loader.py`

In `load_all_tables()`, add parameter `include_basic_info=False`. When True:
- Add to _TABLE_COLUMNS: `"stock_basic": ["ts_code", "industry", "list_date"]`
- Join stock_basic table on ts_code (left join)
- Parse list_date from TEXT (like '20200101') to datetime

## File 4: Modify `model_trainer.py`

In `TriModelTrainer.prepare_data()`:
- Replace the `head(max_stocks)` block (lines 83-85) with:
```python
from universe import StockUniverse
universe = StockUniverse(**UNIVERSE_CONFIG)
n_before = len(df)
df = universe.filter(df)
n_after = len(df)
logger.info("Stock universe: %d -> %d (filtered %.1f%%)", n_before, n_after, (1 - n_after/max(n_before,1))*100)
```
- Change load_all_tables call to include `include_basic_info=True`
- Keep `max_stocks` parameter but pass to universe config

## File 5: Modify `stock_selector.py`

In `select_stocks()`, add optional prefilter step BEFORE existing ST filter:
```python
def select_stocks(self, df_with_scores, top_n=TOP_N_STOCKS, exclude_st=True, prefilter=True):
    df = df_with_scores.copy()
    if prefilter:
        from universe import StockUniverse
        from config import UNIVERSE_CONFIG
        uconfig = {**UNIVERSE_CONFIG, "exclude_limit_board": True}
        universe = StockUniverse(**uconfig)
        df = universe.filter(df)
    # ... existing code continues
```

## File 6: Modify `backtest_engine.py`

No external changes needed (train_all calls prepare_data internally, which now uses universe).
Test data load keeps include_basic_info=False (less data overhead).

## Constraints
1. Python 3.11: /Users/maruohuo/miniforge3/envs/quant/bin/python3.11
2. NO for loops over DataFrame rows - use pandas vectorized ops exclusively
3. Use pathlib for paths
4. Log filter stats at each step with logger.info
5. Backward compatible - don't break train_all() signature
6. Don't modify factor_system.py

## Verification
After implementation, run:
```bash
cd /Users/maruohuo/.openclaw/workspace/quant_trading/lhjy02

# Test 1: import
/Users/maruohuo/miniforge3/envs/quant/bin/python3.11 -c "from universe import StockUniverse; u = StockUniverse(); print('OK')"

# Test 2: data load with basic info
/Users/maruohuo/miniforge3/envs/quant/bin/python3.11 -c "
from data_loader import load_all_tables
df = load_all_tables(start_date='20250101', end_date='20250110', include_basic_info=True)
print(f'Rows: {len(df)}, Has industry: {\"industry\" in df.columns}')
"

# Test 3: filter
/Users/maruohuo/miniforge3/envs/quant/bin/python3.11 -c "
from data_loader import load_all_tables
from universe import StockUniverse
import logging; logging.basicConfig(level=logging.INFO)
df = load_all_tables(start_date='20250101', end_date='20250110', include_basic_info=True)
u = StockUniverse()
f = u.filter(df)
print(f'Before: {len(df)}, After: {len(f)}')
print(f['ts_code'].apply(StockUniverse.get_board).value_counts().to_string())
"
```

If all pass, print "✅ Universe module ready".

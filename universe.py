"""
universe.py - lhjy02 Three-Tier Funnel Stock Universe Filter.

Tier 1 - Hard filter: remove ST/N stamps, new listings (<60d), low turnover
Tier 2 - Tradability ranking: log(float_mv) * sqrt(turnover_rate), pick top N
Tier 3 - Stratified sampling (optional): by Shenwan industry

Uses vectorized pandas ops only (no for loops over DataFrame rows).
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StockUniverse:
    """Three-tier funnel stock pool filter."""

    def __init__(self, **kwargs):
        # Tushare amount is in 千元 (thousands of CNY).
        # 10_000 → 10M CNY, 50_000 → 50M CNY, 100_000 → 100M CNY
        self.min_daily_amount = kwargs.get('min_daily_amount', 50_000)
        self.min_list_days = kwargs.get('min_list_days', 60)
        self.exclude_st = kwargs.get('exclude_st', True)
        self.exclude_limit_board = kwargs.get('exclude_limit_board', False)
        self.exclude_boards = kwargs.get('exclude_boards', [])
        self.top_n = kwargs.get('top_n', 2000)
        self.use_stratified = kwargs.get('use_stratified', False)
        self.stratified_min_per_sector = kwargs.get('stratified_min_per_sector', 20)

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry: hard_filter -> rank_by_tradability -> optional stratified_sample."""
        logger.info("股票池过滤: %d行输入", len(df))
        df = self._hard_filter(df)
        df = self._rank_by_tradability(df)
        if self.use_stratified:
            df = self._stratified_sample(df)
        logger.info("股票池过滤: %d行输出", len(df))
        return df

    def _hard_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply vectorized hard filters: ST/N stamps, new listings, low turnover, limit boards."""
        n0 = len(df)

        # Exclude specific boards (STAR, BSE, ChiNext, SSE, SZSE)
        if self.exclude_boards and 'ts_code' in df.columns:
            n_before = len(df)
            for board in self.exclude_boards:
                if board == 'STAR':
                    df = df[~df['ts_code'].astype(str).str.startswith('688')]
                elif board == 'BSE':
                    df = df[~df['ts_code'].astype(str).str.startswith(('8', '4'))]
                elif board == 'ChiNext':
                    df = df[~df['ts_code'].astype(str).str.startswith(('300', '301'))]
                elif board == 'SSE':
                    df = df[~df['ts_code'].astype(str).str.startswith('6')]
                elif board == 'SZSE':
                    df = df[~df['ts_code'].astype(str).str.startswith('0')]
            n_removed = n_before - len(df)
            logger.info("硬过滤 - 板块%s: 移除%d, 剩余%d",
                       self.exclude_boards, n_removed, len(df))

        # Exclude ST / *ST / PT / N stocks
        if self.exclude_st and 'ts_code' in df.columns:
            st_keywords = ('ST', '*ST', 'PT', 'N', 'st', '*st', 'pt', 'n')
            st_mask = df['ts_code'].astype(str).str.startswith(st_keywords)
            df = df[~st_mask]
            n_removed = n0 - len(df)
            logger.info("硬过滤 - ST/N/PT: 移除%d, 剩余%d", n_removed, len(df))

        # Exclude stocks listed < min_list_days
        if 'list_date' in df.columns:
            n_before = len(df)
            list_dates = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
            if 'date' in df.columns:
                trade_dates = pd.to_datetime(df['date'])
            else:
                trade_dates = pd.Timestamp.now()
            days_listed = (trade_dates - list_dates).dt.days
            valid = days_listed.isna() | (days_listed >= self.min_list_days)
            df = df[valid]
            n_removed = n_before - len(df)
            logger.info("硬过滤 - 上市天数<%d: 移除%d, 剩余%d",
                       self.min_list_days, n_removed, len(df))

        # Exclude stocks with daily amount < min_daily_amount
        if 'amount' in df.columns:
            n_before = len(df)
            valid = df['amount'].isna() | (df['amount'] >= self.min_daily_amount)
            df = df[valid]
            n_removed = n_before - len(df)
            logger.info("硬过滤 - 成交额<%.0e: 移除%d, 剩余%d",
                       self.min_daily_amount, n_removed, len(df))

        # Exclude limit-up/down stocks (cannot trade)
        if self.exclude_limit_board and 'close' in df.columns and 'up_limit' in df.columns and 'down_limit' in df.columns:
            n_before = len(df)
            limit_up = (df['close'] >= df['up_limit']) & (df['up_limit'] > 0)
            limit_down = (df['close'] <= df['down_limit']) & (df['down_limit'] > 0)
            df = df[~(limit_up | limit_down)]
            n_removed = n_before - len(df)
            logger.info("硬过滤 - 涨跌停: 移除%d, 剩余%d", n_removed, len(df))

        logger.info("硬过滤总计: %d -> %d (%.1f%%)",
                   n0, len(df), 100 * len(df) / max(n0, 1))
        return df

    def _rank_by_tradability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per date, compute score = log(float_mv) * sqrt(turnover_rate), pick top_n."""
        n0 = len(df)

        # Use circ_mv/float_mv column
        mv_col = None
        for col in ['circ_mv', 'float_mv']:
            if col in df.columns:
                mv_col = col
                break

        turnover_col = 'turnover_rate' if 'turnover_rate' in df.columns else None

        if mv_col is None and turnover_col is None:
            logger.warning("无市值或换手率列; 跳过可交易性排名")
            return df

        has_date = 'date' in df.columns

        # Fill NaN with median (per-date if date column exists, otherwise global)
        if mv_col:
            if has_date:
                date_med = df.groupby('date')[mv_col].transform('median')
            else:
                date_med = df[mv_col].median()
            df[mv_col] = df[mv_col].fillna(date_med)

        if turnover_col:
            if has_date:
                date_med = df.groupby('date')[turnover_col].transform('median')
            else:
                date_med = df[turnover_col].median()
            df[turnover_col] = df[turnover_col].fillna(date_med)

        # Compute score
        score_parts = []
        if mv_col:
            mv_pos = df[mv_col].clip(lower=1.0)
            score_parts.append(np.log(mv_pos))
        if turnover_col:
            tr = df[turnover_col].clip(lower=0.0)
            score_parts.append(np.sqrt(tr))

        if not score_parts:
            return df

        df['_tradability_score'] = score_parts[0]
        for part in score_parts[1:]:
            df['_tradability_score'] = df['_tradability_score'] * part

        # Pick top_n by tradability score (per-date or global)
        if has_date:
            df = df.sort_values(['date', '_tradability_score'], ascending=[True, False])
            df = df.groupby('date', group_keys=False).head(self.top_n)
        else:
            df = df.sort_values('_tradability_score', ascending=False).head(self.top_n)
        df = df.drop(columns=['_tradability_score'])

        logger.info("可交易性排名: %d -> %d (top_n=%d/日)",
                   n0, len(df), self.top_n)
        return df

    def _stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stratified sampling by industry sector.

        If 'industry' column exists: keep at least stratified_min_per_sector per sector,
        remaining slots allocated by sector market cap proportion.
        If no industry column, fall back to current ranking.
        """
        if 'industry' not in df.columns:
            logger.warning("无'industry'列; 回退到可交易性排名")
            return df

        mv_col = 'float_mv' if 'float_mv' in df.columns else ('circ_mv' if 'circ_mv' in df.columns else None)

        # Min quota per sector
        sector_counts = df.groupby('industry').size()
        min_quota = {}
        for sector, cnt in sector_counts.items():
            min_quota[sector] = min(self.stratified_min_per_sector, cnt)

        guaranteed = sum(min_quota.values())

        # Remaining slots
        remaining_slots = max(0, self.top_n - guaranteed)

        if mv_col and remaining_slots > 0:
            sector_mv = df.groupby('industry')[mv_col].sum()
            total_mv = sector_mv.sum()
            if total_mv > 0:
                sector_weights = sector_mv / total_mv
                # Allocate remaining slots proportionally
                extra_alloc = (sector_weights * remaining_slots).round().clip(lower=0).astype(int)
                # Adjust rounding discrepancies
                while extra_alloc.sum() > remaining_slots:
                    idx = extra_alloc.idxmax()
                    extra_alloc[idx] = max(0, extra_alloc[idx] - 1)
                while extra_alloc.sum() < remaining_slots:
                    idx = extra_alloc.idxmin()
                    extra_alloc[idx] += 1
            else:
                extra_alloc = pd.Series(0, index=sector_mv.index)
        else:
            extra_alloc = pd.Series(0, index=sector_counts.index)

        total_quota = {}
        for sector in sector_counts.index:
            total_quota[sector] = min_quota.get(sector, 0) + extra_alloc.get(sector, 0)
            total_quota[sector] = min(total_quota[sector], sector_counts[sector])

        # Select top tradability stocks within each sector
        if '_tradability_score' not in df.columns:
            # Recompute tradability score for stratification
            score_parts = []
            if mv_col:
                mv_pos = df[mv_col].clip(lower=1.0)
                score_parts.append(np.log(mv_pos))
            if 'turnover_rate' in df.columns:
                tr = df['turnover_rate'].clip(lower=0.0)
                score_parts.append(np.sqrt(tr))
            if score_parts:
                df['_tradability_score'] = score_parts[0]
                for part in score_parts[1:]:
                    df['_tradability_score'] = df['_tradability_score'] * part

        frames = []
        for sector, quota in total_quota.items():
            if quota <= 0:
                continue
            sector_df = df[df['industry'] == sector]
            if '_tradability_score' in sector_df.columns:
                sector_df = sector_df.sort_values('_tradability_score', ascending=False)
            frames.append(sector_df.head(quota))

        result = pd.concat(frames, ignore_index=True)
        result = result.drop(columns=['_tradability_score'], errors='ignore')

        logger.info("分层抽样: %d -> %d (%d个行业)",
                   len(df), len(result), len(frames))
        return result

    @staticmethod
    def get_board(code):
        """Identify board from stock code prefix."""
        code = str(code)
        if code.startswith('688'):
            return 'STAR'
        if code.startswith(('300', '301')):
            return 'ChiNext'
        if code.startswith(('8', '4')):
            return 'BSE'
        if code.startswith('6'):
            return 'SSE'
        if code.startswith('0'):
            return 'SZSE'
        return 'Other'

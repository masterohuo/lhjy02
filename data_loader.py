"""
data_loader.py - lhjy02 quantitative stock selection strategy data loader.

Loads multi-table data from SQLite and merges into a unified DataFrame.
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "quant.db"

# Map logical column names to actual DB column names per table.
# Each entry: table_name -> {actual_db_col: desired_logical_col}
_COLUMN_ALIASES = {
    "stock_daily_basic": {"trade_date": "date", "circ_mv": "float_mv"},
    "stock_moneyflow": {"trade_date": "date"},
    "margin_detail": {"trade_date": "date", "rzye": "margin_balance", "rqye": "short_balance"},
    "stk_limit": {"trade_date": "date"},
}

# Logical column lists for each table (the names you want in the output).
_TABLE_COLUMNS = {
    "stock_daily": [
        "ts_code", "date", "open", "high", "low", "close",
        "pre_close", "change", "pct_chg", "volume", "amount",
    ],
    "stock_daily_basic": [
        "ts_code", "date", "total_mv", "float_mv", "pe",
        "pe_ttm", "pb", "turnover_rate", "volume_ratio",
    ],
    "stock_moneyflow": [
        "ts_code", "date",
        "buy_sm_vol", "sell_sm_vol", "buy_md_vol", "sell_md_vol",
        "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol",
        "net_mf_amount",
    ],
    "adj_factor": ["ts_code", "date", "adj_factor"],
    "stk_limit": ["ts_code", "date", "up_limit", "down_limit"],
}


def _get_connection():
    """Open a connection to the SQLite database."""
    return sqlite3.connect(str(DB_PATH))


def _get_table_columns(table):
    """Return the set of actual column names in a table."""
    with _get_connection() as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _safe_select(table, requested_cols):
    """Return the subset of requested_cols that actually exist in the table."""
    existing = _get_table_columns(table)
    return [c for c in requested_cols if c in existing]


def _parse_date_column(series):
    """Parse integer or string dates into datetime.

    Handles:
      - integer dates (e.g. 20150105 → datetime)
      - string dates (e.g. "20150105" or "2015-01-05" → datetime)
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return pd.to_datetime(numeric.astype(int).astype(str), format="%Y%m%d")
    return pd.to_datetime(series)


def _normalize_date(value):
    """Convert a date value to integer YYYYMMDD."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return int(value.strftime("%Y%m%d"))
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().replace("-", "").replace("/", "")
    return int(s)


def _load_table(conn, table, desired_cols, ts_codes=None, start_date=None, end_date=None):
    """Load a single table with SQL-level filtering.

    Parameters
    ----------
    conn : sqlite3.Connection
    table : str
    desired_cols : list of str
        Logical column names desired in the output.
    ts_codes : list of str, optional
    start_date : int or str, optional
    end_date : int or str, optional

    Returns
    -------
    pd.DataFrame
        The loaded data with logical column names and dates normalized to YYYYMMDD int.
    """
    aliases = _COLUMN_ALIASES.get(table, {})
    reverse_aliases = {v: k for k, v in aliases.items()}
    existing = _get_table_columns(table)

    # Build SELECT clause: map logical → actual column, with AS alias if different
    select_parts = []
    for col in desired_cols:
        actual = reverse_aliases.get(col, col)
        if actual in existing:
            if actual != col:
                select_parts.append(f"{actual} AS {col}")
            else:
                select_parts.append(col)

    if not select_parts:
        return pd.DataFrame()

    # Detect the date column type in the DB: integer vs text
    date_col = reverse_aliases.get("date", "date")
    date_type = None
    if date_col in existing:
        with conn:
            cur = conn.execute(f"SELECT typeof({date_col}) FROM {table} LIMIT 1")
            row = cur.fetchone()
            if row:
                date_type = row[0]

    # Build WHERE clauses
    where_parts = []
    params = []

    if ts_codes is not None and ts_codes:
        placeholders = ",".join(["?"] * len(ts_codes))
        where_parts.append(f"ts_code IN ({placeholders})")
        params.extend(ts_codes)

    sd = _normalize_date(start_date)
    ed = _normalize_date(end_date)

    if date_col in existing:
        if sd is not None:
            if date_type == "integer":
                where_parts.append(f"{date_col} >= ?")
                params.append(sd)
            else:
                where_parts.append(f"CAST({date_col} AS INTEGER) >= ?")
                params.append(sd)

        if ed is not None:
            if date_type == "integer":
                where_parts.append(f"{date_col} <= ?")
                params.append(ed)
            else:
                where_parts.append(f"CAST({date_col} AS INTEGER) <= ?")
                params.append(ed)

    sql = f"SELECT {', '.join(select_parts)} FROM {table}"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    df = pd.read_sql_query(sql, conn, params=params)

    # Normalize the date column to integer YYYYMMDD so all tables match for merging
    if "date" in df.columns:
        df["date"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")

    return df


def load_all_tables(ts_codes=None, start_date=None, end_date=None):
    """Load and merge all stock data tables.

    Parameters
    ----------
    ts_codes : list of str, optional
        Stock codes to filter by (e.g. ['000001.SZ', '600000.SH']).
    start_date : int, str, or pd.Timestamp, optional
        Start date (e.g. 20200101, '20200101', '2020-01-01').
    end_date : int, str, or pd.Timestamp, optional
        End date (same formats as start_date).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns sorted by (ts_code, date).
        Date column is parsed to datetime.
    """
    conn = _get_connection()

    # 1. stock_daily – base table
    df = _load_table(conn, "stock_daily", _TABLE_COLUMNS["stock_daily"],
                     ts_codes=ts_codes, start_date=start_date, end_date=end_date)
    if df.empty:
        conn.close()
        return df

    # 2. stock_daily_basic – merge on (ts_code, date)
    sdb = _load_table(conn, "stock_daily_basic", _TABLE_COLUMNS["stock_daily_basic"],
                      ts_codes=ts_codes, start_date=start_date, end_date=end_date)
    if not sdb.empty:
        df = df.merge(sdb, on=["ts_code", "date"], how="left")

    # 3. stock_moneyflow – merge on (ts_code, date)
    mf = _load_table(conn, "stock_moneyflow", _TABLE_COLUMNS["stock_moneyflow"],
                     ts_codes=ts_codes, start_date=start_date, end_date=end_date)
    if not mf.empty:
        df = df.merge(mf, on=["ts_code", "date"], how="left")

    # 4. adj_factor – merge on (ts_code, date)
    af = _load_table(conn, "adj_factor", _TABLE_COLUMNS["adj_factor"],
                     ts_codes=ts_codes, start_date=start_date, end_date=end_date)
    if not af.empty:
        df = df.merge(af, on=["ts_code", "date"], how="left")

    # 5. stk_limit – optional merge on (ts_code, date)
    sl = _load_table(conn, "stk_limit", _TABLE_COLUMNS["stk_limit"],
                     ts_codes=ts_codes, start_date=start_date, end_date=end_date)
    if not sl.empty:
        df = df.merge(sl, on=["ts_code", "date"], how="left")

    conn.close()

    # Parse date to datetime and sort
    df["date"] = _parse_date_column(df["date"])
    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
    return df


def load_index_daily(start_date=None, end_date=None):
    """Load index daily close prices for benchmark comparison.

    Parameters
    ----------
    start_date : int, str, or pd.Timestamp, optional
    end_date : int, str, or pd.Timestamp, optional

    Returns
    -------
    pd.DataFrame
        Columns: ts_code, date, close, sorted by (ts_code, date).
    """
    existing = _get_table_columns("index_daily")
    idx_cols = [c for c in ["ts_code", "date", "close"] if c in existing]

    conn = _get_connection()
    df = _load_table(conn, "index_daily", idx_cols,
                     start_date=start_date, end_date=end_date)
    conn.close()

    if df.empty:
        return df

    df["date"] = _parse_date_column(df["date"])
    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
    return df

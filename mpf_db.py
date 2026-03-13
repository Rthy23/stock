# ═══════════════════════════════════════════════════════════════════════════════
# mpf_db.py  —  SQLite persistence for MPF holdings
# ═══════════════════════════════════════════════════════════════════════════════
"""
Schema (table: mpf_holdings)
  fund_name        TEXT  PRIMARY KEY
  etf              TEXT
  category         TEXT
  desc             TEXT
  pct              REAL   — portfolio allocation %
  market_value_hkd REAL   — current market value in HKD
  pnl_hkd          REAL   — unrealised P&L in HKD
  units            REAL   — number of units held
  unit_price_hkd   REAL   — NAV per unit in HKD
  updated_at       TEXT   — ISO-8601 timestamp
"""
import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpf_holdings.db")

_COLS = [
    "fund_name", "etf", "category", "desc",
    "pct", "market_value_hkd", "pnl_hkd",
    "units", "unit_price_hkd", "updated_at",
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the holdings table if it does not exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mpf_holdings (
                fund_name        TEXT PRIMARY KEY,
                etf              TEXT    DEFAULT '',
                category         TEXT    DEFAULT '',
                desc             TEXT    DEFAULT '',
                pct              REAL    DEFAULT 0.0,
                market_value_hkd REAL    DEFAULT NULL,
                pnl_hkd          REAL    DEFAULT NULL,
                units            REAL    DEFAULT NULL,
                unit_price_hkd   REAL    DEFAULT NULL,
                updated_at       TEXT    DEFAULT ''
            )
        """)
        conn.commit()


def save_portfolio(portfolio: list[dict]) -> None:
    """
    Upsert every holding in *portfolio* into the DB.
    Extra keys beyond the schema are silently ignored.
    Deletes rows that are no longer in the list.
    """
    if not portfolio:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        names_to_keep = [p["fund_name"] for p in portfolio]

        for p in portfolio:
            conn.execute("""
                INSERT INTO mpf_holdings
                    (fund_name, etf, category, desc,
                     pct, market_value_hkd, pnl_hkd,
                     units, unit_price_hkd, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fund_name) DO UPDATE SET
                    etf              = excluded.etf,
                    category         = excluded.category,
                    desc             = excluded.desc,
                    pct              = excluded.pct,
                    market_value_hkd = excluded.market_value_hkd,
                    pnl_hkd          = excluded.pnl_hkd,
                    units            = excluded.units,
                    unit_price_hkd   = excluded.unit_price_hkd,
                    updated_at       = excluded.updated_at
            """, (
                p.get("fund_name", ""),
                p.get("etf", ""),
                p.get("category", ""),
                p.get("desc", ""),
                float(p.get("pct", 0) or 0),
                _float_or_none(p.get("market_value_hkd")),
                _float_or_none(p.get("pnl_hkd")),
                _float_or_none(p.get("units")),
                _float_or_none(p.get("unit_price_hkd")),
                now,
            ))

        placeholders = ",".join("?" for _ in names_to_keep)
        conn.execute(
            f"DELETE FROM mpf_holdings WHERE fund_name NOT IN ({placeholders})",
            names_to_keep,
        )
        conn.commit()


def load_portfolio() -> list[dict]:
    """Return all holdings from the DB as a list of dicts."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM mpf_holdings ORDER BY rowid"
        ).fetchall()
    return [dict(r) for r in rows]


def clear_portfolio() -> None:
    """Delete all holdings rows."""
    with _connect() as conn:
        conn.execute("DELETE FROM mpf_holdings")
        conn.commit()


def upsert_one(entry: dict) -> None:
    """Insert or update a single holding."""
    save_portfolio([entry])


# ── Helpers ───────────────────────────────────────────────────────────────────
def _float_or_none(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

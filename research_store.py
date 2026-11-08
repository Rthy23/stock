"""PostgreSQL runtime I/O only: no startup DDL, local-history fallbacks or backdating."""
import json
import os
from uuid import uuid4
from datetime import datetime, timezone

import psycopg
from psycopg.types.json import Jsonb
from scientific_factors import METHOD, FACTORS


class ResearchStorageError(RuntimeError):
    pass


def clean(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


class ResearchStore:
    def connect(self):
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise ResearchStorageError("研究持久化資料庫未設定；不會假裝已保存歷史或警報。")
        try:
            return psycopg.connect(url, connect_timeout=5)
        except psycopg.Error as exc:
            raise ResearchStorageError(f"研究資料庫連線失敗（{type(exc).__name__}）。") from None

    def latest(self):
        try:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT id, created_at, payload FROM research_runs "
                    "WHERE method=%s AND COALESCE(payload->>'scope','scan')='scan' "
                    "ORDER BY created_at DESC LIMIT 1", (METHOD,)).fetchone()
            return dict(id=str(row[0]), time=row[1].isoformat(), **row[2]) if row else None
        except psycopg.Error as exc:
            raise ResearchStorageError(f"研究資料表無法讀取（{type(exc).__name__}）；請確認結構已發布。") from None

    def settings(self):
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM research_settings WHERE key='alerts'").fetchone()
        return row[0] if row else {"score_threshold": 1., "sentiment_threshold": 20., "watchlist": []}

    def save_settings(self, settings):
        with self.connect() as conn:
            conn.execute("INSERT INTO research_settings(key,value) VALUES ('alerts',%s) "
                         "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (Jsonb(settings),))

    def append(self, records, universe, failures, sentiment, prices, alerts, background=False, scope="scan"):
        run_id = str(uuid4())
        payload = clean(dict(records=records, universe=universe, failures=failures, sentiment=sentiment, scope=scope))
        with self.connect() as conn:
            stamp = conn.execute(
                "INSERT INTO research_runs(id,background,method,payload) VALUES (%s,%s,%s,%s) RETURNING created_at",
                (run_id, background, METHOD, Jsonb(payload))).fetchone()[0]
            with conn.cursor() as cursor:
                cursor.executemany("INSERT INTO research_scores(run_id,ticker,observed_at,factors) VALUES (%s,%s,%s,%s)",
                                   [(run_id, r["ticker"], stamp, Jsonb(clean(r["factors"]))) for r in records
                                    if "factors" in r])
                cursor.executemany(
                    "INSERT INTO research_prices(ticker,session_close,adjusted_close) VALUES (%s,%s,%s) "
                    "ON CONFLICT(ticker,session_close) DO UPDATE SET adjusted_close=EXCLUDED.adjusted_close, "
                    "observed_at=clock_timestamp()", prices)
                cursor.executemany(
                    "INSERT INTO research_alerts(event_key,ticker,kind,message) VALUES (%s,%s,%s,%s) "
                    "ON CONFLICT(event_key) DO NOTHING",
                    [(a["key"], a.get("ticker"), a["kind"], a["message"]) for a in alerts])
            if scope == "scan":
                # Historical raw inputs/scores remain immutable in research_scores.
                # Only the most recent scan needs a second full UI-payload copy.
                conn.execute("UPDATE research_runs SET payload=payload-'records' "
                             "WHERE id<>%s AND payload ? 'records' "
                             "AND COALESCE(payload->>'scope','scan')='scan'", (run_id,))
            if background:
                symbols = set(universe.get("symbols", []))
                denominator = len(symbols)
                conn.execute("INSERT INTO research_worker(id,completed_at,last_error,coverage) VALUES(1,%s,NULL,%s) "
                             "ON CONFLICT(id) DO UPDATE SET completed_at=EXCLUDED.completed_at,last_error=NULL,"
                             "coverage=EXCLUDED.coverage",
                             (stamp, sum(r["ticker"] in symbols for r in records)/denominator if denominator else 0.))
        return stamp.isoformat()

    def earliest(self):
        with self.connect() as conn:
            return conn.execute("SELECT min(created_at) FROM research_runs WHERE method=%s",
                                (METHOD,)).fetchone()[0]

    def history(self):
        # Only daily-first score projections are needed for IC. Do not pull months
        # of repeated intraday raw-input JSON into the web process.
        projection = ", ".join(
            "'"+key+"', jsonb_build_object('score',s.factors->'"+key+"'->'score')"
            for key in FACTORS)  # constant allowlist, never user-provided SQL
        with self.connect() as conn:
            scores = conn.execute(
                "SELECT DISTINCT ON (s.ticker,(s.observed_at AT TIME ZONE 'America/New_York')::date) "
                "s.ticker,s.observed_at,jsonb_build_object("
                "'composite',s.factors->'composite','universe_member',s.factors->'universe_member',"
                + projection + ") FROM research_scores s JOIN research_runs r "
                "ON r.id=s.run_id WHERE r.method=%s AND s.factors->>'composite' IS NOT NULL ORDER BY s.ticker,"
                "(s.observed_at AT TIME ZONE 'America/New_York')::date,s.observed_at", (METHOD,)).fetchall()
            prices = conn.execute("SELECT ticker,session_close,adjusted_close FROM research_prices "
                                  "ORDER BY ticker,session_close").fetchall()
        return ([dict(ticker=t, time=d.isoformat(), factors=f) for t,d,f in scores],
                [dict(ticker=t, time=d.isoformat(), close=p) for t,d,p in prices])

    def statistics(self):
        with self.connect() as conn:
            count, first = conn.execute(
                "SELECT count(*),min(s.observed_at) FROM research_scores s JOIN research_runs r "
                "ON r.id=s.run_id WHERE r.method=%s", (METHOD,)).fetchone()
        return dict(count=count,first=first.isoformat() if first else None)

    def status(self):
        with self.connect() as conn:
            row = conn.execute("SELECT started_at,completed_at,last_error,coverage FROM research_worker WHERE id=1").fetchone()
        return dict(zip(("started_at","completed_at","last_error","coverage"),
                        [x.isoformat() if isinstance(x, datetime) else x for x in row])) if row else {}

    def mark_worker(self, error=None):
        with self.connect() as conn:
            conn.execute("INSERT INTO research_worker(id,started_at,last_error) VALUES(1,clock_timestamp(),%s) "
                         "ON CONFLICT(id) DO UPDATE SET started_at=EXCLUDED.started_at,last_error=EXCLUDED.last_error",
                         (error,))

    def alerts(self, limit=100):
        with self.connect() as conn:
            return [dict(時間=d.isoformat(), 代碼=t or "大盤", 類型=k, 內容=m)
                    for d,t,k,m in conn.execute(
                        "SELECT created_at,ticker,kind,message FROM research_alerts ORDER BY created_at DESC LIMIT %s",
                        (limit,)).fetchall()]

"""Durable ARK snapshots in Replit's managed PostgreSQL database.

Schema lives in schema/ark_holdings.sql and is applied to development separately;
the application must never create or migrate production tables at startup.
"""

from __future__ import annotations

import os
from datetime import datetime

import psycopg
from psycopg.types.json import Jsonb


class ArkStorageError(RuntimeError):
    """Persistent snapshot storage is unavailable; do not silently use local files."""


class PostgresArkStore:
    def __init__(self, fund: str = "ARKK"):
        self.fund = fund

    def _connect(self):
        # Only the app's PostgreSQL driver receives the runtime-managed URL.
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise ArkStorageError("ARK 持久化資料庫未設定，無法保存跨重新發布的快照。")
        try:
            return psycopg.connect(url, connect_timeout=5)
        except psycopg.Error as exc:
            raise ArkStorageError("ARK 持久化資料庫連線失敗。") from exc

    def status(self) -> dict:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT checked_at, error, unchanged_on_refresh "
                    "FROM ark_fund_refresh WHERE fund = %s",
                    (self.fund,),
                ).fetchone()
        except psycopg.Error as exc:
            raise ArkStorageError("ARK 持久化資料庫無法讀取查詢狀態；請確認資料表已發布。") from exc
        return ({
            "checked_at": row[0].isoformat(),
            "error": row[1],
            "unchanged_on_refresh": row[2],
        } if row else {})

    def latest(self) -> list[dict]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT report_date, snapshot FROM ark_fund_snapshots "
                    "WHERE fund = %s ORDER BY report_date DESC LIMIT 2",
                    (self.fund,),
                ).fetchall()
        except psycopg.Error as exc:
            raise ArkStorageError("ARK 持久化資料庫無法讀取快照；請確認資料表已發布。") from exc
        for report_date, snapshot in rows:
            if (not isinstance(snapshot, dict) or snapshot.get("fund") != self.fund
                    or snapshot.get("date") != report_date.isoformat()
                    or not isinstance(snapshot.get("holdings"), list)):
                raise ArkStorageError("ARK 持久化快照內容與日期不符，停止比較。")
        return [snapshot for _, snapshot in rows]

    def save_snapshot(self, snapshot: dict) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO ark_fund_snapshots (fund, report_date, snapshot) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (fund, report_date) DO UPDATE "
                    "SET snapshot = EXCLUDED.snapshot "
                    "WHERE ark_fund_snapshots.snapshot IS DISTINCT FROM EXCLUDED.snapshot",
                    (self.fund, snapshot["date"], Jsonb(snapshot)),
                )
        except psycopg.Error as exc:
            raise ArkStorageError("ARK 持久化資料庫無法儲存快照；不會改用暫存檔。") from exc

    def save_status(self, status: dict) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO ark_fund_refresh "
                    "(fund, checked_at, error, unchanged_on_refresh) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (fund) DO UPDATE SET "
                    "checked_at = EXCLUDED.checked_at, error = EXCLUDED.error, "
                    "unchanged_on_refresh = EXCLUDED.unchanged_on_refresh",
                    (
                        self.fund, datetime.fromisoformat(status["checked_at"]),
                        status.get("error"), status.get("unchanged_on_refresh", False),
                    ),
                )
        except psycopg.Error as exc:
            raise ArkStorageError("ARK 持久化資料庫無法儲存查詢狀態。") from exc
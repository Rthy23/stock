"""ARKK official daily fund holdings; never inferred personal recommendations."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests


FUND = "ARKK"
SOURCE_URL = (
    "https://assets.ark-funds.com/fund-documents/funds-etf-csv/"
    "ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv"
)
DOCUMENTS_URL = "https://www.ark-funds.com/api/fund/document-table/1004"
CACHE_DIR = Path(".cache/ark_holdings")
REQUIRED_COLUMNS = {
    "date", "fund", "company", "ticker", "cusip", "shares",
    "market value ($)", "weight (%)",
}


class ArkHoldingsError(ValueError):
    """An official source or format problem; never replace with sample data."""


def _decimal(raw: str, field: str, *, optional: bool = False) -> str | None:
    text = (raw or "").strip().replace(",", "").replace("$", "").replace("%", "")
    if optional and not text:
        return None
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ArkHoldingsError(f"ARK CSV 的 {field} 欄位不是數字。") from exc
    if not value.is_finite() or value < 0:
        raise ArkHoldingsError(f"ARK CSV 的 {field} 欄位無效。")
    return str(value)


def parse_holdings(content: bytes, *, today: datetime | None = None) -> dict:
    """Parse the observed official ARKK CSV schema, including its legal footer."""
    if not content or len(content) > 2_000_000:
        raise ArkHoldingsError("ARK CSV 空白或超過大小限制。")
    try:
        stream = io.StringIO(content.decode("utf-8-sig"))
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
            raise ArkHoldingsError("ARK CSV 欄位格式已變更，無法核對基金持倉。")
        rows = []
        reported_date = None
        for row in reader:
            # The official CSV ends in a single quoted, multi-line legal notice.
            if (row.get("date") or "").startswith("Investors should carefully consider") and (
                not row.get("fund")
            ):
                continue
            if None in row or any(row.get(col) is None for col in REQUIRED_COLUMNS):
                raise ArkHoldingsError("ARK CSV 持倉列格式已變更。")
            try:
                report_date = datetime.strptime(row["date"].strip(), "%m/%d/%Y").date()
            except ValueError as exc:
                raise ArkHoldingsError("ARK CSV 持倉日期格式無效。") from exc
            if report_date.weekday() >= 5:
                raise ArkHoldingsError("ARK CSV 日期不是交易日。")
            if row["fund"].strip() != FUND:
                raise ArkHoldingsError("ARK CSV 基金代碼不符 ARKK。")
            if reported_date is not None and report_date != reported_date:
                raise ArkHoldingsError("ARK CSV 含不同的持倉日期。")
            reported_date = report_date
            if not row["company"].strip():
                raise ArkHoldingsError("ARK CSV 缺少公司名稱。")
            cusip = row["cusip"].strip().upper()
            if cusip and not re.fullmatch(r"[A-Z0-9]{9}", cusip):
                raise ArkHoldingsError("ARK CSV 證券識別碼格式已變更。")
            rows.append({
                "company": row["company"].strip(),
                "ticker": row["ticker"].strip().upper(),
                "cusip": cusip,
                "shares": _decimal(row["shares"], "shares", optional=True),
                "market_value": _decimal(row["market value ($)"], "market value ($)"),
                "weight_pct": _decimal(row["weight (%)"], "weight (%)"),
            })
    except UnicodeDecodeError as exc:
        raise ArkHoldingsError("ARK CSV 編碼無法解析。") from exc
    except csv.Error as exc:
        raise ArkHoldingsError("ARK CSV 格式無法解析。") from exc
    if not rows or reported_date is None:
        raise ArkHoldingsError("ARK CSV 沒有可顯示的 ARKK 持倉。")
    reference = (today or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    if reported_date > reference.date():
        raise ArkHoldingsError("ARK CSV 持倉日期在未來。")
    return {
        "fund": FUND,
        "date": reported_date.isoformat(),
        "source_url": SOURCE_URL,
        "holdings": rows,
    }


def fetch_holdings() -> dict:
    try:
        response = requests.get(
            SOURCE_URL, timeout=(5, 18),
            headers={"User-Agent": "US-stock-dashboard/1.0 (official public ARK holdings)"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ArkHoldingsError(f"ARK 官方 CSV 連線失敗：{type(exc).__name__}。") from exc
    if urlparse(response.url).hostname != "assets.ark-funds.com":
        raise ArkHoldingsError("ARK 官方 CSV 重導向到非官方網址。")
    return parse_holdings(response.content)


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as fh:
            value = json.load(fh)
    except FileNotFoundError:
        return {}
    except (ValueError, OSError) as exc:
        raise ArkHoldingsError(f"ARK 本地快照無法讀取：{path.name}。") from exc
    if not isinstance(value, dict):
        raise ArkHoldingsError(f"ARK 本地快照格式錯誤：{path.name}。")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as fh:
            name = fh.name
            json.dump(value, fh, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def compare_snapshots(current: dict, previous: dict | None) -> tuple[list[dict], str | None]:
    """Compare reported share counts by official CUSIP; never infer actual trades."""
    if previous is None or current["date"] <= previous["date"]:
        return [], "尚未保存兩個不同交易日的官方快照，無法比較持倉變化。"
    if current["fund"] != previous["fund"]:
        return [], "基金不一致，無法比較。"
    maps = []
    for snapshot in (previous, current):
        mapping = {}
        for row in snapshot["holdings"]:
            key = row.get("cusip")
            shares = row.get("shares")
            if not key or shares is None or key in mapping or not re.fullmatch(r"[A-Z0-9]{9}", key):
                return [], "證券識別碼或股數缺失／重複，無法可靠比較。"
            try:
                mapping[key] = (row, Decimal(shares))
            except InvalidOperation:
                return [], "股數欄位無法核對，無法比較。"
        maps.append(mapping)
    old, new = maps
    changes = []
    for key in sorted(old.keys() | new.keys()):
        old_row, old_shares = old.get(key, ({}, Decimal(0)))
        new_row, new_shares = new.get(key, ({}, Decimal(0)))
        difference = new_shares - old_shares
        if difference == 0:
            continue
        changes.append({
            "cusip": key,
            "company": new_row.get("company") or old_row["company"],
            "ticker": new_row.get("ticker") or old_row.get("ticker", ""),
            "previous_shares": str(old_shares),
            "current_shares": str(new_shares),
            "change_shares": str(difference),
            "status": ("新增於持倉清單" if key not in old else
                       "已不在持倉清單" if key not in new else
                       "股數增加" if difference > 0 else "股數減少"),
        })
    return changes, None


def get_ark_holdings(
    *, cache_dir: Path | None = None, now: datetime | None = None, fetcher=None
) -> dict:
    """Refresh at most once per UTC day on success; retain dated actual snapshots."""
    directory = cache_dir if cache_dir is not None else CACHE_DIR
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    fetcher = fetcher or fetch_holdings
    status_path = directory / "status.json"
    status = _read_json(status_path)
    snapshots = sorted(directory.glob(f"{FUND}_????-??-??.json"), reverse=True)
    current = _read_json(snapshots[0]) if snapshots else None
    previous = _read_json(snapshots[1]) if len(snapshots) > 1 else None
    checked_at = status.get("checked_at")
    recent = False
    if checked_at:
        try:
            checked = datetime.fromisoformat(checked_at)
            ny_day = now.astimezone(ZoneInfo("America/New_York")).date()
            source_is_older = bool(current and ny_day.weekday() < 5
                                   and current["date"] < ny_day.isoformat())
            recent = (now - checked < timedelta(hours=1) if
                      status.get("error") or source_is_older else checked.date() == now.date())
        except (ValueError, TypeError):
            pass
    if not recent or (current is None and not status.get("error")):
        try:
            fetched = fetcher()
            if (fetched.get("fund") != FUND or fetched.get("source_url") != SOURCE_URL
                    or not fetched.get("holdings")):
                raise ArkHoldingsError("ARK 官方資料缺少 ARKK 基金持倉。")
            report_date = datetime.strptime(fetched["date"], "%Y-%m-%d").date()
            if report_date.weekday() >= 5 or report_date > now.astimezone(
                ZoneInfo("America/New_York")
            ).date():
                raise ArkHoldingsError("ARK 官方持倉日期異常。")
            if current and fetched["date"] < current["date"]:
                raise ArkHoldingsError("ARK 官網 CSV 日期早於最近成功快照。")
            if not current or fetched["date"] != current["date"] or fetched != current:
                _write_json(directory / f"{FUND}_{fetched['date']}.json", fetched)
            if current and fetched["date"] > current["date"]:
                previous = current
            unchanged = bool(current and fetched["date"] == current["date"] and checked_at)
            current = fetched
            status = {
                "checked_at": now.isoformat(),
                "error": None,
                "unchanged_on_refresh": unchanged,
            }
        except (ArkHoldingsError, requests.RequestException, ValueError) as exc:
            status = {
                "checked_at": now.isoformat(),
                "error": str(exc),
                "unchanged_on_refresh": False,
            }
        _write_json(status_path, status)
    changes, comparison_note = compare_snapshots(current, previous) if current else ([], None)
    return {
        "current": current,
        "previous": previous,
        "changes": changes,
        "comparison_note": comparison_note,
        "error": status.get("error"),
        "checked_at": status.get("checked_at"),
        "unchanged_on_refresh": status.get("unchanged_on_refresh", False),
    }
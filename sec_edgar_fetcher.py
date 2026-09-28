"""Read verifiable institutional 13F snapshots, never analyst recommendations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
import threading
import time
from xml.etree import ElementTree as ET

import requests


# Verified against SEC filing index pages; re-check the submissions name at runtime.
# Gotham is 1510387, NOT the 1512466 from the initial request.
MANAGERS = (
    ("warren_buffett", "Berkshire Hathaway Inc.", "1067983", ("BERKSHIRE HATHAWAY",)),
    ("bill_ackman", "Pershing Square Capital Management", "1336528", ("PERSHING SQUARE",)),
    ("michael_burry", "Scion Asset Management", "1649339", ("SCION ASSET MANAGEMENT",)),
    ("seth_klarman", "The Baupost Group", "1061768", ("BAUPOST",)),
    ("david_tepper", "Appaloosa Management", "1656456", ("APPALOOSA",)),
    ("stanley_druckenmiller", "Duquesne Family Office", "1536411", ("DUQUESNE",)),
    # The supplied 1173334 CIK has older Pabrai filings; the recent 13F filer
    # is Dalal Street, LLC (its SEC filings identify Mohnish Pabrai).
    ("mohnish_pabrai", "Dalal Street, LLC（Pabrai 相關申報機構）", "1549575", ("DALAL STREET",)),
    ("joel_greenblatt", "Gotham Asset Management", "1510387", ("GOTHAM ASSET MANAGEMENT",)),
)
MANAGER_IDS = frozenset(manager[0] for manager in MANAGERS)
_lock = threading.Lock()
_last_request = 0.0
_AGENT_PATTERN = re.compile(r"^[^\s@]+(?:\s+[^\s@]+)*\s+[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$")


class EdgarError(Exception):
    """SEC data was unavailable, incomplete or not attributable to this filer."""


def validate_user_agent(user_agent: str) -> str:
    """SEC asks for an identifying application name and a real contact address."""
    value = user_agent.strip()
    if not _AGENT_PATTERN.fullmatch(value) or "example.com" in value.lower():
        raise EdgarError("請在 Secrets 設定 SEC_USER_AGENT，格式為「應用名稱 可聯絡的電子郵件」；不能使用範例地址。")
    return value


def _get(url: str, user_agent: str) -> requests.Response:
    global _last_request
    with _lock:
        wait = 0.12 - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
    try:
        response = requests.get(
            url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=(5, 20),
        )
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise EdgarError(f"SEC 讀取失敗（{url}）：{exc}") from exc


def _filings(data: dict) -> list[dict]:
    """Flatten the SEC column-oriented recent submissions array."""
    recent = data.get("filings", {}).get("recent", {})
    columns = ("form", "accessionNumber", "filingDate", "reportDate", "primaryDocument")
    size = len(recent.get("form", []))
    if any(len(recent.get(column, [])) != size for column in columns):
        raise EdgarError("SEC 申報欄位不完整，無法辨識報告期。")
    return [dict(zip(columns, row)) for row in zip(*(recent[c] for c in columns))
            if row[0] in ("13F-HR", "13F-HR/A") and row[3]]


def _candidate_periods(data: dict, agent: str) -> list[dict]:
    rows = _filings(data)
    # SEC recent lists may not contain two 13F periods for infrequent filers.
    if len({row["reportDate"] for row in rows}) < 2:
        for item in data.get("filings", {}).get("files", []):
            name = item.get("name", "")
            if not re.fullmatch(r"CIK\d+-submissions-\d+\.json", name):
                continue
            rows += _filings(_get(f"https://data.sec.gov/submissions/{name}", agent).json())
            if len({row["reportDate"] for row in rows}) >= 2:
                break
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["reportDate"]].append(row)
    return [sorted(groups[period], key=lambda r: (r["filingDate"], r["accessionNumber"]), reverse=True)
            for period in sorted(groups, reverse=True)]


def _text(node: ET.Element, name: str) -> str:
    for child in node:
        if child.tag.rsplit("}", 1)[-1].lower() == name.lower():
            return (child.text or "").strip()
    return ""


def _amount(text: str) -> Decimal:
    try:
        value = Decimal(text.replace(",", ""))
        if not value.is_finite() or value < 0:
            raise ValueError(text)
        return value
    except (InvalidOperation, ValueError):
        raise EdgarError(f"SEC 資訊表數量欄位無效：{text!r}") from None


def parse_information_table(xml: bytes) -> list[dict]:
    """Parse the SEC XML informationTable without inferring exchange tickers."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise EdgarError("SEC 資訊表 XML 無法解析。") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "informationtable":
        raise EdgarError("檔案不是 SEC 13F informationTable。")
    holdings = {}
    for node in root:
        if node.tag.rsplit("}", 1)[-1].lower() != "infotable":
            continue
        cusip = _text(node, "cusip").upper()
        issuer = _text(node, "nameOfIssuer")
        title = _text(node, "titleOfClass")
        option = _text(node, "putCall").upper()
        quantity_node = next((child for child in node if child.tag.rsplit("}", 1)[-1].lower() == "shrsorprnamt"), None)
        quantity = _text(quantity_node, "sshPrnamt") if quantity_node is not None else ""
        unit = _text(quantity_node, "sshPrnamtType").upper() if quantity_node is not None else ""
        if not cusip or not issuer or not quantity or unit not in ("SH", "PRN"):
            raise EdgarError("SEC 資訊表缺少 CUSIP、名稱或有效的持有數量單位。")
        key = (cusip, title.upper(), option, unit)
        if key not in holdings:
            holdings[key] = {"cusip": cusip, "issuer": issuer, "class": title,
                             "option": option, "unit": unit, "quantity": Decimal(0),
                             "value_reported": Decimal(0)}
        holdings[key]["quantity"] += _amount(quantity)
        holdings[key]["value_reported"] += _amount(_text(node, "value"))
    if not holdings:
        raise EdgarError("SEC 申報沒有可解析的持倉明細，可能是僅通知申報。")
    return sorted(holdings.values(), key=lambda h: h["value_reported"], reverse=True)


def compare_holdings(current: list[dict], previous: list[dict]) -> list[dict]:
    """Compare like-for-like CUSIP/class/option/unit; reported value is not price-adjusted."""
    def key(item: dict) -> tuple:
        return (item["cusip"], item["class"].upper(), item["option"], item["unit"])

    old = {key(item): item for item in previous}
    new = {key(item): item for item in current}
    changes = []
    for identity in new.keys() | old.keys():
        now, before = new.get(identity), old.get(identity)
        qty_now = now["quantity"] if now else Decimal(0)
        qty_before = before["quantity"] if before else Decimal(0)
        status = ("新增" if before is None else "出清" if now is None else
                  "加碼" if qty_now > qty_before else "減碼" if qty_now < qty_before else "未變")
        item = now or before
        changes.append({**{k: item[k] for k in ("cusip", "issuer", "class", "option", "unit")},
                        "status": status, "previous_quantity": qty_before, "quantity": qty_now,
                        "previous_value_reported": before["value_reported"] if before else Decimal(0),
                        "value_reported": now["value_reported"] if now else Decimal(0)})
    return sorted(changes, key=lambda item: (item["status"] == "未變", item["issuer"], item["cusip"]))


def _amendment_type(base: str, primary_document: str, agent: str) -> str:
    """Read the actual SEC primary document; unknown amendments are not complete."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.xml", primary_document, re.I):
        return "UNKNOWN"
    try:
        root = ET.fromstring(_get(base + "/" + primary_document, agent).content)
    except (ET.ParseError, EdgarError):
        return "UNKNOWN"
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1].lower() == "amendmenttype":
            value = (node.text or "").strip().upper()
            if value in ("RESTATEMENT", "NEW HOLDINGS"):
                return value
    return "UNKNOWN"


def _read_period(cik: str, rows: list[dict], agent: str) -> tuple[dict, str]:
    for row in rows:
        base = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{row['accessionNumber'].replace('-', '')}")
        if row["form"] == "13F-HR/A":
            amendment = _amendment_type(base, row["primaryDocument"], agent)
            if amendment != "RESTATEMENT":
                # If the latest filing adds entries, falling back to the original
                # would claim incomplete holdings are the full quarter. Fail closed.
                raise EdgarError(
                    f"報告期 {row['reportDate']} 有"
                    f"{'新增持倉' if amendment == 'NEW HOLDINGS' else '無法辨識'}修正申報；"
                    "未合併前不提供完整持倉或兩期變化。"
                )
        index = _get(base + "/index.json", agent).json()
        names = [item.get("name", "") for item in index.get("directory", {}).get("item", [])]
        for name in names:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.xml", name, re.I) or name.lower() == row["primaryDocument"].lower():
                continue
            response = _get(f"{base}/{name}", agent)
            try:
                holdings = parse_information_table(response.content)
            except EdgarError as exc:
                if "檔案不是 SEC 13F" in str(exc):
                    continue
                raise
            return {
                "report_date": row["reportDate"], "filing_date": row["filingDate"],
                "form": row["form"], "source_url": base + "/" + name,
                "filing_url": base + "/" + row["accessionNumber"] + "-index.htm",
                "holdings": holdings,
            }, ""
        raise EdgarError(f"申報 {row['accessionNumber']} 未找到可解析的 information table XML。")
    raise EdgarError("此報告期沒有可讀取的完整持倉申報。")


def fetch_manager(manager: tuple[str, str, str, tuple[str, ...]], user_agent: str) -> dict:
    agent = validate_user_agent(user_agent)
    manager_id, label, cik, expected_names = manager
    try:
        data = _get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", agent).json()
    except ValueError as exc:
        raise EdgarError("SEC 申報清單回傳了無效 JSON。") from exc
    actual_name = data.get("name", "")
    if str(data.get("cik", "")).lstrip("0") != str(int(cik)) or not any(
        alias in actual_name.upper() for alias in expected_names
    ):
        raise EdgarError(f"CIK {cik} 的 SEC 申報名稱「{actual_name}」與預期的 {label} 不符；已停止顯示。")
    groups = _candidate_periods(data, agent)
    if not groups:
        raise EdgarError(f"{label} 沒有可用的 13F-HR 申報。")
    current, note = _read_period(cik, groups[0], agent)
    previous = None
    if len(groups) > 1:
        try:
            previous, previous_note = _read_period(cik, groups[1], agent)
            note = "；".join(filter(None, (note, previous_note)))
        except EdgarError as exc:
            note = "；".join(filter(None, (note, f"前一期無法讀取：{exc}")))
    return {
        "manager_id": manager_id, "label": label, "sec_name": actual_name,
        "cik": cik, "current": current, "previous": previous,
        "changes": compare_holdings(current["holdings"], previous["holdings"]) if previous else [],
        "note": note, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
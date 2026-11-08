"""Small, source-labelled research comparisons; never substitute invented inputs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yfinance as yf

from market_data import get_history, get_fundamentals, get_sp500, number

SECTOR_ETFS = {
    "Technology": "XLK", "Information Technology": "XLK",
    "Healthcare": "XLV", "Health Care": "XLV",
    "Financial Services": "XLF", "Financials": "XLF",
    "Consumer Cyclical": "XLY", "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Communication Services": "XLC", "Industrials": "XLI",
    "Energy": "XLE", "Utilities": "XLU", "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Materials": "XLB",
}


def source_window(values):
    stamps = []
    for value in values:
        if value is None:
            continue
        stamp = pd.to_datetime(value, unit="s", utc=True, errors="coerce") if isinstance(
            value, (int, float)) else pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(stamp):
            stamps.append(stamp)
    return f"{min(stamps).isoformat()} ～ {max(stamps).isoformat()}" if stamps else "來源時間未知"


def vix_percentile(series):
    """Midrank empirical percentile vs preceding 252 sessions; no future values."""
    def rank(window):
        past, value = window[:-1], window[-1]
        return ((past < value).sum() + .5 * (past == value).sum()) / len(past) * 100
    return series.dropna().rolling(253, min_periods=253).apply(rank, raw=True)


def score_text(value, decimals=2, missing="數據不足"):
    return f"{value:+.{decimals}f}" if number(value) is not None else missing


def relative_returns(stock, benchmark):
    """Return differences over 21/63/126 *common daily intervals*, in percentage points."""
    if stock is None or benchmark is None:
        return {m: None for m in (1, 3, 6)}
    def daily(df):
        s = df["Close"].dropna().copy()
        s.index = pd.to_datetime([t.date() for t in s.index])
        return s[~s.index.duplicated(keep="last")]
    common = pd.concat([daily(stock), daily(benchmark)], axis=1, join="inner").dropna()
    result = {}
    for months, bars in ((1, 21), (3, 63), (6, 126)):
        result[months] = (float(((common.iloc[-1, 0] / common.iloc[-bars-1, 0] - 1)
                                - (common.iloc[-1, 1] / common.iloc[-bars-1, 1] - 1)) * 100)
                          if len(common) > bars else None)
    return result


def trend_structure(hist, interval="1d"):
    """Slope is MA change over five bars, not annualised regression."""
    close = hist["Close"].dropna() if hist is not None else pd.Series(dtype=float)
    result = {"unit": "日" if interval == "1d" else "根 K 線",
              "time": str(hist.index[-1]) if hist is not None and not hist.empty else "未知"}
    for window in (50, 200):
        ma = close.rolling(window).mean()
        slope = (float((ma.iloc[-1] / ma.iloc[-6] - 1) * 100)
                 if len(close) >= window + 5 and ma.iloc[-6] != 0 else None)
        result[window] = {
            "value": number(ma.iloc[-1]) if len(ma) else None,
            "slope": slope,
            "above": bool(close.iloc[-1] > ma.iloc[-1]) if len(close) >= window else None}
    a, b = result[50], result[200]
    if None in (a["value"], b["value"], a["slope"], b["slope"]):
        result["trend"] = "數據不足"
    elif a["above"] and b["above"] and a["value"] > b["value"] and a["slope"] > 0 and b["slope"] > 0:
        result["trend"] = "上升趨勢（價格、排列與斜率一致）"
    elif not a["above"] and not b["above"] and a["value"] < b["value"] and a["slope"] < 0 and b["slope"] < 0:
        result["trend"] = "下降趨勢（價格、排列與斜率一致）"
    else:
        result["trend"] = "混合／過渡結構"
    return result


def breadth_record(ticker, hist):
    # A daily bar may be incomplete during trading: explicitly labelled in UI.
    close = hist["Close"].dropna()
    if len(close) < 2:
        raise ValueError("日線不足")
    return {"ticker": ticker, "change": float(close.iloc[-1] - close.iloc[-2]),
            "above50": bool(close.iloc[-1] > close.tail(50).mean()) if len(close) >= 50 else None,
            "above200": bool(close.iloc[-1] > close.tail(200).mean()) if len(close) >= 200 else None,
            "time": str(close.index[-1])}


def summarize_breadth(records, total):
    # Do not combine different source sessions into a supposedly same-day breadth.
    date = max((str(r["time"])[:10] for r in records), default=None)
    rows = [r for r in records if str(r["time"])[:10] == date]
    result = {"date": date, "valid": len(rows), "total": total,
              "coverage": len(rows)/total if total else 0,
              "advances": sum(r["change"] > 0 for r in rows),
              "declines": sum(r["change"] < 0 for r in rows),
              "unchanged": sum(r["change"] == 0 for r in rows)}
    for window in (50, 200):
        values = [r[f"above{window}"] for r in rows if r.get(f"above{window}") is not None]
        result[f"above{window}"] = sum(values)/len(values) if values else None
        result[f"count{window}"] = len(values)
    return result


def fetch_breadth(progress=None):
    pool = get_sp500()["symbols"]
    records, errors = [], {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = {executor.submit(get_history, t): t for t in pool}
        for job in as_completed(jobs):
            ticker = jobs[job]
            try:
                records.append(breadth_record(ticker, job.result()))
            except Exception as exc:
                errors[ticker] = str(exc)
            if progress:
                progress(len(records)+len(errors), len(pool))
    return {"summary": summarize_breadth(records, len(pool)), "errors": errors,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "records": records}


@st.cache_data(ttl=900)
def sector_strength():
    spy = get_history("SPY")
    rows = []
    for etf in sorted(set(SECTOR_ETFS.values())):
        try:
            hist = get_history(etf)
            rows.append({"板塊 ETF": etf, **{
                f"{m}M 相對 SPY（百分點）": value
                for m, value in relative_returns(hist, spy).items()},
                "ETF 日線時間": str(hist.index[-1]), "SPY 日線時間": str(spy.index[-1]),
                "抓取時間（非來源）": hist.attrs.get("received_at")})
        except Exception:
            rows.append({"板塊 ETF": etf, "狀態": "資料不足"})
    return rows


def historical_forward_pe(table):
    """Only a genuine historical Forward PE series is acceptable, never trailing PE."""
    if table is None or table.empty:
        return pd.Series(dtype=float)
    labels = {"".join(c for c in str(i).lower() if c.isalnum()): i for i in table.index}
    key = labels.get("forwardperatio") or labels.get("forwardpe")
    if key is None:
        return pd.Series(dtype=float)
    raw = table.loc[key]
    values = {}
    for label, value in raw.items():
        date = pd.to_datetime(label, errors="coerce")
        value = number(value)
        if pd.notna(date) and value is not None and value > 0:
            values[date] = value
    return pd.Series(values, dtype=float).sort_index()


@st.cache_data(ttl=900)
def valuation_comparison(ticker, sector):
    current = get_fundamentals(ticker)
    etf = SECTOR_ETFS.get(sector)
    members = get_sp500().get("sectors", {})
    peers = [t for t, s in members.items() if etf and SECTOR_ETFS.get(s) == etf]
    values, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = {executor.submit(get_fundamentals, t): t for t in peers}
        for job in as_completed(jobs):
            try:
                fact = job.result()
                pe = number(fact.get("forward_pe"))
                if pe is not None and pe > 0:
                    values.append({"ticker": jobs[job], "pe": pe,
                                   "received_at": fact.get("received_at")})
            except Exception:
                failures.append(jobs[job])
    history_error = None
    try:
        history = historical_forward_pe(yf.Ticker(ticker).get_valuation_measures(
            freq="quarterly", periods=None))
    except Exception as exc:
        history = pd.Series(dtype=float)
        history_error = str(exc)
    pe = number(current.get("forward_pe"))
    median = float(pd.Series([v["pe"] for v in values]).median()) if len(values) >= 5 else None
    return {"current": pe, "median": median, "sample": len(values), "pool": len(peers),
            "premium": (pe / median - 1)*100 if pe is not None and pe > 0 and median else None,
            "history": history, "history_error": history_error, "failures": failures,
            "received_at": current.get("received_at"), "samples": values}


@st.cache_data(ttl=900)
def earnings_event(ticker):
    facts = get_fundamentals(ticker)
    raw = facts.get("raw", {})
    # An earnings window is a window, not a confirmed point timestamp.
    start = raw.get("earningsTimestampStart") or raw.get("earningsTimestamp")
    end = raw.get("earningsTimestampEnd") or start
    return {"start": start, "end": end, "received_at": facts.get("received_at"),
            "confirmed": bool(raw.get("isEarningsDateEstimate") is False)}

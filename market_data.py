"""Replaceable, free market-data adapter. Receipt times are NEVER quote times."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import StringIO
from threading import RLock
from typing import Protocol
from zoneinfo import ZoneInfo
import math

import exchange_calendars as xcals
import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
QUOTE_TTL = 60
FUNDAMENTAL_TTL = 900
HISTORY_TTL = 900
_cache: dict = {}
_lock = RLock()


def number(value):
    """Preserve real zeros; reject absent/non-finite values."""
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (ValueError, TypeError):
        return None


def source_time(value):
    try:
        return datetime.fromtimestamp(float(value), UTC).isoformat() if value else None
    except (ValueError, TypeError, OverflowError):
        return None


def market_status(now=None):
    now = now or datetime.now(UTC)
    local = now.astimezone(ET)
    calendar = xcals.get_calendar("XNYS")
    day = pd.Timestamp(local.date())
    if not calendar.is_session(day):
        return {"label": "休市", "open": False, "elapsed": None, "now": local}
    opening = calendar.session_open(day).to_pydatetime()
    closing = calendar.session_close(day).to_pydatetime()
    if opening <= now < closing:
        label = "盤中"
    elif local.hour < 4 or local.hour >= 20:
        label = "休市"
    elif now < opening:
        label = "盤前"
    else:
        label = "盤後"
    elapsed = min(1., max(0., (now - opening).total_seconds() /
                         (closing - opening).total_seconds()))
    return {"label": label, "open": label == "盤中", "elapsed": elapsed,
            "now": local, "session_open": opening, "session_close": closing}


def relative_volume(volume, avg_volume, now=None):
    state = market_status(now)
    if volume is None or avg_volume is None or avg_volume <= 0:
        return None
    # A linear intraday approximation, not an empirically fitted volume curve.
    fraction = state["elapsed"] if state["open"] else 1.
    if state["label"] == "盤前" or not fraction:
        return None
    return volume / (avg_volume * fraction)


def quote_volume_ratio(quote, avg_volume, now=None):
    now = now or datetime.now(UTC)
    stamp = quote.get("regular", {}).get("source_time")
    if not stamp:
        return None
    source = datetime.fromisoformat(stamp)
    state = market_status(now)
    if source.astimezone(ET).date() != now.astimezone(ET).date():
        return None
    if "session_open" not in state or source < state["session_open"]:
        return None
    return relative_volume(quote.get("volume"), avg_volume, source)


class MarketProvider(Protocol):
    def quote(self, ticker: str) -> dict: ...
    def fundamentals(self, ticker: str) -> dict: ...
    def history(self, ticker: str, period: str, interval: str) -> pd.DataFrame: ...


class YahooProvider:
    def quote(self, ticker):
        stock = yf.Ticker(ticker)
        info = {}
        fast = {}
        try:
            fast = {"lastPrice": stock.fast_info.last_price,
                    "previousClose": stock.fast_info.previous_close,
                    "lastVolume": stock.fast_info.last_volume}
        except Exception:
            pass
        try:
            info = stock.get_info() or {}
        except Exception:
            pass
        regular = number(info.get("regularMarketPrice"))
        stamp = source_time(info.get("regularMarketTime"))
        if regular is None:
            regular = number(fast.get("lastPrice"))
            stamp = None  # fast_info has no trustworthy last-trade timestamp.
        if regular is None:
            raise ValueError(f"{ticker}：Yahoo 沒有可用報價")
        previous = number(info.get("regularMarketPreviousClose"))
        if previous is None:
            previous = number(fast.get("previousClose"))
        return {
            "ticker": ticker, "source": "Yahoo Finance / yfinance",
            "regular": {"price": regular, "source_time": stamp},
            "pre": {"price": number(info.get("preMarketPrice")),
                    "source_time": source_time(info.get("preMarketTime"))},
            "post": {"price": number(info.get("postMarketPrice")),
                     "source_time": source_time(info.get("postMarketTime"))},
            "previous_close": previous,
            "change_pct": ((regular / previous - 1) * 100
                           if previous is not None and previous > 0 else None),
            "volume": number(info.get("regularMarketVolume", fast.get("lastVolume"))),
        }

    def fundamentals(self, ticker):
        info = yf.Ticker(ticker).get_info() or {}
        if not info.get("symbol") and not info.get("longName"):
            raise ValueError(f"{ticker}：基本面資料不足")
        quarter = source_time(info.get("mostRecentQuarter"))
        quarter = quarter[:10] if quarter else None
        return {
            "ticker": ticker, "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "資料不足",
            "market_cap": number(info.get("marketCap")),
            "net_margin": number(info.get("profitMargins")),
            "pe_ratio": number(info.get("trailingPE")),
            "forward_pe": number(info.get("forwardPE")),
            "revenue_growth": number(info.get("revenueGrowth")),
            "52w_high": number(info.get("fiftyTwoWeekHigh")),
            "52w_low": number(info.get("fiftyTwoWeekLow")),
            "dividend_yield": number(info.get("dividendYield")),
            "eps": number(info.get("trailingEps")), "beta": number(info.get("beta")),
            "financial_period": quarter,
            "margin_period": f"TTM，截至 {quarter}" if quarter else "財報期間資料不足",
            "growth_period": f"最近季度 {quarter}（YoY）" if quarter else "財報期間資料不足",
            "eps_period": f"TTM，截至 {quarter}" if quarter else "財報期間資料不足",
            "period_note": "以 Yahoo mostRecentQuarter 標示，未逐欄對帳原始財報。",
            "raw": info,
        }

    def history(self, ticker, period="1y", interval="1d"):
        kwargs = {"period": period, "interval": interval}
        if period == "3y":
            end = datetime.now(UTC).date()
            kwargs = {"start": (end - timedelta(days=1096)).isoformat(),
                      "end": (end + timedelta(days=1)).isoformat(), "interval": interval}
        frame = yf.Ticker(ticker).history(**kwargs, auto_adjust=True, prepost=False)
        if frame is None or frame.empty:
            raise ValueError(f"{ticker}：adjusted OHLCV 資料不足")
        frame.attrs["price_basis"] = "adjusted"
        return frame


provider: MarketProvider = YahooProvider()


def clear_market_cache():
    with _lock:
        _cache.clear()
    from yfinance.data import YfData
    YfData().cache_get.cache_clear()


def _get(key, ttl, fetch, force=False):
    now = datetime.now(UTC)
    with _lock:
        entry = _cache.get(key)
    if not force and entry and (now - entry[0]).total_seconds() < ttl:
        result = deepcopy(entry[1])
        hit = True
    else:
        result = fetch()  # Failed requests are never cached as successful data.
        now = datetime.now(UTC)
        with _lock:
            _cache[key] = (now, deepcopy(result))
        hit = False
    if isinstance(result, dict):
        result["cache_hit"] = hit
        result["received_at"] = (entry[0] if hit else now).isoformat()
    elif isinstance(result, pd.DataFrame):
        result.attrs["cache_hit"] = hit
        result.attrs["received_at"] = (entry[0] if hit else now).isoformat()
    return result


def get_quote(ticker, force=False):
    return _get(("quote", ticker), QUOTE_TTL, lambda: provider.quote(ticker), force)


def get_fundamentals(ticker, force=False):
    return _get(("fundamentals", ticker), FUNDAMENTAL_TTL,
                lambda: provider.fundamentals(ticker), force)


def get_history(ticker, period="1y", interval="1d", force=False):
    return _get(("history", ticker, period, interval), HISTORY_TTL,
                lambda: provider.history(ticker, period, interval), force)


def get_sp500(force=False):
    def fetch():
        sources = [
            ("Wikipedia S&P 500 成分表", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
            ("公開 S&P 500 成分鏡像（datasets/s-and-p-500-companies）",
             "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"),
        ]
        for label, url in sources:
            try:
                r = requests.get(url, timeout=25, headers={"User-Agent": "StockResearchDashboard/1.0"})
                r.raise_for_status()
                table = (pd.read_html(StringIO(r.text))[0] if "wikipedia" in url
                         else pd.read_csv(StringIO(r.text)))
                symbols = sorted(set(table["Symbol"].str.replace(".", "-", regex=False)))
                if not 490 <= len(symbols) <= 520:
                    raise ValueError("成分數量無法核對")
                sector_col = next((c for c in ("GICS Sector", "Sector") if c in table), None)
                sectors = ({str(row["Symbol"]).replace(".", "-"): row[sector_col]
                            for _, row in table.iterrows()} if sector_col else {})
                return {"symbols": symbols, "sectors": sectors, "source": label, "url": url}
            except Exception:
                continue
        raise ValueError("S&P 500 成分表無法取得；不會以舊的 93 支名單替代。")
    return _get(("sp500",), 86400, fetch, force)

"""Honest quote labels and a market-hours-only, 60-second quote refresh loop."""
from datetime import datetime, timezone
import math
import streamlit as st
from market_data import ET, clear_market_cache, get_quote, get_history, market_status, quote_volume_ratio


def fmt(value, decimals=2, suffix=""):
    if value is None:
        return "資料不足"
    try:
        return f"{value:,.{decimals}f}{suffix}" if math.isfinite(float(value)) else "資料不足"
    except (ValueError, TypeError):
        return "資料不足"


def quote_label(quote, session="regular", now=None):
    part = (quote or {}).get(session, {})
    stamp = part.get("source_time")
    receipt = (quote or {}).get("received_at")
    now = now or datetime.now(timezone.utc)
    reused = bool(receipt and (now - datetime.fromisoformat(receipt)).total_seconds() > 2)
    cached = " · 快取" if (quote or {}).get("cache_hit") or reused else ""
    if not stamp and not receipt:
        return "報價來源時間資料不足 · 年齡未知" + cached
    source = datetime.fromisoformat(stamp or receipt)
    seconds = (now - source).total_seconds()
    if seconds < -60:
        return ("來源時間異常" if stamp else "抓取時間異常") + " · 年齡未知" + cached
    age = max(0, int(seconds))
    text = f"{age} 秒前" if age < 60 else f"{age//60} 分鐘前"
    if not stamp:
        fetch_age = f"{age} 秒" if age < 60 else f"{age//60} 分鐘"
        return (f"抓取時間 {source.astimezone(ET):%Y-%m-%d %H:%M:%S} ET（非報價時間）"
                f" · 距抓取 {fetch_age} · 報價來源時間未知{cached}")
    stale = " · 可能過期" if market_status(now)["open"] and age > 300 else ""
    return f"報價時間 {source.astimezone(ET):%Y-%m-%d %H:%M:%S} ET · {text}{cached}{stale}"


def render_quote_details(ticker, quote=None):
    try:
        quote = quote or get_quote(ticker)
    except Exception as exc:
        st.warning(f"{ticker}：資料不足（{exc}）")
        return
    for col, session, name in zip(st.columns(3), ("regular", "pre", "post"),
                                  ("Regular 正常時段", "Pre-market 盤前", "Post-market 盤後")):
        with col:
            st.metric(f"{ticker} · {name}", fmt(quote[session]["price"]))
            st.caption(quote_label(quote, session))
    st.caption("Yahoo Finance；以上為各時段原始成交報價，不混合。技術分析／回測使用 adjusted close。")


def _active_tickers():
    page = st.session_state.get("page_radio", "")
    rows = st.session_state.get("results") or []
    if "回測" in page:
        return list(dict.fromkeys(r["ticker"] for r in rows))
    if "Micro" in page:
        data = st.session_state.get("diag_stock_info") or {}
        return [data["ticker"]] if data.get("ticker") else []
    if "持倉" in page and "官方" not in page:
        return list((st.session_state.get("portfolio") or {}).keys())
    return [st.session_state.get("benchmark", "VOO")]


def _refresh(tickers, force=False):
    errors = {}
    quotes = st.session_state.setdefault("live_quotes", {})
    if force:
        st.cache_data.clear()
        clear_market_cache()
        for ticker in tickers:
            for key in (f"factor7_{ticker}", f"factor7_data_{ticker}", f"factor7_{ticker}_timing"):
                st.session_state.pop(key, None)
    for ticker in tickers:
        try:
            quote = get_quote(ticker, force=True)
            quotes[ticker] = quote
            for key in ("scan_records", "results"):
                for row in st.session_state.get(key) or []:
                    if row["ticker"] == ticker:
                        row.update(quote=quote, price=quote["regular"]["price"],
                                   change_pct=quote["change_pct"], volume=quote["volume"],
                                   volume_ratio=quote_volume_ratio(quote, row.get("avg_volume20")))
            diag = st.session_state.get("diag_stock_info") or {}
            if diag.get("ticker") == ticker:
                diag.update(price=quote["regular"]["price"], quote=quote)
            if force:
                from data_fetcher import get_stock_info
                from screener_engine import build_record
                if diag.get("ticker") == ticker:
                    st.session_state["diag_stock_info"] = get_stock_info(ticker)
                    st.session_state["diag_hist"] = get_history(
                        ticker, st.session_state.get("diag_period", "1y"),
                        "1h" if st.session_state.get("diag_period") == "1d" else "1d", force=True)
                for row in st.session_state.get("scan_records") or []:
                    if row["ticker"] == ticker:
                        row.update(build_record(ticker, force=True))
        except Exception as exc:
            errors[ticker] = str(exc)  # Keep previous source timestamp, never stamp stale values as new.
    st.session_state["quote_errors"] = errors
    st.session_state["quote_request_at"] = datetime.now(timezone.utc).isoformat()


@st.fragment(run_every=1)
def render_market_header():
    state = market_status()
    st.markdown(f"### 市場狀態：{state['label']} · {state['now']:%Y-%m-%d %H:%M:%S} ET")
    tickers = _active_tickers()
    if st.button("立即刷新", key="force_market_refresh", help="清除所有應用快取，重新請求目前顯示的報價與相關資料"):
        with st.spinner("繞過應用快取，重新抓取已顯示股票…"):
            _refresh(tickers, force=True)
        st.rerun()
    stamp = st.session_state.get("quote_request_at")
    age = ((datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()
           if stamp else 60)
    if tickers and ((state["open"] and age >= 60) or stamp is None):
        _refresh(tickers)
        st.rerun()
    if state["open"] and tickers:
        st.caption(f"報價自動刷新倒數：{max(0, 60-int(age))} 秒 · 僅刷新目前顯示股票")
    else:
        st.caption("非正常交易時段，已停止自動抓取報價；仍可立即刷新。")
    if stamp:
        st.caption(f"最後請求完成：{datetime.fromisoformat(stamp).astimezone(ET):%Y-%m-%d %H:%M:%S} ET"
                   "（不是報價來源時間）")
    st.caption("歷史圖／技術指標／回測統一 adjusted close（含拆股與股息調整）；"
               "現貨 regular／盤前／盤後報價保留實際成交價格。")
    errors = st.session_state.get("quote_errors") or {}
    if errors:
        st.warning("報價刷新失敗，不能視為最新：" + "、".join(errors))
    if tickers:
        with st.expander("已顯示標的的各時段報價與來源時間", expanded=False):
            chosen = st.selectbox("查看各時段報價", tickers, key="header_quote_ticker")
            for ticker in [chosen]:
                quote = (st.session_state.get("live_quotes") or {}).get(ticker)
                if quote is None:
                    quote = next((r.get("quote") for r in st.session_state.get("results") or []
                                  if r["ticker"] == ticker), None)
                if quote is None:
                    diag = st.session_state.get("diag_stock_info") or {}
                    quote = diag.get("quote") if diag.get("ticker") == ticker else None
                if quote:
                    render_quote_details(ticker, quote)
                else:
                    st.caption(f"{ticker}：報價資料不足，按立即刷新取得")

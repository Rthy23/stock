# ═══════════════════════════════════════════════════════════════════════════════
# gemini_helper.py — Centralised Gemini API layer
#
# Provides:
#   call_gemini_cached()  — @st.cache_data(ttl=3600) + tenacity retry
#   call_gemini_raw()     — retry only (no cache; for image / non-text flows)
#   is_quota_error()      — classify exception
#   render_quota_error()  — user-friendly Streamlit error widget
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import time
from datetime import datetime, timedelta

import streamlit as st
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)

_DEFAULT_MODEL = "gemini-2.0-flash"
_MAX_ATTEMPTS  = 3          # total tries
_WAIT_MIN      = 4          # seconds before first retry
_WAIT_MAX      = 60         # cap on backoff delay
_WAIT_MULT     = 2          # exponential multiplier


# ── Error classification ──────────────────────────────────────────────────────

def is_quota_error(e: Exception) -> bool:
    """Return True for 429 / RESOURCE_EXHAUSTED responses."""
    msg = str(e).lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "quota" in msg
        or "rate limit" in msg
        or "rateLimitExceeded" in msg.lower()
    )


def is_auth_error(e: Exception) -> bool:
    """Return True for 403 / API key errors."""
    msg = str(e).lower()
    return "403" in msg or "leaked" in msg or "api_key_invalid" in msg


def is_retryable(e: Exception) -> bool:
    """Return True if the error is transient and worth retrying."""
    if is_auth_error(e):
        return False          # 403 will never self-heal — don't waste retries
    msg = str(e).lower()
    return (
        is_quota_error(e)
        or "timeout" in msg
        or "connection" in msg
        or "503" in msg
        or "500" in msg
    )


# ── Low-level raw caller with tenacity ───────────────────────────────────────

def _do_call(prompt: str, api_key: str, model_name: str) -> str:
    """Single attempt — raises on any error."""
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=api_key)
    model    = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


@retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_exponential(multiplier=_WAIT_MULT, min=_WAIT_MIN, max=_WAIT_MAX),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    reraise=True,
)
def _do_call_with_retry(prompt: str, api_key: str, model_name: str) -> str:
    """Tenacity-wrapped single-call. Retries on 429 / timeout with exponential backoff."""
    return _do_call(prompt, api_key, model_name)


def call_gemini_raw(
    prompt: str,
    api_key: str,
    model_name: str = _DEFAULT_MODEL,
) -> str:
    """
    Call Gemini with retry but WITHOUT caching.
    Use this for image / binary payloads or truly dynamic results.
    Returns the response text, or raises a descriptive exception.
    """
    try:
        return _do_call_with_retry(prompt, api_key, model_name)
    except RetryError as re:
        raise re.last_attempt.exception() from re  # unwrap tenacity wrapper


# ── Cached wrapper (cache key = prompt + model) ───────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def call_gemini_cached(
    prompt: str,
    api_key: str,
    model_name: str = _DEFAULT_MODEL,
) -> str:
    """
    Call Gemini with 1-hour result cache + exponential-backoff retry.

    • Identical prompts within the same hour return instantly from cache.
    • On 429 / timeout, retries up to 3 times with exponential backoff.
    • Raises exception on permanent failure — caller decides how to display it.

    Cache key is (prompt, api_key, model_name) — different tickers / portfolios
    automatically get separate cache slots.
    """
    return call_gemini_raw(prompt, api_key, model_name)


# ── Prompt builders (pure functions, no I/O) ─────────────────────────────────

def build_stock_prompt(stock_info: dict, hist, ticker: str) -> str:
    """Build the individual-stock analysis prompt string."""
    price      = stock_info.get("price", "N/A")
    pe         = stock_info.get("pe_ratio", "N/A")
    net_margin = round((stock_info.get("net_margin") or 0) * 100, 2)
    rev_growth = round((stock_info.get("revenue_growth") or 0) * 100, 2)
    mkt_cap    = stock_info.get("market_cap", "N/A")
    eps        = stock_info.get("eps", "N/A")
    beta       = stock_info.get("beta", "N/A")
    sector     = stock_info.get("sector", "N/A")
    name       = stock_info.get("name", ticker)

    hist_summary = ""
    if hist is not None and not hist.empty:
        ret_1y  = round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 1)
        hi_52   = round(float(hist["High"].max()), 2)
        lo_52   = round(float(hist["Low"].min()),  2)
        hist_summary = (
            f"過去1年報酬率：{ret_1y}%，52週高點：${hi_52}，"
            f"52週低點：${lo_52}，近期收盤：${hist['Close'].iloc[-1]:.2f}"
        )

    return (
        f"你是一位華爾街資深量化分析師，請以繁體中文對以下美股進行深度分析：\n\n"
        f"【公司】{name}（{ticker}），板塊：{sector}\n"
        f"【基本面】股價：${price}，P/E：{pe}，市值：${mkt_cap}，"
        f"EPS：${eps}，Beta：{beta}\n"
        f"淨利率：{net_margin}%，營收增長：{rev_growth}%\n"
        f"【歷史表現】{hist_summary}\n\n"
        f"請提供：\n"
        f"1. 綜合評分（1-10）及理由\n"
        f"2. 核心投資亮點（2-3點）\n"
        f"3. 主要風險因素（2-3點）\n"
        f"4. 進場時機建議\n"
        f"5. 目標價位區間及止損建議\n"
        f"請以結構化格式輸出，每節使用 **粗體標題**，內容精簡但具體。"
    )


def build_portfolio_prompt(portfolio: dict, prices: dict) -> str:
    """Build the portfolio analysis prompt string."""
    if not portfolio:
        return ""
    rows = []
    total_cost = total_val = 0.0
    for t, data in portfolio.items():
        buy = data.get("buy_price", 0)
        qty = data.get("qty", 0)
        cur = prices.get(t, buy)
        pnl_pct     = (cur - buy) / buy * 100 if buy else 0
        cost        = buy * qty
        val         = cur * qty
        total_cost += cost
        total_val  += val
        rows.append(
            f"  {t}: 買入${buy:.2f}×{qty}股，現價${cur:.2f}，"
            f"盈虧{pnl_pct:+.1f}%，市值${val:,.0f}"
        )
    total_pnl    = (total_val - total_cost) / total_cost * 100 if total_cost else 0
    holdings_txt = "\n".join(rows)
    return (
        f"你是一位華爾街資深投資組合經理，請以繁體中文對以下持倉進行深度分析：\n\n"
        f"【投資組合總覽】\n"
        f"總成本：${total_cost:,.0f}，現值：${total_val:,.0f}，"
        f"整體盈虧：{total_pnl:+.1f}%\n\n"
        f"【個股持倉明細】\n{holdings_txt}\n\n"
        f"請提供：\n"
        f"1. 整體組合評估（分散度、風險集中度）\n"
        f"2. 表現最佳與最差持倉分析\n"
        f"3. 組合再平衡建議（哪些應加碼、減碼或停損）\n"
        f"4. 宏觀環境下的整體風險提示\n"
        f"5. 下一步行動建議（具體可執行）\n"
        f"請以結構化格式輸出，每節使用 **粗體標題**，內容精簡但具體。"
    )


# ── Friendly error UI widgets ─────────────────────────────────────────────────

def _next_refresh_str(minutes: int = 60) -> str:
    return (datetime.now() + timedelta(minutes=minutes)).strftime("%H:%M")


def render_quota_error(container=None, refresh_minutes: int = 60) -> None:
    """
    Render a user-friendly 'quota exhausted' warning inside `container`
    (defaults to st main area). Shows the expected auto-refresh time.
    """
    target = container or st
    next_t = _next_refresh_str(refresh_minutes)
    target.markdown(
        f"<div style='background:#2D1B00; border-left:4px solid #F0883E; "
        f"border-radius:8px; padding:14px 16px; margin:8px 0;'>"
        f"<b style='color:#F0883E;'>⏳ AI 分析服務目前繁忙（配額已達上限）</b><br>"
        f"<span style='color:#E6EDF3; font-size:13px;'>"
        f"系統將於 <b>{next_t}</b> 後自動刷新快取，屆時可重新生成分析報告。<br>"
        f"目前已自動載入此標的最近一次的歷史分析結果供您參考。"
        f"</span></div>",
        unsafe_allow_html=True,
    )


def render_auth_error(container=None) -> None:
    """Render a user-friendly API key error."""
    target = container or st
    target.markdown(
        "<div style='background:#2D1B1B; border-left:4px solid #DA3633; "
        "border-radius:8px; padding:14px 16px; margin:8px 0;'>"
        "<b style='color:#DA3633;'>🔑 API Key 錯誤</b><br>"
        "<span style='color:#E6EDF3; font-size:13px;'>"
        "請至 Replit Secrets 更新 <code>GEMINI_API_KEY</code>，"
        "或確認 Key 未洩漏或被撤銷。"
        "</span></div>",
        unsafe_allow_html=True,
    )


def render_generic_error(e: Exception, container=None) -> None:
    """Render a generic AI failure notice."""
    target = container or st
    target.markdown(
        f"<div style='background:#1C2128; border-left:4px solid #8B949E; "
        f"border-radius:8px; padding:12px 14px; margin:8px 0;'>"
        f"<b style='color:#8B949E;'>⚠️ AI 分析暫時不可用</b><br>"
        f"<span style='color:#8B949E; font-size:12px;'>{e}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def handle_gemini_error(e: Exception, cache_key: str | None = None,
                        container=None) -> str | None:
    """
    Classify `e` and render the appropriate Streamlit error widget.
    Returns the cached result string if one exists under `cache_key`,
    otherwise returns None so the caller can display a fallback.
    """
    cached = st.session_state.get(cache_key) if cache_key else None
    if is_auth_error(e):
        render_auth_error(container)
    elif is_quota_error(e):
        render_quota_error(container)
    else:
        render_generic_error(e, container)
    return cached   # may be None if no previous result

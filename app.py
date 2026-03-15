# ═══════════════════════════════════════════════════════════════════════════════
# main.py  —  Entry point, page routing, sidebar (美股選股儀表板)
# ═══════════════════════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import time
from datetime import datetime

st.set_page_config(
    page_title="美股選股儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from analysis import (
    compute_technicals, screen_stocks, format_market_cap,
    calc_buy_zone, calc_exit_strategy, classify_investment_horizon,
    plot_relative_strength, plot_four_quadrant,
    calculate_seven_factors, plot_factor_radar, build_factor_prompt,
)
from data_fetcher import (
    get_stock_info, get_historical_data, get_analyst_data,
    get_combined_sentiment, get_market_benchmark,
    save_watchlist, save_portfolio,
    get_factor_data, standardize_timezone,
    SCREENER_STOCKS, BENCHMARK_LABELS, SECTOR_ETFS,
)
from ui_components import (
    init_session, navigate_to_diagnosis,
    render_diagnosis, render_trade_plan,
    render_news_intelligence, render_portfolio_dashboard,
    render_opportunity_banner, render_macro_sentiment_dashboard,
    plot_price_chart, plot_volume_chart,
    plot_analyst_targets, plot_analyst_recs, plot_radar,
    zone_progress_bar,
)
import backtest_engine as be
from mpf_assistant import render_mpf_page
from kol_whitelist import render_kol_section
from ocr_module import generate_quant_report
from notifier import (
    run_all_checks, get_current_prices,
    load_notification_log, send_telegram_notification,
)
from user_config import (
    load_order, save_order, get_section_labels, SECTION_META,
    load_kol_whitelist, add_kol, remove_kol,
    save_gemini_key, load_gemini_key, clear_gemini_key, gemini_key_days_remaining,
)

_MODULE = "main"

def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


_DARK_CARD = "background:#1C2128; border-radius:6px"
_DARK_GREEN_CARD = "background:#0D2E1A; border-radius:6px"
_DARK_RED_CARD   = "background:#2D1B1B; border-radius:6px"
_DARK_BLUE_CARD  = "background:#1B2A3D; border-radius:6px"
_DARK_AMBER_CARD = "background:#2D1B00; border-radius:6px"


from gemini_helper import (
    call_gemini_cached,
    build_stock_prompt,
    build_portfolio_prompt,
    is_quota_error, is_auth_error,
    render_quota_error, render_auth_error, render_generic_error,
    handle_gemini_error,
)


def _call_gemini_stock_ai(stock_info: dict, hist, ticker: str, api_key: str) -> str | None:
    """
    Generate Gemini AI individual stock analysis.
    Uses 1-hour cache + exponential-backoff retry on 429/timeout.
    """
    try:
        prompt = build_stock_prompt(stock_info, hist, ticker)
        return call_gemini_cached(prompt, api_key)
    except Exception as e:
        if is_auth_error(e):
            return f"❌ API Key 錯誤：請在 Replit Secrets 中更新 GEMINI_API_KEY。"
        if is_quota_error(e):
            return "__QUOTA__"
        return f"❌ AI分析失敗：{e}"


def _call_gemini_portfolio_ai(portfolio: dict, prices: dict, api_key: str) -> str | None:
    """
    Generate Gemini AI portfolio analysis.
    Uses 1-hour cache + exponential-backoff retry on 429/timeout.
    """
    if not portfolio:
        return "❌ 投資組合為空，無法生成分析。"
    try:
        prompt = build_portfolio_prompt(portfolio, prices)
        return call_gemini_cached(prompt, api_key)
    except Exception as e:
        if is_auth_error(e):
            return f"❌ API Key 錯誤：請在 Replit Secrets 中更新 GEMINI_API_KEY。"
        if is_quota_error(e):
            return "__QUOTA__"
        return f"❌ AI分析失敗：{e}"


def _get_gemini_key() -> str:
    """
    Return Gemini API key with priority:
      1. Replit Secrets panel (st.secrets)  — most secure
      2. User-entered key in user_config.json (30-day TTL)
      3. os.environ fallback (e.g. .replit [userenv.shared])
    User-entered key is intentionally ranked above os.environ so that
    a freshly entered key always overrides a stale key from .replit.
    """
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    user_key = load_gemini_key()
    if user_key:
        return user_key
    return os.environ.get("GEMINI_API_KEY", "")


def _get_telegram_creds() -> tuple[str, str]:
    """Return (bot_token, chat_id) from st.secrets or env vars."""
    try:
        token   = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_USER_ID", "")
        if token and chat_id:
            return token, str(chat_id)
    except Exception:
        pass
    return (
        os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        os.environ.get("TELEGRAM_USER_ID", ""),
    )


def _telegram_configured() -> bool:
    t, c = _get_telegram_creds()
    return bool(t and c)


def _inject_global_css() -> None:
    """Dark theme — black background, white text, Inter font."""
    font_size  = st.session_state.get("font_size", 13)
    font_stack = "'Inter', 'Roboto', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

    st.markdown(
        f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

        /* ── Dark base ── */
        .stApp {{
            background-color: #0D1117 !important;
        }}
        html, body {{
            background-color: #0D1117 !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #161B22 !important;
        }}

        /* ── Universal colour override ── */
        * {{
            color: #E6EDF3 !important;
        }}

        /* ── Font override: block/inline elements — span intentionally omitted ── */
        html, body, .stApp, div, p, button, label, input,
        select, textarea, td, th, li, a,
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stCaption, .stText {{
            font-family: {font_stack} !important;
        }}

        /* ── Text-span font: lower specificity than icon selector below ── */
        div span, p span, label span, button span,
        .stMarkdown span, .stCaption span, .stText span, li span {{
            font-family: {font_stack} !important;
        }}

        /* ── Material icon spans: restore icon font (data-testid is the real selector).
               Specificity (0,0,1,1) > div span (0,0,0,2) so this wins for icon spans. ── */
        span[data-testid="stIconMaterial"] {{
            font-family: "Material Symbols Rounded" !important;
            font-style: normal !important;
            font-weight: 400 !important;
            -webkit-font-feature-settings: "liga" 1 !important;
            font-feature-settings: "liga" 1 !important;
            color: #8B949E !important;
        }}

        /* ── Line-height fix ── */
        div, p, span, li, label, td, th {{
            line-height: 1.6 !important;
        }}

        /* ── Font sizes ── */
        html, body, .stApp, p, span, label, div, li,
        .stMarkdown, .stCaption, .stText {{
            font-size: {font_size}px !important;
        }}
        h1 {{ font-size: {font_size + 10}px !important; font-weight: 700 !important; }}
        h2 {{ font-size: {font_size + 6}px  !important; font-weight: 700 !important; }}
        h3 {{ font-size: {font_size + 3}px  !important; font-weight: 600 !important; }}

        /* ── Inputs ── */
        .stTextInput input, .stNumberInput input,
        .stSelectbox select, .stTextArea textarea,
        input[type="text"], input[type="number"] {{
            background-color: #1C2128 !important;
            border-color: #30363D !important;
            color: #E6EDF3 !important;
            font-size: {font_size}px !important;
            border-radius: 6px !important;
        }}

        /* ── Tables ── */
        .stDataFrame, .dataframe, table {{
            font-size: {font_size}px !important;
            background-color: #1C2128 !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            font-size: {font_size}px !important;
            border-radius: 6px !important;
            background-color: #21262D !important;
            border-color: #30363D !important;
            color: #E6EDF3 !important;
        }}
        .stButton > button[kind="primary"] {{
            background-color: #1F6FEB !important;
            border-color: #1F6FEB !important;
            color: #ffffff !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab"] {{
            font-size: {font_size}px !important;
            background-color: #161B22 !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background-color: #161B22 !important;
        }}

        /* ── Metric cards ── */
        [data-testid="metric-container"] {{
            background-color: #1C2128 !important;
            border: 1px solid #30363D !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }}

        /* ── Expander ── */
        details {{
            background-color: #1C2128 !important;
            border: 1px solid #30363D !important;
            border-radius: 6px !important;
        }}
        details summary {{
            background-color: #1C2128 !important;
        }}

        /* ── Selectbox / dropdown ── */
        [data-baseweb="select"] * {{
            background-color: #1C2128 !important;
            border-color: #30363D !important;
        }}

        /* ── Sidebar radio buttons ── */
        [data-testid="stSidebar"] label {{
            color: #E6EDF3 !important;
        }}

        /* ── Dividers ── */
        hr {{
            border-color: #30363D !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )



# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    init_session()
    _inject_global_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("📈 美股選股")
    st.sidebar.markdown("---")

    # ── Gemini API Key management ─────────────────────────────────────────────
    # NOTE: Replit injects [userenv.shared] values into BOTH os.environ AND
    # st.secrets, so we cannot use either to detect "user intentionally set key".
    # The sidebar exclusively uses user_config.json (explicitly saved by the user)
    # as the source of truth for display. _get_gemini_key() still falls back to
    # os.environ for backend calls, but the UI never surfaces it.
    _ui_stored_key = load_gemini_key()   # only keys saved by the user via this UI

    with st.sidebar.expander("🔑 Gemini API Key", expanded=not bool(_ui_stored_key)):
        if _ui_stored_key:
            _days_left = gemini_key_days_remaining()
            st.success(f"✅ API Key 生效中（剩餘 **{_days_left}** 天到期）")
            st.caption("30 天後自動失效，屆時重新貼入即可。")
            if st.button("🗑️ 清除 API Key", key="clear_gemini_key_btn",
                         use_container_width=True):
                clear_gemini_key()
                st.rerun()
        else:
            st.warning("⚠️ 請輸入 Gemini API Key 以啟用 AI 功能。")
            st.caption(
                "前往 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) "
                "免費申請，貼入下方儲存（本地保存 30 天，不上傳）。"
            )
            _key_input = st.text_input(
                "貼入 API Key", type="password",
                placeholder="AIzaSy...",
                key="gemini_key_input_field",
            )
            if st.button("💾 儲存 API Key", key="save_gemini_key_btn",
                         use_container_width=True, type="primary"):
                _raw = _key_input.strip()
                if len(_raw) < 20:
                    st.error("格式錯誤，Key 通常以 AIzaSy 開頭，請確認完整貼入。")
                else:
                    save_gemini_key(_raw)
                    st.success("✅ 已儲存！有效期 30 天。")
                    st.rerun()

    st.sidebar.markdown("---")

    # ── Resolve active API key (after potential user-input above) ─────────────
    _api_key = _get_gemini_key()

    _PAGES = [
        "📡 總體市場 (Macro)",
        "🔬 個股診斷 (Micro)",
        "💼 我的持倉",
        "📊 美股回測",
        "🛡️ MPF 智投",
    ]
    _nav   = st.session_state.get("nav_page", _PAGES[0])
    _idx   = _PAGES.index(_nav) if _nav in _PAGES else 0

    page = st.sidebar.radio(
        "功能選擇",
        _PAGES,
        index=_idx,
        key="page_radio",
    )
    st.session_state["nav_page"] = page

    # Portfolio badge + quick export in sidebar
    portfolio = st.session_state.get("portfolio", {})
    if portfolio:
        sb_pf_col, sb_ex_col = st.sidebar.columns([3, 2])
        sb_pf_col.markdown(
            f"<div style='background:#1A2E1A; border:1px solid #00FF7F44; "
            f"border-radius:6px; padding:6px 10px; font-size:12px; color:#00FF7F;'>"
            f"💼 持倉 {len(portfolio)} 支</div>",
            unsafe_allow_html=True,
        )
        sb_ex_col.download_button(
            label="⬇️ 導出",
            data=json.dumps(portfolio, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="portfolio.json",
            mime="application/json",
            use_container_width=True,
            key="sb_export",
        )

    # ── Display controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("🎨 顯示設定", expanded=False):
        st.toggle(
            "🌙 夜間模式 (Dark Mode)",
            value=st.session_state.get("dark_mode", True),
            key="dark_mode",
        )
        st.slider(
            "字體大小 (px)",
            min_value=12, max_value=18,
            value=st.session_state.get("font_size", 13),
            step=1, key="font_size",
        )

    # ── Screener parameters ───────────────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ 篩選設定", expanded=False):
        min_cap = st.number_input(
            "市值下限 (十億美元 $B)", min_value=1, max_value=500, value=10, step=1,
            help="篩選市值大於此數值（單位：十億美元）的股票",
        )
        min_margin = st.number_input(
            "最低淨利率 (%)", min_value=1, max_value=50, value=10, step=1,
            help="篩選淨利率高於此百分比的股票",
        )
        min_growth = st.number_input(
            "最低營收增長 (%)", min_value=1, max_value=100, value=15, step=1,
            help="篩選年營收增長率高於此百分比的股票",
        )
        max_pe = st.number_input(
            "P/E Ratio 上限", min_value=5, max_value=200, value=30, step=1,
            help="篩選本益比低於此數值的股票",
        )
    params = (min_cap, min_margin, min_growth, max_pe)

    # ── Telegram notification settings ────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("📱 Telegram 即時通知", expanded=False):
        _tg_ok = _telegram_configured()
        if _tg_ok:
            st.success("✅ Telegram 已連接")
        else:
            st.warning("⚠️ 請在 Secrets 中設定 TELEGRAM_BOT_TOKEN 及 TELEGRAM_USER_ID")

        alert_threshold = st.slider(
            "止盈/止損閾值 (%)",
            min_value=5, max_value=30, value=10, step=1,
            help="持倉漲跌超過此百分比時觸發通知",
            key="tg_threshold",
        )
        auto_check = st.toggle(
            "開盤後自動檢查（首次訪問觸發）",
            value=False, key="tg_auto_check",
        )
        st.caption("每項警報有 4 小時冷卻期，避免重複推送")

        if _tg_ok:
            if st.button("📤 發送測試訊息", key="tg_test", use_container_width=True):
                _tok, _cid = _get_telegram_creds()
                _ok = send_telegram_notification(
                    _tok, _cid,
                    "✅ *美股選股儀表板* — Telegram 通知連接測試成功！\n"
                    "您的即時警報服務已就緒。",
                )
                if _ok:
                    st.success("測試訊息已發送！")
                else:
                    st.error("發送失敗，請確認 Bot Token 和 User ID 正確。")

            if st.button("🔍 立即執行全面檢查", key="tg_run_now",
                         use_container_width=True, type="primary"):
                st.session_state["tg_run_checks"] = True
                st.rerun()
        else:
            st.info("💡 設定步驟：\n1. 向 @BotFather 申請 Bot Token\n"
                    "2. 向 @userinfobot 取得您的 User ID\n"
                    "3. 在 Replit Secrets 中設定\n"
                    "   TELEGRAM_BOT_TOKEN\n"
                    "   TELEGRAM_USER_ID")

    # ── Personalisation Settings (module order + watchlist + KOL whitelist) ──
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ 個人化設定", expanded=False):

        # ── Tab 1: Module ordering ────────────────────────────────────────────
        st.markdown("**📐 個股診斷模組排序**")
        st.caption("修改「顯示順序」數字（1 = 最頂部），點擊「儲存排序」生效。")
        _sec_labels = get_section_labels()
        _cur_order  = load_order()
        _cfg_df = pd.DataFrame({
            "模組名稱": [_sec_labels[k] for k in _cur_order],
            "顯示順序": list(range(1, len(_cur_order) + 1)),
            "_key":     _cur_order,
        })
        _edited_cfg = st.data_editor(
            _cfg_df[["模組名稱", "顯示順序"]],
            hide_index=True,
            use_container_width=True,
            key="module_order_editor",
            column_config={
                "模組名稱": st.column_config.TextColumn("模組", disabled=True),
                "顯示順序": st.column_config.NumberColumn(
                    "順序", min_value=1, max_value=len(_cur_order),
                    step=1, help="輸入數字決定顯示位置（1 = 最頂部）",
                ),
            },
        )
        if st.button("💾 儲存排序", key="save_module_order_btn",
                     use_container_width=True):
            _merged = _edited_cfg.copy()
            _merged["_key"] = _cur_order
            _sorted_cfg = _merged.sort_values("顯示順序").reset_index(drop=True)
            _new_order  = _sorted_cfg["_key"].tolist()
            save_order(_new_order)
            st.success("✅ 排序已儲存！")
            st.rerun()

        st.markdown("---")

        # ── Tab 2: Watchlist management ───────────────────────────────────────
        st.markdown("**🌟 收藏股票清單**")
        _wl = list(st.session_state.get("watchlist", []))
        if _wl:
            st.caption(f"目前 {len(_wl)} 支：" + "、".join(_wl))
        else:
            st.caption("清單目前為空。")

        _wl_col1, _wl_col2 = st.columns([3, 1])
        with _wl_col1:
            _wl_new = st.text_input(
                "新增股票代號", placeholder="AAPL",
                key="sidebar_wl_add_input", label_visibility="collapsed",
            )
        with _wl_col2:
            if st.button("＋", key="sidebar_wl_add_btn", use_container_width=True):
                _ticker_add = _wl_new.strip().upper()
                if _ticker_add and _ticker_add not in _wl:
                    _wl.append(_ticker_add)
                    st.session_state["watchlist"] = _wl
                    save_watchlist(_wl)
                    st.rerun()

        if _wl:
            _rm_choice = st.selectbox(
                "移除股票", options=["（選擇）"] + _wl,
                key="sidebar_wl_remove_select",
            )
            if st.button("🗑️ 移除", key="sidebar_wl_remove_btn",
                         use_container_width=True):
                if _rm_choice != "（選擇）" and _rm_choice in _wl:
                    _wl.remove(_rm_choice)
                    st.session_state["watchlist"] = _wl
                    save_watchlist(_wl)
                    st.rerun()

        st.markdown("---")

        # ── Tab 3: KOL whitelist management ──────────────────────────────────
        st.markdown("**📝 KOL / 分析師白名單**")
        st.caption("輸入 X 或 Threads 帳號（含 @ 符號），加入爬蟲監控清單。")
        _kol_list = load_kol_whitelist()
        if _kol_list:
            for _kh in _kol_list:
                _kc1, _kc2 = st.columns([3, 1])
                with _kc1:
                    st.markdown(f"`{_kh}`")
                with _kc2:
                    if st.button("✕", key=f"kol_rm_{_kh}",
                                 use_container_width=True):
                        remove_kol(_kh)
                        st.rerun()
        else:
            st.caption("尚未加入任何 KOL。")

        _kol_col1, _kol_col2 = st.columns([3, 1])
        with _kol_col1:
            _kol_input = st.text_input(
                "KOL 帳號", placeholder="@handle",
                key="sidebar_kol_add_input", label_visibility="collapsed",
            )
        with _kol_col2:
            if st.button("加入", key="sidebar_kol_add_btn",
                         use_container_width=True):
                _ok, _msg = add_kol(_kol_input)
                if _ok:
                    st.success(_msg)
                    st.rerun()
                else:
                    st.error(_msg)

    # ── Benchmark monitor ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 大盤監測儀")
    benchmark = st.sidebar.selectbox(
        "宏觀基準指數",
        ["VOO", "QQQ", "IWM", "DIA"],
        index=["VOO", "QQQ", "IWM", "DIA"].index(
            st.session_state.get("benchmark", "VOO")
        ),
        key="benchmark",
        help="切換基準指數以更新黃金/死亡交叉提示及個股相對強弱比較",
    )

    with st.sidebar:
        with st.spinner("更新大盤數據…"):
            try:
                bm = get_market_benchmark(benchmark, "1y")
            except Exception:
                bm = {"ticker": benchmark, "label": benchmark,
                      "sma50": None, "sma200": None,
                      "golden_cross": None, "price": None, "perf_1y": None}

    if bm.get("golden_cross") is True:
        sma50_str  = f"{bm['sma50']:.2f}"  if bm["sma50"]  else "N/A"
        sma200_str = f"{bm['sma200']:.2f}" if bm["sma200"] else "N/A"
        st.sidebar.markdown(
            f"<div style='background:#0D2E1A; border:1px solid #238636; "
            f"border-radius:6px; padding:8px 10px; font-size:12px;'>"
            f"✅ <b>宏觀多頭（黃金交叉）</b><br>"
            f"SMA50 {sma50_str} &gt; SMA200 {sma200_str}"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif bm.get("golden_cross") is False:
        sma50_str  = f"{bm['sma50']:.2f}"  if bm["sma50"]  else "N/A"
        sma200_str = f"{bm['sma200']:.2f}" if bm["sma200"] else "N/A"
        st.sidebar.markdown(
            f"<div style='background:#2D1B1B; border:1px solid #DA3633; "
            f"border-radius:6px; padding:8px 10px; font-size:12px;'>"
            f"⚠️ <b>宏觀空頭（死亡交叉）</b><br>"
            f"SMA50 {sma50_str} &lt; SMA200 {sma200_str}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.caption("⏳ 大盤數據載入中…")

    if bm.get("price") and bm.get("perf_1y") is not None:
        perf = bm["perf_1y"]
        pc   = "#00FF7F" if perf >= 0 else "#FF4B4B"
        sign = "+" if perf >= 0 else ""
        perf_text_col = "green" if perf >= 0 else "red"
        st.sidebar.markdown(
            f"<div style='font-size:11px; margin-top:5px;'>"
            f"{bm['label']}　"
            f"<b>${bm['price']:.2f}</b>　"
            f"<span style='color:{perf_text_col};'>{sign}{perf:.1f}% (1Y)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Telegram auto / manual check trigger ──────────────────────────────────
    _run_now = st.session_state.pop("tg_run_checks", False)
    _auto_ok = (
        st.session_state.get("tg_auto_check", False)
        and _telegram_configured()
        and not st.session_state.get("tg_auto_done_today", False)
    )
    if (_run_now or _auto_ok) and _telegram_configured():
        _tg_tok, _tg_cid  = _get_telegram_creds()
        _tg_gem           = _get_gemini_key()
        _tg_portfolio     = st.session_state.get("portfolio", {})
        _tg_watchlist     = st.session_state.get("watchlist", [])
        _tg_threshold     = st.session_state.get("tg_threshold", 10)
        _tg_prices        = get_current_prices(list(_tg_portfolio.keys()))
        _tg_fear          = None
        _tg_golden        = bm.get("golden_cross")
        try:
            _tg_sent_data = get_combined_sentiment("SPY")
            _tg_fear      = _tg_sent_data.get("combined")
        except Exception:
            pass
        _tg_mpf_signals: list = []
        with st.sidebar:
            with st.spinner("📡 檢查警報條件…"):
                _tg_sent = run_all_checks(
                    portfolio     = _tg_portfolio,
                    watchlist     = _tg_watchlist,
                    prices        = _tg_prices,
                    fear_index    = _tg_fear,
                    golden_cross  = _tg_golden,
                    mpf_signals   = _tg_mpf_signals,
                    bot_token     = _tg_tok,
                    chat_id       = _tg_cid,
                    gemini_key    = _tg_gem,
                    threshold_pct = float(_tg_threshold),
                )
        st.session_state["tg_last_sent"]    = _tg_sent
        st.session_state["tg_auto_done_today"] = True
        if _tg_sent:
            st.sidebar.success(f"📱 已發送 {len(_tg_sent)} 則 Telegram 通知")
        else:
            st.sidebar.info("📱 暫無新警報（所有條件在冷卻期內）")

    # ── Watchlist sidebar ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    watchlist = st.session_state.get("watchlist", [])
    st.sidebar.markdown("### ⭐ 我的收藏")
    if watchlist:
        for wl_ticker in watchlist:
            col_wl, col_rm = st.sidebar.columns([3, 1])
            cached_results = st.session_state.get("results") or []
            wl_info = next((s for s in cached_results if s["ticker"] == wl_ticker), None)
            if col_wl.button(f"📌 {wl_ticker}", key=f"wl_goto_{wl_ticker}",
                             use_container_width=True):
                if wl_info:
                    navigate_to_diagnosis(wl_info)
                else:
                    st.session_state["diag_ticker"]      = wl_ticker
                    st.session_state["diag_stock_info"]  = None
                    st.session_state["diag_hist"]        = None
                    st.session_state["auto_fetch"]       = True
                    st.session_state["nav_page"]         = "🔬 個股診斷 (Micro)"
                    st.rerun()
            if col_rm.button("✕", key=f"wl_rm_{wl_ticker}"):
                st.session_state["watchlist"].remove(wl_ticker)
                save_watchlist(st.session_state["watchlist"])
                st.rerun()
    else:
        st.sidebar.caption("尚未收藏任何股票\n在個股診斷頁面點擊「添加到收藏夾」")

    st.sidebar.markdown("---")

    # ── 術語注解清單 ──────────────────────────────────────────────────────────
    with st.sidebar.expander("📚 術語注解清單", expanded=False):
        st.markdown(
            """
**RSI（相對強弱指標）**
> 股票的「體力值」。RSI > 70 表示超買，容易出現回調；RSI < 30 表示超賣，容易出現反彈。

**MACD（指數平滑移動平均線）**
> 趨勢指南針。短期均線向上穿越長期均線為「金叉」，看漲；向下穿越為「死叉」，看跌。

**乖離率（Bias）**
> 股價與均線之間的「橡皮筋」距離。拉得越遠越容易彈回，是衡量超漲超跌的輔助指標。

**黃金交叉（Golden Cross）**
> 短期均線（如 SMA50）向上穿越長期均線（如 SMA200），為多頭趨勢的強烈訊號。

**做 T（T+0 操作）**
> 當天低點買入、高點賣出，賺取當日波動價差，以降低平均持倉成本的操作技巧。

**做空（Short Selling）**
> 先向券商借入股票賣出，再於低價買回還券。預期股價下跌時使用，可從跌勢中獲利。
            """,
            unsafe_allow_html=False,
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Opportunity banner (all pages)
    render_opportunity_banner()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Screener Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    if page == "📡 總體市場 (Macro)":
        st.title("🏠 美股選股儀表板")

        if bm.get("golden_cross") is True:
            st.markdown(
                f"<div style='background:#0D2E1A; border:1.5px solid #238636; "
                f"border-radius:8px; padding:10px 18px; margin-bottom:12px; font-size:14px;'>"
                f"✅ &nbsp;<b style='color:#3FB950;'>市場趨勢：宏觀多頭（黃金交叉）</b>"
                f"&nbsp;—&nbsp;"
                f"{bm['label']} ({benchmark})　"
                f"SMA50 <b style='color:#E6EDF3;'>{bm['sma50']:.2f}</b> &gt; "
                f"SMA200 <b style='color:#E6EDF3;'>{bm['sma200']:.2f}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif bm.get("golden_cross") is False:
            st.markdown(
                f"<div style='background:#2D1B1B; border:1.5px solid #DA3633; "
                f"border-radius:8px; padding:10px 18px; margin-bottom:12px; font-size:14px;'>"
                f"⚠️ &nbsp;<b style='color:#FF7B72;'>市場趨勢：宏觀空頭（死亡交叉）</b>"
                f"&nbsp;—&nbsp;"
                f"{bm['label']} ({benchmark})　"
                f"SMA50 <b style='color:#E6EDF3;'>{bm['sma50']:.2f}</b> &lt; "
                f"SMA200 <b style='color:#E6EDF3;'>{bm['sma200']:.2f}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

        try:
            render_macro_sentiment_dashboard(bm_data=bm)
        except Exception as _macro_err:
            st.warning(f"⚠️ 宏觀情緒圖表加載失敗，請重新整理頁面。（{_macro_err}）")

        # ── Financial Calendar ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📅 財經月曆")
        st.caption("重要財報、FOMC 會議、經濟數據發布（🔴高/🟡中/🟢低重要度）")

        _CALENDAR = [
            {"date": "2026-03-19", "event": "FOMC 利率決議",             "imp": "🔴", "impact": "聯準會宣布利率決定，影響全市場風險偏好"},
            {"date": "2026-03-28", "event": "美國 PCE 通脹數據",          "imp": "🔴", "impact": "聯準會首選通脹指標，直接影響加息預期"},
            {"date": "2026-04-02", "event": "美國非農就業 (NFP)",          "imp": "🔴", "impact": "就業數據反映經濟健康，超預期數值驅動美元升值"},
            {"date": "2026-04-10", "event": "美國 CPI 通脹數據",           "imp": "🔴", "impact": "核心通脹指標，高於預期往往壓制股市"},
            {"date": "2026-04-15", "event": "摩根大通財報 (JPM)",          "imp": "🟡", "impact": "金融業財報季開始，設定市場基調"},
            {"date": "2026-04-22", "event": "特斯拉財報 (TSLA)",           "imp": "🟡", "impact": "電動車龍頭業績影響成長股情緒"},
            {"date": "2026-04-29", "event": "Meta 財報 (META)",            "imp": "🟡", "impact": "AI 廣告支出與用戶增長指引"},
            {"date": "2026-04-30", "event": "FOMC 利率決議 + 記者會",      "imp": "🔴", "impact": "Powell 記者會措辭影響市場全年走向"},
            {"date": "2026-05-07", "event": "蘋果財報 (AAPL)",             "imp": "🟡", "impact": "服務收入與 iPhone 出貨指引"},
            {"date": "2026-05-08", "event": "Amazon 財報 (AMZN)",          "imp": "🟡", "impact": "AWS 雲端增長與電商邊際利潤"},
            {"date": "2026-05-08", "event": "美國非農就業 (NFP)",          "imp": "🔴", "impact": "就業數據影響降息時機預期"},
            {"date": "2026-05-21", "event": "NVIDIA 財報 (NVDA)",          "imp": "🔴", "impact": "AI 晶片需求指引，半導體板塊風向標"},
            {"date": "2026-05-28", "event": "GDP 初值 Q1 2026",            "imp": "🟡", "impact": "確認經濟擴張或衰退路徑"},
            {"date": "2026-06-11", "event": "FOMC 利率決議 + 點陣圖更新",  "imp": "🔴", "impact": "最新利率路徑預期，全年最重要的 Fed 會議之一"},
            {"date": "2026-06-12", "event": "美國 CPI 通脹數據",           "imp": "🔴", "impact": "中期通脹走向確認"},
            {"date": "2026-07-10", "event": "美國 CPI 通脹數據",           "imp": "🟡", "impact": "Q3 開局通脹觀察"},
            {"date": "2026-07-29", "event": "FOMC 利率決議",               "imp": "🔴", "impact": "Q3 利率決定"},
            {"date": "2026-09-16", "event": "FOMC 利率決議 + 點陣圖更新",  "imp": "🔴", "impact": "秋季利率路徑更新，影響年底行情"},
        ]
        today = datetime.today().date()
        upcoming = [e for e in _CALENDAR if datetime.strptime(e["date"], "%Y-%m-%d").date() >= today][:8]

        if upcoming:
            _cal_cols = st.columns(2)
            for ci, ev in enumerate(upcoming):
                ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
                days_to = (ev_date - today).days
                days_str = f"還有 {days_to} 天" if days_to > 0 else "今日"
                border = "#FF4B4B" if ev["imp"] == "🔴" else "#FFD700" if ev["imp"] == "🟡" else "#00FF7F"
                _cal_cols[ci % 2].markdown(
                    f"<div style='border-left:4px solid {border}; border-radius:6px; "
                    f"background:#1C2128; padding:10px 14px; margin-bottom:8px;'>"
                    f"<div style='font-size:12px; color:#8B949E;'>{ev['date']}  ·  {days_str}</div>"
                    f"<div style='font-weight:700; font-size:14px;'>{ev['imp']} {ev['event']}</div>"
                    f"<div style='font-size:12px; color:#8B949E; margin-top:4px;'>{ev['impact']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── KOL / Analyst Tracker ──────────────────────────────────────────
        st.markdown("---")
        render_kol_section(api_key=_api_key)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Stock Diagnosis
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔬 個股診斷 (Micro)":
        st.title("🔬 個股診斷 (Micro)")

        col_input, col_period, col_btn = st.columns([2, 1, 1])
        with col_input:
            ticker_input = st.text_input(
                "輸入股票代碼",
                placeholder="例：AAPL  MSFT  GOOGL",
                value=st.session_state.get("diag_ticker", ""),
            ).upper().strip()
        with col_period:
            period = st.selectbox("歷史數據", ["1y", "6mo", "2y", "5y"], index=0)
        with col_btn:
            analyze_btn = st.button("🩺 診斷分析", type="primary",
                                    use_container_width=True)

        if ticker_input:
            st.session_state["diag_ticker"] = ticker_input

        if ticker_input and analyze_btn:
            with st.spinner(f"正在獲取 {ticker_input} 的完整數據…"):
                fetched_info = get_stock_info(ticker_input)
                fetched_hist = get_historical_data(ticker_input, period)
            if not fetched_info or not fetched_info.get("price"):
                st.error(f"❌ 無法找到股票代碼「{ticker_input}」，請確認後重試。")
            else:
                st.session_state["diag_stock_info"] = fetched_info
                st.session_state["diag_hist"]       = fetched_hist
                st.session_state["diag_period"]     = period
                st.session_state["auto_fetch"]      = False

        if st.session_state.get("auto_fetch"):
            auto_ticker = st.session_state.get("diag_ticker", "")
            with st.spinner(f"正在載入 {auto_ticker} 的完整數據…"):
                try:
                    if not st.session_state.get("diag_stock_info"):
                        fetched_info = get_stock_info(auto_ticker)
                        if fetched_info and fetched_info.get("price"):
                            st.session_state["diag_stock_info"] = fetched_info
                        else:
                            st.error(
                                f"❌ 無法取得「{auto_ticker}」的報價，"
                                f"請確認代碼正確或稍後重試。"
                            )
                    fetched_hist = get_historical_data(auto_ticker, period)
                    st.session_state["diag_hist"] = fetched_hist
                except Exception as _ae:
                    st.error(f"❌ 自動載入數據失敗：{_ae}")
            st.session_state["auto_fetch"] = False

        stock_info = st.session_state.get("diag_stock_info")
        hist       = st.session_state.get("diag_hist")

        if stock_info and stock_info["ticker"] != ticker_input and ticker_input:
            st.info("👆 代碼已更改，請按「診斷分析」重新獲取數據。")
            stock_info = None

        if not stock_info:
            st.info("👆 輸入股票代碼後按「診斷分析」，或在篩選結果中點擊代碼直接跳轉。")

        if stock_info and stock_info.get("price"):
            price  = stock_info["price"]
            ticker = stock_info["ticker"]

            # Header + watchlist button
            header_col, wl_col = st.columns([4, 1])
            with header_col:
                st.markdown(f"## {stock_info['name']} ({ticker})")
                st.caption(f"產業：{stock_info['sector']}　｜　目前股價：**${price:.2f}**")
            with wl_col:
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                already_in_wl = ticker in st.session_state["watchlist"]
                if already_in_wl:
                    if st.button("⭐ 已在收藏夾", use_container_width=True):
                        st.session_state["watchlist"].remove(ticker)
                        save_watchlist(st.session_state["watchlist"])
                        st.rerun()
                else:
                    if st.button("☆ 添加到收藏夾", type="primary",
                                 use_container_width=True):
                        st.session_state["watchlist"].append(ticker)
                        save_watchlist(st.session_state["watchlist"])
                        st.rerun()

            # Key metrics
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("股價",     f"${price:.2f}")
            m2.metric("市值",     format_market_cap(stock_info["market_cap"]))
            m3.metric("P/E",      f"{stock_info['pe_ratio']:.1f}" if stock_info["pe_ratio"] else "N/A")
            m4.metric("淨利率",   f"{stock_info['net_margin']*100:.1f}%" if stock_info["net_margin"] else "N/A")
            m5.metric("營收增長", f"{stock_info['revenue_growth']*100:.1f}%" if stock_info["revenue_growth"] else "N/A")
            m6.metric("EPS",      f"${stock_info['eps']:.2f}" if stock_info["eps"] else "N/A")
            m7.metric("Beta",     f"{stock_info['beta']:.2f}" if stock_info["beta"] else "N/A")

            # ── [Moved] 歷史表現對比分析 now rendered via dynamic section loop below ─

            # ── 決策速查表 ─────────────────────────────────────────────────────
            st.markdown("---")
            _ts = datetime.now().strftime("%H:%M")
            _bm_ticker_q = st.session_state.get("benchmark", "VOO")
            _alpha_q = _is_out_q = None
            if hist is not None and not hist.empty:
                try:
                    _bm_hist_q = bm.get("hist")
                    if _bm_hist_q is not None:
                        _, _alpha_q, _is_out_q, _ = plot_relative_strength(
                            hist, _bm_hist_q, ticker, _bm_ticker_q
                        )
                except Exception:
                    pass

            try:
                _sent      = get_combined_sentiment("SPY")
                _combined_q = _sent.get("combined", 50.0)
            except Exception:
                _combined_q = 50.0

            _ad_q  = {}
            try:
                _ad_q = get_analyst_data(ticker) or {}
            except Exception:
                pass
            _rec_q    = (_ad_q.get("recommendation") or "").lower()
            _is_buy_q = _rec_q in ("buy", "strong_buy")

            _gc = bm.get("golden_cross")
            if   _gc is True:  _gc_label, _gc_color = "✅ 黃金交叉", "#00FF7F"
            elif _gc is False: _gc_label, _gc_color = "⚠️ 死亡交叉", "#FF4B4B"
            else:              _gc_label, _gc_color = "— 載入中",   "#aaa"

            if _alpha_q is not None:
                _a_label = f"{_alpha_q:+.1f}%"
                _a_color = "#00FF7F" if _alpha_q >= 0 else "#FF4B4B"
                _a_delta = "強於大盤 ▲" if _alpha_q >= 0 else "弱於大盤 ▼"
            else:
                _a_label, _a_color, _a_delta = "計算中…", "#aaa", None

            if   _combined_q >= 70: _s_color, _s_label = "#00FF7F", "極度貪婪"
            elif _combined_q >= 50: _s_color, _s_label = "#7FFF00", "貪婪"
            elif _combined_q >= 30: _s_color, _s_label = "#FFD700", "中性"
            else:                   _s_color, _s_label = "#FF4B4B", "恐慌"

            st.markdown(
                f"<div style='background:#0A0D1A; border:1px solid #2A2D3E; "
                f"border-radius:8px; padding:10px 18px; margin-bottom:6px;'>"
                f"<span style='color:#8B949E; font-size:11px;'>⚡ 決策速查表　更新時間 {_ts}</span>"
                f"<div style='display:flex; gap:40px; margin-top:8px;'>"
                f"  <div><span style='color:#aaa; font-size:11px;'>大盤趨勢</span><br>"
                f"    <span style='color:{_gc_color}; font-size:14px; font-weight:700;'>"
                f"    {_gc_label}</span>"
                f"    <span style='color:#aaa; font-size:10px;'> ({_bm_ticker_q})</span></div>"
                f"  <div><span style='color:#aaa; font-size:11px;'>情緒溫度</span><br>"
                f"    <span style='color:{_s_color}; font-size:14px; font-weight:700;'>"
                f"    {_combined_q:.0f}/100</span>"
                f"    <span style='color:{_s_color}; font-size:11px;'> {_s_label}</span></div>"
                f"  <div><span style='color:#aaa; font-size:11px;'>個股 Alpha (vs {_bm_ticker_q})</span><br>"
                f"    <span style='color:{_a_color}; font-size:14px; font-weight:700;'>"
                f"    {_a_label}</span>"
                f"    <span style='color:{_a_color}; font-size:11px;'>"
                f"    {'  ' + _a_delta if _a_delta else ''}</span></div>"
                f"  <div><span style='color:#aaa; font-size:11px;'>分析師評級</span><br>"
                f"    <span style='color:{'#7FFF00' if _is_buy_q else '#FFD700'}; "
                f"    font-size:14px; font-weight:700;'>"
                f"    {_rec_q.replace('_',' ').title() if _rec_q and _rec_q!='n/a' else 'N/A'}"
                f"    </span></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            # ── 融合訊號 ───────────────────────────────────────────────────────
            _buy_consensus = (_is_out_q is True)  and _is_buy_q
            _oversold_warn = (_is_out_q is False) and (_combined_q < 30)
            if _buy_consensus:
                st.markdown(
                    "<div style='background:#0D2E1A; border:2px solid #238636; "
                    "border-radius:8px; padding:10px 18px; margin-bottom:6px; "
                    "font-size:14px; text-align:center; color:#E6EDF3;'>"
                    "✅ <b style='font-size:15px;'>買入共識訊號</b>"
                    "　—　個股相對強勢 + 分析師評級買入，條件同步觸發。"
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif _oversold_warn:
                st.markdown(
                    "<div style='background:#2D1B00; border:2px solid #D29922; "
                    "border-radius:8px; padding:10px 18px; margin-bottom:6px; "
                    "font-size:14px; text-align:center; color:#E6EDF3;'>"
                    "⚠️ <b style='font-size:15px;'>超跌警示（補跌風險）</b>"
                    "　—　個股跑輸大盤且市場情緒恐慌，謹防進一步補跌。"
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Section 1: Diagnosis + Radar
            try:
                col_diag, col_radar = st.columns([1, 1])
                with col_diag:
                    render_diagnosis(stock_info, params)
                with col_radar:
                    st.plotly_chart(plot_radar(stock_info), use_container_width=True)
            except Exception as _e1:
                st.error(f"⚠️ 診斷評分渲染異常：{_e1}")

            # Section 2: Buy Zone + Trade Plan
            st.markdown("---")
            if hist is not None and not hist.empty:
                try:
                    render_trade_plan(stock_info, hist)
                except Exception as _e2:
                    st.error(f"⚠️ 交易計畫計算異常：{_e2}")
            else:
                st.warning("⚠️ 無法獲取歷史數據，無法計算技術指標。")

            # Section 2b: Investment Horizon Classification
            st.markdown("---")
            st.markdown("### 🧭 投資屬性分類")
            try:
                _sma50_h, _sma200_h, _ = compute_technicals(hist) if hist is not None and not hist.empty else (None, None, None)
                _hz = classify_investment_horizon(price, _sma50_h, _sma200_h, hist)
                _acc = _hz["accent"]
                _bg  = _hz["bg"]
                _bdr = _hz["border"]
                _bullet_html = "".join(
                    f"<li style='margin:5px 0; font-size:13px; color:#ccc;'>{r}</li>"
                    for r in _hz["reasons"]
                )
                st.markdown(
                    f"""
                    <div style="
                        background:{_bg};
                        border-left:4px solid {_bdr};
                        border-radius:8px;
                        padding:18px 22px;
                        margin-top:4px;
                    ">
                      <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                        <span style="font-size:28px; line-height:1;">{_hz["icon"]}</span>
                        <div>
                          <div style="font-size:13px; color:#aaa; letter-spacing:.5px; margin-bottom:2px;">
                            投資屬性分類
                          </div>
                          <div style="font-size:20px; font-weight:700; color:{_acc};">
                            {_hz["label"]}
                          </div>
                        </div>
                      </div>
                      <div style="
                          background:#ffffff12;
                          border-radius:6px;
                          padding:9px 14px;
                          margin-bottom:12px;
                          font-size:13px;
                          color:{_acc};
                          font-weight:600;
                          letter-spacing:.3px;
                      ">
                        🗓 {_hz["hold_period"]}
                      </div>
                      <ul style="margin:0; padding-left:20px; list-style:disc;">
                        {_bullet_html}
                      </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as _e_hz:
                st.warning(f"⚠️ 投資屬性分類計算異常：{_e_hz}")

            # Section 3: K-line chart
            if hist is not None and not hist.empty:
                with st.expander("📈 查看 K 線圖表", expanded=False):
                    try:
                        sma50_v, sma200_v, low20_v = compute_technicals(hist)
                        stop_v, target_v, _ = (
                            calc_exit_strategy(price, sma200_v, low20_v)
                            if sma200_v else (None, None, None)
                        )
                        buy_lower_v, buy_upper_v, _, _ = calc_buy_zone(price, sma50_v, sma200_v)
                        pf_entry = st.session_state["portfolio"].get(ticker)
                        pf_buy   = pf_entry["buy_price"] if pf_entry else None
                        st.plotly_chart(
                            plot_price_chart(
                                ticker, hist,
                                sma50=sma50_v, sma200=sma200_v,
                                stop_price=stop_v, target_price=target_v,
                                buy_lower=buy_lower_v, buy_upper=buy_upper_v,
                                portfolio_buy_price=pf_buy,
                                current_price=price,
                            ),
                            use_container_width=True,
                        )
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("期間最高",   f"${hist['High'].max():.2f}")
                        s2.metric("期間最低",   f"${hist['Low'].min():.2f}")
                        ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                        s3.metric("期間報酬率", f"{ret:.1f}%", delta=f"{ret:.1f}%")
                        s4.metric("平均成交量", f"{hist['Volume'].mean():,.0f}")
                    except Exception:
                        st.error("⚠️ K 線圖表渲染異常，請稍後再試。")

            # Section 3.5: Volume Momentum
            st.markdown("---")
            st.markdown("### 📊 成交量動能分析")
            if hist is not None and not hist.empty:
                try:
                    vol_fig, vol_ratio, is_inst = plot_volume_chart(ticker, hist)
                    avg20_val = hist["Volume"].rolling(20).mean().iloc[-1]
                    if is_inst:
                        st.markdown(
                            f"<div style='background:#2D1B1B; border:1px solid #DA3633; "
                            f"border-radius:7px; padding:9px 16px; margin-bottom:8px; font-size:13px; color:#E6EDF3;'>"
                            f"⚠️ <b>機構級資金活動偵測</b>"
                            f"　今日成交量為 20 日均量的 "
                            f"<b>{vol_ratio:.1f}×</b>"
                            f"，異常放量，建議密切關注方向性突破。</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        ratio_label = "高" if vol_ratio >= 1.2 else "正常"
                        st.markdown(
                            f"<div style='background:#1C2128; border-radius:7px; "
                            f"padding:8px 16px; margin-bottom:8px; font-size:13px; color:#E6EDF3;'>"
                            f"今日成交量為 20 日均量的 "
                            f"<b>{vol_ratio:.2f}×</b>"
                            f"　（均量：{avg20_val:,.0f}，成交{ratio_label}）</div>",
                            unsafe_allow_html=True,
                        )
                    vm1, vm2, vm3 = st.columns(3)
                    vm1.metric("今日成交量",      f"{hist['Volume'].iloc[-1]:,.0f}")
                    vm2.metric("20 日均量",       f"{avg20_val:,.0f}")
                    vm3.metric("量比（今 / 均）",
                               f"{vol_ratio:.2f}×",
                               delta="異常放量 ⚠️" if is_inst else "正常")
                    st.plotly_chart(vol_fig, use_container_width=True,
                                    key=f"vol_{ticker}")
                except Exception:
                    st.caption("⏳ 成交量數據載入失敗，請稍後再試。")

            # Section 3.6: Sector ETF Correlation
            st.markdown("---")
            _sector       = stock_info.get("sector", "")
            _sect_etf_info = SECTOR_ETFS.get(_sector)
            if _sect_etf_info is None and any(
                kw in (stock_info.get("name", "") + _sector).lower()
                for kw in ("semiconductor", "chip", "semi")
            ):
                _sect_etf_info = ("SMH", "半導體 ETF")

            if _sect_etf_info:
                _sect_etf_ticker, _sect_etf_label = _sect_etf_info
                st.markdown(
                    f"### 🏭 板塊關聯監測　"
                    f"<span style='color:#aaa; font-size:13px;'>"
                    f"{ticker} vs {_sect_etf_ticker}（{_sect_etf_label}）</span>",
                    unsafe_allow_html=True,
                )
                try:
                    with st.spinner(f"載入 {_sect_etf_ticker} 板塊數據…"):
                        _sect_bm   = get_market_benchmark(_sect_etf_ticker, "1y")
                        _sect_hist = _sect_bm.get("hist")
                    if _sect_hist is not None and hist is not None and not hist.empty:
                        _sf, _sa, _so, _sr = plot_relative_strength(
                            hist, _sect_hist, ticker, _sect_etf_ticker
                        )
                        if _sf:
                            _sa_sign  = f"{_sa:+.1f}%" if _sa is not None else "N/A"
                            _sa_color = "#00FF7F" if (_sa or 0) >= 0 else "#FF4B4B"
                            _sa_text  = "強於板塊 ▲" if (_sa or 0) >= 0 else "弱於板塊 ▼"
                            st.markdown(
                                f"<div style='background:#13162A; border-radius:7px; "
                                f"padding:7px 16px; margin-bottom:6px; font-size:13px;'>"
                                f"板塊 Alpha：<b style='color:{_sa_color};'>{_sa_sign}</b>"
                                f"　<span style='color:{_sa_color};'>{_sa_text}</span>"
                                f"　vs <b>{_sect_etf_ticker}</b>（{_sect_etf_label}）"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.plotly_chart(_sf, use_container_width=True,
                                            key=f"sect_{ticker}_1y")
                    else:
                        st.caption(f"⏳ {_sect_etf_ticker} 板塊數據暫不可用。")
                except Exception:
                    st.caption(f"⏳ 板塊數據載入失敗，請稍後再試。")
            else:
                st.markdown("### 🏭 板塊關聯監測")
                st.caption(f"⏳ 未能找到「{_sector}」對應的板塊 ETF 數據。")

            # Section 4: Portfolio Entry
            st.markdown("---")
            st.markdown("### ➕ 加入投資組合")
            pf_existing = st.session_state["portfolio"].get(ticker)
            with st.expander(
                f"{'✏️ 修改持倉' if pf_existing else '📥 新增到投資組合'} — {ticker}",
                expanded=bool(pf_existing),
            ):
                pf_default_price = float(pf_existing["buy_price"]) if pf_existing else float(price)
                pf_default_qty   = int(pf_existing["qty"])         if pf_existing else 1
                pf_c1, pf_c2, pf_c3 = st.columns([1.5, 1.5, 1])
                with pf_c1:
                    pf_buy_price = st.number_input(
                        "買入均價 (USD)", min_value=0.01, max_value=99999.0,
                        value=pf_default_price, step=0.5,
                        format="%.2f", key=f"pf_price_{ticker}",
                    )
                with pf_c2:
                    pf_qty = st.number_input(
                        "持倉數量 (股)", min_value=1, max_value=1_000_000,
                        value=pf_default_qty, step=1, key=f"pf_qty_{ticker}",
                    )
                with pf_c3:
                    st.metric("持倉成本", f"${pf_buy_price * pf_qty:,.2f}")

                col_add, col_del = st.columns([2, 1])
                if col_add.button(
                    f"{'💾 更新持倉' if pf_existing else '✅ 加入投資組合'}",
                    type="primary", use_container_width=True,
                    key=f"pf_add_{ticker}",
                ):
                    st.session_state["portfolio"][ticker] = {
                        "buy_price": pf_buy_price,
                        "qty":       int(pf_qty),
                    }
                    save_portfolio(st.session_state["portfolio"])
                    st.success(
                        f"✅ {ticker} 已加入投資組合！"
                        f"買入均價 ${pf_buy_price:.2f}，{int(pf_qty)} 股"
                    )
                    st.rerun()

                if pf_existing:
                    if col_del.button("🗑️ 移除持倉", use_container_width=True,
                                      key=f"pf_rm_{ticker}"):
                        del st.session_state["portfolio"][ticker]
                        save_portfolio(st.session_state["portfolio"])
                        st.rerun()

                if pf_existing:
                    pnl_val = (price - pf_buy_price) / pf_buy_price * 100
                    pnl_sym = f"+{pnl_val:.2f}%" if pnl_val >= 0 else f"{pnl_val:.2f}%"
                    pnl_col = "#00FF7F" if pnl_val >= 0 else "#FF4B4B"
                    pnl_html_col = "#00FF7F" if pnl_val >= 0 else "#FF4B4B"
                    st.markdown(
                        f"<div style='background:#1C2128; border-left:4px solid {pnl_html_col}; "
                        f"border-radius:6px; padding:8px 14px; margin-top:8px; color:#E6EDF3;'>"
                        f"目前盈虧：<span style='color:{pnl_html_col}; font-weight:700; font-size:16px;'>"
                        f"{pnl_sym}</span>　｜　"
                        f"浮動盈虧金額：<span style='color:{pnl_html_col}; font-weight:700;'>"
                        f"${(price - pf_buy_price) * pf_existing['qty']:+,.2f}</span></div>",
                        unsafe_allow_html=True,
                    )

            # Section 5: News
            render_news_intelligence(ticker, stock_info["name"])

            # Section 6: Relative Strength + Four-Quadrant
            st.markdown("---")
            bm_ticker   = st.session_state.get("benchmark", "VOO")
            bm_label    = BENCHMARK_LABELS.get(bm_ticker, bm_ticker)
            diag_period = st.session_state.get("diag_period", "1y")

            hdr_l6, hdr_r6 = st.columns([3, 1])
            with hdr_l6:
                st.markdown("### 📊 績效相對強弱 & 四象限分析")
                st.caption(
                    f"個股 **{ticker}** vs 基準 **{bm_ticker}**（{bm_label}）"
                    f"　兩者起始均設為 0%，直接看超額報酬。"
                )
            with hdr_r6:
                rs_period = st.selectbox(
                    "比較區間",
                    ["1y", "6mo", "2y"],
                    index=["1y", "6mo", "2y"].index(diag_period)
                         if diag_period in ["1y", "6mo", "2y"] else 0,
                    key="rs_period",
                )

            try:
                with st.spinner(f"載入 {bm_ticker} 數據…"):
                    bm_rs      = get_market_benchmark(bm_ticker, rs_period)
                    bm_hist_rs = bm_rs.get("hist")

                if bm_hist_rs is not None and hist is not None and not hist.empty:
                    rs_fig, alpha_pct, is_out, stock_ret = plot_relative_strength(
                        hist, bm_hist_rs, ticker, bm_ticker
                    )
                    if rs_fig and alpha_pct is not None and stock_ret is not None:
                        a_sign = f"+{alpha_pct:.1f}" if alpha_pct >= 0 else f"{alpha_pct:.1f}"
                        r_sign = f"+{stock_ret:.1f}"  if stock_ret  >= 0 else f"{stock_ret:.1f}"
                        if is_out:
                            st.markdown(
                                f"<div style='background:#0D2E1A; border:1px solid #238636; "
                                f"border-radius:7px; padding:8px 18px; margin-bottom:6px; font-size:13px; color:#E6EDF3;'>"
                                f"🚀 <b>強於大盤 (Alpha {a_sign}%)</b>"
                                f"　{ticker} 報酬 <b>{r_sign}%</b>"
                                f"　vs {bm_ticker} 報酬"
                                f" <b>{stock_ret - alpha_pct:+.1f}%</b>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div style='background:#2D1B00; border:1px solid #D29922; "
                                f"border-radius:7px; padding:8px 18px; margin-bottom:6px; font-size:13px; color:#E6EDF3;'>"
                                f"🐢 <b>弱於大盤 (Alpha {a_sign}%)</b>"
                                f"　{ticker} 報酬 <b>{r_sign}%</b>"
                                f"　vs {bm_ticker} 報酬"
                                f" <b>{stock_ret - alpha_pct:+.1f}%</b>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        chart_col, quad_col = st.columns([3, 2])
                        with chart_col:
                            st.plotly_chart(rs_fig, use_container_width=True,
                                            key=f"rs_{ticker}_{rs_period}")
                        with quad_col:
                            qfig, q_label, q_action, dot_color = plot_four_quadrant(
                                stock_ret, alpha_pct, ticker, bm_ticker
                            )
                            st.plotly_chart(qfig, use_container_width=True,
                                            key=f"quad_{ticker}_{rs_period}")
                            st.markdown(
                                f"<div style='background:#13162A; border:1px solid {dot_color}55; "
                                f"border-radius:7px; padding:8px 14px; text-align:center;'>"
                                f"<div style='color:{dot_color}; font-size:14px; font-weight:700;'>"
                                f"{q_label}</div>"
                                f"<div style='color:#aaa; font-size:12px; margin-top:3px;'>"
                                f"系統建議：<b style='color:{dot_color};'>{q_action}</b>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning("⚠️ 日期區間對齊資料不足，無法繪製比較圖。")
                else:
                    st.warning(f"⚠️ {bm_ticker} 基準數據暫時不可用，請稍後重試。")
            except Exception:
                st.error("⚠️ 相對強弱計算異常，請稍後再試。")

            # Section 7: Analyst Consensus
            st.markdown("---")
            _analyst_ts = datetime.now().strftime("%H:%M:%S")
            st.markdown(
                f"### 🏦 分析師共識評級 & 目標價分析"
                f"<span style='color:#8B949E; font-size:12px; margin-left:12px;'>"
                f"⏱ 資料載入時間 {_analyst_ts}</span>",
                unsafe_allow_html=True,
            )
            try:
                with st.spinner("載入分析師數據…"):
                    ad = get_analyst_data(ticker)
                if not ad:
                    st.caption("⏳ 分析師數據暫時不可用，請稍後再試。")
                else:
                    rec_key = (ad.get("recommendation") or "n/a").lower()
                    rec_map = {
                        "strong_buy":   ("強力買入 ★★★★★", "#00FF7F", "#0D2E0D"),
                        "buy":          ("買入 ★★★★",      "#7FFF00", "#1A2E0D"),
                        "hold":         ("持有 ★★★",        "#FFD700", "#2E2A0D"),
                        "underperform": ("表現落後 ★★",     "#FFA500", "#2E1A0D"),
                        "sell":         ("賣出 ★",           "#FF4B4B", "#2E0D0D"),
                    }
                    r_label, r_color, r_bg = rec_map.get(
                        rec_key, ("中性 ★★★", "#aaa", "#1A1D2E"))
                    num_a    = ad.get("num_analysts", 0) or 0
                    rec_mean = ad.get("rec_mean")
                    rm_text  = f"　評分：{rec_mean:.2f}/5.00" if rec_mean else ""

                    an_l, an_r = st.columns([3, 1])
                    with an_l:
                        st.markdown(
                            f"<div style='background:{r_bg}; border:1px solid {r_color}44; "
                            f"border-radius:8px; padding:10px 18px;'>"
                            f"<span style='color:{r_color}; font-size:16px; font-weight:800;'>"
                            f"{r_label}</span>"
                            f"<span style='color:#aaa; font-size:12px;'>"
                            f"　共 {num_a} 位分析師{rm_text}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with an_r:
                        rec_mean_disp = f"{rec_mean:.1f}" if rec_mean else "N/A"
                        st.metric("分析師共識分", rec_mean_disp,
                                  help="1=強力買入 ··· 5=強力賣出")

                    t_mean = ad.get("target_mean")
                    t_high = ad.get("target_high")
                    t_low  = ad.get("target_low")
                    t_med  = ad.get("target_median")
                    if t_mean and t_high and t_low and t_low < t_high:
                        tp_metrics = st.columns(4)
                        tp_metrics[0].metric("目標低價",   f"${t_low:.2f}")
                        tp_metrics[1].metric("目標中位數", f"${t_med:.2f}" if t_med else "N/A")
                        tp_metrics[2].metric("目標均價",   f"${t_mean:.2f}")
                        tp_metrics[3].metric("目標高價",   f"${t_high:.2f}")
                        upside_pct = (t_mean - price) / price * 100 if price else 0
                        upside_col = "#00FF7F" if upside_pct >= 0 else "#FF4B4B"
                        st.markdown(
                            f"<div style='background:#13162A; border-radius:7px; "
                            f"padding:8px 16px; font-size:13px; margin:4px 0 8px;'>"
                            f"距目標均價上行空間："
                            f"<b style='color:{upside_col}; font-size:15px;'>"
                            f"{upside_pct:+.1f}%</b>"
                            f"　（現價 ${price:.2f} → 目標均 ${t_mean:.2f}）"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            plot_analyst_targets(price, t_low, t_mean, t_high, ticker),
                            use_container_width=True, key=f"at_{ticker}",
                        )
                    else:
                        st.caption("⏳ 目標價數據尚未公佈。")

                    recs_df = ad.get("recs_df")
                    if recs_df is not None and not recs_df.empty:
                        st.plotly_chart(
                            plot_analyst_recs(recs_df),
                            use_container_width=True, key=f"ar_{ticker}",
                        )
            except Exception:
                st.caption("⏳ 分析師數據暫時不可用，請稍後再試。")

            # ══════════════════════════════════════════════════════════════
            # DYNAMIC SECTION RENDERING — order controlled by user_config.json
            # ══════════════════════════════════════════════════════════════

            # ── Nested render functions (closure over ticker/hist/stock_info/bm) ─

            def _section_comparison():
                st.markdown("---")
                st.markdown("### 📅 歷史表現對比分析")
                _PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
                _sel_period = st.radio(
                    "選擇時間區間", list(_PERIOD_MAP.keys()),
                    index=1, horizontal=True,
                    key=f"hist_compare_period_{ticker}",
                )
                _yf_period   = _PERIOD_MAP[_sel_period]
                _bm_ticker_c = st.session_state.get("benchmark", "VOO")
                with st.spinner(f"載入 {_sel_period} 歷史數據…"):
                    _comp_hist = get_historical_data(ticker, _yf_period)
                    _bm_data_c = get_market_benchmark(_bm_ticker_c, _yf_period)
                if _comp_hist is not None and not _comp_hist.empty:
                    _cc = _comp_hist["Close"].dropna()
                    _stock_ret = float((_cc.iloc[-1] / _cc.iloc[0] - 1) * 100)
                    _stock_vol = float(_cc.pct_change().dropna().std() * (252 ** 0.5) * 100)
                    _dd_roll   = (_cc - _cc.cummax()) / _cc.cummax() * 100
                    _stock_dd  = float(_dd_roll.min())
                    _bm_hist_c = _bm_data_c.get("hist")
                    _bm_ret_c  = None
                    if _bm_hist_c is not None and not _bm_hist_c.empty:
                        _bmc = _bm_hist_c["Close"].dropna()
                        _bm_ret_c = float((_bmc.iloc[-1] / _bmc.iloc[0] - 1) * 100)
                    _rs = f"+{_stock_ret:.2f}%" if _stock_ret >= 0 else f"{_stock_ret:.2f}%"
                    _ma1, _ma2, _ma3, _ma4, _ma5 = st.columns(5)
                    _ma1.metric(f"區間報酬 ({_sel_period})", _rs, delta=_rs, delta_color="normal")
                    if _bm_ret_c is not None:
                        _alpha_c = _stock_ret - _bm_ret_c
                        _bm_sign = f"+{_bm_ret_c:.2f}%" if _bm_ret_c >= 0 else f"{_bm_ret_c:.2f}%"
                        _al_sign = f"+{_alpha_c:.2f}%" if _alpha_c >= 0 else f"{_alpha_c:.2f}%"
                        _ma2.metric(f"{_bm_ticker_c} 基準", _bm_sign, delta=_bm_sign, delta_color="normal")
                        _ma3.metric("超額報酬 Alpha", _al_sign, delta=_al_sign, delta_color="normal")
                    else:
                        _ma2.metric(f"{_bm_ticker_c} 基準", "N/A")
                        _ma3.metric("超額報酬 Alpha", "N/A")
                    _ma4.metric("最大回撤", f"{_stock_dd:.2f}%")
                    _ma5.metric("年化波動率", f"{_stock_vol:.1f}%")
                    _norm_df = pd.DataFrame({"日期": _cc.index,
                                             ticker: (_cc / _cc.iloc[0] * 100).values})
                    if _bm_hist_c is not None and not _bm_hist_c.empty:
                        _bmc2 = _bm_hist_c["Close"].dropna()
                        _bmc2a = _bmc2.reindex(_cc.index, method="ffill").dropna()
                        if not _bmc2a.empty:
                            _norm_df[_bm_ticker_c] = (
                                _bmc2a / _bmc2a.iloc[0] * 100
                            ).values[: len(_norm_df)]
                    _fig_norm = go.Figure()
                    _fig_norm.add_trace(go.Scatter(
                        x=_norm_df["日期"], y=_norm_df[ticker], mode="lines", name=ticker,
                        line=dict(color="#1F6FEB", width=2.5),
                        hovertemplate=f"{ticker}: %{{y:.1f}}<extra></extra>",
                    ))
                    if _bm_ticker_c in _norm_df.columns:
                        _fig_norm.add_trace(go.Scatter(
                            x=_norm_df["日期"], y=_norm_df[_bm_ticker_c],
                            mode="lines", name=_bm_ticker_c,
                            line=dict(color="#FF9500", width=2, dash="dot"),
                            hovertemplate=f"{_bm_ticker_c}: %{{y:.1f}}<extra></extra>",
                        ))
                    _fig_norm.add_hline(y=100, line_dash="dash", line_color="#444", line_width=1)
                    _fig_norm.update_layout(
                        title=f"{ticker} vs {_bm_ticker_c} — 累計報酬率對比（{_sel_period}，以 100 為基準）",
                        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#1A1D2E",
                        height=320, margin=dict(t=45, b=30, l=60, r=20),
                        hovermode="x unified", legend=dict(x=0.01, y=0.99),
                        yaxis_title="累計報酬率（基準=100）",
                    )
                    st.plotly_chart(_fig_norm, use_container_width=True,
                                    key=f"norm_chart_{ticker}_{_sel_period}")
                    st.markdown("**區間細分表現**")
                    _sub_periods = {"1 週": 5, "1 個月": 21, "3 個月": 63}
                    _sub_tabs = st.tabs(list(_sub_periods.keys()))
                    for _stab, (_slbl, _sdays) in zip(_sub_tabs, _sub_periods.items()):
                        with _stab:
                            _sl = _comp_hist.tail(_sdays)
                            if len(_sl) < 2:
                                st.caption("資料不足，無法計算此區間。")
                                continue
                            _so  = float(_sl["Close"].iloc[0])
                            _sc2 = float(_sl["Close"].iloc[-1])
                            _sh  = float(_sl["High"].max())
                            _slo2 = float(_sl["Low"].min())
                            _sr  = (_sc2 - _so) / _so * 100
                            _sv  = float(_sl["Volume"].mean())
                            _scolor = "#00FF7F" if _sr >= 0 else "#FF4B4B"
                            _ssign  = f"+{_sr:.2f}%" if _sr >= 0 else f"{_sr:.2f}%"
                            _sc1, _sc2c, _sc3, _sc4 = st.columns(4)
                            _sc1.metric(f"報酬（{_slbl}）", _ssign, delta=_ssign, delta_color="normal")
                            _sc2c.metric("區間最高", f"${_sh:.2f}")
                            _sc3.metric("區間最低", f"${_slo2:.2f}")
                            _sc4.metric("均日成交量", f"{_sv:,.0f}")
                            _sfig = px.line(
                                _sl.reset_index(), x=_sl.index, y="Close",
                                title=f"{ticker} — {_slbl}收盤走勢",
                                labels={"Close": "收盤價 (USD)", "x": "日期"},
                                color_discrete_sequence=[_scolor],
                            )
                            _sfig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#0E1117", plot_bgcolor="#1A1D2E",
                                height=230, margin=dict(t=38, b=30, l=60, r=20),
                                hovermode="x unified",
                            )
                            st.plotly_chart(_sfig, use_container_width=True,
                                            key=f"sub_chart_{ticker}_{_slbl}_{_sel_period}")
                else:
                    st.warning(f"無法載入 {ticker} 的 {_sel_period} 歷史數據。")

            def _section_factor_system():
                st.markdown("---")
                st.markdown("### 📊 7-Factor 多因子分析系統")
                _f7_key      = f"factor7_{ticker}"
                _f7_data_key = f"factor7_data_{ticker}"
                if st.button("🔬 計算七大因子", key=f"f7_btn_{ticker}",
                             type="primary", use_container_width=True):
                    with st.spinner(f"正在計算 {ticker} 七大因子…"):
                        _fd = get_factor_data(ticker)
                        _f7 = calculate_seven_factors(stock_info, hist, _fd)
                    st.session_state[_f7_key]      = _f7
                    st.session_state[_f7_data_key] = _fd
                _f7 = st.session_state.get(_f7_key)
                if _f7:
                    _comp   = _f7.get("composite", 0)
                    _signal = _f7.get("signal", "HOLD")
                    _sig_colors = {
                        "STRONG BUY":  "#00FF7F", "BUY":  "#7CFC00",
                        "HOLD":        "#FFD700", "SELL": "#FF8C00",
                        "STRONG SELL": "#FF4B4B",
                    }
                    _sc = _sig_colors.get(_signal, "#FFD700")
                    st.markdown(
                        f"<div style='background:#161B22; border:1px solid #30363D; "
                        f"border-radius:12px; padding:20px 28px; margin-bottom:16px;'>"
                        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                        f"<div>"
                        f"<div style='font-size:24px; font-weight:700; color:#E6EDF3;'>{ticker}</div>"
                        f"<div style='color:{_sc}; font-size:18px; font-weight:700; margin-top:4px;'>{_signal}</div>"
                        f"<div style='color:#8B949E; font-size:12px; margin-top:2px;'>7-Factor Composite Score</div>"
                        f"</div>"
                        f"<div style='text-align:right;'>"
                        f"<div style='font-size:48px; font-weight:700; color:{_sc};'>{_comp:+.2f}</div>"
                        f"<div style='color:#8B949E; font-size:12px;'>Range: -5 to +5</div>"
                        f"<div style='width:180px; height:6px; background:#21262D; "
                        f"border-radius:3px; margin-top:6px; overflow:hidden;'>"
                        f"<div style='width:{max(0,min(100,(_comp+5)/10*100)):.1f}%; "
                        f"height:100%; background:{_sc}; border-radius:3px;'></div>"
                        f"</div></div></div></div>",
                        unsafe_allow_html=True,
                    )
                    _fcol1, _fcol2 = st.columns([1, 1])
                    with _fcol1:
                        st.markdown("**Factor Exposure Radar 因子雷達圖**")
                        st.plotly_chart(plot_factor_radar(_f7, ticker),
                                        use_container_width=True, key=f"f7_radar_{ticker}")
                    with _fcol2:
                        st.markdown("**Group Scores 因子組評分**")
                        for _fk in ["Momentum","Value","Quality","Growth","Volatility","Sentiment","Macro"]:
                            _fg  = _f7.get(_fk, {})
                            _flbl = _fg.get("label", _fk)
                            _fs  = _fg.get("score", 0.0)
                            _fc  = "#00FF7F" if _fs >= 0 else "#FF4B4B"
                            _bwp = (_fs / 5) * 50
                            _bl  = "50%" if _fs >= 0 else f"{50+_bwp:.1f}%"
                            _bw  = f"{_bwp:.1f}%" if _fs >= 0 else f"{-_bwp:.1f}%"
                            st.markdown(
                                f"<div style='margin-bottom:10px;'>"
                                f"<div style='display:flex;justify-content:space-between;"
                                f"align-items:center;margin-bottom:3px;'>"
                                f"<span style='color:#C9D1D9;font-size:13px;'>{_flbl}</span>"
                                f"<span style='color:{_fc};font-weight:700;font-size:13px;'>{_fs:+.2f}</span>"
                                f"</div>"
                                f"<div style='position:relative;height:8px;background:#21262D;"
                                f"border-radius:4px;overflow:hidden;'>"
                                f"<div style='position:absolute;left:50%;height:100%;width:1px;background:#444;'></div>"
                                f"<div style='position:absolute;left:{_bl};width:{_bw};height:100%;"
                                f"background:{_fc};border-radius:4px;'></div>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                    st.markdown("---")

                    def _ftbl(factor_key):
                        fg  = _f7.get(factor_key, {})
                        fs  = fg.get("score", 0.0)
                        fc  = "#00FF7F" if fs >= 0 else "#FF4B4B"
                        lbl = fg.get("label", factor_key)
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"align-items:baseline;margin-bottom:6px;'>"
                            f"<span style='font-weight:700;font-size:15px;color:#E6EDF3;'>{lbl}</span>"
                            f"<span style='font-size:18px;font-weight:700;color:{fc};'>{fs:+.2f}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        tbl = (
                            "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
                            "<thead><tr style='color:#8B949E;border-bottom:1px solid #30363D;'>"
                            "<th style='text-align:left;padding:4px 6px;'>FACTOR 因子</th>"
                            "<th style='text-align:right;padding:4px 6px;'>VALUE 數值</th>"
                            "<th style='text-align:right;padding:4px 6px;'>SCORE 評分</th>"
                            "</tr></thead><tbody>"
                        )
                        for itm in fg.get("items", []):
                            sc   = itm.get("score", 0.0)
                            sc_c = "#00FF7F" if sc > 0 else ("#FF4B4B" if sc < 0 else "#8B949E")
                            tbl += (
                                f"<tr style='border-bottom:1px solid #21262D;'>"
                                f"<td style='padding:5px 6px;color:#C9D1D9;'>{itm['name']}</td>"
                                f"<td style='padding:5px 6px;text-align:right;color:#8B949E;'>"
                                f"{itm['value_str']}</td>"
                                f"<td style='padding:5px 6px;text-align:right;"
                                f"font-weight:700;color:{sc_c};'>{sc:+.1f}</td>"
                                f"</tr>"
                            )
                        tbl += "</tbody></table>"
                        st.markdown(
                            f"<div style='background:#161B22;border:1px solid #21262D;"
                            f"border-radius:8px;padding:12px 14px;margin-bottom:12px;'>{tbl}</div>",
                            unsafe_allow_html=True,
                        )

                    _r1c1, _r1c2, _r1c3 = st.columns(3)
                    with _r1c1: _ftbl("Momentum")
                    with _r1c2: _ftbl("Value")
                    with _r1c3: _ftbl("Quality")
                    _r2c1, _r2c2, _r2c3 = st.columns(3)
                    with _r2c1: _ftbl("Growth")
                    with _r2c2: _ftbl("Volatility")
                    with _r2c3: _ftbl("Sentiment")
                    _ftbl("Macro")

                    st.markdown("---")
                    st.markdown("**🤖 AI 量化因子報告**")
                    _f7ai_key = _get_gemini_key()
                    if _f7ai_key:
                        if st.button("✨ 生成 AI 因子分析報告",
                                     key=f"f7_ai_btn_{ticker}", type="primary"):
                            with st.spinner("Gemini AI 正在解讀七大因子…"):
                                _f7_prompt = build_factor_prompt(
                                    ticker, stock_info.get("name", ticker), _f7)
                                try:
                                    _f7_ai_txt = call_gemini_cached(_f7_prompt, _f7ai_key)
                                except Exception as _f7e:
                                    _f7_ai_txt = ("__QUOTA__" if is_quota_error(_f7e)
                                                  else f"❌ {_f7e}")
                            st.session_state[f"f7_ai_{ticker}"] = _f7_ai_txt
                            st.rerun()
                        _f7_rpt = st.session_state.get(f"f7_ai_{ticker}")
                        if _f7_rpt:
                            if _f7_rpt == "__QUOTA__":
                                render_quota_error()
                            elif _f7_rpt.startswith("❌"):
                                render_auth_error() if "Key" in _f7_rpt else st.error(_f7_rpt)
                            else:
                                st.markdown(
                                    f"<div style='background:#1B2A3D;border-left:4px solid #1F6FEB;"
                                    f"border-radius:8px;padding:16px;font-size:14px;"
                                    f"color:#E6EDF3;line-height:1.8;'>{_f7_rpt}</div>",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption("💡 AI 報告需要有效的 GEMINI_API_KEY，請在 Secrets 中配置。")

            def _section_ai_report():
                st.markdown("---")
                st.markdown("### 🤖 Gemini AI 個股深度分析")
                _ai_key = _get_gemini_key()
                if _ai_key:
                    if st.button("✨ 生成 AI 分析報告", key=f"diag_ai_btn_{ticker}",
                                 type="primary"):
                        with st.spinner(f"正在調用 Gemini AI 分析 {ticker}…"):
                            _ai_text = _call_gemini_stock_ai(stock_info, hist, ticker, _ai_key)
                        st.session_state[f"diag_ai_report_{ticker}"] = _ai_text
                        st.rerun()
                    if st.session_state.get(f"diag_ai_report_{ticker}"):
                        _report = st.session_state[f"diag_ai_report_{ticker}"]
                        if _report == "__QUOTA__":
                            render_quota_error()
                        elif _report.startswith("❌"):
                            render_auth_error() if "Key" in _report else st.error(_report)
                        else:
                            st.markdown(
                                f"<div style='background:#1B2A3D;border-left:4px solid #1F6FEB;"
                                f"border-radius:8px;padding:16px;font-size:14px;"
                                f"color:#E6EDF3;line-height:1.8;'>{_report}</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("💡 AI 分析需要有效的 GEMINI_API_KEY，請在 Secrets 中配置。")

            # ── Dynamic rendering loop ─────────────────────────────────────
            _SECTION_RENDER_MAP = {
                "Comparison":   _section_comparison,
                "FactorSystem": _section_factor_system,
                "AIReport":     _section_ai_report,
            }
            _diag_display_order = load_order()
            for _diag_section_key in _diag_display_order:
                _fn = _SECTION_RENDER_MAP.get(_diag_section_key)
                if _fn:
                    _fn()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Portfolio Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "💼 我的持倉":
        with st.spinner("正在更新持倉市值…"):
            render_portfolio_dashboard()

        # ── Gemini AI Portfolio Analysis ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Gemini AI 持倉組合分析")
        _pf_ai_key = _get_gemini_key()
        if _pf_ai_key:
            _pf_data = st.session_state.get("portfolio", {})
            if _pf_data:
                if st.button("✨ 生成 AI 投組分析", key="pf_ai_btn", type="primary"):
                    with st.spinner("正在調用 Gemini AI 分析持倉組合…"):
                        import yfinance as _yf_pf
                        _pf_prices = {}
                        for _t in _pf_data:
                            try:
                                _ph = standardize_timezone(
                                    _yf_pf.Ticker(_t).history(period="1d")
                                )
                                if not _ph.empty:
                                    _pf_prices[_t] = float(_ph["Close"].iloc[-1])
                            except Exception:
                                _pf_prices[_t] = _pf_data[_t].get("buy_price", 0)
                        _pf_ai_text = _call_gemini_portfolio_ai(_pf_data, _pf_prices, _pf_ai_key)
                    st.session_state["pf_ai_report"] = _pf_ai_text
                    st.rerun()
                if st.session_state.get("pf_ai_report"):
                    _pf_report = st.session_state["pf_ai_report"]
                    if _pf_report == "__QUOTA__":
                        render_quota_error()
                    elif _pf_report.startswith("❌"):
                        render_auth_error() if "Key" in _pf_report else st.error(_pf_report)
                    else:
                        st.markdown(
                            f"<div style='background:#1B2A3D; border-left:4px solid #1F6FEB; "
                            f"border-radius:8px; padding:16px; font-size:14px; "
                            f"color:#E6EDF3; line-height:1.8;'>{_pf_report}</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("💡 尚無持倉數據，請先在「個股診斷」頁面添加股票到投資組合。")
        else:
            st.caption("💡 AI 分析需要有效的 GEMINI_API_KEY，請在 Secrets 中配置。")

        # ── Telegram notification history ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📱 Telegram 通知紀錄")
        if _telegram_configured():
            _notif_log = load_notification_log()
            if _notif_log:
                _recent = list(reversed(_notif_log[-10:]))
                _ntype_icons = {
                    "take_profit":        "💰",
                    "stop_loss":          "⚠️",
                    "watchlist_rsi":      "📉",
                    "watchlist_breakout": "📈",
                    "mpf_rebalance":      "🛡️",
                    "macro_fear":         "😱",
                    "macro_crash":        "🚨",
                }
                for _n in _recent:
                    _ts    = _n.get("time", "")[:16].replace("T", " ")
                    _ntype = _n.get("type", "alert")
                    _ntic  = _n.get("ticker", "")
                    _npct  = _n.get("pct", "")
                    _nicon = _ntype_icons.get(_ntype, "📢")
                    _pct_part = (f" ({float(_npct):+.1f}%)" if _npct not in ("", None, "N/A") else "")
                    _label = (
                        f"{_nicon} **{_ts}** — {_ntype}"
                        + (f" — {_ntic}" if _ntic else "")
                        + _pct_part
                    )
                    with st.expander(_label, expanded=False):
                        st.caption(_n.get("message", "（無訊息內容）"))
            else:
                st.info("📭 尚無 Telegram 通知紀錄。\n\n"
                        "點擊側邊欄的「🔍 立即執行全面檢查」按鈕以觸發警報檢查。")
        else:
            st.info("💡 請先在側邊欄的 📱 Telegram 即時通知 中設定 Bot Token 及 User ID。")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — Backtest  (RSI strategy + multi-stock + Gemini + tech analysis)
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📊 美股回測":
        st.markdown("## 📊 進階量化策略回測平台")
        st.caption("多股批量回測 · RSI均值回歸策略 · SMA200濾網 · 動態止損 · Gemini AI分析報告")

        # ── From screener banner ───────────────────────────────────────────
        if st.session_state.get("bt_from_screener"):
            screened_str = st.session_state.get("bt_tickers", "")
            st.markdown(
                f"<div style='background:#0D2E1A; border:1px solid #238636; "
                f"border-radius:6px; padding:8px 14px; margin-bottom:10px; font-size:13px; color:#E6EDF3;'>"
                f"✅ 已從篩選器導入股票：<b>{screened_str}</b>　"
                f"可在下方修改後執行回測。</div>",
                unsafe_allow_html=True,
            )
            st.session_state["bt_from_screener"] = False

        (tab0,) = st.tabs(["🔍 股票篩選"])

        # ──────────────────────────────────────────────────────────────────
        # Tab 0 — Stock Screener (migrated from Macro page)
        # ──────────────────────────────────────────────────────────────────
        with tab0:
            st.markdown("### 🔍 優質美股篩選器")
            st.caption("根據自定義基本面條件，自動篩選符合條件的優質股票，並可一鍵送往回測引擎")

            sc_c1, sc_c2, sc_c3, sc_c4 = st.columns(4)
            sc_c1.metric("市值門檻",  f"> ${min_cap}B")
            sc_c2.metric("淨利率",    f"> {min_margin}%")
            sc_c3.metric("營收增長",  f"> {min_growth}%")
            sc_c4.metric("P/E Ratio", f"< {max_pe}")
            st.caption("💡 在左側邊欄「⚙️ 篩選設定」調整上述條件")
            st.markdown("---")

            if st.button("🔄 開始篩選（需要約1-2分鐘）",
                         type="primary", use_container_width=True, key="bt_screen_run"):
                st.session_state["screening"]     = True
                st.session_state["results"]       = None
                st.session_state["screen_params"] = params

            if st.session_state.get("screening") and not st.session_state.get("results"):
                progress_bar = st.progress(0)
                status_text  = st.empty()
                stocks_data  = []
                total        = len(SCREENER_STOCKS)
                screen_err   = None
                try:
                    for i, ticker in enumerate(SCREENER_STOCKS):
                        status_text.text(f"正在分析 {ticker}… ({i+1}/{total})")
                        progress_bar.progress((i + 1) / total)
                        try:
                            data = get_stock_info(ticker)
                            if data:
                                stocks_data.append(data)
                        except Exception:
                            pass
                        time.sleep(0.05)
                    used_params = st.session_state.get("screen_params", params)
                    results     = screen_stocks(stocks_data, *used_params)
                    st.session_state["results"] = results
                except Exception as _se:
                    screen_err = str(_se)
                    st.session_state["results"] = []
                finally:
                    st.session_state["screening"] = False
                    progress_bar.empty()
                    status_text.empty()
                if screen_err:
                    st.error(f"❌ 篩選過程發生錯誤：{screen_err}　請重試或稍後再試。")

            if st.session_state.get("results"):
                results = st.session_state["results"]
                st.success(f"✅ 篩選完成！找到 **{len(results)}** 支符合條件的股票")

                hcols = st.columns([1, 2.5, 1.5, 1.2, 1, 1, 1, 0.8, 0.8])
                for txt, col in zip(
                    ["代碼", "公司名稱", "產業", "市值", "股價", "淨利率", "營收增長", "P/E", "⭐"],
                    hcols,
                ):
                    col.markdown(f"<span style='color:#888; font-size:12px;'>{txt}</span>",
                                 unsafe_allow_html=True)
                st.markdown("<hr style='margin:4px 0 8px 0; border-color:#ddd;'>",
                            unsafe_allow_html=True)

                for stock in results:
                    t     = stock["ticker"]
                    in_wl = t in st.session_state["watchlist"]
                    rcols = st.columns([1, 2.5, 1.5, 1.2, 1, 1, 1, 0.8, 0.8])
                    if rcols[0].button(f"**{t}**", key=f"sc_goto_{t}", use_container_width=True):
                        navigate_to_diagnosis(stock)
                    rcols[1].markdown(f"<span style='font-size:13px;'>{stock['name']}</span>",
                                      unsafe_allow_html=True)
                    rcols[2].markdown(f"<span style='color:#8B949E; font-size:12px;'>{stock['sector']}</span>",
                                      unsafe_allow_html=True)
                    rcols[3].markdown(f"<span style='font-size:13px;'>{format_market_cap(stock['market_cap'])}</span>",
                                      unsafe_allow_html=True)
                    rcols[4].markdown(f"<span style='color:#0066CC; font-size:13px;'>${stock['price']:.2f}</span>",
                                      unsafe_allow_html=True)
                    rcols[5].markdown(f"<span style='color:#009900; font-size:13px;'>{stock['net_margin']*100:.1f}%</span>",
                                      unsafe_allow_html=True)
                    rcols[6].markdown(f"<span style='color:#CC7700; font-size:13px;'>{stock['revenue_growth']*100:.1f}%</span>",
                                      unsafe_allow_html=True)
                    rcols[7].markdown(f"<span style='font-size:13px;'>{stock['pe_ratio']:.1f}</span>",
                                      unsafe_allow_html=True)
                    wl_label = "⭐" if in_wl else "☆"
                    if rcols[8].button(wl_label, key=f"sc_wl_{t}"):
                        if in_wl:
                            st.session_state["watchlist"].remove(t)
                        else:
                            st.session_state["watchlist"].append(t)
                        save_watchlist(st.session_state["watchlist"])
                        st.rerun()
                    st.markdown("<hr style='margin:2px 0; border-color:#eee;'>",
                                unsafe_allow_html=True)

                df_sc = pd.DataFrame(results)
                sc_tickers = [s["ticker"] for s in results]
                sc_bt1, sc_bt2 = st.columns([3, 1])
                with sc_bt1:
                    st.info(f"📊 篩選出 **{len(sc_tickers)}** 支股票，可向下滾動至「⚙️ 策略設定」執行批量回測")
                with sc_bt2:
                    if st.button("🚀 送往回測", type="primary",
                                 key="sc_to_bt", use_container_width=True):
                        top5 = sc_tickers[:5]
                        st.session_state["bt_tickers"]       = ", ".join(top5)
                        st.session_state["bt_from_screener"] = True
                        st.rerun()

                st.markdown("### 📊 篩選結果圖表")
                try:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_cap = px.bar(
                            df_sc.head(10), x="ticker", y="market_cap",
                            title="市值比較",
                            color="market_cap", color_continuous_scale="Blues",
                            labels={"market_cap": "市值 (USD)", "ticker": "代碼"},
                        )
                        fig_cap.update_layout(
                            paper_bgcolor="#0d1117", plot_bgcolor="#161B22",
                            font=dict(color="#ddd"),
                            showlegend=False, margin=dict(t=40, b=20),
                        )
                        fig_cap.update_yaxes(tickformat=".2s",
                                             tickfont=dict(color="#aaa"),
                                             gridcolor="#2a2a3a")
                        fig_cap.update_xaxes(tickfont=dict(color="#aaa"))
                        st.plotly_chart(fig_cap, use_container_width=True)
                    with c2:
                        fig_sc = px.scatter(
                            df_sc, x="pe_ratio", y="net_margin",
                            size="market_cap", color="revenue_growth",
                            hover_data=["ticker", "name"],
                            title="P/E vs 淨利率（氣泡=市值）",
                            labels={"pe_ratio": "P/E", "net_margin": "淨利率",
                                    "revenue_growth": "營收增長"},
                            color_continuous_scale="Teal",
                        )
                        fig_sc.update_layout(
                            paper_bgcolor="#0d1117", plot_bgcolor="#161B22",
                            font=dict(color="#ddd"),
                            margin=dict(t=40, b=20),
                        )
                        fig_sc.update_xaxes(tickfont=dict(color="#aaa"), gridcolor="#2a2a3a")
                        fig_sc.update_yaxes(tickfont=dict(color="#aaa"), gridcolor="#2a2a3a")
                        st.plotly_chart(fig_sc, use_container_width=True)
                except Exception as _ce:
                    st.warning(f"⚠️ 圖表加載失敗：{_ce}")

            elif not st.session_state.get("screening"):
                st.info("👆 點擊上方按鈕開始篩選股票")
                with st.expander("📖 篩選邏輯說明"):
                    st.markdown(f"""
| 指標 | 條件 | 說明 |
|------|------|------|
| 市值 | > ${min_cap}B | 確保流動性充足 |
| 淨利率 | > {min_margin}% | 良好獲利能力 |
| 營收增長 | > {min_growth}% | 確保成長動能 |
| P/E Ratio | < {max_pe} | 合理估值範圍 |

資料來源：Yahoo Finance (yfinance)
                    """)

        # ──────────────────────────────────────────────────────────────────
        # Section 1 — Strategy Setup  (scrollable)
        # ──────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## ⚙️ 策略設定")
        with st.container():
            st.markdown("### 📌 資產類別與 RSI 參數")

            asset_class = st.radio(
                "資產類別（自動調整 RSI 閾值）",
                list(be.get_rsi_presets().keys()),
                horizontal=True,
                key="bt_asset_class",
            )
            preset = be.get_rsi_presets()[asset_class]

            preset_info_cols = st.columns(4)
            for col_w, k, label in [
                (preset_info_cols[0], "buy",        "RSI 買入閾值"),
                (preset_info_cols[1], "sell",       "RSI 賣出閾值"),
                (preset_info_cols[2], "sma_period", "SMA 濾網週期"),
                (preset_info_cols[3], "stop_loss",  "動態止損"),
            ]:
                val = f"{preset[k]*100:.0f}%" if k == "stop_loss" else str(preset[k])
                col_w.markdown(
                    f"<div style='background:#1B2A3D; border-left:3px solid #1F6FEB; "
                    f"border-radius:6px; padding:10px; text-align:center;'>"
                    f"<div style='font-size:12px; color:#8B949E;'>{label}</div>"
                    f"<div style='font-size:18px; font-weight:700; color:#E6EDF3;'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🎯 回測參數")

            with st.form("backtest_form_v2"):
                row1a, row1b, row1c = st.columns([4, 1, 2])
                with row1a:
                    raw_tickers = st.text_input(
                        "股票代碼（最多 5 支，逗號分隔）",
                        value=st.session_state.get("bt_tickers", "AAPL, MSFT, NVDA"),
                        placeholder="AAPL, MSFT, NVDA, TSLA, GOOG",
                        help="最多 5 支，超出部分自動截取前 5 支",
                    )
                with row1b:
                    years = st.selectbox(
                        "回測年期", [1, 2, 3, 5, 10],
                        index=[1,2,3,5,10].index(
                            st.session_state.get("bt_years", 3)
                        ),
                    )
                with row1c:
                    benchmark = st.selectbox(
                        "基準指數",
                        ["SPY", "QQQ", "QUAL", "MTUM", "VT"],
                        index=["SPY","QQQ","QUAL","MTUM","VT"].index(
                            st.session_state.get("bt_benchmark", "SPY")
                        ),
                    )

                row2a, row2b = st.columns([2, 3])
                with row2a:
                    strategy_mode = st.radio(
                        "策略模式",
                        ["買入持有", "RSI均值回歸"],
                        index=0 if st.session_state.get("bt_strategy", "買入持有") == "買入持有" else 1,
                        horizontal=True,
                    )
                with row2b:
                    if asset_class == "自定義":
                        rsi_buy_custom  = st.slider("RSI 買入閾值", 10, 45, preset["buy"],   key="rsi_buy_c")
                        rsi_sell_custom = st.slider("RSI 賣出閾值", 55, 90, preset["sell"],  key="rsi_sell_c")
                        stop_loss_custom= st.slider("止損比例 %",    5,  25, int(preset["stop_loss"]*100), key="sl_c")
                        rsi_buy_v   = rsi_buy_custom
                        rsi_sell_v  = rsi_sell_custom
                        stop_loss_v = stop_loss_custom / 100
                    else:
                        rsi_buy_v   = preset["buy"]
                        rsi_sell_v  = preset["sell"]
                        stop_loss_v = preset["stop_loss"]
                        if strategy_mode == "RSI均值回歸":
                            st.info(
                                f"使用 {asset_class} 預設：RSI {rsi_buy_v}/{rsi_sell_v}　"
                                f"SMA{preset['sma_period']} 濾網　止損 {stop_loss_v*100:.0f}%"
                            )

                submitted_bt = st.form_submit_button("🚀 執行回測", type="primary",
                                                     use_container_width=True)

            if submitted_bt:
                tickers_raw = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
                tickers = tickers_raw[:5]
                st.session_state["bt_tickers"]   = raw_tickers
                st.session_state["bt_years"]     = years
                st.session_state["bt_benchmark"] = benchmark
                st.session_state["bt_strategy"]  = strategy_mode
                if not tickers:
                    st.error("請至少輸入一個股票代碼。")
                elif len(tickers_raw) > 5:
                    st.warning(f"⚠️ 最多支援 5 支股票，已自動使用前 5 支：{', '.join(tickers)}")
                if tickers:
                    with st.spinner(f"正在執行 {strategy_mode} 回測（{years}年期），請稍候…"):
                        result = be.run_backtest(
                            tickers, years, benchmark,
                            strategy_mode=strategy_mode,
                            rsi_buy=rsi_buy_v, rsi_sell=rsi_sell_v,
                            sma_period=preset.get("sma_period", 200),
                            stop_loss_pct=stop_loss_v,
                        )
                    st.session_state["bt_result"] = result
                    if not result.get("error"):
                        st.success("✅ 回測完成！請向下滾動查看「績效報告」。")
                    else:
                        st.error(f"⚠️ 回測失敗：{result['error']}")

        # ──────────────────────────────────────────────────────────────────
        # Section 2 — Performance Report  (scrollable)
        # ──────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📈 績效報告")
        with st.container():
            def _fmt_pct(v):
                return f"{v*100:+.2f}%" if v is not None else "N/A"
            def _fmt_f(v, decimals=2):
                return f"{v:.{decimals}f}" if v is not None else "N/A"

            result = st.session_state.get("bt_result")
            if not result:
                st.info("💡 請先在上方「策略設定」設定參數並執行回測。")
            elif result.get("error"):
                st.error(f"⚠️ 回測失敗：{result['error']}")
            else:
                pf_s         = result["portfolio_series"]
                bm_s         = result["benchmark_series"]
                pf_m         = result["portfolio_metrics"]
                bm_m         = result["benchmark_metrics"]
                alpha        = result["alpha"]
                hv           = result["high_vol_flags"]
                tickers_used = result["pf_tickers"]
                bm_ticker    = result["benchmark_ticker"]
                strategy_lbl = result.get("strategy_mode", "買入持有")
                contrib      = result.get("contribution", {})
                dd_periods   = result.get("drawdown_periods", [])

                st.markdown(
                    f"<div style='background:#1B2A3D; border-left:4px solid #1F6FEB; "
                    f"border-radius:8px; padding:8px 14px; margin-bottom:10px; font-size:13px; color:#E6EDF3;'>"
                    f"📊 策略：<b>{strategy_lbl}</b>　"
                    f"標的：<b>{', '.join(tickers_used)}</b>　"
                    f"基準：<b>{bm_ticker}</b>　"
                    f"年期：<b>{result['window_years']}年</b></div>",
                    unsafe_allow_html=True,
                )

                # ── Performance chart ──────────────────────────────────────
                st.markdown("### 📈 績效走勢對比")
                fig = go.Figure()
                pf_color = "#FF4B4B" if strategy_lbl == "買入持有" else "#FFD700"

                # Individual ticker lines (dotted, dimmer) — only when multi-stock
                _IND_COLORS = ["#FF9F7F", "#7FD4FF", "#CCFF99", "#FFD700", "#CC99FF",
                               "#FF77AA", "#99FFEE", "#FFC0CB", "#AAD4FF", "#FFE066"]
                ind_series = result.get("individual_series", {})
                if len(tickers_used) > 1:
                    for idx, t in enumerate(tickers_used):
                        s_ind = ind_series.get(t)
                        if s_ind is not None and not s_ind.empty:
                            fig.add_trace(go.Scatter(
                                x=s_ind.index, y=s_ind.values, mode="lines",
                                name=f"{t}（個別）",
                                line=dict(color=_IND_COLORS[idx % len(_IND_COLORS)],
                                          width=1.2, dash="dot"),
                                opacity=0.7,
                                hovertemplate=f"{t} %{{x|%Y-%m-%d}}: %{{y:.1f}}<extra></extra>",
                            ))

                # Combined portfolio line (bold)
                if pf_s is not None and not pf_s.empty:
                    lbl = (f"{strategy_lbl}（{tickers_used[0]}）"
                           if len(tickers_used) == 1
                           else f"{strategy_lbl} 等權組合")
                    fig.add_trace(go.Scatter(
                        x=pf_s.index, y=pf_s.values, mode="lines",
                        name=lbl,
                        line=dict(color=pf_color, width=2.5),
                        hovertemplate="日期: %{x|%Y-%m-%d}<br>淨值: %{y:.1f}<extra></extra>",
                    ))

                # Benchmark line
                if bm_s is not None and not bm_s.empty:
                    fig.add_trace(go.Scatter(
                        x=bm_s.index, y=bm_s.values, mode="lines",
                        name=f"基準：{bm_ticker}",
                        line=dict(color="#00D4FF", width=2, dash="dot"),
                        hovertemplate="日期: %{x|%Y-%m-%d}<br>淨值: %{y:.1f}<extra></extra>",
                    ))

                fig.update_layout(
                    height=420, margin=dict(l=0, r=0, t=20, b=20),
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(size=11, color="#ddd")),
                    xaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                               tickfont=dict(size=12, color="#aaa")),
                    yaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                               tickfont=dict(size=12, color="#aaa"),
                               title=dict(text="淨值（初始=100）",
                                          font=dict(size=12, color="#aaa"))),
                    font=dict(size=12, color="#ddd"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Metrics cards ──────────────────────────────────────────
                st.markdown("### 📋 績效指標報告")
                alpha_color = "#00FF7F" if (alpha is not None and alpha > 0) else "#FF4B4B"
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                for col_m, title, val, color in [
                    (mc1, "策略 CAGR",   _fmt_pct(pf_m.get("cagr")),   "#00FF7F"),
                    (mc2, "夏普比率",    _fmt_f(pf_m.get("sharpe")),   "#00D4FF"),
                    (mc3, "最大回撤",    _fmt_pct(pf_m.get("max_dd")), "#FF4B4B"),
                    (mc4, "年化波動",    _fmt_pct(pf_m.get("vol")),    "#FFD700"),
                    (mc5, "超額報酬 α",  _fmt_pct(alpha),              alpha_color),
                ]:
                    col_m.markdown(
                        f"<div style='background:#1C2128; border-left:3px solid {color}; "
                        f"border-radius:6px; padding:12px; text-align:center;'>"
                        f"<div style='font-size:12px; color:#8B949E;'>{title}</div>"
                        f"<div style='font-size:18px; font-weight:700; color:#E6EDF3;'>{val}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                bc1, bc2, bc3, bc4 = st.columns(4)
                for col_b, title, val in [
                    (bc1, f"{bm_ticker} CAGR",     _fmt_pct(bm_m.get("cagr"))),
                    (bc2, f"{bm_ticker} 夏普",     _fmt_f(bm_m.get("sharpe"))),
                    (bc3, f"{bm_ticker} 最大回撤",  _fmt_pct(bm_m.get("max_dd"))),
                    (bc4, f"{bm_ticker} 年化波動",  _fmt_pct(bm_m.get("vol"))),
                ]:
                    col_b.markdown(
                        f"<div style='background:#1C2128; border:1px solid #30363D; "
                        f"border-radius:6px; padding:10px; text-align:center;'>"
                        f"<div style='font-size:12px; color:#8B949E;'>{title}</div>"
                        f"<div style='font-size:16px; font-weight:600; color:#E6EDF3;'>{val}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Per-ticker contribution table ──────────────────────────
                if result.get("per_ticker_metrics"):
                    st.markdown("---")
                    st.markdown("### 🔬 個股績效與貢獻度")
                    rows_t = []
                    total_portfolio_ret = pf_m.get("cagr")
                    for t, m in result["per_ticker_metrics"].items():
                        vol_flag  = hv.get(t, False)
                        c_data    = contrib.get(t, {})
                        rows_t.append({
                            "代碼":       t,
                            "CAGR":       _fmt_pct(m.get("cagr")),
                            "夏普比率":   _fmt_f(m.get("sharpe")),
                            "最大回撤":   _fmt_pct(m.get("max_dd")),
                            "總回報":     f"{c_data.get('total_return_pct',0):+.1f}%",
                            "組合貢獻":   f"{c_data.get('contribution_pct',0):+.1f}%",
                            "風險標記":   "🔴 高波動" if vol_flag else "✅ 正常",
                        })
                    st.dataframe(pd.DataFrame(rows_t),
                                 use_container_width=True, hide_index=True)

                # ── Drawdown period analysis ───────────────────────────────
                if dd_periods:
                    st.markdown("---")
                    st.markdown("### 📉 最大回撤週期分析")
                    worst = dd_periods[0]
                    prolonged = worst["duration_days"] > 180 and not worst["recovered"]
                    banner_color = "#FF4B4B" if prolonged else "#FFD700"
                    banner_text  = (
                        "⚠️ 偵測到長期單邊下跌持倉過久（未恢復超過 180 天）——"
                        "建議啟用動態止損以避免深度回撤。"
                        if prolonged else
                        f"✅ 最大回撤週期 {worst['duration_days']} 天，"
                        f"{'已恢復' if worst['recovered'] else '尚未恢復'}。"
                    )
                    st.markdown(
                        f"<div style='background:#1a0d0d; border-left:4px solid {banner_color}; "
                        f"border-radius:6px; padding:10px 14px; margin-bottom:8px; "
                        f"font-size:13px; color:{banner_color};'>{banner_text}</div>",
                        unsafe_allow_html=True,
                    )
                    dd_rows = []
                    for d in dd_periods[:6]:
                        dd_rows.append({
                            "開始日期":     str(d["start_date"])[:10],
                            "結束日期":     str(d["end_date"])[:10],
                            "持續天數":     d["duration_days"],
                            "最大跌幅":     f"{d['max_loss_pct']:.1f}%",
                            "是否恢復":     "✅ 是" if d["recovered"] else "❌ 未恢復",
                        })
                    st.dataframe(pd.DataFrame(dd_rows),
                                 use_container_width=True, hide_index=True)

                # ── Gemini AI analysis report ──────────────────────────────
                st.markdown("---")
                st.markdown("### 🤖 Gemini AI 量化分析報告")
                gemini_key_val = _get_gemini_key()
                if gemini_key_val:
                    if st.button("✨ 生成 AI 分析報告", key="gen_gemini_report"):
                        with st.spinner("正在調用 Gemini AI 分析回測數據…"):
                            report_text = generate_quant_report(result, gemini_key_val)
                        if report_text:
                            st.session_state["bt_gemini_report"] = report_text
                            st.rerun()
                        else:
                            st.warning("AI 報告生成失敗，請確認 GEMINI_API_KEY 已正確設定於 Secrets。")
                    _bt_report = st.session_state.get("bt_gemini_report")
                    if _bt_report:
                        if _bt_report == "__QUOTA__":
                            render_quota_error()
                        elif _bt_report.startswith("❌"):
                            render_auth_error() if "Key" in _bt_report else st.error(_bt_report)
                        else:
                            st.markdown(
                                f"<div style='background:#1B2A3D; border-left:4px solid #1F6FEB; "
                                f"border-radius:8px; padding:16px; font-size:14px; "
                                f"color:#E6EDF3; line-height:1.8;'>"
                                f"{_bt_report}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("💡 AI 報告需要有效的 GEMINI_API_KEY 配置。")

        # ──────────────────────────────────────────────────────────────────
        # Section 3 — Technical Indicator Comparison  (scrollable)
        # ──────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📉 技術指標對比")
        with st.container():
            result = st.session_state.get("bt_result")
            if not result or result.get("error"):
                st.info("💡 請先在上方「策略設定」完成回測。")
            else:
                tickers_used = result["pf_tickers"]
                years_used   = result.get("window_years", 3)

                indicator = st.radio(
                    "選擇指標",
                    ["RSI", "MACD", "Bollinger Bands"],
                    horizontal=True, key="bt_indicator",
                )

                colors = ["#FF4B4B", "#00D4FF", "#FFD700", "#00FF7F", "#FF8C00"]

                if indicator == "RSI":
                    fig_rsi = go.Figure()
                    preset_now = be.get_rsi_presets().get(
                        st.session_state.get("bt_asset_class", "自定義"),
                        be.get_rsi_presets()["自定義"]
                    )
                    for i, t in enumerate(tickers_used):
                        with st.spinner(f"載入 {t} RSI…"):
                            ohlcv = be.fetch_ohlcv(t, years_used)
                        if ohlcv.empty:
                            continue
                        rsi_s = be.calc_rsi(ohlcv["Close"])
                        fig_rsi.add_trace(go.Scatter(
                            x=rsi_s.index, y=rsi_s.values,
                            mode="lines", name=t,
                            line=dict(color=colors[i % len(colors)], width=1.5),
                            hovertemplate=f"{t} RSI: %{{y:.1f}}<extra></extra>",
                        ))
                    for level, color, label in [
                        (preset_now["buy"],  "#00FF7F", f"買入線 {preset_now['buy']}"),
                        (preset_now["sell"], "#FF4B4B", f"賣出線 {preset_now['sell']}"),
                        (50, "#888888", "中線 50"),
                    ]:
                        fig_rsi.add_hline(
                            y=level, line_dash="dash",
                            line_color=color,
                            annotation_text=label,
                            annotation_font_color=color,
                        )
                    fig_rsi.update_layout(
                        height=380, title="RSI 對比",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        legend=dict(orientation="h", y=1.05, font=dict(size=12, color="#ddd")),
                        xaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                   tickfont=dict(size=12, color="#aaa")),
                        yaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                   tickfont=dict(size=12, color="#aaa"),
                                   range=[0, 100]),
                        font=dict(size=12, color="#ddd"),
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)

                elif indicator == "MACD":
                    for i, t in enumerate(tickers_used):
                        with st.spinner(f"載入 {t} MACD…"):
                            ohlcv = be.fetch_ohlcv(t, years_used)
                        if ohlcv.empty:
                            continue
                        macd_l, signal_l, hist_l = be.calc_macd(ohlcv["Close"])
                        fig_m = go.Figure()
                        fig_m.add_trace(go.Scatter(
                            x=macd_l.index, y=macd_l.values, mode="lines",
                            name="MACD", line=dict(color="#00D4FF", width=1.5)))
                        fig_m.add_trace(go.Scatter(
                            x=signal_l.index, y=signal_l.values, mode="lines",
                            name="Signal", line=dict(color="#FFD700", width=1.5)))
                        hist_colors = ["#00FF7F" if v >= 0 else "#FF4B4B"
                                       for v in hist_l.values]
                        fig_m.add_trace(go.Bar(
                            x=hist_l.index, y=hist_l.values,
                            name="Histogram", marker_color=hist_colors, opacity=0.6))
                        fig_m.add_hline(y=0, line_color="#555", line_dash="dot")
                        fig_m.update_layout(
                            height=300, title=f"{t} MACD",
                            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                            legend=dict(orientation="h", y=1.05,
                                        font=dict(size=12, color="#ddd")),
                            xaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                       tickfont=dict(size=12, color="#aaa")),
                            yaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                       tickfont=dict(size=12, color="#aaa")),
                            font=dict(size=12, color="#ddd"),
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        st.plotly_chart(fig_m, use_container_width=True)

                else:  # Bollinger Bands
                    for i, t in enumerate(tickers_used[:3]):
                        with st.spinner(f"載入 {t} Bollinger…"):
                            ohlcv = be.fetch_ohlcv(t, years_used)
                        if ohlcv.empty:
                            continue
                        upper, mid, lower = be.calc_bollinger(ohlcv["Close"])
                        fig_bb = go.Figure()
                        fig_bb.add_trace(go.Scatter(
                            x=upper.index, y=upper.values, mode="lines",
                            name="上軌", line=dict(color="#FF4B4B", width=1, dash="dash")))
                        fig_bb.add_trace(go.Scatter(
                            x=mid.index, y=mid.values, mode="lines",
                            name="中軌 SMA20", line=dict(color="#00D4FF", width=1.5)))
                        fig_bb.add_trace(go.Scatter(
                            x=lower.index, y=lower.values, mode="lines",
                            name="下軌", line=dict(color="#00FF7F", width=1, dash="dash"),
                            fill="tonexty", fillcolor="rgba(0,212,255,0.04)"))
                        fig_bb.add_trace(go.Scatter(
                            x=ohlcv["Close"].index, y=ohlcv["Close"].values,
                            mode="lines", name="收盤價",
                            line=dict(color="#FFD700", width=1.5)))
                        fig_bb.update_layout(
                            height=320, title=f"{t} Bollinger Bands (20,2)",
                            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                            legend=dict(orientation="h", y=1.05,
                                        font=dict(size=12, color="#ddd")),
                            xaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                       tickfont=dict(size=12, color="#aaa")),
                            yaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                       tickfont=dict(size=12, color="#aaa")),
                            font=dict(size=12, color="#ddd"),
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        st.plotly_chart(fig_bb, use_container_width=True)

        # ──────────────────────────────────────────────────────────────────
        # Section 4 — Fund Flow Analysis  (scrollable)
        # ──────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 💹 資金流入分析")
        with st.container():
            st.caption("OBV（能量潮）反映資金累積趨勢；MFI（資金流量指標）衡量買賣力道強弱")
            result = st.session_state.get("bt_result")
            if not result or result.get("error"):
                st.info("💡 請先在「策略設定」完成回測。")
            else:
                tickers_used = result["pf_tickers"]
                years_used   = result.get("window_years", 3)
                colors_ff    = ["#00D4FF", "#FFD700", "#FF4B4B", "#00FF7F", "#FF8C00"]

                flow_mode = st.radio(
                    "指標選擇", ["OBV（能量潮）", "MFI（資金流量）"],
                    horizontal=True, key="bt_flow_mode",
                )

                if flow_mode == "OBV（能量潮）":
                    fig_obv = go.Figure()
                    for i, t in enumerate(tickers_used):
                        with st.spinner(f"載入 {t} OBV…"):
                            ohlcv = be.fetch_ohlcv(t, years_used)
                        if ohlcv.empty:
                            continue
                        obv_s = be.calc_obv(ohlcv["Close"], ohlcv["Volume"])
                        obv_norm = (obv_s / obv_s.abs().max() * 100).rename(t)
                        fig_obv.add_trace(go.Scatter(
                            x=obv_norm.index, y=obv_norm.values, mode="lines",
                            name=t, line=dict(color=colors_ff[i % len(colors_ff)], width=1.5),
                            hovertemplate=f"{t} OBV(norm): %{{y:.1f}}<extra></extra>",
                        ))
                    fig_obv.add_hline(y=0, line_color="#555", line_dash="dot",
                                      annotation_text="基準線", annotation_font_color="#888")
                    fig_obv.update_layout(
                        height=380, title="OBV 能量潮對比（已正規化）",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        legend=dict(orientation="h", y=1.05,
                                    font=dict(size=12, color="#ddd")),
                        xaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                   tickfont=dict(size=12, color="#aaa")),
                        yaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                   tickfont=dict(size=12, color="#aaa")),
                        font=dict(size=12, color="#ddd"),
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_obv, use_container_width=True)
                    st.caption(
                        "📌 OBV 持續上升 = 資金持續流入（牛市訊號）；"
                        "OBV 下行背離股價 = 警惕潛在賣壓。"
                    )

                else:  # MFI
                    for i, t in enumerate(tickers_used):
                        with st.spinner(f"載入 {t} MFI…"):
                            ohlcv = be.fetch_ohlcv(t, years_used)
                        if ohlcv.empty:
                            continue
                        mfi_s = be.calc_mfi(
                            ohlcv["High"], ohlcv["Low"],
                            ohlcv["Close"], ohlcv["Volume"]
                        )
                        # latest MFI gauge
                        latest_mfi = float(mfi_s.dropna().iloc[-1]) if not mfi_s.dropna().empty else 50
                        fig_mfi = go.Figure()
                        fig_mfi.add_trace(go.Scatter(
                            x=mfi_s.index, y=mfi_s.values, mode="lines",
                            name=f"{t} MFI",
                            line=dict(color=colors_ff[i % len(colors_ff)], width=1.5),
                            hovertemplate=f"MFI: %{{y:.1f}}<extra></extra>",
                        ))
                        fig_mfi.add_hline(y=80, line_color="#FF4B4B", line_dash="dash",
                                          annotation_text="超買 80",
                                          annotation_font_color="#FF4B4B")
                        fig_mfi.add_hline(y=20, line_color="#00FF7F", line_dash="dash",
                                          annotation_text="超賣 20",
                                          annotation_font_color="#00FF7F")
                        status = "超買" if latest_mfi >= 80 else "超賣" if latest_mfi <= 20 else "中性"
                        status_c = "#FF4B4B" if status == "超買" else "#00FF7F" if status == "超賣" else "#FFD700"
                        fig_mfi.update_layout(
                            height=300,
                            title=f"{t} MFI — 最新值: {latest_mfi:.1f} "
                                  f"<span style='color:{status_c}'>({status})</span>",
                            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                            legend=dict(orientation="h", y=1.05,
                                        font=dict(size=12, color="#ddd")),
                            xaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                       tickfont=dict(size=12, color="#aaa")),
                            yaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                                       tickfont=dict(size=12, color="#aaa"),
                                       range=[0, 100]),
                            font=dict(size=12, color="#ddd"),
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        st.plotly_chart(fig_mfi, use_container_width=True)
                    st.caption(
                        "📌 MFI > 80 = 資金過度集中（超買）；MFI < 20 = 籌碼出清（超賣）。"
                        "配合 RSI 使用可提高入場準確度。"
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — MPF 智投
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🛡️ MPF 智投":
        render_mpf_page()


if __name__ == "__main__":
    try:
        main()
    except Exception as _top_err:
        st.error(f"❌ 程式初始化失敗，請重新整理頁面。（{_top_err}）")

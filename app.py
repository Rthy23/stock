# ═══════════════════════════════════════════════════════════════════════════════
# main.py  —  Entry point, page routing, sidebar (美股選股儀表板)
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import plotly.express as px
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
    calc_buy_zone, calc_exit_strategy,
    plot_relative_strength, plot_four_quadrant,
)
from data_fetcher import (
    get_stock_info, get_historical_data, get_analyst_data,
    get_combined_sentiment, get_market_benchmark,
    save_watchlist, save_portfolio,
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

_MODULE = "main"

def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    init_session()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("📈 美股選股")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "功能選擇",
        ["📡 總體市場 (Macro)", "🔬 個股診斷 (Micro)", "💼 我的持倉"],
        key="page",
    )

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
            f"<div style='background:#0D2E0D; border:1px solid #00FF7F55; "
            f"border-radius:6px; padding:8px 10px; font-size:12px;'>"
            f"✅ <b style='color:#00FF7F;'>宏觀多頭（黃金交叉）</b><br>"
            f"<span style='color:#aaa;'>SMA50 {sma50_str} &gt; SMA200 {sma200_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif bm.get("golden_cross") is False:
        sma50_str  = f"{bm['sma50']:.2f}"  if bm["sma50"]  else "N/A"
        sma200_str = f"{bm['sma200']:.2f}" if bm["sma200"] else "N/A"
        st.sidebar.markdown(
            f"<div style='background:#2E0D0D; border:1px solid #FF4B4B55; "
            f"border-radius:6px; padding:8px 10px; font-size:12px;'>"
            f"⚠️ <b style='color:#FF4B4B;'>宏觀空頭（死亡交叉）</b><br>"
            f"<span style='color:#aaa;'>SMA50 {sma50_str} &lt; SMA200 {sma200_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.caption("⏳ 大盤數據載入中…")

    if bm.get("price") and bm.get("perf_1y") is not None:
        perf = bm["perf_1y"]
        pc   = "#00FF7F" if perf >= 0 else "#FF4B4B"
        sign = "+" if perf >= 0 else ""
        st.sidebar.markdown(
            f"<div style='color:#aaa; font-size:11px; margin-top:5px;'>"
            f"{bm['label']}　"
            f"<b style='color:#00D4FF;'>${bm['price']:.2f}</b>　"
            f"<span style='color:{pc};'>{sign}{perf:.1f}% (1Y)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

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
                    st.session_state["page"]             = "🔬 個股診斷 (Micro)"
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
                f"<div style='background:linear-gradient(90deg,#0D2E0D,#112911); "
                f"border:1.5px solid #00FF7F; border-radius:8px; "
                f"padding:10px 18px; margin-bottom:12px; font-size:14px;'>"
                f"✅ &nbsp;<b style='color:#00FF7F;'>市場趨勢：宏觀多頭（黃金交叉）</b>"
                f"&nbsp;—&nbsp;"
                f"<span style='color:#aaa;'>{bm['label']} ({benchmark})　"
                f"SMA50 <b style='color:#fff;'>{bm['sma50']:.2f}</b> &gt; "
                f"SMA200 <b style='color:#fff;'>{bm['sma200']:.2f}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif bm.get("golden_cross") is False:
            st.markdown(
                f"<div style='background:linear-gradient(90deg,#2E0D0D,#291111); "
                f"border:1.5px solid #FF4B4B; border-radius:8px; "
                f"padding:10px 18px; margin-bottom:12px; font-size:14px;'>"
                f"⚠️ &nbsp;<b style='color:#FF4B4B;'>市場趨勢：宏觀空頭（死亡交叉）</b>"
                f"&nbsp;—&nbsp;"
                f"<span style='color:#aaa;'>{bm['label']} ({benchmark})　"
                f"SMA50 <b style='color:#fff;'>{bm['sma50']:.2f}</b> &lt; "
                f"SMA200 <b style='color:#fff;'>{bm['sma200']:.2f}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        render_macro_sentiment_dashboard(bm_data=bm)

        st.markdown("根據您自定義的基本面條件，自動篩選符合條件的優質美股")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("市值門檻",  f"> ${min_cap}B")
        col2.metric("淨利率",    f"> {min_margin}%")
        col3.metric("營收增長",  f"> {min_growth}%")
        col4.metric("P/E Ratio", f"< {max_pe}")

        st.markdown("---")

        if st.button("🔄 開始篩選（需要約1-2分鐘）",
                     type="primary", use_container_width=True):
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
            st.success(
                f"✅ 篩選完成！找到 **{len(results)}** 支符合條件的股票，"
                f"點擊代碼即可查看完整診斷"
            )
            hcols = st.columns([1, 2.5, 1.5, 1.2, 1, 1, 1, 0.8, 0.8])
            for txt, col in zip(
                ["代碼", "公司名稱", "產業", "市值", "股價", "淨利率", "營收增長", "P/E", "⭐"],
                hcols,
            ):
                col.markdown(
                    f"<span style='color:#aaa; font-size:12px;'>{txt}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("<hr style='margin:4px 0 8px 0; border-color:#2A2D3E;'>",
                        unsafe_allow_html=True)

            for stock in results:
                t     = stock["ticker"]
                in_wl = t in st.session_state["watchlist"]
                rcols = st.columns([1, 2.5, 1.5, 1.2, 1, 1, 1, 0.8, 0.8])
                if rcols[0].button(f"**{t}**", key=f"goto_{t}",
                                   use_container_width=True):
                    navigate_to_diagnosis(stock)
                rcols[1].markdown(
                    f"<span style='font-size:13px;'>{stock['name']}</span>",
                    unsafe_allow_html=True)
                rcols[2].markdown(
                    f"<span style='color:#aaa; font-size:12px;'>{stock['sector']}</span>",
                    unsafe_allow_html=True)
                rcols[3].markdown(
                    f"<span style='font-size:13px;'>{format_market_cap(stock['market_cap'])}</span>",
                    unsafe_allow_html=True)
                rcols[4].markdown(
                    f"<span style='color:#00D4FF; font-size:13px;'>${stock['price']:.2f}</span>",
                    unsafe_allow_html=True)
                rcols[5].markdown(
                    f"<span style='color:#00FF7F; font-size:13px;'>{stock['net_margin']*100:.1f}%</span>",
                    unsafe_allow_html=True)
                rcols[6].markdown(
                    f"<span style='color:#FFD700; font-size:13px;'>{stock['revenue_growth']*100:.1f}%</span>",
                    unsafe_allow_html=True)
                rcols[7].markdown(
                    f"<span style='font-size:13px;'>{stock['pe_ratio']:.1f}</span>",
                    unsafe_allow_html=True)
                wl_label = "⭐" if in_wl else "☆"
                if rcols[8].button(wl_label, key=f"wl_tog_{t}"):
                    if in_wl:
                        st.session_state["watchlist"].remove(t)
                    else:
                        st.session_state["watchlist"].append(t)
                    save_watchlist(st.session_state["watchlist"])
                    st.rerun()
                st.markdown("<hr style='margin:2px 0; border-color:#1E2130;'>",
                            unsafe_allow_html=True)

            df = pd.DataFrame(results)
            st.markdown("### 📊 篩選結果圖表")
            c1, c2 = st.columns(2)
            with c1:
                fig_cap = px.bar(
                    df.head(10), x="ticker", y="market_cap",
                    title="市值比較",
                    color="market_cap", color_continuous_scale="Blues",
                    labels={"market_cap": "市值 (USD)", "ticker": "股票代碼"},
                )
                fig_cap.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117", plot_bgcolor="#1A1D2E",
                    showlegend=False,
                    hovermode="closest",
                )
                fig_cap.update_yaxes(tickformat=".2s")
                st.plotly_chart(fig_cap, use_container_width=True)
            with c2:
                fig_scatter = px.scatter(
                    df, x="pe_ratio", y="net_margin",
                    size="market_cap", color="revenue_growth",
                    hover_data=["ticker", "name"],
                    title="P/E vs 淨利率（氣泡=市值）",
                    labels={"pe_ratio": "P/E Ratio", "net_margin": "淨利率",
                            "revenue_growth": "營收增長"},
                    color_continuous_scale="Teal",
                )
                fig_scatter.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117", plot_bgcolor="#1A1D2E",
                    hovermode="closest",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        elif not st.session_state.get("screening"):
            st.info("👆 點擊上方按鈕開始篩選股票")
            with st.expander("📖 篩選邏輯說明"):
                st.markdown(f"""
| 指標 | 條件 | 說明 |
|------|------|------|
| 市值 | > ${min_cap}B | 確保流動性 |
| 淨利率 | > {min_margin}% | 良好獲利能力 |
| 營收增長 | > {min_growth}% | 確保成長動能 |
| P/E Ratio | < {max_pe} | 合理估值 |

資料來源：Yahoo Finance (yfinance)
                """)

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

            # ── 1週 / 1個月 / 3個月 對比分析 ─────────────────────────────────
            if hist is not None and not hist.empty:
                st.markdown("---")
                st.markdown("### 📅 歷史表現對比分析（1週 / 1個月 / 3個月）")
                _periods = {"1 週": 5, "1 個月": 21, "3 個月": 63}
                _tabs = st.tabs(list(_periods.keys()))
                for _tab, (_label, _days) in zip(_tabs, _periods.items()):
                    with _tab:
                        _slice = hist.tail(_days)
                        if len(_slice) < 2:
                            st.caption("資料不足，無法計算此區間。")
                            continue
                        _open_p   = float(_slice["Close"].iloc[0])
                        _close_p  = float(_slice["Close"].iloc[-1])
                        _high_p   = float(_slice["High"].max())
                        _low_p    = float(_slice["Low"].min())
                        _ret_p    = (_close_p - _open_p) / _open_p * 100
                        _avg_vol  = float(_slice["Volume"].mean())
                        _ret_col  = "#00FF7F" if _ret_p >= 0 else "#FF4B4B"
                        _ret_sign = f"+{_ret_p:.2f}%" if _ret_p >= 0 else f"{_ret_p:.2f}%"
                        _c1, _c2, _c3, _c4 = st.columns(4)
                        _c1.metric(f"區間報酬（{_label}）", _ret_sign,
                                   delta=_ret_sign, delta_color="normal")
                        _c2.metric("區間最高",  f"${_high_p:.2f}")
                        _c3.metric("區間最低",  f"${_low_p:.2f}")
                        _c4.metric("均日成交量", f"{_avg_vol:,.0f}")
                        _fig_p = px.line(
                            _slice.reset_index(),
                            x=_slice.index, y="Close",
                            title=f"{ticker} — {_label}收盤走勢",
                            labels={"Close": "收盤價 (USD)", "x": "日期"},
                            color_discrete_sequence=[_ret_col],
                        )
                        _fig_p.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D2E",
                            height=250, margin=dict(t=38, b=30, l=60, r=20),
                            hovermode="x unified",
                        )
                        st.plotly_chart(_fig_p, use_container_width=True,
                                        key=f"period_chart_{ticker}_{_label}")

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
                f"<span style='color:#666; font-size:11px;'>⚡ 決策速查表　更新時間 {_ts}</span>"
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
                    "<div style='background:#0D2E0D; border:2px solid #00FF7F; "
                    "border-radius:8px; padding:10px 18px; margin-bottom:6px; "
                    "font-size:14px; text-align:center;'>"
                    "✅ <b style='color:#00FF7F; font-size:15px;'>買入共識訊號</b>"
                    "　—　個股相對強勢 + 分析師評級買入，條件同步觸發。"
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif _oversold_warn:
                st.markdown(
                    "<div style='background:#2E1A0D; border:2px solid #FF8C00; "
                    "border-radius:8px; padding:10px 18px; margin-bottom:6px; "
                    "font-size:14px; text-align:center;'>"
                    "⚠️ <b style='color:#FFA500; font-size:15px;'>超跌警示（補跌風險）</b>"
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
                            f"<div style='background:#2E0D0D; border:1px solid #FF4B4B; "
                            f"border-radius:7px; padding:9px 16px; margin-bottom:8px; font-size:13px;'>"
                            f"⚠️ <b style='color:#FF4B4B;'>機構級資金活動偵測</b>"
                            f"　今日成交量為 20 日均量的 "
                            f"<b style='color:#FF4B4B;'>{vol_ratio:.1f}×</b>"
                            f"，異常放量，建議密切關注方向性突破。</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        ratio_color = "#00FF7F" if vol_ratio >= 1.2 else "#aaa"
                        st.markdown(
                            f"<div style='background:#13162A; border-radius:7px; "
                            f"padding:8px 16px; margin-bottom:8px; font-size:13px;'>"
                            f"今日成交量為 20 日均量的 "
                            f"<b style='color:{ratio_color};'>{vol_ratio:.2f}×</b>"
                            f"　（均量：{avg20_val:,.0f}）</div>",
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
                    st.markdown(
                        f"<div style='background:#1A1D2E; border-radius:6px; "
                        f"padding:8px 14px; margin-top:8px;'>"
                        f"目前盈虧：<span style='color:{pnl_col}; font-weight:700; font-size:16px;'>"
                        f"{pnl_sym}</span>　｜　"
                        f"浮動盈虧金額：<span style='color:{pnl_col}; font-weight:700;'>"
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
                                f"<div style='background:#0D2E0D; border:1px solid #00FF7F55; "
                                f"border-radius:7px; padding:8px 18px; margin-bottom:6px; font-size:13px;'>"
                                f"🚀 <b style='color:#00FF7F;'>強於大盤 (Alpha {a_sign}%)</b>"
                                f"　{ticker} 報酬 <b style='color:#fff;'>{r_sign}%</b>"
                                f"　vs {bm_ticker} 報酬"
                                f" <b style='color:#00D4FF;'>{stock_ret - alpha_pct:+.1f}%</b>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div style='background:#2E1A0D; border:1px solid #FF8C0055; "
                                f"border-radius:7px; padding:8px 18px; margin-bottom:6px; font-size:13px;'>"
                                f"🐢 <b style='color:#FFA500;'>弱於大盤 (Alpha {a_sign}%)</b>"
                                f"　{ticker} 報酬 <b style='color:#fff;'>{r_sign}%</b>"
                                f"　vs {bm_ticker} 報酬"
                                f" <b style='color:#00D4FF;'>{stock_ret - alpha_pct:+.1f}%</b>"
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
                f"<span style='color:#555; font-size:12px; margin-left:12px;'>"
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

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Portfolio Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "💼 我的持倉":
        with st.spinner("正在更新持倉市值…"):
            render_portfolio_dashboard()


if __name__ == "__main__":
    main()

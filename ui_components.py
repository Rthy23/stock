# ═══════════════════════════════════════════════════════════════════════════════
# ui_components.py  —  Plotly charts, Streamlit render panels, session init
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import json
from datetime import datetime

from analysis import (
    classify_sentiment, sentiment_badge, news_impact_summary,
    compute_technicals, screen_stocks, format_market_cap,
    calc_buy_zone, calc_exit_strategy, classify_investment_horizon,
    _vix_to_greed_score, _score_to_label,
    plot_relative_strength, plot_four_quadrant,
    plot_sentiment_gauge, plot_fear_timeline,
)
from data_fetcher import (
    load_watchlist, load_portfolio, save_watchlist, save_portfolio,
    get_stock_news, get_analyst_data, get_social_sentiment,
    get_combined_sentiment, get_vix_history, get_market_benchmark,
    parse_ibkr_screenshot, fmt_usd_hkd, _get_rate,
    MACRO_EVENTS, BENCHMARK_LABELS, SECTOR_ETFS,
)

_MODULE = "ui_components"

def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── Session state init ─────────────────────────────────────────────────────────
def init_session() -> None:
    """Initialise all required session_state keys (idempotent)."""
    try:
        defaults = {
            "watchlist":      load_watchlist,
            "portfolio":      load_portfolio,
            "screening":      lambda: False,
            "results":        lambda: None,
            "diag_ticker":    lambda: "",
            "diag_stock_info":lambda: None,
            "diag_hist":      lambda: None,
            "diag_period":    lambda: "1y",
            "auto_fetch":     lambda: False,
            "nav_page":       lambda: "📡 總體市場 (Macro)",
            "benchmark":      lambda: "VOO",
        }
        for key, factory in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = factory()
    except Exception as e:
        print(_err("init_session", e))


# ── Navigation helper ──────────────────────────────────────────────────────────
def navigate_to_diagnosis(stock_info: dict) -> None:
    """Jump to diagnosis page and flag auto-fetch for historical data."""
    try:
        st.session_state["diag_ticker"]      = stock_info["ticker"]
        st.session_state["diag_stock_info"]  = stock_info
        st.session_state["diag_hist"]        = None
        st.session_state["auto_fetch"]       = True
        st.session_state["nav_page"]         = "🔬 個股診斷 (Micro)"
        st.rerun()
    except Exception as e:
        print(_err("navigate_to_diagnosis", e))


# ── Sentiment mutation alert ───────────────────────────────────────────────────
def _check_sentiment_mutation(ticker: str, current_score: float) -> None:
    """Compare current social score vs cached; fire alert if drop ≥ 30%."""
    try:
        key  = f"prev_social_{ticker}"
        prev = st.session_state.get(key)
        if prev is not None and prev > 1:
            drop = (prev - current_score) / prev
            if drop >= 0.30:
                st.error(
                    f"🚨 **情緒突變警報！** {ticker} 的社交情緒得分在本次更新中"
                    f"驟降 **{drop*100:.1f}%**（{prev:.0f} → {current_score:.0f} / 100）。"
                    f"　社交媒體出現大量空頭情緒，請注意下行風險！"
                )
        st.session_state[key] = current_score
    except Exception as e:
        print(_err("_check_sentiment_mutation", e))


# ── Zone progress bar ──────────────────────────────────────────────────────────
def zone_progress_bar(pct_position, buy_lower, buy_upper, price) -> None:
    """Render a horizontal HTML progress bar for the buy-zone position."""
    if pct_position is None:
        return
    try:
        bar_color = "#00D4FF" if 0 <= pct_position <= 100 else "#FFA500"
        clamp     = max(0, min(100, pct_position))
        bar_html  = f"""
        <div style="margin: 16px 0 8px 0;">
          <div style="display:flex; justify-content:space-between;
                      color:#aaa; font-size:13px; margin-bottom:4px;">
            <span>SMA 200 (下限) ${buy_lower:.2f}</span>
            <span style="color:#00D4FF; font-weight:600;">
              目前價格 ${price:.2f}（{clamp:.0f}%）
            </span>
            <span>SMA 50 (上限) ${buy_upper:.2f}</span>
          </div>
          <div style="background:#2A2D3E; border-radius:8px;
                      height:22px; width:100%; position:relative; overflow:hidden;">
            <div style="background:linear-gradient(90deg,#00D4FF,{bar_color});
                        width:{clamp}%; height:100%; border-radius:8px;
                        transition:width .5s ease;"></div>
            <div style="position:absolute; top:50%; left:{clamp}%;
                        transform:translate(-50%,-50%);
                        width:12px; height:12px; border-radius:50%;
                        background:white; box-shadow:0 0 6px rgba(255,255,255,0.8);">
            </div>
          </div>
          <div style="display:flex; justify-content:space-between;
                      color:#555; font-size:13px; margin-top:4px;">
            <span>強力支撐區</span>
            <span>理想買入區</span>
            <span>偏高區域</span>
          </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)
    except Exception as e:
        print(_err("zone_progress_bar", e))


# ── Price / candlestick chart ──────────────────────────────────────────────────
def plot_price_chart(
    ticker, hist,
    sma50=None, sma200=None,
    stop_price=None, target_price=None,
    buy_lower=None, buy_upper=None,
    portfolio_buy_price=None,
    current_price=None,
) -> go.Figure:
    """Full K-line chart with SMAs, stop/target, buy-zone shading, and PnL line."""
    try:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"],
            low=hist["Low"],   close=hist["Close"],
            name="K 線",
            increasing_line_color="#00FF7F", decreasing_line_color="#FF4B4B",
        ))
        if sma50 is not None:
            fig.add_hline(
                y=sma50, line_color="#FFD700", line_width=1.5, line_dash="dash",
                annotation_text="  SMA 50",
                annotation_font_color="#FFD700",
                annotation_position="right",
            )
        if sma200 is not None:
            fig.add_hline(
                y=sma200, line_color="#FF8C00", line_width=1.5, line_dash="dot",
                annotation_text="  SMA 200",
                annotation_font_color="#FF8C00",
                annotation_position="right",
            )
        if buy_lower and buy_upper:
            fig.add_hrect(
                y0=buy_lower, y1=buy_upper,
                fillcolor="rgba(0,212,255,0.06)",
                line_width=0,
                annotation_text="買入區間",
                annotation_font_color="#00D4FF",
                annotation_position="right",
            )
        if stop_price is not None:
            fig.add_hline(
                y=stop_price, line_color="#FF4B4B", line_width=2, line_dash="dot",
                annotation_text="  止損",
                annotation_font_color="#FF4B4B",
                annotation_position="left",
            )
        if target_price is not None:
            fig.add_hline(
                y=target_price, line_color="#00FF7F", line_width=2, line_dash="dot",
                annotation_text="  目標",
                annotation_font_color="#00FF7F",
                annotation_position="left",
            )
        if portfolio_buy_price is not None:
            pnl_pct    = ((current_price or 0) - portfolio_buy_price) / portfolio_buy_price * 100
            cost_color = "#00FF7F" if pnl_pct >= 0 else "#FF4B4B"
            ann_text   = f"  持倉成本 ${portfolio_buy_price:.2f}"
            fig.add_hline(
                y=portfolio_buy_price,
                line=dict(color=cost_color, dash="dot", width=2.2),
                annotation_text=ann_text,
                annotation_font_color=cost_color,
                annotation_position="left",
            )
        fig.update_layout(
            title=f"{ticker} 股價走勢",
            xaxis_title="日期", yaxis_title="價格 (USD)",
            template="plotly_dark",
            plot_bgcolor="#1A1D2E", paper_bgcolor="#0E1117",
            xaxis_rangeslider_visible=False,
            height=520,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            annotations=[
                dict(
                    xref="paper", yref="paper",
                    x=0.0, y=-0.08,
                    showarrow=False,
                    text=(
                        "📗 <b style='color:#00FF7F'>陽線</b>（收 > 開，綠色）　"
                        "📕 <b style='color:#FF4B4B'>陰線</b>（收 < 開，紅色）　"
                        "— SMA50（金）　·· SMA200（橙）"
                    ),
                    font=dict(size=12, color="#888"),
                    align="left",
                    xanchor="left",
                ),
            ],
        )
        return fig
    except Exception as e:
        print(_err("plot_price_chart", e))
        return go.Figure()


# ── Volume chart ───────────────────────────────────────────────────────────────
def plot_volume_chart(ticker: str, hist: pd.DataFrame) -> tuple:
    """
    Enhanced volume chart with 20-day avg overlay and institutional flag.
    Returns (fig, vol_ratio, is_institutional).
    """
    try:
        avg20  = hist["Volume"].rolling(20).mean()
        colors = ["#00D4FF" if c >= o else "#FF4B4B"
                  for c, o in zip(hist["Close"], hist["Open"])]
        last_vol = float(hist["Volume"].iloc[-1])
        last_avg = float(avg20.iloc[-1]) if not pd.isna(avg20.iloc[-1]) else last_vol
        vol_ratio       = last_vol / last_avg if last_avg > 0 else 1.0
        is_institutional = vol_ratio >= 2.0

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"], marker_color=colors, name="成交量",
            hovertemplate="日期：%{x}<br>成交量：%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=hist.index, y=avg20, mode="lines", name="20日均量",
            line=dict(color="#FFD700", width=1.8, dash="dash"),
            hovertemplate="20日均量：%{y:,.0f}<extra></extra>",
        ))
        flag_text  = (f"⚠️ 今日量 {vol_ratio:.1f}× 均量<br>機構級資金活動"
                      if is_institutional else f"今日量 {vol_ratio:.1f}× 均量")
        flag_color = "#FF4B4B" if is_institutional else "#FFD700"
        fig.add_annotation(
            x=hist.index[-1], y=last_vol, text=flag_text,
            showarrow=True, arrowhead=2, arrowcolor=flag_color,
            font=dict(color=flag_color, size=12),
            bgcolor="rgba(0,0,0,0.7)", ax=-60, ay=-40,
        )
        fig.update_layout(
            title=dict(text=f"📊 {ticker} 成交量動能分析", font=dict(size=13, color="#fff")),
            xaxis=dict(showgrid=True, gridcolor="#1E2130", tickfont=dict(color="#aaa")),
            yaxis=dict(showgrid=True, gridcolor="#1E2130", tickfont=dict(color="#aaa"),
                       title=dict(text="成交量", font=dict(color="#aaa", size=11))),
            legend=dict(font=dict(color="#aaa"), bgcolor="rgba(0,0,0,0)"),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
            height=290, margin=dict(t=42, b=55, l=65, r=20), barmode="overlay",
            hovermode="x unified",
            annotations=[
                dict(
                    xref="paper", yref="paper",
                    x=0.0, y=-0.18,
                    showarrow=False,
                    text=(
                        "🔵 <b style='color:#00D4FF'>藍色柱</b>：當日陽線成交（收 > 開）　"
                        "🔴 <b style='color:#FF4B4B'>紅色柱</b>：當日陰線成交（收 < 開）　"
                        "— <b style='color:#FFD700'>黃線</b>：20日均量"
                    ),
                    font=dict(size=12, color="#888"),
                    align="left",
                    xanchor="left",
                ),
            ],
        )
        return fig, vol_ratio, is_institutional
    except Exception as e:
        print(_err("plot_volume_chart", e))
        return go.Figure(), 1.0, False


# ── Analyst target range chart ─────────────────────────────────────────────────
def plot_analyst_targets(
    current_price: float, t_low: float, t_mean: float,
    t_high: float, ticker: str,
) -> go.Figure:
    """Horizontal bar: analyst target range with mean and current price lines."""
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[t_high - t_low], y=["目標價區間"], base=[t_low], orientation="h",
            marker_color="rgba(0,212,255,0.2)",
            marker_line=dict(color="#00D4FF", width=1),
            name="分析師目標區間",
            hovertemplate=(f"目標低：${t_low:.2f}<br>目標高：${t_high:.2f}<extra></extra>"),
        ))
        axis_min = min(current_price, t_low)  * 0.93
        axis_max = max(current_price, t_high) * 1.07
        fig.add_vline(x=t_mean, line_color="#FFD700", line_width=2,
                      annotation_text=f"  均值 ${t_mean:.2f}",
                      annotation_font=dict(color="#FFD700", size=11),
                      annotation_position="top")
        fig.add_vline(x=current_price, line_color="#fff", line_width=2, line_dash="dash",
                      annotation_text=f"  現價 ${current_price:.2f}",
                      annotation_font=dict(color="#fff", size=11),
                      annotation_position="bottom")
        upside      = (t_mean - current_price) / current_price * 100
        upside_color = "#00FF7F" if upside >= 0 else "#FF4B4B"
        fig.update_layout(
            title=dict(
                text=(f"分析師目標價區間　上行空間 "
                      f"<span style='color:{upside_color}'>{upside:+.1f}%</span>"),
                font=dict(size=12, color="#fff"),
            ),
            xaxis=dict(range=[axis_min, axis_max], tickprefix="$",
                       showgrid=True, gridcolor="#1E2130",
                       tickfont=dict(color="#aaa", size=12)),
            yaxis=dict(tickfont=dict(color="#aaa", size=12)),
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=140, margin=dict(t=48, b=30, l=90, r=20), showlegend=False,
        )
        return fig
    except Exception as e:
        print(_err("plot_analyst_targets", e))
        return go.Figure()


# ── Analyst recommendation history ────────────────────────────────────────────
def plot_analyst_recs(recs_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of analyst recommendations history (recent periods)."""
    try:
        period_col = "period" if "period" in recs_df.columns else recs_df.columns[0]
        labels     = [str(p) for p in recs_df[period_col]]
        labels_map = {"0m": "本月", "-1m": "上月", "-2m": "2月前",
                      "-3m": "3月前", "-4m": "4月前", "-5m": "5月前"}
        labels = [labels_map.get(l, l) for l in labels]

        def _col(df, *names):
            for n in names:
                if n in df.columns:
                    return df[n].fillna(0).astype(int).tolist()
            return [0] * len(df)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="強力買入", x=labels,
                             y=_col(recs_df, "strongBuy", "strong_buy"),
                             marker_color="#00FF7F"))
        fig.add_trace(go.Bar(name="買入", x=labels,
                             y=_col(recs_df, "buy"), marker_color="#7FFF00"))
        fig.add_trace(go.Bar(name="持有", x=labels,
                             y=_col(recs_df, "hold"), marker_color="#FFD700"))
        fig.add_trace(go.Bar(name="賣出", x=labels,
                             y=_col(recs_df, "sell"), marker_color="#FF8C00"))
        fig.add_trace(go.Bar(name="強力賣出", x=labels,
                             y=_col(recs_df, "strongSell", "strong_sell"),
                             marker_color="#FF4B4B"))
        fig.update_layout(
            title=dict(text="分析師評級歷史（近期各月）",
                       font=dict(size=12, color="#fff")),
            barmode="stack",
            xaxis=dict(tickfont=dict(color="#aaa", size=12)),
            yaxis=dict(title=dict(text="分析師數", font=dict(color="#aaa", size=12)),
                       tickfont=dict(color="#aaa", size=12),
                       showgrid=True, gridcolor="#1E2130"),
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            legend=dict(font=dict(color="#aaa", size=12),
                        orientation="h", yanchor="bottom", y=1.02),
            height=260, margin=dict(t=55, b=30, l=50, r=20),
        )
        return fig
    except Exception as e:
        print(_err("plot_analyst_recs", e))
        return go.Figure()


# ── Radar chart ────────────────────────────────────────────────────────────────
def plot_radar(stock_info: dict) -> go.Figure:
    """Pentagonal fundamental radar for a stock."""
    try:
        pe_score     = max(0, min(100, (30 - stock_info["pe_ratio"]) / 30 * 100)) if stock_info["pe_ratio"] > 0 else 0
        margin_score = min(100, stock_info["net_margin"] * 300)
        growth_score = min(100, stock_info["revenue_growth"] * 200)
        cap_score    = min(100, stock_info["market_cap"] / 2_000_000_000_000 * 100)
        div_score    = min(100, (stock_info["dividend_yield"] or 0) * 2000)
        fig = go.Figure(go.Scatterpolar(
            r=[pe_score, margin_score, growth_score, cap_score, div_score],
            theta=["本益比評分", "淨利率", "營收增長", "市值規模", "股息殖利率"],
            fill="toself", line_color="#00D4FF",
            fillcolor="rgba(0,212,255,0.2)",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2A2D3E"),
                angularaxis=dict(gridcolor="#2A2D3E"),
                bgcolor="#1A1D2E",
            ),
            showlegend=False, template="plotly_dark",
            paper_bgcolor="#0E1117", height=350, title="基本面雷達圖",
        )
        return fig
    except Exception as e:
        print(_err("plot_radar", e))
        return go.Figure()


# ── Diagnosis checklist ────────────────────────────────────────────────────────
def render_diagnosis(stock_info: dict, params: tuple) -> None:
    """Render the per-stock diagnostic checklist against screener thresholds."""
    try:
        ticker = stock_info["ticker"]
        min_cap, min_margin, min_growth, max_pe = params
        st.markdown("### 🩺 個股診斷 — 選股條件逐項檢查")
        checks = [
            (f"市值 > ${min_cap}B",
             stock_info["market_cap"] > min_cap * 1_000_000_000,
             format_market_cap(stock_info["market_cap"])),
            (f"淨利率 > {min_margin}%",
             stock_info["net_margin"] > min_margin / 100,
             f"{stock_info['net_margin']*100:.1f}%"),
            (f"過去一年營收增長 > {min_growth}%",
             stock_info["revenue_growth"] > min_growth / 100,
             f"{stock_info['revenue_growth']*100:.1f}%"),
            (f"P/E Ratio < {max_pe}",
             0 < stock_info["pe_ratio"] < max_pe,
             f"{stock_info['pe_ratio']:.1f}"),
        ]
        cols = st.columns([3, 1, 2])
        cols[0].markdown("**條件**"); cols[1].markdown("**結果**"); cols[2].markdown("**實際數值**")
        st.markdown("---")
        all_pass = True
        for label, passed, value in checks:
            c0, c1, c2 = st.columns([3, 1, 2])
            c0.markdown(label)
            c1.markdown("✅ 達標" if passed else "❌ 未達標")
            c2.markdown(f"`{value}`")
            if not passed:
                all_pass = False
        st.markdown("---")
        if all_pass:
            st.success(f"🎯 **{ticker}** 完全符合您設定的選股條件！")
        else:
            failed = sum(1 for _, p, _ in checks if not p)
            st.warning(f"⚠️ **{ticker}** 有 {failed} 項條件未達標，不建議納入核心持倉。")
    except Exception as e:
        st.error(f"MODULE_ERROR: [ui_components] | FUNCTION: [render_diagnosis] | ERROR: {e}")


# ── Trade plan renderer ────────────────────────────────────────────────────────
def render_trade_plan(stock_info: dict, hist: pd.DataFrame) -> tuple:
    """Render buy-zone + stop/target trade plan cards."""
    try:
        price  = stock_info["price"]
        ticker = stock_info["ticker"]
        sma50, sma200, low20 = compute_technicals(hist)
        buy_lower, buy_upper, zone_label, pct = calc_buy_zone(price, sma50, sma200)
        stop, target, risk_pct = (
            calc_exit_strategy(price, sma200, low20) if sma200 else (None, None, None)
        )
        st.markdown("### 📊 建議買入區間 (Buy Zone)")
        st.info(zone_label)
        if buy_lower and buy_upper:
            zone_progress_bar(pct, buy_lower, buy_upper, price)
        st.markdown("### 📋 交易計畫建議")
        c1, c2, c3 = st.columns(3)
        buy_limit  = buy_upper if buy_upper else price
        with c1:
            st.markdown(f"""
            <div style="background:#1A2E1A; border:1px solid #00FF7F33;
                        border-radius:10px; padding:16px; text-align:center;">
              <div style="color:#aaa; font-size:12px; margin-bottom:4px;">🟢 建議買入上限</div>
              <div style="color:#00FF7F; font-size:22px; font-weight:700;">${buy_limit:.2f}</div>
              <div style="color:#666; font-size:13px;">(SMA 50)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            if stop:
                st.markdown(f"""
                <div style="background:#2E1A1A; border:1px solid #FF4B4B33;
                            border-radius:10px; padding:16px; text-align:center;">
                  <div style="color:#aaa; font-size:12px; margin-bottom:4px;">🔴 建議止損價</div>
                  <div style="color:#FF4B4B; font-size:22px; font-weight:700;">${stop:.2f}</div>
                  <div style="color:#666; font-size:13px;">(SMA200 × 97% 或 20日低 × 98%)</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("資料不足")
        with c3:
            if target:
                st.markdown(f"""
                <div style="background:#1A1A2E; border:1px solid #00D4FF33;
                            border-radius:10px; padding:16px; text-align:center;">
                  <div style="color:#aaa; font-size:12px; margin-bottom:4px;">🎯 預計目標價</div>
                  <div style="color:#00D4FF; font-size:22px; font-weight:700;">${target:.2f}</div>
                  <div style="color:#666; font-size:13px;">(買入點盈虧比 1:2)</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("資料不足")
        if stop:
            st.markdown("---")
            entry_price = buy_limit
            if price < stop:
                st.error(
                    f"🚨 **已觸發止損 / 不建議此買入價**：目前股價 ${price:.2f} 已低於止損線 "
                    f"${stop:.2f}，技術上已跌破止損位。"
                    f"**強烈不建議**在此價位買入，請等待趨勢確認後再重新評估。"
                )
            elif risk_pct is not None:
                loss_pct = abs(risk_pct) * 100
                gain_pct = loss_pct * 2
                formula_pct = ((stop - entry_price) / entry_price) * 100
                col_warn, col_ratio = st.columns([3, 1])
                with col_warn:
                    st.error(
                        f"⚠️ **風險提示**：以建議買入價 ${entry_price:.2f} 計算，"
                        f"跌至止損線 ${stop:.2f} 將虧損約 **{loss_pct:.1f}%**。"
                        f"（風控公式：(止損 − 買入) ÷ 買入 = {formula_pct:.1f}%）"
                        f"　請確保此倉位的虧損在您的整體資金管理範圍內。"
                    )
                with col_ratio:
                    st.markdown(f"""
                    <div style="background:#1A1D2E; border:1px solid #444;
                                border-radius:8px; padding:12px; text-align:center;">
                      <div style="color:#aaa; font-size:13px;">風險報酬比</div>
                      <div style="color:#FFD700; font-size:20px; font-weight:700;">1 : 2</div>
                      <div style="color:#aaa; font-size:13px;">
                        損 {loss_pct:.1f}% → 獲 {gain_pct:.1f}%
                      </div>
                    </div>""", unsafe_allow_html=True)
        # ── Investment horizon classification box ──────────────────────────
        horizon = classify_investment_horizon(price, sma50, sma200, hist)
        acc     = horizon["accent"]
        bg      = horizon["bg"]
        bdr     = horizon["border"]
        icon    = horizon["icon"]
        label   = horizon["label"]
        period  = horizon["hold_period"]
        reasons = horizon["reasons"]

        bullet_html = "".join(
            f"<li style='margin:4px 0; font-size:13px; color:#ccc;'>{r}</li>"
            for r in reasons
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div style="
                background:{bg};
                border-left:4px solid {bdr};
                border-radius:8px;
                padding:16px 20px;
                margin-top:8px;
            ">
              <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <span style="font-size:24px;">{icon}</span>
                <div>
                  <span style="font-size:13px; color:#aaa; letter-spacing:.5px;">
                    投資屬性分類
                  </span><br>
                  <span style="font-size:18px; font-weight:700; color:{acc};">
                    {label}
                  </span>
                </div>
              </div>
              <div style="
                  background:#ffffff0d;
                  border-radius:6px;
                  padding:8px 12px;
                  margin-bottom:10px;
                  font-size:13px;
                  color:{acc};
                  font-weight:600;
              ">
                🗓 {period}
              </div>
              <ul style="margin:0; padding-left:18px; list-style:disc;">
                {bullet_html}
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return sma50, sma200, stop, target, buy_lower, buy_upper
    except Exception as e:
        st.error(f"MODULE_ERROR: [ui_components] | FUNCTION: [render_trade_plan] | ERROR: {e}")
        return None, None, None, None, None, None


# ── News intelligence ──────────────────────────────────────────────────────────
def render_news_intelligence(ticker: str, company_name: str) -> None:
    """Fetch + display news with keyword sentiment and macro impact summary."""
    st.markdown("---")
    _news_ts = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"### 📰 新聞與影響分析 (News Intelligence)"
        f"<span style='color:#555; font-size:12px; margin-left:12px;'>"
        f"⏱ 資料載入時間 {_news_ts}</span>",
        unsafe_allow_html=True,
    )
    try:
        with st.spinner(f"正在獲取 {ticker} 最新新聞…"):
            raw_news = get_stock_news(ticker)
    except Exception as e:
        st.warning(f"⚠️ 新聞獲取發生錯誤：{e}")
        return

    if not raw_news:
        st.info("ℹ️ 暫時無法獲取近期新聞，請稍後重試。")
        return

    try:
        news_items = []
        for item in raw_news:
            content   = item.get("content", {})
            title = pub_date = url = publisher = ""
            if isinstance(content, dict):
                title     = content.get("title", "") or content.get("headline", "")
                pub_date  = content.get("pubDate", "") or content.get("displayTime", "")
                provider  = content.get("provider", {})
                if isinstance(provider, dict):
                    publisher = provider.get("displayName", "")
                canonical = content.get("canonicalUrl", {})
                if isinstance(canonical, dict):
                    url = canonical.get("url", "")
            else:
                title     = item.get("title", "")
                pub_date  = item.get("providerPublishTime", "")
                url       = item.get("link", "")
                publisher = item.get("publisher", "")
            if not title:
                continue
            news_items.append({
                "title":     title,
                "publisher": publisher,
                "pub_date":  pub_date,
                "url":       url,
                "sentiment": classify_sentiment(title),
            })

        if not news_items:
            st.info("ℹ️ 暫時無法解析新聞內容，請稍後重試。")
            return

        result = news_impact_summary(news_items)
        if len(result) == 2:
            st.info(result[1])
            return

        overall, icon, verdict, summary, pos, neg, neu, total = result
        verdict_colors = {
            "bullish":        ("#00FF7F", "#1A2E1A"),
            "mildly_bullish": ("#FFD700", "#2E2A1A"),
            "neutral":        ("#aaa",    "#1A1D2E"),
            "bearish":        ("#FF4B4B", "#2E1A1A"),
        }
        v_color, v_bg = verdict_colors.get(overall, ("#aaa", "#1A1D2E"))
        pos_w = pos / total * 100
        neg_w = neg / total * 100
        neu_w = neu / total * 100

        st.markdown(f"""
        <div style="background:{v_bg}; border:1px solid {v_color}44;
                    border-radius:12px; padding:20px; margin-bottom:16px;">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <span style="font-size:28px;">{icon}</span>
            <div>
              <div style="color:{v_color}; font-size:18px; font-weight:700;">
                宏觀情緒研判：{verdict}
              </div>
              <div style="color:#aaa; font-size:12px;">
                基於近期 {total} 則新聞分析 · {company_name} ({ticker})
              </div>
            </div>
          </div>
          <p style="color:#ccc; font-size:14px; line-height:1.7; margin:0 0 16px 0;">{summary}</p>
          <div style="margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between;
                        color:#aaa; font-size:12px; margin-bottom:4px;">
              <span>🟢 正面 {pos} 則</span>
              <span>⚪ 中性 {neu} 則</span>
              <span>🔴 負面 {neg} 則</span>
            </div>
            <div style="display:flex; height:10px; border-radius:6px; overflow:hidden;">
              <div style="background:#00FF7F; width:{pos_w:.1f}%;"></div>
              <div style="background:#555;    width:{neu_w:.1f}%;"></div>
              <div style="background:#FF4B4B; width:{neg_w:.1f}%;"></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 📋 近期新聞清單")
        for item in news_items:
            badge_html    = sentiment_badge(item["sentiment"])
            title_display = item["title"]
            if item.get("url"):
                title_display = (
                    f'<a href="{item["url"]}" target="_blank" '
                    f'style="color:#00D4FF; text-decoration:none;">'
                    f'{item["title"]}</a>'
                )
            pub_info = ""
            if item.get("publisher"):
                pub_info += item["publisher"]
            if item.get("pub_date"):
                try:
                    if isinstance(item["pub_date"], (int, float)):
                        dt = datetime.fromtimestamp(int(item["pub_date"]))
                        pub_info += f" · {dt.strftime('%m/%d %H:%M')}"
                    elif isinstance(item["pub_date"], str) and item["pub_date"]:
                        pub_info += f" · {item['pub_date'][:10]}"
                except Exception:
                    pass
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:10px;
                        padding:10px 0; border-bottom:1px solid #2A2D3E;">
              <div style="padding-top:2px; flex-shrink:0;">{badge_html}</div>
              <div>
                <div style="font-size:14px; line-height:1.5;">{title_display}</div>
                <div style="color:#666; font-size:13px; margin-top:3px;">{pub_info}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"⚠️ 新聞資料解析發生錯誤，部分數據可能無法顯示。({e})")


# ── Portfolio dashboard ────────────────────────────────────────────────────────
def render_portfolio_dashboard() -> None:
    """Full portfolio page: IBKR sync, holdings table, pie chart, notes."""
    st.title("💼 我的投資組合")

    exp_col, note_col = st.columns([1, 3])
    with exp_col:
        portfolio_snapshot = st.session_state.get("portfolio", {})
        if portfolio_snapshot:
            st.download_button(
                label="⬇️ 導出持倉 JSON",
                data=json.dumps(portfolio_snapshot, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="portfolio_export.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("（尚無持倉可導出）")
    with note_col:
        st.info(
            "💾 持倉數據儲存於本次瀏覽器 Session。"
            "關閉頁面前請使用左側按鈕導出 JSON 備份，"
            "或確認伺服器端可寫入時數據已自動保存。",
            icon="ℹ️",
        )

    st.markdown("---")

    # ── IBKR Screenshot Sync ───────────────────────────────────────────────────
    with st.expander("📸 同步盈透證券截圖（AI 自動辨識）", expanded=False):
        st.markdown(
            "<span style='color:#aaa; font-size:13px;'>"
            "上傳盈透證券持倉截圖，AI 將自動提取代碼、持倉量、平均成本與未實現盈虧。"
            "解析完成後請確認後再同步。"
            "</span>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "選擇截圖檔案（JPG / PNG / WebP）",
            type=["jpg", "jpeg", "png", "webp"],
            key="ibkr_upload",
        )
        if uploaded is not None:
            img_bytes = uploaded.read()
            mime_map  = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                         "png": "image/png",  "webp": "image/webp"}
            ext       = uploaded.name.rsplit(".", 1)[-1].lower()
            mime_type = mime_map.get(ext, "image/png")
            col_img, col_btn = st.columns([3, 1])
            with col_img:
                st.image(img_bytes, caption="上傳的截圖預覽", use_container_width=True)
            parse_clicked = col_btn.button(
                "🤖 AI 解析截圖", type="primary",
                use_container_width=True, key="ibkr_parse_btn",
            )
            if parse_clicked or st.session_state.get("ibkr_parsed_rows"):
                if parse_clicked:
                    with st.spinner("🔍 正在使用 Gemini Vision 解析截圖，請稍候（最長 90 秒）…"):
                        try:
                            holdings, detected_rate = parse_ibkr_screenshot(img_bytes, mime_type)
                            st.session_state["ibkr_parsed_rows"] = holdings
                            if detected_rate is not None:
                                st.session_state["usd_to_hkd"] = detected_rate
                                st.info(f"💱 截圖中偵測到匯率：1 USD = {detected_rate} HKD，已自動套用。")
                        except ValueError as e:
                            st.error(f"❌ 解析失敗：{e}")
                            st.session_state.pop("ibkr_parsed_rows", None)
                        except Exception as e:
                            st.error(
                                f"❌ 發生未預期的錯誤：{type(e).__name__} — {e}\n"
                                f"請檢查網路連線或截圖格式後重試。"
                            )
                            st.session_state.pop("ibkr_parsed_rows", None)

                parsed_rows = st.session_state.get("ibkr_parsed_rows", [])
                if parsed_rows:
                    st.markdown("#### 📋 AI 解析結果預覽（請核對後再確認同步）")
                    rate       = _get_rate()
                    preview_df = pd.DataFrame([{
                        "代碼":             r.get("ticker", ""),
                        "持倉量":           r.get("qty"),
                        "平均成本(USD)":     r.get("avg_price"),
                        f"平均成本(HKD@{rate})": (
                            round(float(r["avg_price"]) * rate, 2)
                            if r.get("avg_price") is not None else None
                        ),
                        "未實現盈虧(USD)":  r.get("unrealized_pnl"),
                        f"未實現盈虧(HKD@{rate})": (
                            round(float(r["unrealized_pnl"]) * rate, 2)
                            if r.get("unrealized_pnl") is not None else None
                        ),
                    } for r in parsed_rows])
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    if st.button("✅ 確認同步到投資組合", type="primary",
                                 use_container_width=True, key="ibkr_confirm"):
                        synced = skipped = 0
                        for r in parsed_rows:
                            ticker    = str(r.get("ticker") or "").strip().upper()
                            qty       = r.get("qty")
                            avg_price = r.get("avg_price")
                            if ticker and qty is not None and avg_price is not None:
                                try:
                                    st.session_state["portfolio"][ticker] = {
                                        "buy_price": float(avg_price),
                                        "qty":       int(qty),
                                    }
                                    synced += 1
                                except (ValueError, TypeError):
                                    skipped += 1
                            else:
                                skipped += 1
                        save_portfolio(st.session_state["portfolio"])
                        st.session_state.pop("ibkr_parsed_rows", None)
                        msg = f"✅ 已成功同步 {synced} 筆持倉！"
                        if skipped:
                            msg += f"（{skipped} 筆因資料不完整已略過）"
                        st.success(msg)
                        st.rerun()

    st.markdown("---")

    # ── Portfolio overview ─────────────────────────────────────────────────────
    portfolio = st.session_state.get("portfolio", {})
    if not portfolio:
        st.info("📭 投資組合尚為空。請前往『個股診斷』頁面，輸入買入價格與持倉數量後加入，或使用上方截圖同步功能。")
        return

    rows       = []
    fetch_errs = []
    tickers    = list(portfolio.keys())
    with st.spinner(f"正在更新 {len(tickers)} 支股票的即時報價…"):
        for t, pos in portfolio.items():
            buy_price = pos.get("buy_price", 0)
            qty       = pos.get("qty", 0)
            cur_price = 0
            name      = t
            try:
                info      = yf.Ticker(t).fast_info
                cur_price = getattr(info, "last_price", None) or 0
                if not cur_price:
                    full_info = yf.Ticker(t).info
                    cur_price = (full_info.get("regularMarketPrice")
                                 or full_info.get("currentPrice") or 0)
                    name = full_info.get("shortName", t)
            except Exception as _fe:
                fetch_errs.append(f"{t}（{_fe}）")
            cost_value = buy_price * qty
            cur_value  = cur_price * qty
            pnl        = cur_value - cost_value
            ret_pct    = (pnl / cost_value * 100) if cost_value > 0 else 0
            rows.append({
                "ticker": t, "name": name,
                "buy_price": buy_price, "qty": qty,
                "cur_price": cur_price,
                "cost_value": cost_value, "cur_value": cur_value,
                "pnl": pnl, "ret_pct": ret_pct,
                "price_ok": cur_price > 0,
            })
    if fetch_errs:
        st.warning(
            f"⚠️ 以下股票報價取得失敗，盈虧計算可能不準確：{', '.join(fetch_errs)}"
        )

    total_cost = sum(r["cost_value"] for r in rows)
    total_cur  = sum(r["cur_value"]  for r in rows)
    total_pnl  = total_cur - total_cost
    total_ret  = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("持倉股票數",  f"{len(rows)} 支")
    s2.metric("總投入成本",  fmt_usd_hkd(total_cost))
    s3.metric("目前總市值",  fmt_usd_hkd(total_cur))
    pnl_str = (f"+{fmt_usd_hkd(total_pnl)}" if total_pnl >= 0
               else f"-{fmt_usd_hkd(abs(total_pnl))}")
    s4.metric("總盈虧 (P&L)", pnl_str, delta=f"{total_ret:+.2f}%")

    st.markdown("---")
    st.markdown("### 📋 持倉明細")

    _HDR_STYLE = "color:#aaa; font-size:12px; font-weight:600;"
    col_w      = [0.9, 1.5, 1.8, 0.8, 2.2, 2.2, 0.7]
    hcols      = st.columns(col_w)
    for txt, col in zip(
        ["代碼", "持倉量", "平均成本(USD)", "報酬率", "當前盈虧(USD)", "當前盈虧(HKD)", "操作"],
        hcols,
    ):
        col.markdown(f"<span style='{_HDR_STYLE}'>{txt}</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0 6px 0; border-color:#2A2D3E;'>",
                unsafe_allow_html=True)

    for r in rows:
        rp      = r["ret_pct"]
        rate    = _get_rate()
        if rp <= -10:
            row_bg      = ("background:rgba(255,75,75,0.12); border-left:3px solid #FF4B4B;"
                           " border-radius:4px; padding:4px 6px; margin-bottom:4px;")
            alert_label = "⚠️ 虧損超過 10%！"
            ret_col     = "#FF4B4B"
        elif rp >= 20:
            row_bg      = ("background:rgba(0,255,127,0.10); border-left:3px solid #00FF7F;"
                           " border-radius:4px; padding:4px 6px; margin-bottom:4px;")
            alert_label = "🚀 獲利超過 20%，考慮分批獲利！"
            ret_col     = "#00FF7F"
        else:
            row_bg      = ""
            alert_label = ""
            ret_col     = "#FF4B4B" if rp < 0 else "#00FF7F"

        if alert_label:
            st.markdown(
                f"<div style='{row_bg}'>"
                f"<span style='color:{ret_col}; font-size:12px;'>{alert_label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        ret_sym     = f"+{rp:.2f}%" if rp >= 0 else f"{rp:.2f}%"
        pnl_usd     = r["pnl"]
        pnl_hkd     = pnl_usd * rate
        pnl_usd_str = (f"+${pnl_usd:,.2f} USD" if pnl_usd >= 0
                       else f"-${abs(pnl_usd):,.2f} USD")
        pnl_hkd_str = (f"+${pnl_hkd:,.1f} HKD" if pnl_hkd >= 0
                       else f"-${abs(pnl_hkd):,.1f} HKD")

        rcols = st.columns(col_w)
        if rcols[0].button(f"**{r['ticker']}**", key=f"pf_goto_{r['ticker']}",
                           use_container_width=True):
            st.session_state["diag_ticker"]      = r["ticker"]
            st.session_state["diag_stock_info"]  = None
            st.session_state["diag_hist"]        = None
            st.session_state["auto_fetch"]       = True
            st.session_state["nav_page"]         = "🔬 個股診斷 (Micro)"
            st.rerun()

        rcols[1].markdown(
            f"<span style='font-size:13px;'>{r['qty']} 股</span>",
            unsafe_allow_html=True)
        rcols[2].markdown(
            f"<span style='font-size:13px;'>${r['buy_price']:.2f} USD<br>"
            f"<span style='color:#aaa; font-size:13px;'>≈ ${r['buy_price']*rate:.1f} HKD</span></span>",
            unsafe_allow_html=True)
        rcols[3].markdown(
            f"<span style='color:{ret_col}; font-size:14px; font-weight:700;'>{ret_sym}</span>",
            unsafe_allow_html=True)
        rcols[4].markdown(
            f"<span style='color:{ret_col}; font-size:13px;'>{pnl_usd_str}</span>",
            unsafe_allow_html=True)
        rcols[5].markdown(
            f"<span style='color:{ret_col}; font-size:13px;'>{pnl_hkd_str}</span>",
            unsafe_allow_html=True)
        if rcols[6].button("🗑️", key=f"pf_del_{r['ticker']}",
                           help=f"從投資組合移除 {r['ticker']}"):
            del st.session_state["portfolio"][r["ticker"]]
            save_portfolio(st.session_state["portfolio"])
            st.rerun()

        # ── Per-position notes ────────────────────────────────────────────────
        if "portfolio_notes" not in st.session_state:
            st.session_state["portfolio_notes"] = {}
        _note_key  = f"note_ta_{r['ticker']}"
        _prev_note = st.session_state["portfolio_notes"].get(r["ticker"], "")
        with st.expander(f"📝 {r['ticker']} 筆記", expanded=bool(_prev_note)):
            _note_val = st.text_area(
                "買入邏輯 / 停損條件 / 觀察要點",
                value=_prev_note,
                placeholder="例如：Q3 earnings catalyst, stop-loss at $180, target $220...",
                height=80,
                label_visibility="collapsed",
                key=_note_key,
            )
            if _note_val != _prev_note:
                st.session_state["portfolio_notes"][r["ticker"]] = _note_val

        st.markdown("<hr style='margin:2px 0; border-color:#1E2130;'>",
                    unsafe_allow_html=True)

    # ── Pie chart ──────────────────────────────────────────────────────────────
    if len(rows) > 1:
        st.markdown("### 📊 持倉分佈（按市值）")
        fig_pie = px.pie(
            values=[r["cur_value"] for r in rows],
            names=[r["ticker"] for r in rows],
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4,
        )
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            showlegend=True, height=380,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)


# ── Opportunity banner ─────────────────────────────────────────────────────────
def render_opportunity_banner() -> None:
    """Show a green banner for watchlist stocks currently in the ideal buy zone."""
    try:
        watchlist = st.session_state.get("watchlist", [])
        if not watchlist:
            return
        price_lookup = {}
        for s in (st.session_state.get("results") or []):
            price_lookup[s["ticker"]] = s
        cached_diag = st.session_state.get("diag_stock_info")
        if cached_diag and cached_diag.get("ticker"):
            price_lookup[cached_diag["ticker"]] = cached_diag
        portfolio    = st.session_state.get("portfolio", {})
        opportunities = []
        for t in watchlist:
            info   = price_lookup.get(t)
            if not info:
                continue
            price  = info.get("price", 0)
            low52  = info.get("52w_low") or 0
            high52 = info.get("52w_high") or 0
            if not price or not low52 or not high52:
                continue
            range52 = high52 - low52
            if range52 > 0 and (price - low52) / range52 <= 0.35:
                opportunities.append((t, price, t in portfolio))
        if not opportunities:
            return
        parts = "　　".join(
            f"🟢 <b>{t}</b> ${p:.2f}{'  💼' if in_pf else ''}"
            for t, p, in_pf in opportunities
        )
        st.markdown(
            f"""<div style="background:linear-gradient(90deg,#0A2010,#0D2E18);
                        border:2px solid #00FF7F; border-radius:10px;
                        padding:14px 22px; margin-bottom:18px;">
              <span style="font-size:15px; color:#00FF7F; font-weight:700;">
                🚨 機會偵測：{parts}
                目前處於理想買入區間，請關注！
              </span>
            </div>""",
            unsafe_allow_html=True,
        )
    except Exception as e:
        print(_err("render_opportunity_banner", e))


# ── Macro sentiment dashboard ──────────────────────────────────────────────────
def render_macro_sentiment_dashboard(bm_data: dict | None = None) -> None:
    """
    Compact 3-column macro dashboard (SMA status | gauge | sentiment breakdown)
    + full-width VIX fear timeline.
    """
    try:
        with st.container():
            hdr_l, hdr_r = st.columns([2, 1])
            with hdr_l:
                st.markdown("### 🌡️ 宏觀市場情緒溫度計")
            with hdr_r:
                st.markdown(
                    "<div style='text-align:right; color:#666; font-size:13px; "
                    "padding-top:10px;'>StockTwits 60% ＋ 新聞 40%｜每 10 分鐘更新</div>",
                    unsafe_allow_html=True,
                )

            with st.spinner("更新情緒數據…"):
                try:
                    sentiment = get_combined_sentiment("SPY")
                except Exception:
                    sentiment = {"combined": 50.0, "news_score": 50.0,
                                 "social_score": 50.0,
                                 "social_data": {"bull_count": 0, "bear_count": 0, "total": 0}}

            combined     = sentiment["combined"]
            news_score   = sentiment["news_score"]
            social_score = sentiment["social_score"]
            sd           = sentiment["social_data"]
            _check_sentiment_mutation("SPY", social_score)

            label_c, col_c = _score_to_label(combined)
            label_n, col_n = _score_to_label(news_score)
            label_s, col_s = _score_to_label(social_score)

            col_sma, col_gauge, col_meta = st.columns([1.1, 1.6, 1.1])

            with col_sma:
                st.markdown("**📊 大盤趨勢狀態**")
                if bm_data and bm_data.get("golden_cross") is True:
                    sma50v  = bm_data.get("sma50",  0) or 0
                    sma200v = bm_data.get("sma200", 0) or 0
                    perf    = bm_data.get("perf_1y", 0) or 0
                    bm_lbl  = bm_data.get("label", "ETF")
                    bm_tick = bm_data.get("ticker", "")
                    st.markdown(
                        f"<div style='background:#0D2E0D; border:1px solid #00FF7F44; "
                        f"border-radius:7px; padding:10px 12px;'>"
                        f"<div style='color:#00FF7F; font-weight:700; font-size:13px;'>"
                        f"✅ 黃金交叉</div>"
                        f"<div style='color:#aaa; font-size:13px; margin-top:4px;'>"
                        f"{bm_lbl}<br>{bm_tick}</div>"
                        f"<hr style='border-color:#1A3A1A; margin:6px 0;'>"
                        f"<div style='font-size:13px;'>"
                        f"SMA50 &nbsp;<b style='color:#fff;'>{sma50v:.1f}</b><br>"
                        f"SMA200 <b style='color:#fff;'>{sma200v:.1f}</b></div>"
                        f"<div style='color:#00FF7F; font-size:12px; margin-top:4px;'>"
                        f"1Y: +{perf:.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                elif bm_data and bm_data.get("golden_cross") is False:
                    sma50v  = bm_data.get("sma50",  0) or 0
                    sma200v = bm_data.get("sma200", 0) or 0
                    perf    = bm_data.get("perf_1y", 0) or 0
                    bm_lbl  = bm_data.get("label", "ETF")
                    bm_tick = bm_data.get("ticker", "")
                    pc      = "#00FF7F" if perf >= 0 else "#FF4B4B"
                    st.markdown(
                        f"<div style='background:#2E0D0D; border:1px solid #FF4B4B44; "
                        f"border-radius:7px; padding:10px 12px;'>"
                        f"<div style='color:#FF4B4B; font-weight:700; font-size:13px;'>"
                        f"⚠️ 死亡交叉</div>"
                        f"<div style='color:#aaa; font-size:13px; margin-top:4px;'>"
                        f"{bm_lbl}<br>{bm_tick}</div>"
                        f"<hr style='border-color:#3A1A1A; margin:6px 0;'>"
                        f"<div style='font-size:13px;'>"
                        f"SMA50 &nbsp;<b style='color:#fff;'>{sma50v:.1f}</b><br>"
                        f"SMA200 <b style='color:#fff;'>{sma200v:.1f}</b></div>"
                        f"<div style='color:{pc}; font-size:12px; margin-top:4px;'>"
                        f"1Y: {perf:+.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("⏳ 大盤數據載入中…")

            with col_gauge:
                gauge_fig = plot_sentiment_gauge(combined, "S&P 500 情緒")
                gauge_fig.update_layout(height=200, margin=dict(t=40, b=5, l=10, r=10))
                st.plotly_chart(gauge_fig, use_container_width=True, key="gauge_main")

            with col_meta:
                st.markdown("**🎯 情緒細項**")
                st.markdown(
                    f"<div style='background:#13162A; border-radius:7px; padding:10px 12px; font-size:12px;'>"
                    f"<div style='display:flex; justify-content:space-between; padding:3px 0;'>"
                    f"  <span style='color:#aaa;'>📰 新聞（40%）</span>"
                    f"  <span style='color:{col_n}; font-weight:700;'>{news_score:.0f} ({label_n})</span></div>"
                    f"<div style='display:flex; justify-content:space-between; padding:3px 0;'>"
                    f"  <span style='color:#aaa;'>💬 社交（60%）</span>"
                    f"  <span style='color:{col_s}; font-weight:700;'>{social_score:.0f} ({label_s})</span></div>"
                    f"<hr style='border-color:#2A2D3E; margin:5px 0;'>"
                    f"<div style='display:flex; justify-content:space-between; padding:2px 0;'>"
                    f"  <span style='color:#fff; font-weight:600;'>🎯 綜合</span>"
                    f"  <span style='color:{col_c}; font-weight:800; font-size:15px;'>{combined:.0f} ({label_c})</span></div>"
                    f"<hr style='border-color:#2A2D3E; margin:5px 0;'>"
                    f"<div style='color:#555; font-size:13px;'>"
                    f"🟢 看多 {sd.get('bull_count',0)}&nbsp;&nbsp;"
                    f"🔴 看空 {sd.get('bear_count',0)}&nbsp;&nbsp;"
                    f"共 {sd.get('total',0)} 則</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with st.spinner("載入 VIX 走勢…"):
                try:
                    vix_df = get_vix_history()
                except Exception:
                    vix_df = pd.DataFrame()
            st.plotly_chart(
                plot_fear_timeline(vix_df, MACRO_EVENTS),
                use_container_width=True, key="fear_timeline",
            )

        st.markdown("---")
    except Exception as e:
        st.error(f"MODULE_ERROR: [ui_components] | FUNCTION: [render_macro_sentiment_dashboard] | ERROR: {e}")

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import numpy as np
import json
import os

WATCHLIST_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")
PORTFOLIO_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")

ALERT_LOSS_THRESHOLD = -0.10   # -10% triggers red warning row


def load_watchlist():
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []


def save_watchlist(watchlist):
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(watchlist, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_portfolio():
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def save_portfolio(portfolio):
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

st.set_page_config(
    page_title="美股選股儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

SCREENER_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "HD", "CVX", "MRK", "ABBV", "PEP",
    "COST", "AVGO", "LLY", "TMO", "MCD", "ACN", "DHR", "NEE", "TXN",
    "ADBE", "NKE", "PM", "CRM", "QCOM", "AMD", "INTC", "INTU",
    "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "ADI",
    "MA", "AXP", "GS", "BLK", "SPGI", "ICE", "CME",
    "UNP", "UPS", "FDX", "NSC", "CSX",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ISRG", "BSX", "MDT", "SYK",
    "LOW", "TGT", "SBUX", "YUM", "CMG",
    "SO", "DUK", "AEP", "EXC", "D",
    "AMT", "PLD", "CCI", "EQIX", "PSA",
    "LIN", "APD", "ECL", "SHW", "PPG",
    "RTX", "LMT", "GD", "NOC", "BA",
    "ELV", "HUM", "CI", "CVS", "MCK"
]

# ── Sentiment keywords ──────────────────────────────────────────────────────
POSITIVE_WORDS = [
    "beat", "surge", "record", "growth", "profit", "upgrade", "bullish",
    "rally", "strong", "outperform", "revenue", "dividend", "buyback",
    "partnership", "launch", "innovation", "approval", "agreement", "gains",
    "rises", "jumps", "soars", "exceeds", "boosts", "expands", "tops",
    "milestone", "breakthrough", "acquisition", "deal", "wins", "buy",
    "高", "增長", "突破", "創新", "合作", "獲利", "回購"
]
NEGATIVE_WORDS = [
    "miss", "fall", "drop", "decline", "loss", "downgrade", "bearish",
    "sell", "weak", "underperform", "recall", "lawsuit", "fine",
    "investigation", "layoff", "bankruptcy", "cut", "warning", "risk",
    "falls", "drops", "plunges", "tumbles", "slides", "slumps", "misses",
    "lowers", "cuts", "delays", "fears", "concern", "probe", "fraud",
    "下跌", "虧損", "裁員", "調查", "風險", "降級"
]


# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────
def init_session():
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist()
    if "screening" not in st.session_state:
        st.session_state["screening"] = False
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "diag_ticker" not in st.session_state:
        st.session_state["diag_ticker"] = ""
    if "diag_stock_info" not in st.session_state:
        st.session_state["diag_stock_info"] = None
    if "diag_hist" not in st.session_state:
        st.session_state["diag_hist"] = None
    if "diag_period" not in st.session_state:
        st.session_state["diag_period"] = "1y"
    if "auto_fetch" not in st.session_state:
        st.session_state["auto_fetch"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "🏠 選股儀表板"
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = load_portfolio()


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get("marketCap", 0) or 0
        net_margin = info.get("profitMargins", 0) or 0
        pe_ratio = info.get("trailingPE", 0) or 0
        revenue_growth = info.get("revenueGrowth", 0) or 0
        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": market_cap,
            "net_margin": net_margin,
            "pe_ratio": pe_ratio,
            "revenue_growth": revenue_growth,
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)) or 0,
            "52w_high": info.get("fiftyTwoWeekHigh", 0) or 0,
            "52w_low": info.get("fiftyTwoWeekLow", 0) or 0,
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "eps": info.get("trailingEps", 0) or 0,
            "beta": info.get("beta", 0) or 0,
        }
    except Exception:
        return None


def get_historical_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception:
        return None


def get_stock_news(ticker):
    """Fetch recent news from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []
        return news[:5]
    except Exception:
        return []


def compute_technicals(hist):
    """Return SMA50, SMA200, and 20-day low."""
    if hist is None or len(hist) < 20:
        return None, None, None
    sma50  = hist["Close"].rolling(50).mean().iloc[-1]  if len(hist) >= 50  else None
    sma200 = hist["Close"].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
    low20  = hist["Low"].tail(20).min()
    return sma50, sma200, low20


def screen_stocks(stocks_data, min_cap, min_margin, min_growth, max_pe):
    filtered = []
    for stock in stocks_data:
        if stock is None:
            continue
        if (stock["market_cap"] > min_cap * 1_000_000_000
                and stock["net_margin"] > min_margin / 100
                and stock["revenue_growth"] > min_growth / 100
                and 0 < stock["pe_ratio"] < max_pe):
            filtered.append(stock)
    return sorted(filtered, key=lambda x: x["market_cap"], reverse=True)[:15]


def format_market_cap(val):
    if val >= 1_000_000_000_000:
        return f"${val/1_000_000_000_000:.2f}T"
    elif val >= 1_000_000_000:
        return f"${val/1_000_000_000:.2f}B"
    return f"${val/1_000_000:.2f}M"


# ──────────────────────────────────────────────
# Sentiment analysis
# ──────────────────────────────────────────────
def classify_sentiment(text):
    """Keyword-based sentiment: positive / negative / neutral."""
    text_lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


def sentiment_badge(sentiment):
    colors = {
        "positive": ("#00FF7F", "#1A2E1A", "▲ 正面"),
        "negative": ("#FF4B4B", "#2E1A1A", "▼ 負面"),
        "neutral":  ("#FFA500", "#2E2A1A", "● 中性"),
    }
    color, bg, label = colors.get(sentiment, colors["neutral"])
    return f"""<span style="background:{bg}; color:{color}; border:1px solid {color}44;
                border-radius:4px; padding:2px 8px; font-size:11px;
                font-weight:600; white-space:nowrap;">{label}</span>"""


def news_impact_summary(news_items):
    """Generate a macro impact summary from the sentiment distribution."""
    if not news_items:
        return "neutral", "暫無近期新聞數據。"

    sentiments = [item["sentiment"] for item in news_items]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    total = len(sentiments)

    pos_pct = pos / total * 100
    neg_pct = neg / total * 100

    if pos_pct >= 60:
        overall = "bullish"
        icon = "🟢"
        verdict = "整體偏多"
        summary = (
            f"近期 {total} 則新聞中，{pos} 則（{pos_pct:.0f}%）屬正面消息。"
            "市場情緒偏樂觀，基本面支撐明確，短期股價上行動能較強。"
            "建議關注財報、產品發布等催化劑，可考慮逢低分批建倉。"
        )
    elif neg_pct >= 60:
        overall = "bearish"
        icon = "🔴"
        verdict = "整體偏空"
        summary = (
            f"近期 {total} 則新聞中，{neg} 則（{neg_pct:.0f}%）屬負面消息。"
            "市場情緒偏悲觀，存在潛在下行風險。"
            "建議謹慎操作，等待明確轉折信號，嚴格遵守止損紀律。"
        )
    elif pos_pct >= 40:
        overall = "mildly_bullish"
        icon = "🟡"
        verdict = "溫和偏多"
        summary = (
            f"近期 {total} 則新聞中，正負面消息各占一定比例（正面 {pos_pct:.0f}%，負面 {neg_pct:.0f}%）。"
            "整體情緒溫和偏正面，建議結合技術面（買入區間、SMA 支撐）做決策，"
            "分批建倉並嚴格控制倉位大小。"
        )
    else:
        overall = "neutral"
        icon = "⚪"
        verdict = "中性觀望"
        summary = (
            f"近期 {total} 則新聞中，正面 {pos} 則、負面 {neg} 則、中性 {neu} 則。"
            "市場方向不明確，建議等待更清晰的基本面或技術面信號，"
            "暫不積極加倉，持觀望態度。"
        )

    return overall, icon, verdict, summary, pos, neg, neu, total


# ──────────────────────────────────────────────
# Buy-zone & exit-strategy calculations
# ──────────────────────────────────────────────
def calc_buy_zone(price, sma50, sma200):
    """Return (lower, upper, zone_label, pct_position)."""
    if sma50 is None or sma200 is None:
        return None, None, "資料不足", None

    lower = min(sma200, sma50)
    upper = max(sma200, sma50)

    if price > sma50 * 1.05:
        label = "📛 目前股價稍高，建議等待回調"
    elif lower <= price <= upper:
        label = "✅ 目前處於理想分批建倉區間"
    elif price < lower:
        label = "⚠️ 股價低於 SMA 200，趨勢偏弱，謹慎操作"
    else:
        label = "ℹ️ 股價略高，留意風險"

    zone_width = upper - lower
    if zone_width > 0:
        pct = (price - lower) / zone_width * 100
        pct = max(0, min(100, pct))
    else:
        pct = 50.0

    return lower, upper, label, pct


def calc_exit_strategy(price, sma200, low20):
    """
    Stop-loss: SMA200 * 0.97  OR  low20 * 0.98 — choose the higher (tighter).
    Take-profit: risk*2 above entry (1:2 R:R).
    """
    if sma200 is None or low20 is None:
        return None, None, None

    stop1 = sma200 * 0.97
    stop2 = low20  * 0.98
    stop  = max(stop1, stop2)

    risk_pct = (price - stop) / price
    target   = price * (1 + risk_pct * 2)

    return stop, target, risk_pct


# ──────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────
def plot_price_chart(ticker, hist, sma50=None, sma200=None,
                     stop_price=None, target_price=None,
                     buy_lower=None, buy_upper=None,
                     portfolio_buy_price=None, current_price=None):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        name=ticker,
        increasing_line_color="#00D4FF",
        decreasing_line_color="#FF4B4B"
    ))

    if sma50 is not None:
        ma50_series = hist["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=ma50_series,
            name="SMA 50", line=dict(color="#FFA500", width=1.5)
        ))
    if sma200 is not None:
        ma200_series = hist["Close"].rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=ma200_series,
            name="SMA 200", line=dict(color="#FF69B4", width=1.5)
        ))

    if buy_lower is not None and buy_upper is not None:
        fig.add_hrect(
            y0=buy_lower, y1=buy_upper,
            fillcolor="rgba(255,215,0,0.12)",
            line_width=0,
            annotation_text="建議買入區間",
            annotation_position="top left",
            annotation_font_color="#FFD700"
        )

    if stop_price is not None:
        fig.add_hline(
            y=stop_price,
            line=dict(color="#FF4B4B", dash="dash", width=1.8),
            annotation_text=f"  止損 ${stop_price:.2f}",
            annotation_font_color="#FF4B4B",
            annotation_position="right"
        )

    if target_price is not None:
        fig.add_hline(
            y=target_price,
            line=dict(color="#00FF7F", dash="dash", width=1.8),
            annotation_text=f"  目標 ${target_price:.2f}",
            annotation_font_color="#00FF7F",
            annotation_position="right"
        )

    if portfolio_buy_price is not None:
        if current_price is not None and portfolio_buy_price > 0:
            pnl_pct = (current_price - portfolio_buy_price) / portfolio_buy_price * 100
            cost_color = "#00FF7F" if pnl_pct >= 0 else "#FF4B4B"
            pnl_str   = f"+{pnl_pct:.1f}%" if pnl_pct >= 0 else f"{pnl_pct:.1f}%"
            ann_text  = f"  持倉成本 ${portfolio_buy_price:.2f}  ({pnl_str})"
        else:
            cost_color = "#FFD700"
            ann_text   = f"  持倉成本 ${portfolio_buy_price:.2f}"
        fig.add_hline(
            y=portfolio_buy_price,
            line=dict(color=cost_color, dash="dot", width=2.2),
            annotation_text=ann_text,
            annotation_font_color=cost_color,
            annotation_position="left"
        )

    fig.update_layout(
        title=f"{ticker} 股價走勢",
        xaxis_title="日期", yaxis_title="價格 (USD)",
        template="plotly_dark",
        plot_bgcolor="#1A1D2E", paper_bgcolor="#0E1117",
        xaxis_rangeslider_visible=False,
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    return fig


def plot_volume_chart(ticker, hist):
    colors = ["#00D4FF" if c >= o else "#FF4B4B"
              for c, o in zip(hist["Close"], hist["Open"])]
    fig = go.Figure(go.Bar(x=hist.index, y=hist["Volume"],
                           marker_color=colors, name="成交量"))
    fig.update_layout(
        title=f"{ticker} 成交量",
        xaxis_title="日期", yaxis_title="成交量",
        template="plotly_dark",
        plot_bgcolor="#1A1D2E", paper_bgcolor="#0E1117",
        height=250
    )
    return fig


def plot_radar(stock_info):
    pe_score     = max(0, min(100, (30 - stock_info["pe_ratio"]) / 30 * 100)) if stock_info["pe_ratio"] > 0 else 0
    margin_score = min(100, stock_info["net_margin"] * 300)
    growth_score = min(100, stock_info["revenue_growth"] * 200)
    cap_score    = min(100, stock_info["market_cap"] / 2_000_000_000_000 * 100)
    div_score    = min(100, (stock_info["dividend_yield"] or 0) * 2000)

    fig = go.Figure(go.Scatterpolar(
        r=[pe_score, margin_score, growth_score, cap_score, div_score],
        theta=["本益比評分", "淨利率", "營收增長", "市值規模", "股息殖利率"],
        fill="toself",
        line_color="#00D4FF",
        fillcolor="rgba(0,212,255,0.2)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2A2D3E"),
            angularaxis=dict(gridcolor="#2A2D3E"),
            bgcolor="#1A1D2E"
        ),
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        height=350,
        title="基本面雷達圖"
    )
    return fig


# ──────────────────────────────────────────────
# Zone progress bar (custom HTML)
# ──────────────────────────────────────────────
def zone_progress_bar(pct_position, buy_lower, buy_upper, price):
    if pct_position is None:
        return

    bar_color = "#00D4FF" if 0 <= pct_position <= 100 else "#FFA500"
    clamp = max(0, min(100, pct_position))

    bar_html = f"""
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
                  color:#555; font-size:11px; margin-top:4px;">
        <span>強力支撐區</span>
        <span>理想買入區</span>
        <span>偏高區域</span>
      </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Diagnosis checklist
# ──────────────────────────────────────────────
def render_diagnosis(stock_info, params):
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
    cols[0].markdown("**條件**")
    cols[1].markdown("**結果**")
    cols[2].markdown("**實際數值**")
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


# ──────────────────────────────────────────────
# Trade plan card
# ──────────────────────────────────────────────
def render_trade_plan(stock_info, hist):
    price  = stock_info["price"]
    ticker = stock_info["ticker"]

    sma50, sma200, low20 = compute_technicals(hist)
    buy_lower, buy_upper, zone_label, pct = calc_buy_zone(price, sma50, sma200)
    stop, target, risk_pct = calc_exit_strategy(price, sma200, low20) if sma200 else (None, None, None)

    st.markdown("### 📊 建議買入區間 (Buy Zone)")
    st.info(zone_label)

    if buy_lower and buy_upper:
        zone_progress_bar(pct, buy_lower, buy_upper, price)

    st.markdown("### 📋 交易計畫建議")

    c1, c2, c3 = st.columns(3)
    buy_limit = buy_upper if buy_upper else price

    with c1:
        st.markdown(f"""
        <div style="background:#1A2E1A; border:1px solid #00FF7F33;
                    border-radius:10px; padding:16px; text-align:center;">
          <div style="color:#aaa; font-size:12px; margin-bottom:4px;">🟢 建議買入上限</div>
          <div style="color:#00FF7F; font-size:22px; font-weight:700;">${buy_limit:.2f}</div>
          <div style="color:#666; font-size:11px;">(SMA 50)</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if stop:
            st.markdown(f"""
            <div style="background:#2E1A1A; border:1px solid #FF4B4B33;
                        border-radius:10px; padding:16px; text-align:center;">
              <div style="color:#aaa; font-size:12px; margin-bottom:4px;">🔴 建議止損價</div>
              <div style="color:#FF4B4B; font-size:22px; font-weight:700;">${stop:.2f}</div>
              <div style="color:#666; font-size:11px;">(SMA200 × 97% 或 20日低 × 98%)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("資料不足")

    with c3:
        if target:
            st.markdown(f"""
            <div style="background:#1A1A2E; border:1px solid #00D4FF33;
                        border-radius:10px; padding:16px; text-align:center;">
              <div style="color:#aaa; font-size:12px; margin-bottom:4px;">🎯 預計目標價</div>
              <div style="color:#00D4FF; font-size:22px; font-weight:700;">${target:.2f}</div>
              <div style="color:#666; font-size:11px;">(買入點盈虧比 1:2)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("資料不足")

    if stop and risk_pct is not None:
        st.markdown("---")
        loss_pct = risk_pct * 100
        gain_pct = risk_pct * 2 * 100
        col_warn, col_ratio = st.columns([3, 1])
        with col_warn:
            st.error(
                f"⚠️ **風險提示**：若以目前價格 ${price:.2f} 買入，"
                f"跌至止損線 ${stop:.2f} 將虧損約 **{loss_pct:.1f}%**。"
                f"請確保此倉位的虧損在您的整體資金管理範圍內。"
            )
        with col_ratio:
            st.markdown(f"""
            <div style="background:#1A1D2E; border:1px solid #444;
                        border-radius:8px; padding:12px; text-align:center;">
              <div style="color:#aaa; font-size:11px;">風險報酬比</div>
              <div style="color:#FFD700; font-size:20px; font-weight:700;">1 : 2</div>
              <div style="color:#aaa; font-size:11px;">
                損 {loss_pct:.1f}% → 獲 {gain_pct:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

    return sma50, sma200, stop, target, buy_lower, buy_upper


# ──────────────────────────────────────────────
# News Intelligence module
# ──────────────────────────────────────────────
def render_news_intelligence(ticker, company_name):
    st.markdown("---")
    st.markdown("### 📰 新聞與影響分析 (News Intelligence)")

    try:
        with st.spinner(f"正在獲取 {ticker} 最新新聞…"):
            raw_news = get_stock_news(ticker)
    except Exception as e:
        st.warning(f"⚠️ 新聞獲取發生錯誤：{e}")
        return

    if not raw_news:
        st.info("ℹ️ 暫時無法獲取近期新聞，請稍後重試。")
        return

    # Enrich with sentiment + render everything inside one try-except
    try:
        news_items = []
        for item in raw_news:
            content = item.get("content", {})
            title = ""
            pub_date = ""
            url = ""
            publisher = ""

            if isinstance(content, dict):
                title = content.get("title", "") or content.get("headline", "")
                pub_date = content.get("pubDate", "") or content.get("displayTime", "")
                url = ""
                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    publisher = provider.get("displayName", "")
                canonical = content.get("canonicalUrl", {})
                if isinstance(canonical, dict):
                    url = canonical.get("url", "")
            else:
                title = item.get("title", "")
                pub_date = item.get("providerPublishTime", "")
                url = item.get("link", "")
                publisher = item.get("publisher", "")

            if not title:
                continue

            sentiment = classify_sentiment(title)
            news_items.append({
                "title": title,
                "publisher": publisher,
                "pub_date": pub_date,
                "url": url,
                "sentiment": sentiment,
            })

        if not news_items:
            st.info("ℹ️ 暫時無法解析新聞內容，請稍後重試。")
            return

        # Impact summary
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

        # Sentiment distribution bar
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
          <p style="color:#ccc; font-size:14px; line-height:1.7; margin:0 0 16px 0;">
            {summary}
          </p>
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
        </div>
        """, unsafe_allow_html=True)

        # News list
        st.markdown("#### 📋 近期新聞清單")
        for item in news_items:
            badge_html = sentiment_badge(item["sentiment"])
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
                <div style="color:#666; font-size:11px; margin-top:3px;">{pub_info}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"⚠️ 新聞資料解析發生錯誤，部分數據可能無法顯示。({e})")


# ──────────────────────────────────────────────
# Portfolio dashboard renderer
# ──────────────────────────────────────────────
def render_portfolio_dashboard():
    st.title("💼 我的投資組合")
    portfolio = st.session_state.get("portfolio", {})

    if not portfolio:
        st.info("📭 投資組合尚為空。請前往『個股診斷』頁面，輸入買入價格與持倉數量後加入。")
        return

    # ── Fetch current prices for all holdings ────────────────────────────────
    rows = []
    for t, pos in portfolio.items():
        buy_price = pos.get("buy_price", 0)
        qty       = pos.get("qty", 0)
        try:
            info = yf.Ticker(t).info
            cur_price = info.get("regularMarketPrice") or info.get("currentPrice") or 0
            name      = info.get("shortName", t)
        except Exception:
            cur_price = 0
            name      = t
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
        })

    total_cost = sum(r["cost_value"] for r in rows)
    total_cur  = sum(r["cur_value"]  for r in rows)
    total_pnl  = total_cur - total_cost
    total_ret  = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    # ── Summary cards ─────────────────────────────────────────────────────────
    pnl_color = "#00FF7F" if total_pnl >= 0 else "#FF4B4B"
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("持倉股票數",   f"{len(rows)} 支")
    s2.metric("總投入成本",   f"${total_cost:,.0f}")
    s3.metric("目前總市值",   f"${total_cur:,.0f}")
    pnl_str = f"+${total_pnl:,.0f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.0f}"
    s4.metric("總盈虧 (P&L)", pnl_str,
              delta=f"{total_ret:+.2f}%")

    st.markdown("---")

    # ── Holdings table ────────────────────────────────────────────────────────
    st.markdown("### 📋 持倉明細")
    hcols = st.columns([1, 2.5, 1.2, 1, 1.2, 1.2, 1.4, 1.4, 0.8])
    for txt, col in zip(
        ["代碼", "公司", "買入均價", "數量", "目前股價", "持倉市值", "盈虧金額", "報酬率", "操作"],
        hcols
    ):
        col.markdown(f"<span style='color:#aaa; font-size:12px;'>{txt}</span>",
                     unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0 6px 0; border-color:#2A2D3E;'>",
                unsafe_allow_html=True)

    for r in rows:
        is_alert = r["ret_pct"] <= ALERT_LOSS_THRESHOLD * 100
        row_bg   = "background:#2E1A1A; border-radius:6px; padding:2px 4px;" if is_alert else ""
        ret_col  = "#FF4B4B" if r["ret_pct"] < 0 else "#00FF7F"
        ret_sym  = f"+{r['ret_pct']:.2f}%" if r["ret_pct"] >= 0 else f"{r['ret_pct']:.2f}%"
        pnl_sym  = f"+${r['pnl']:,.2f}"   if r["pnl"]     >= 0 else f"-${abs(r['pnl']):,.2f}"

        if is_alert:
            st.markdown(
                f"<div style='{row_bg}'>"
                f"<span style='color:#FF4B4B; font-size:12px;'>⚠️ 虧損超過 {abs(ALERT_LOSS_THRESHOLD)*100:.0f}%，請注意風險！</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        rcols = st.columns([1, 2.5, 1.2, 1, 1.2, 1.2, 1.4, 1.4, 0.8])

        if rcols[0].button(f"**{r['ticker']}**", key=f"pf_goto_{r['ticker']}",
                           use_container_width=True):
            # Navigate to diagnosis for this stock
            st.session_state["diag_ticker"]     = r["ticker"]
            st.session_state["diag_stock_info"] = None
            st.session_state["diag_hist"]       = None
            st.session_state["auto_fetch"]      = True
            st.session_state["page"]            = "🔍 個股診斷 & 交易計畫"
            st.rerun()

        rcols[1].markdown(
            f"<span style='font-size:12px;'>{r['name']}</span>",
            unsafe_allow_html=True)
        rcols[2].markdown(f"<span style='font-size:13px;'>${r['buy_price']:.2f}</span>",
                          unsafe_allow_html=True)
        rcols[3].markdown(f"<span style='font-size:13px;'>{r['qty']}</span>",
                          unsafe_allow_html=True)
        rcols[4].markdown(
            f"<span style='color:#00D4FF; font-size:13px;'>${r['cur_price']:.2f}</span>",
            unsafe_allow_html=True)
        rcols[5].markdown(
            f"<span style='font-size:13px;'>${r['cur_value']:,.0f}</span>",
            unsafe_allow_html=True)
        rcols[6].markdown(
            f"<span style='color:{ret_col}; font-size:13px;'>{pnl_sym}</span>",
            unsafe_allow_html=True)
        rcols[7].markdown(
            f"<span style='color:{ret_col}; font-size:14px; font-weight:700;'>{ret_sym}</span>",
            unsafe_allow_html=True)

        if rcols[8].button("🗑️", key=f"pf_del_{r['ticker']}",
                           help=f"從投資組合移除 {r['ticker']}"):
            del st.session_state["portfolio"][r["ticker"]]
            save_portfolio(st.session_state["portfolio"])
            st.rerun()

        st.markdown("<hr style='margin:2px 0; border-color:#1E2130;'>",
                    unsafe_allow_html=True)

    # ── Portfolio pie chart ───────────────────────────────────────────────────
    if len(rows) > 1:
        st.markdown("### 📊 持倉分佈")
        fig_pie = px.pie(
            values=[r["cur_value"] for r in rows],
            names=[r["ticker"] for r in rows],
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4,
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            showlegend=True,
            height=380,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)


# ──────────────────────────────────────────────
# Opportunity banner (watchlist buy-zone alert)
# ──────────────────────────────────────────────
def render_opportunity_banner():
    """
    Show a prominent green banner for watchlist stocks currently in the
    ideal buy zone.  Uses cached screener results + diag_stock_info to
    avoid extra network calls.
    """
    watchlist = st.session_state.get("watchlist", [])
    if not watchlist:
        return

    # Build price lookup from cached data
    price_lookup = {}
    for s in (st.session_state.get("results") or []):
        price_lookup[s["ticker"]] = s

    cached_diag = st.session_state.get("diag_stock_info")
    if cached_diag and cached_diag.get("ticker"):
        price_lookup[cached_diag["ticker"]] = cached_diag

    portfolio = st.session_state.get("portfolio", {})

    opportunities = []
    for t in watchlist:
        info = price_lookup.get(t)
        if not info:
            continue
        price  = info.get("price", 0)
        low52  = info.get("52w_low") or 0
        high52 = info.get("52w_high") or 0
        if not price or not low52 or not high52:
            continue
        # Proxy: lower 35% of 52-week range ≈ near SMA200 support zone
        range52 = high52 - low52
        if range52 > 0 and (price - low52) / range52 <= 0.35:
            in_pf = t in portfolio
            opportunities.append((t, price, in_pf))

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


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def navigate_to_diagnosis(stock_info):
    """Jump to diagnosis page and flag auto-fetch for hist data."""
    st.session_state["diag_ticker"]     = stock_info["ticker"]
    st.session_state["diag_stock_info"] = stock_info
    st.session_state["diag_hist"]       = None
    st.session_state["auto_fetch"]      = True
    st.session_state["page"]            = "🔍 個股診斷 & 交易計畫"
    st.rerun()


def main():
    init_session()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("📈 美股選股")
    st.sidebar.markdown("---")

    # key="page" so radio reads/writes st.session_state["page"] directly,
    # allowing navigate_to_diagnosis() to switch pages via session state.
    page = st.sidebar.radio(
        "功能選擇",
        ["🏠 選股儀表板", "🔍 個股診斷 & 交易計畫", "💼 我的持倉"],
        key="page"
    )

    # Portfolio summary badge in sidebar
    portfolio = st.session_state.get("portfolio", {})
    if portfolio:
        st.sidebar.markdown(
            f"<div style='background:#1A2E1A; border:1px solid #00FF7F44; "
            f"border-radius:6px; padding:6px 10px; margin-top:4px; font-size:12px; color:#00FF7F;'>"
            f"💼 持倉 {len(portfolio)} 支股票</div>",
            unsafe_allow_html=True
        )

    # ── Collapsible screener parameters ──────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ 篩選設定", expanded=False):
        min_cap = st.number_input(
            "市值下限 (十億美元 $B)", min_value=1, max_value=500,
            value=10, step=1,
            help="篩選市值大於此數值（單位：十億美元）的股票"
        )
        min_margin = st.number_input(
            "最低淨利率 (%)", min_value=1, max_value=50,
            value=10, step=1,
            help="篩選淨利率高於此百分比的股票"
        )
        min_growth = st.number_input(
            "最低營收增長 (%)", min_value=1, max_value=100,
            value=15, step=1,
            help="篩選年營收增長率高於此百分比的股票"
        )
        max_pe = st.number_input(
            "P/E Ratio 上限", min_value=5, max_value=200,
            value=30, step=1,
            help="篩選本益比低於此數值的股票"
        )
    params = (min_cap, min_margin, min_growth, max_pe)

    # ── Watchlist sidebar section ───────────────────────────────────────────
    st.sidebar.markdown("---")
    watchlist = st.session_state.get("watchlist", [])
    st.sidebar.markdown("### ⭐ 我的收藏")

    if watchlist:
        for wl_ticker in watchlist:
            col_wl, col_rm = st.sidebar.columns([3, 1])
            # Look up cached info for watchlist navigation
            cached_results = st.session_state.get("results") or []
            wl_info = next((s for s in cached_results if s["ticker"] == wl_ticker), None)
            if col_wl.button(f"📌 {wl_ticker}", key=f"wl_goto_{wl_ticker}",
                             use_container_width=True):
                if wl_info:
                    navigate_to_diagnosis(wl_info)
                else:
                    # No cached info — navigate with auto full-fetch
                    st.session_state["diag_ticker"]     = wl_ticker
                    st.session_state["diag_stock_info"] = None
                    st.session_state["diag_hist"]       = None
                    st.session_state["auto_fetch"]      = True
                    st.session_state["page"]            = "🔍 個股診斷 & 交易計畫"
                    st.rerun()
            if col_rm.button("✕", key=f"wl_rm_{wl_ticker}"):
                st.session_state["watchlist"].remove(wl_ticker)
                save_watchlist(st.session_state["watchlist"])
                st.rerun()
    else:
        st.sidebar.caption("尚未收藏任何股票\n在個股診斷頁面點擊「添加到收藏夾」")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── Opportunity detection banner (shown on all pages) ────────────────────
    render_opportunity_banner()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Screener Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    if page == "🏠 選股儀表板":
        st.title("🏠 美股選股儀表板")
        st.markdown("根據您自定義的基本面條件，自動篩選符合條件的優質美股")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("市值門檻", f"> ${min_cap}B")
        col2.metric("淨利率",   f"> {min_margin}%")
        col3.metric("營收增長", f"> {min_growth}%")
        col4.metric("P/E Ratio", f"< {max_pe}")

        st.markdown("---")

        if st.button("🔄 開始篩選（需要約1-2分鐘）",
                     type="primary", use_container_width=True):
            st.session_state["screening"] = True
            st.session_state["results"]   = None
            st.session_state["screen_params"] = params

        if st.session_state.get("screening") and not st.session_state.get("results"):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            stocks_data  = []
            total = len(SCREENER_STOCKS)

            for i, ticker in enumerate(SCREENER_STOCKS):
                status_text.text(f"正在分析 {ticker}… ({i+1}/{total})")
                progress_bar.progress((i + 1) / total)
                data = get_stock_info(ticker)
                if data:
                    stocks_data.append(data)
                time.sleep(0.05)

            used_params = st.session_state.get("screen_params", params)
            results = screen_stocks(stocks_data, *used_params)
            st.session_state["results"]   = results
            st.session_state["screening"] = False
            progress_bar.empty()
            status_text.empty()

        if st.session_state.get("results"):
            results = st.session_state["results"]
            st.success(f"✅ 篩選完成！找到 **{len(results)}** 支符合條件的股票，點擊代碼即可查看完整診斷")

            # ── Interactive results table ─────────────────────────────────────
            # Header row
            hcols = st.columns([1, 2.5, 1.5, 1.2, 1, 1, 1, 0.8, 0.8])
            for txt, col in zip(
                ["代碼", "公司名稱", "產業", "市值", "股價", "淨利率", "營收增長", "P/E", "⭐"],
                hcols
            ):
                col.markdown(f"<span style='color:#aaa; font-size:12px;'>{txt}</span>",
                             unsafe_allow_html=True)
            st.markdown("<hr style='margin:4px 0 8px 0; border-color:#2A2D3E;'>",
                        unsafe_allow_html=True)

            for stock in results:
                t = stock["ticker"]
                in_wl = t in st.session_state["watchlist"]
                rcols = st.columns([1, 2.5, 1.5, 1.2, 1, 1, 1, 0.8, 0.8])

                # Clickable ticker button → navigate to diagnosis
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

                # Watchlist toggle
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

            # Charts
            df = pd.DataFrame(results)
            st.markdown("### 📊 篩選結果圖表")
            c1, c2 = st.columns(2)
            with c1:
                fig_cap = px.bar(
                    df.head(10), x="ticker", y="market_cap",
                    title="市值比較",
                    color="market_cap", color_continuous_scale="Blues",
                    labels={"market_cap": "市值 (USD)", "ticker": "股票代碼"}
                )
                fig_cap.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117", plot_bgcolor="#1A1D2E",
                    showlegend=False
                )
                fig_cap.update_yaxes(tickformat=".2s")
                st.plotly_chart(fig_cap, use_container_width=True)

            with c2:
                fig_scatter = px.scatter(
                    df, x="pe_ratio", y="net_margin",
                    size="market_cap", color="revenue_growth",
                    hover_data=["ticker", "name"],
                    title="P/E vs 淨利率（氣泡=市值）",
                    labels={
                        "pe_ratio": "P/E Ratio",
                        "net_margin": "淨利率",
                        "revenue_growth": "營收增長"
                    },
                    color_continuous_scale="Teal"
                )
                fig_scatter.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117", plot_bgcolor="#1A1D2E"
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
    # PAGE 2 — Stock Diagnosis & Trade Plan (linear, no tabs)
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔍 個股診斷 & 交易計畫":
        st.title("🔍 個股診斷 & 交易計畫")

        col_input, col_period, col_btn = st.columns([2, 1, 1])
        with col_input:
            ticker_input = st.text_input(
                "輸入股票代碼",
                placeholder="例：AAPL  MSFT  GOOGL",
                value=st.session_state.get("diag_ticker", "")
            ).upper().strip()
        with col_period:
            period = st.selectbox("歷史數據", ["1y", "6mo", "2y", "5y"], index=0)
        with col_btn:
            analyze_btn = st.button("🩺 診斷分析", type="primary",
                                    use_container_width=True)

        if ticker_input:
            st.session_state["diag_ticker"] = ticker_input

        # ── Manual button fetch ───────────────────────────────────────────────
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

        # ── Auto-fetch triggered by clicking ticker in screener ───────────────
        if st.session_state.get("auto_fetch"):
            auto_ticker = st.session_state.get("diag_ticker", "")
            with st.spinner(f"正在載入 {auto_ticker} 的完整數據…"):
                if not st.session_state.get("diag_stock_info"):
                    fetched_info = get_stock_info(auto_ticker)
                    if fetched_info and fetched_info.get("price"):
                        st.session_state["diag_stock_info"] = fetched_info
                fetched_hist = get_historical_data(auto_ticker, period)
                st.session_state["diag_hist"] = fetched_hist
            st.session_state["auto_fetch"] = False

        # ── Load from cache ───────────────────────────────────────────────────
        stock_info = st.session_state.get("diag_stock_info")
        hist       = st.session_state.get("diag_hist")

        if stock_info and stock_info["ticker"] != ticker_input and ticker_input:
            st.info("👆 代碼已更改，請按「診斷分析」重新獲取數據。")
            stock_info = None

        if not stock_info:
            st.info("👆 輸入股票代碼後按「診斷分析」，或在篩選結果中點擊代碼直接跳轉。")

        # ══════════════════════════════════════════════════════════════════════
        # LINEAR DIAGNOSIS LAYOUT — all sections on one page
        # ══════════════════════════════════════════════════════════════════════
        if stock_info and stock_info.get("price"):
            price  = stock_info["price"]
            ticker = stock_info["ticker"]

            # ── Header + Watchlist button ────────────────────────────────────
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

            # ── Key metrics row ──────────────────────────────────────────────
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("股價",     f"${price:.2f}")
            m2.metric("市值",     format_market_cap(stock_info["market_cap"]))
            m3.metric("P/E",      f"{stock_info['pe_ratio']:.1f}" if stock_info["pe_ratio"] else "N/A")
            m4.metric("淨利率",   f"{stock_info['net_margin']*100:.1f}%" if stock_info["net_margin"] else "N/A")
            m5.metric("營收增長", f"{stock_info['revenue_growth']*100:.1f}%" if stock_info["revenue_growth"] else "N/A")
            m6.metric("EPS",      f"${stock_info['eps']:.2f}" if stock_info["eps"] else "N/A")
            m7.metric("Beta",     f"{stock_info['beta']:.2f}" if stock_info["beta"] else "N/A")

            st.markdown("---")

            # ── Section 1: Diagnosis + Radar ─────────────────────────────────
            col_diag, col_radar = st.columns([1, 1])
            with col_diag:
                render_diagnosis(stock_info, params)
            with col_radar:
                st.plotly_chart(plot_radar(stock_info), use_container_width=True)

            # ── Section 2: Buy Zone + Trade Plan ─────────────────────────────
            st.markdown("---")
            if hist is not None and not hist.empty:
                render_trade_plan(stock_info, hist)
            else:
                st.warning("⚠️ 無法獲取歷史數據，無法計算技術指標。")

            # ── Section 3: K-line chart (collapsible) ────────────────────────
            if hist is not None and not hist.empty:
                with st.expander("📈 查看 K 線圖表與成交量", expanded=False):
                    sma50_v, sma200_v, low20_v = compute_technicals(hist)
                    stop_v, target_v, _ = calc_exit_strategy(price, sma200_v, low20_v) \
                        if sma200_v else (None, None, None)
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
                            current_price=price
                        ),
                        use_container_width=True
                    )
                    st.plotly_chart(plot_volume_chart(ticker, hist),
                                    use_container_width=True)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("期間最高", f"${hist['High'].max():.2f}")
                    s2.metric("期間最低", f"${hist['Low'].min():.2f}")
                    ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                    s3.metric("期間報酬率", f"{ret:.1f}%", delta=f"{ret:.1f}%")
                    s4.metric("平均成交量", f"{hist['Volume'].mean():,.0f}")

            # ── Section 4: Portfolio Entry ────────────────────────────────────
            st.markdown("---")
            st.markdown("### ➕ 加入投資組合")
            pf_existing = st.session_state["portfolio"].get(ticker)
            with st.expander(
                f"{'✏️ 修改持倉' if pf_existing else '📥 新增到投資組合'} — {ticker}",
                expanded=bool(pf_existing)
            ):
                pf_default_price = float(pf_existing["buy_price"]) if pf_existing else float(price)
                pf_default_qty   = int(pf_existing["qty"])         if pf_existing else 1
                pf_c1, pf_c2, pf_c3 = st.columns([1.5, 1.5, 1])
                with pf_c1:
                    pf_buy_price = st.number_input(
                        "買入均價 (USD)", min_value=0.01, max_value=99999.0,
                        value=pf_default_price, step=0.5,
                        format="%.2f", key=f"pf_price_{ticker}"
                    )
                with pf_c2:
                    pf_qty = st.number_input(
                        "持倉數量 (股)", min_value=1, max_value=1_000_000,
                        value=pf_default_qty, step=1,
                        key=f"pf_qty_{ticker}"
                    )
                with pf_c3:
                    cost_display = pf_buy_price * pf_qty
                    st.metric("持倉成本", f"${cost_display:,.2f}")

                col_add, col_del = st.columns([2, 1])
                if col_add.button(
                    f"{'💾 更新持倉' if pf_existing else '✅ 加入投資組合'}",
                    type="primary", use_container_width=True,
                    key=f"pf_add_{ticker}"
                ):
                    st.session_state["portfolio"][ticker] = {
                        "buy_price": pf_buy_price,
                        "qty": int(pf_qty),
                    }
                    save_portfolio(st.session_state["portfolio"])
                    st.success(f"✅ {ticker} 已加入投資組合！買入均價 ${pf_buy_price:.2f}，{int(pf_qty)} 股")
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
                        unsafe_allow_html=True
                    )

            # ── Section 5: News Intelligence ──────────────────────────────────
            st.markdown("---")
            render_news_intelligence(ticker, stock_info["name"])

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Portfolio Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "💼 我的持倉":
        with st.spinner("正在更新持倉市值…"):
            render_portfolio_dashboard()


if __name__ == "__main__":
    main()

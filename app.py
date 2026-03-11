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

WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")


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
                     buy_lower=None, buy_upper=None):
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

    with st.spinner("正在獲取最新新聞資訊…"):
        raw_news = get_stock_news(ticker)

    if not raw_news:
        st.info("ℹ️ 暫時無法獲取近期新聞，請稍後重試。")
        return

    # Enrich with sentiment
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
        "bullish":       ("#00FF7F", "#1A2E1A"),
        "mildly_bullish": ("#FFD700", "#2E2A1A"),
        "neutral":       ("#aaa",    "#1A1D2E"),
        "bearish":       ("#FF4B4B", "#2E1A1A"),
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
            title_display = f'<a href="{item["url"]}" target="_blank" style="color:#00D4FF; text-decoration:none;">{item["title"]}</a>'

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


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    init_session()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("📈 美股選股")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "功能選擇",
        ["🏠 選股儀表板", "🔍 個股診斷 & 交易計畫"]
    )

    # ── Dynamic screener parameters ─────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 自定義篩選條件")

    min_cap = st.sidebar.number_input(
        "市值下限 (十億美元 $B)", min_value=1, max_value=500,
        value=10, step=1,
        help="篩選市值大於此數值（單位：十億美元）的股票"
    )
    min_margin = st.sidebar.number_input(
        "最低淨利率 (%)", min_value=1, max_value=50,
        value=10, step=1,
        help="篩選淨利率高於此百分比的股票"
    )
    min_growth = st.sidebar.number_input(
        "最低營收增長 (%)", min_value=1, max_value=100,
        value=15, step=1,
        help="篩選年營收增長率高於此百分比的股票"
    )
    max_pe = st.sidebar.number_input(
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
            if col_wl.button(f"📌 {wl_ticker}", key=f"wl_goto_{wl_ticker}",
                             use_container_width=True):
                st.session_state["diag_ticker"] = wl_ticker
                st.session_state["page_jump"] = "🔍 個股診斷 & 交易計畫"
                st.rerun()
            if col_rm.button("✕", key=f"wl_rm_{wl_ticker}"):
                st.session_state["watchlist"].remove(wl_ticker)
                save_watchlist(st.session_state["watchlist"])
                st.rerun()
    else:
        st.sidebar.caption("尚未收藏任何股票\n在個股診斷頁面點擊「添加到收藏夾」")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Handle watchlist page jump
    if st.session_state.get("page_jump"):
        page = st.session_state.pop("page_jump")

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
            st.success(f"✅ 篩選完成！找到 **{len(results)}** 支符合條件的股票")

            df = pd.DataFrame(results)
            display_df = pd.DataFrame({
                "股票代碼":   df["ticker"],
                "公司名稱":   df["name"],
                "產業":       df["sector"],
                "市值":       df["market_cap"].apply(format_market_cap),
                "股價 (USD)": df["price"].apply(lambda x: f"${x:.2f}" if x else "N/A"),
                "淨利率":     df["net_margin"].apply(lambda x: f"{x*100:.1f}%"),
                "營收增長":   df["revenue_growth"].apply(lambda x: f"{x*100:.1f}%"),
                "P/E Ratio":  df["pe_ratio"].apply(lambda x: f"{x:.1f}"),
                "EPS":        df["eps"].apply(lambda x: f"${x:.2f}" if x else "N/A"),
                "Beta":       df["beta"].apply(lambda x: f"{x:.2f}" if x else "N/A"),
            })
            st.dataframe(display_df, use_container_width=True,
                         hide_index=True, height=420)

            # Quick-add to watchlist from screener results
            st.markdown("#### ⭐ 快速添加篩選結果至收藏夾")
            ticker_list = df["ticker"].tolist()
            selected_wl = []
            cols_per_row = 5
            rows = [ticker_list[i:i+cols_per_row]
                    for i in range(0, len(ticker_list), cols_per_row)]
            for row in rows:
                row_cols = st.columns(cols_per_row)
                for j, t in enumerate(row):
                    already = t in st.session_state["watchlist"]
                    label = f"✅ {t}" if already else f"+ {t}"
                    if row_cols[j].button(label, key=f"add_wl_{t}"):
                        if not already:
                            st.session_state["watchlist"].append(t)
                            save_watchlist(st.session_state["watchlist"])
                            st.rerun()

            # Charts
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
    # PAGE 2 — Stock Diagnosis & Trade Plan
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

        if ticker_input and analyze_btn:
            with st.spinner(f"正在獲取 {ticker_input} 的數據…"):
                stock_info = get_stock_info(ticker_input)
                hist       = get_historical_data(ticker_input, period)

            if not stock_info or not stock_info.get("price"):
                st.error(f"❌ 無法找到股票代碼「{ticker_input}」，請確認後重試。")
                return

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
                        st.success(f"✅ {ticker} 已添加到收藏夾！")
                        st.rerun()

            # ── Tab layout ───────────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs(
                ["🩺 個股診斷", "📊 交易計畫", "📈 K線圖表", "📰 新聞分析"]
            )

            with tab1:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("股價",     f"${price:.2f}")
                m2.metric("市值",     format_market_cap(stock_info["market_cap"]))
                m3.metric("P/E",      f"{stock_info['pe_ratio']:.1f}" if stock_info["pe_ratio"] else "N/A")
                m4.metric("淨利率",   f"{stock_info['net_margin']*100:.1f}%" if stock_info["net_margin"] else "N/A")
                m5.metric("營收增長", f"{stock_info['revenue_growth']*100:.1f}%" if stock_info["revenue_growth"] else "N/A")

                st.markdown("")
                col_diag, col_radar = st.columns([1, 1])
                with col_diag:
                    render_diagnosis(stock_info, params)
                with col_radar:
                    st.plotly_chart(plot_radar(stock_info), use_container_width=True)

                st.markdown("### 📌 其他指標")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("EPS",       f"${stock_info['eps']:.2f}" if stock_info["eps"] else "N/A")
                e2.metric("Beta",      f"{stock_info['beta']:.2f}"  if stock_info["beta"] else "N/A")
                e3.metric("52週高點", f"${stock_info['52w_high']:.2f}" if stock_info["52w_high"] else "N/A")
                e4.metric("52週低點", f"${stock_info['52w_low']:.2f}"  if stock_info["52w_low"] else "N/A")

            with tab2:
                if hist is not None and not hist.empty:
                    render_trade_plan(stock_info, hist)
                else:
                    st.warning("⚠️ 無法獲取歷史數據，無法計算技術指標")

            with tab3:
                if hist is not None and not hist.empty:
                    sma50_v, sma200_v, low20_v = compute_technicals(hist)
                    stop_v, target_v, _ = calc_exit_strategy(price, sma200_v, low20_v) \
                        if sma200_v else (None, None, None)
                    buy_lower_v, buy_upper_v, _, _ = calc_buy_zone(price, sma50_v, sma200_v)

                    st.plotly_chart(
                        plot_price_chart(
                            ticker, hist,
                            sma50=sma50_v, sma200=sma200_v,
                            stop_price=stop_v, target_price=target_v,
                            buy_lower=buy_lower_v, buy_upper=buy_upper_v
                        ),
                        use_container_width=True
                    )
                    st.plotly_chart(plot_volume_chart(ticker, hist),
                                    use_container_width=True)

                    st.markdown("### 📈 期間統計")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("最高價", f"${hist['High'].max():.2f}")
                    s2.metric("最低價", f"${hist['Low'].min():.2f}")
                    ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                    s3.metric("期間報酬率", f"{ret:.1f}%", delta=f"{ret:.1f}%")
                    s4.metric("平均成交量", f"{hist['Volume'].mean():,.0f}")
                else:
                    st.warning("⚠️ 無法獲取歷史數據")

            with tab4:
                render_news_intelligence(ticker, stock_info["name"])


if __name__ == "__main__":
    main()

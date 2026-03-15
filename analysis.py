# ═══════════════════════════════════════════════════════════════════════════════
# analysis.py  —  Pure calculation logic (no Streamlit, no network calls)
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go

_MODULE = "analysis"

def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── Sentiment keyword lists ────────────────────────────────────────────────────
POSITIVE_WORDS = [
    "beat", "surge", "record", "growth", "profit", "upgrade", "bullish",
    "rally", "strong", "outperform", "revenue", "dividend", "buyback",
    "partnership", "launch", "innovation", "approval", "agreement", "gains",
    "rises", "jumps", "soars", "exceeds", "boosts", "expands", "tops",
    "milestone", "breakthrough", "acquisition", "deal", "wins", "buy",
    "高", "增長", "突破", "創新", "合作", "獲利", "回購",
]
NEGATIVE_WORDS = [
    "miss", "fall", "drop", "decline", "loss", "downgrade", "bearish",
    "sell", "weak", "underperform", "recall", "lawsuit", "fine",
    "investigation", "layoff", "bankruptcy", "cut", "warning", "risk",
    "falls", "drops", "plunges", "tumbles", "slides", "slumps", "misses",
    "lowers", "cuts", "delays", "fears", "concern", "probe", "fraud",
    "下跌", "虧損", "裁員", "調查", "風險", "降級",
]


# ── Screener & technical calculations ─────────────────────────────────────────
def compute_technicals(hist: pd.DataFrame):
    """Return (sma50, sma200, low20). All may be None on insufficient data."""
    try:
        if hist is None or hist.empty or len(hist) < 20:
            return None, None, None
        close = hist["Close"].dropna()
        low   = hist["Low"].dropna()
        if len(close) < 20:
            return None, None, None
        sma50_raw  = close.rolling(50).mean().iloc[-1]  if len(close) >= 50  else None
        sma200_raw = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        sma50  = float(sma50_raw)  if sma50_raw  is not None and not pd.isna(sma50_raw)  else None
        sma200 = float(sma200_raw) if sma200_raw is not None and not pd.isna(sma200_raw) else None
        low20_raw = low.tail(20).min()
        low20 = float(low20_raw) if not pd.isna(low20_raw) else None
        return sma50, sma200, low20
    except Exception as e:
        print(_err("compute_technicals", e))
        return None, None, None


def screen_stocks(stocks_data: list, min_cap, min_margin, min_growth, max_pe) -> list:
    """Filter stocks list against fundamental thresholds; return top 15 by market cap."""
    try:
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
    except Exception as e:
        print(_err("screen_stocks", e))
        return []


def format_market_cap(val: float) -> str:
    try:
        if val >= 1_000_000_000_000:
            return f"${val/1_000_000_000_000:.2f}T"
        if val >= 1_000_000_000:
            return f"${val/1_000_000_000:.2f}B"
        return f"${val/1_000_000:.2f}M"
    except Exception:
        return "N/A"


# ── Sentiment helpers ──────────────────────────────────────────────────────────
def classify_sentiment(text: str) -> str:
    """Keyword-based sentiment: positive / negative / neutral."""
    try:
        text_lower = text.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"
    except Exception as e:
        print(_err("classify_sentiment", e))
        return "neutral"


def sentiment_badge(sentiment: str) -> str:
    """Return an HTML badge string for a sentiment label."""
    colors = {
        "positive": ("#00FF7F", "#1A2E1A", "▲ 正面"),
        "negative": ("#FF4B4B", "#2E1A1A", "▼ 負面"),
        "neutral":  ("#FFA500", "#2E2A1A", "● 中性"),
    }
    color, bg, label = colors.get(sentiment, colors["neutral"])
    return (f'<span style="background:{bg}; color:{color}; border:1px solid {color}44;'
            f' border-radius:4px; padding:2px 8px; font-size:13px;'
            f' font-weight:600; white-space:nowrap;">{label}</span>')


def news_impact_summary(news_items: list) -> tuple:
    """Generate a macro impact summary from the sentiment distribution."""
    try:
        if not news_items:
            return "neutral", "暫無近期新聞數據。"

        sentiments = [item["sentiment"] for item in news_items]
        pos   = sentiments.count("positive")
        neg   = sentiments.count("negative")
        neu   = sentiments.count("neutral")
        total = len(sentiments)
        pos_pct = pos / total * 100
        neg_pct = neg / total * 100

        if pos_pct >= 60:
            return ("bullish", "🟢", "整體偏多",
                    f"近期 {total} 則新聞中，{pos} 則（{pos_pct:.0f}%）屬正面消息。"
                    "市場情緒偏樂觀，基本面支撐明確，短期股價上行動能較強。"
                    "建議關注財報、產品發布等催化劑，可考慮逢低分批建倉。",
                    pos, neg, neu, total)
        if neg_pct >= 60:
            return ("bearish", "🔴", "整體偏空",
                    f"近期 {total} 則新聞中，{neg} 則（{neg_pct:.0f}%）屬負面消息。"
                    "市場情緒偏悲觀，存在潛在下行風險。"
                    "建議謹慎操作，等待明確轉折信號，嚴格遵守止損紀律。",
                    pos, neg, neu, total)
        if pos_pct >= 40:
            return ("mildly_bullish", "🟡", "溫和偏多",
                    f"近期 {total} 則新聞中，正負面消息各占一定比例"
                    f"（正面 {pos_pct:.0f}%，負面 {neg_pct:.0f}%）。"
                    "整體情緒溫和偏正面，建議結合技術面（買入區間、SMA 支撐）做決策，"
                    "分批建倉並嚴格控制倉位大小。",
                    pos, neg, neu, total)
        return ("neutral", "⚪", "中性觀望",
                f"近期 {total} 則新聞中，正面 {pos} 則、負面 {neg} 則、中性 {neu} 則。"
                "市場方向不明確，建議等待更清晰的基本面或技術面信號，"
                "暫不積極加倉，持觀望態度。",
                pos, neg, neu, total)
    except Exception as e:
        print(_err("news_impact_summary", e))
        return "neutral", "新聞分析發生錯誤。"


# ── VIX / sentiment scoring ────────────────────────────────────────────────────
def _vix_to_greed_score(vix: float) -> float:
    """Map VIX value to a 0-100 Greed score (low VIX = high greed)."""
    score = 92 - (vix - 10) / 30 * 87
    return round(max(5.0, min(95.0, score)), 1)


def _score_to_label(score: float) -> tuple:
    """Return (label_zh, color_hex) for a 0-100 greed score."""
    if score < 20:  return "極度恐慌", "#FF2D2D"
    if score < 40:  return "恐慌",     "#FF8C00"
    if score < 60:  return "中性",     "#FFD700"
    if score < 80:  return "貪婪",     "#7CFC00"
    return "極度貪婪", "#00FF7F"


# ── Buy-zone & exit-strategy ───────────────────────────────────────────────────
def calc_buy_zone(price, sma50, sma200) -> tuple:
    """Return (lower, upper, zone_label, pct_position)."""
    try:
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
        pct = (price - lower) / zone_width * 100 if zone_width > 0 else 50.0
        pct = max(0.0, min(100.0, pct))
        return lower, upper, label, pct
    except Exception as e:
        print(_err("calc_buy_zone", e))
        return None, None, "計算錯誤", None


def calc_exit_strategy(price, sma200, low20) -> tuple:
    """
    Stop-loss: max(SMA200×0.97, low20×0.98).
    Take-profit: 1:2 R:R from entry.
    risk_pct = (止損價 - 買入價) / 買入價 × 100  (negative, representing loss %).
    Returns (stop, target, risk_pct).
    """
    try:
        if sma200 is None or low20 is None:
            return None, None, None
        stop    = max(sma200 * 0.97, low20 * 0.98)
        risk    = (stop - price) / price          # negative value (loss %)
        target  = price * (1 - risk * 2)          # price + 2× the upside equivalent
        return stop, target, risk
    except Exception as e:
        print(_err("calc_exit_strategy", e))
        return None, None, None


# ── Investment horizon classifier ─────────────────────────────────────────────
def classify_investment_horizon(
    price: float,
    sma50,
    sma200,
    hist: pd.DataFrame,
) -> dict:
    """
    Classify investment horizon as long-term (長期) or short-term (短期)
    based on trend position, moving-average relationship, and volatility.

    Scoring (+/−):
      +2  price between SMA50 and SMA200 (ideal accumulation zone)
      +1  SMA50 > SMA200 (golden-cross territory)
      +1  price > SMA200 (above long-term trend)
      −1  price > SMA50 × 1.05 (extended, momentum-chasing risk)
      −1  price < SMA200 (below long-term trend)
      −1  30-day daily-return std > 3 % (high volatility)

    Score ≥ 2 → 長期 (Long-Term)
    Score  < 2 → 短期 (Short-Term)

    Returns dict with:
      horizon      "long" | "short"
      label        "長期投資" | "短期交易"
      icon         emoji string
      hold_period  human-readable holding range
      reasons      list[str]  bullet points explaining the classification
      bg           CSS background colour
      border       CSS border colour
      accent       CSS text-highlight colour
    """
    try:
        score   = 0
        reasons = []

        if sma50 is not None and sma200 is not None:
            in_zone = (min(sma50, sma200) <= price <= max(sma50, sma200))
            if in_zone:
                score += 2
                reasons.append("股價位於 SMA50–SMA200 理想建倉區間")
            if sma50 > sma200:
                score += 1
                reasons.append("SMA50 > SMA200（黃金交叉趨勢）")
            if price > sma200:
                score += 1
                reasons.append("股價站上 SMA200 長期均線")
            else:
                score -= 1
                reasons.append("股價低於 SMA200，長期趨勢偏弱")
            if price > sma50 * 1.05:
                score -= 1
                reasons.append("股價短期漲幅偏大（> SMA50 × 1.05），回調風險較高")
        else:
            reasons.append("均線資料不足，以波動率作為主要依據")

        if hist is not None and not hist.empty and len(hist) >= 10:
            returns = hist["Close"].pct_change().dropna()
            vol30   = returns.tail(30).std() * 100
            if vol30 > 3.0:
                score -= 1
                reasons.append(f"近30日日波動率 {vol30:.1f}%，高波動適合短期操作")
            else:
                reasons.append(f"近30日日波動率 {vol30:.1f}%，波動穩定")

        if score >= 2:
            return {
                "horizon":     "long",
                "label":       "長期投資",
                "icon":        "🌱",
                "hold_period": "建議持有週期：6 個月 ～ 2 年以上",
                "reasons":     reasons,
                "bg":          "#0F2318",
                "border":      "#00FF7F",
                "accent":      "#00FF7F",
            }
        else:
            return {
                "horizon":     "short",
                "label":       "短期交易",
                "icon":        "⚡",
                "hold_period": "建議持有週期：2 週 ～ 3 個月",
                "reasons":     reasons,
                "bg":          "#0F1A2E",
                "border":      "#00D4FF",
                "accent":      "#00D4FF",
            }
    except Exception as e:
        print(_err("classify_investment_horizon", e))
        return {
            "horizon":     "short",
            "label":       "短期交易",
            "icon":        "⚡",
            "hold_period": "建議持有週期：2 週 ～ 3 個月",
            "reasons":     ["分類計算發生錯誤，預設顯示短期"],
            "bg":          "#0F1A2E",
            "border":      "#00D4FF",
            "accent":      "#00D4FF",
        }


# ── Relative strength chart ────────────────────────────────────────────────────
def plot_relative_strength(
    stock_hist:   pd.DataFrame,
    bm_hist:      pd.DataFrame,
    stock_ticker: str,
    bm_ticker:    str,
) -> tuple:
    """
    Normalize both series to 0% from first common date.
    Returns (fig, alpha_pct, is_outperforming, stock_return).
    Returns (None, None, None, None) on error.
    """
    try:
        def _strip_tz(series: pd.Series) -> pd.Series:
            idx = pd.to_datetime(series.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            s = series.copy()
            s.index = idx
            return s

        s_close = _strip_tz(stock_hist["Close"])
        b_close = _strip_tz(bm_hist["Close"])
        common  = s_close.index.intersection(b_close.index)
        if len(common) < 5:
            return None, None, None, None

        s_pct = (s_close.loc[common] / s_close.loc[common].iloc[0] - 1) * 100
        b_pct = (b_close.loc[common] / b_close.loc[common].iloc[0] - 1) * 100

        stock_return = float(s_pct.iloc[-1])
        alpha        = float(s_pct.iloc[-1] - b_pct.iloc[-1])
        is_out       = alpha > 0
        line_color   = "#00FF7F" if is_out else "#FF4B4B"
        fill_color   = "rgba(0,255,127,0.07)" if is_out else "rgba(255,75,75,0.07)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(common) + list(common[::-1]),
            y=list(s_pct)  + list(b_pct[::-1]),
            fill="toself", fillcolor=fill_color,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1.2)
        fig.add_trace(go.Scatter(
            x=common, y=b_pct, mode="lines", name=bm_ticker,
            line=dict(color="#00D4FF", width=2, dash="dash"),
            hovertemplate=f"{bm_ticker}: %{{y:+.2f}}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=common, y=s_pct, mode="lines", name=stock_ticker,
            line=dict(color=line_color, width=2.5),
            hovertemplate=f"{stock_ticker}: %{{y:+.2f}}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[common[-1]], y=[stock_return],
            mode="markers+text",
            text=[f"  {stock_return:+.1f}%"], textposition="middle right",
            textfont=dict(color=line_color, size=11),
            marker=dict(size=8, color=line_color),
            showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(
            title=dict(
                text=f"📈 累計報酬率對比：{stock_ticker} vs {bm_ticker}（起始均為 0%）",
                font=dict(size=13, color="#fff"),
            ),
            xaxis=dict(showgrid=True, gridcolor="#1E2130", tickfont=dict(color="#aaa")),
            yaxis=dict(showgrid=True, gridcolor="#1E2130", tickfont=dict(color="#aaa"),
                       ticksuffix="%",
                       title=dict(text="累計報酬率 (%)", font=dict(color="#aaa", size=11))),
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font=dict(color="#fff"),
            legend=dict(font=dict(color="#fff", size=11), bgcolor="rgba(0,0,0,0)",
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=340, margin=dict(t=55, b=40, l=65, r=60),
            hovermode="x unified",
        )
        return fig, alpha, is_out, stock_return
    except Exception as e:
        print(_err("plot_relative_strength", e))
        return None, None, None, None


# ── Four-quadrant matrix ───────────────────────────────────────────────────────
def plot_four_quadrant(
    stock_return: float,
    alpha: float,
    stock_ticker: str,
    bm_ticker: str,
) -> tuple:
    """2x2 quadrant scatter. Returns (fig, q_label, q_action, dot_color)."""
    try:
        is_pos_r = stock_return >= 0
        is_pos_a = alpha >= 0
        if is_pos_r and is_pos_a:
            q_label, q_action, dot_color = "強勢上升", "建議加倉 ✅", "#00FF7F"
        elif not is_pos_r and is_pos_a:
            q_label, q_action, dot_color = "抗跌強勢", "可以持有 💪", "#00D4FF"
        elif not is_pos_r and not is_pos_a:
            q_label, q_action, dot_color = "補跌弱勢", "考慮止損 ⚠️", "#FF4B4B"
        else:
            q_label, q_action, dot_color = "弱勢反彈", "謹慎觀望 🔍", "#FFD700"

        max_r = max(abs(stock_return) * 1.4, 20)
        max_a = max(abs(alpha) * 1.4, 15)

        fig = go.Figure()

        fig.add_shape(type="rect", x0=0,      y0=0,      x1=max_r,  y1=max_a,
                      fillcolor="rgba(0,255,127,0.06)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=-max_r, y0=0,      x1=0,      y1=max_a,
                      fillcolor="rgba(0,212,255,0.06)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=-max_r, y0=-max_a, x1=0,      y1=0,
                      fillcolor="rgba(255,75,75,0.06)",  line_width=0, layer="below")
        fig.add_shape(type="rect", x0=0,      y0=-max_a, x1=max_r,  y1=0,
                      fillcolor="rgba(255,215,0,0.06)",  line_width=0, layer="below")

        fig.add_annotation(
            x=max_r * 0.55, y=max_a * 0.78,
            text="🚀 強勢上升<br><b>建議加倉</b>",
            font=dict(color="#00FF7F", size=12),
            showarrow=False,
            bgcolor="rgba(0,0,0,0.5)",
        )
        fig.add_annotation(
            x=-max_r * 0.55, y=max_a * 0.78,
            text="💪 抗跌強勢<br><b>可以持有</b>",
            font=dict(color="#00D4FF", size=12),
            showarrow=False,
            bgcolor="rgba(0,0,0,0.5)",
        )
        fig.add_annotation(
            x=-max_r * 0.55, y=-max_a * 0.78,
            text="⚠️ 補跌弱勢<br><b>考慮止損</b>",
            font=dict(color="#FF4B4B", size=12),
            showarrow=False,
            bgcolor="rgba(0,0,0,0.5)",
        )
        fig.add_annotation(
            x=max_r * 0.55, y=-max_a * 0.78,
            text="🔍 弱勢反彈<br><b>謹慎觀望</b>",
            font=dict(color="#FFD700", size=12),
            showarrow=False,
            bgcolor="rgba(0,0,0,0.5)",
        )

        fig.add_hline(y=0, line_color="#444", line_width=1.5)
        fig.add_vline(x=0, line_color="#444", line_width=1.5)

        dot_x = max(-max_r * 0.93, min(max_r * 0.93, stock_return))
        dot_y = max(-max_a * 0.93, min(max_a * 0.93, alpha))

        fig.add_trace(go.Scatter(
            x=[dot_x], y=[dot_y],
            mode="markers+text",
            text=[f"  {stock_ticker}"],
            textposition="middle right",
            textfont=dict(color="#fff", size=11),
            marker=dict(size=14, color=dot_color, symbol="circle",
                        line=dict(color="#fff", width=2)),
            showlegend=False,
            hovertemplate=(
                f"<b>{stock_ticker}</b><br>"
                f"期間報酬：{stock_return:+.1f}%<br>"
                f"Alpha：{alpha:+.1f}%<br>"
                f"象限：{q_label}<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=dict(text="四象限強弱矩陣", font=dict(size=12, color="#fff")),
            xaxis=dict(
                title=dict(text=f"{stock_ticker} 期間報酬 (%)", font=dict(color="#aaa", size=12)),
                range=[-max_r, max_r], showgrid=False, zeroline=False,
                tickfont=dict(color="#aaa", size=12), ticksuffix="%",
            ),
            yaxis=dict(
                title=dict(text="超額 Alpha (%)", font=dict(color="#aaa", size=12)),
                range=[-max_a, max_a], showgrid=False, zeroline=False,
                tickfont=dict(color="#aaa", size=12), ticksuffix="%",
            ),
            paper_bgcolor="#0E1117", plot_bgcolor="#12151F",
            font=dict(color="#fff"), height=340,
            margin=dict(t=38, b=50, l=60, r=20),
            showlegend=False,
        )
        return fig, q_label, q_action, dot_color
    except Exception as e:
        print(_err("plot_four_quadrant", e))
        return None, "N/A", "N/A", "#aaa"
# ── Sentiment gauge chart ──────────────────────────────────────────────────────
def plot_sentiment_gauge(score: float, title_prefix: str = "市場宏觀情緒") -> go.Figure:
    """Plotly Gauge: 0 = Extreme Fear, 100 = Extreme Greed."""
    try:
        label, bar_color = _score_to_label(score)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"{title_prefix}　{label}", "font": {"size": 15, "color": "#FFFFFF"}},
            number={"font": {"color": bar_color, "size": 38}, "suffix": " / 100"},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["極恐慌", "恐慌", "中性", "貪婪", "極貪婪"],
                    "tickcolor": "#aaa", "tickfont": {"size": 10, "color": "#aaa"},
                },
                "bar":         {"color": bar_color, "thickness": 0.3},
                "bgcolor":     "#1E2130", "borderwidth": 0,
                "steps": [
                    {"range": [0,   20], "color": "#3D0D0D"},
                    {"range": [20,  40], "color": "#3D200D"},
                    {"range": [40,  60], "color": "#2E2A0D"},
                    {"range": [60,  80], "color": "#152E0D"},
                    {"range": [80, 100], "color": "#0D2E0D"},
                ],
                "threshold": {"line": {"color": bar_color, "width": 3},
                              "thickness": 0.8, "value": score},
            },
        ))
        fig.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font={"color": "#FFFFFF"}, height=240,
            margin=dict(t=50, b=5, l=20, r=20),
        )
        return fig
    except Exception as e:
        print(_err("plot_sentiment_gauge", e))
        return go.Figure()


# ── VIX fear timeline ──────────────────────────────────────────────────────────
def plot_fear_timeline(vix_df: pd.DataFrame, macro_events: list) -> go.Figure:
    """
    Line chart of last-N-day VIX converted to Greed Score (0-100),
    with macro_events annotated as vertical markers.
    """
    try:
        fig = go.Figure()
        if vix_df.empty:
            fig.add_annotation(
                text="⚠️ VIX 數據暫不可用", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font={"color": "#aaa", "size": 14},
            )
        else:
            greed_series = vix_df["VIX"].apply(_vix_to_greed_score)
            for lo, hi, zone_color in [(0,20,"rgba(255,45,45,0.08)"),
                                        (20,40,"rgba(255,140,0,0.06)"),
                                        (40,60,"rgba(255,215,0,0.04)"),
                                        (60,80,"rgba(124,252,0,0.06)"),
                                        (80,100,"rgba(0,255,127,0.08)")]:
                fig.add_hrect(y0=lo, y1=hi, fillcolor=zone_color, line_width=0, layer="below")

            fig.add_trace(go.Scatter(
                x=vix_df.index, y=greed_series,
                mode="lines+markers", name="市場貪婪/恐慌指數",
                line=dict(color="#00D4FF", width=2.5),
                marker=dict(size=5, color="#00D4FF"),
                hovertemplate="%{x|%m/%d}<br>情緒分：%{y:.0f}<br>VIX：%{customdata:.1f}<extra></extra>",
                customdata=vix_df["VIX"].values,
            ))

            chart_start = vix_df.index.min().date()
            chart_end   = vix_df.index.max().date()
            for ev_date_str, ev_label, ev_severity in macro_events:
                try:
                    ev_date = pd.to_datetime(ev_date_str).date()
                except Exception:
                    continue
                if not (chart_start <= ev_date <= chart_end):
                    continue
                ev_color = "#FF4B4B" if ev_severity == "high" else (
                           "#FFA500" if ev_severity == "medium" else "#FFD700")
                ev_ms = int(pd.Timestamp(ev_date_str).timestamp() * 1000)
                fig.add_vline(x=ev_ms, line_width=1.5, line_dash="dot", line_color=ev_color)
                try:
                    ev_ts    = pd.Timestamp(ev_date_str)
                    closest  = vix_df.iloc[(vix_df.index - ev_ts).abs().argsort()[:1]]
                    ann_y    = _vix_to_greed_score(float(closest["VIX"].iloc[0]))
                except Exception:
                    ann_y = 85
                fig.add_annotation(
                    x=ev_ms, y=ann_y, text=f"<b>{ev_label}</b>",
                    showarrow=True, arrowhead=2, arrowcolor=ev_color, arrowwidth=1.2,
                    font={"size": 9, "color": ev_color},
                    bgcolor="rgba(0,0,0,0.55)", bordercolor=ev_color,
                    borderwidth=1, borderpad=3,
                )

        fig.update_layout(
            title={"text": "📉 市場恐慌指數走勢（過去 3 個月）",
                   "font": {"size": 14, "color": "#FFFFFF"}},
            xaxis=dict(showgrid=True, gridcolor="#1E2130", tickfont={"color": "#aaa"},
                       title=dict(text="日期", font={"color": "#aaa"})),
            yaxis=dict(range=[0,100], showgrid=True, gridcolor="#1E2130",
                       tickfont={"color": "#aaa"},
                       title=dict(text="情緒得分（0=極恐 / 100=極貪）",
                                  font={"color": "#aaa"}),
                       tickvals=[0,20,40,60,80,100],
                       ticktext=["0 極恐","20 恐慌","40 中性","60 貪婪","80 極貪","100"]),
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font={"color": "#FFFFFF"}, height=340,
            margin=dict(t=50, b=40, l=60, r=20),
            showlegend=False, hovermode="x unified",
        )
        return fig
    except Exception as e:
        print(_err("plot_fear_timeline", e))
        return go.Figure()


# ══════════════════════════════════════════════════════════════════════════════
# 7-FACTOR MULTI-FACTOR ANALYSIS SYSTEM
# Scores each factor sub-metric to [-5, +5] using linear interpolation
# between domain breakpoints; groups them into 7 composite factor scores.
# ══════════════════════════════════════════════════════════════════════════════

def _interp_score(value: float, breakpoints: list) -> float:
    """
    Linear interpolation between (threshold, score) breakpoints.
    Result is clamped to [-5, +5].
    """
    if value <= breakpoints[0][0]:
        return float(breakpoints[0][1])
    if value >= breakpoints[-1][0]:
        return float(breakpoints[-1][1])
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0) if x1 != x0 else 0.0
            return round(y0 + t * (y1 - y0), 1)
    return 0.0


def calculate_factor_score(value, metric_type: str) -> float:
    """
    Map a raw financial metric value to a score in [-5, +5].
    Returns 0.0 when value is None / NaN.
    """
    import math
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0

    # ── Momentum ───────────────────────────────────────────────────────────
    if metric_type == "momentum_1m":
        return _interp_score(value, [(-20,-5),(-10,-3),(-5,-1),(0,0),(3,1),(8,3),(15,5)])
    if metric_type == "momentum_3m":
        return _interp_score(value, [(-35,-5),(-15,-3),(-5,-1),(0,0),(5,1),(20,3),(40,5)])
    if metric_type == "momentum_6m":
        return _interp_score(value, [(-50,-5),(-20,-3),(-5,-1),(0,0),(10,1),(30,3),(60,5)])
    if metric_type == "momentum_12m":
        return _interp_score(value, [(-60,-5),(-25,-3),(-5,-1),(0,0),(15,1),(40,3),(80,5)])
    if metric_type == "rsi":
        # RSI momentum: 50-65 = ideal (trend + not overbought)
        return _interp_score(value, [(10,-3),(25,-1),(35,0),(50,2),(60,5),(70,2),(80,-1),(90,-3)])
    if metric_type == "macd_signal":
        # +1 if bullish crossover, -1 if bearish (or 0 neutral)
        return max(-5.0, min(5.0, value * 5.0))

    # ── Value (lower multiple = higher score for PE/PB/PS; reverse for yield) ─
    if metric_type == "pe":
        if value <= 0:
            return -1.0   # negative earnings
        return _interp_score(value, [(5,5),(10,4),(15,3),(20,2),(25,1),(35,-1),(50,-3),(80,-5)])
    if metric_type == "pb":
        if value <= 0:
            return -1.0
        return _interp_score(value, [(0.5,5),(1,4),(2,2),(3,1),(5,0),(8,-2),(15,-4),(30,-5)])
    if metric_type == "ev_ebitda":
        if value <= 0:
            return -1.0
        return _interp_score(value, [(5,5),(8,4),(12,3),(15,2),(20,0),(25,-2),(35,-4),(50,-5)])
    if metric_type == "ps":
        if value <= 0:
            return -1.0
        return _interp_score(value, [(0.5,5),(1,4),(3,2),(5,1),(8,0),(15,-2),(25,-4),(50,-5)])
    if metric_type == "dividend_yield":
        return _interp_score(value, [(0,0),(0.5,1),(1,2),(2,3),(3,4),(5,5),(8,4),(12,2)])

    # ── Quality ───────────────────────────────────────────────────────────
    if metric_type == "ROE":
        return _interp_score(value, [(-30,-5),(-5,-3),(0,-1),(5,0),(10,1),(15,3),(20,4),(30,5)])
    if metric_type == "roa":
        return _interp_score(value, [(-15,-5),(0,-2),(3,-1),(5,0),(8,2),(12,4),(18,5)])
    if metric_type == "gross_margin":
        return _interp_score(value, [(0,-3),(10,-1),(20,0),(30,1),(40,2),(55,4),(70,5)])
    if metric_type == "op_margin":
        return _interp_score(value, [(-20,-5),(0,-2),(5,-1),(10,0),(15,2),(20,4),(30,5)])
    if metric_type == "debt_equity":
        return _interp_score(value, [(0,5),(20,4),(50,3),(80,2),(100,1),(150,-1),(200,-3),(300,-5)])
    if metric_type == "current_ratio":
        return _interp_score(value, [(0,-5),(0.5,-3),(0.8,-1),(1.0,0),(1.5,2),(2.0,4),(3.0,5)])

    # ── Growth ────────────────────────────────────────────────────────────
    if metric_type == "rev_growth":
        return _interp_score(value, [(-30,-5),(-5,-2),(0,-1),(3,0),(8,2),(15,4),(25,5)])
    if metric_type == "eps_growth":
        return _interp_score(value, [(-50,-5),(-10,-2),(0,-1),(5,0),(10,2),(20,4),(40,5)])
    if metric_type == "fwd_eps_growth":
        return _interp_score(value, [(-20,-4),(0,-1),(5,0),(10,2),(15,3),(25,5)])
    if metric_type == "analyst_rating":
        # 1=Strong Buy, 5=Strong Sell
        return _interp_score(value, [(1,5),(1.5,4),(2,3),(2.5,1),(3,0),(3.5,-2),(4,-4),(5,-5)])

    # ── Volatility ────────────────────────────────────────────────────────
    if metric_type == "beta":
        return _interp_score(value, [(0,2),(0.5,4),(0.8,4),(1.0,3),(1.2,1),(1.5,-1),(2.0,-3),(3.0,-5)])
    if metric_type == "vol_30d":
        return _interp_score(value, [(0,5),(8,4),(12,3),(18,1),(25,-1),(35,-3),(50,-5)])
    if metric_type == "max_dd":
        # max_dd is negative % (e.g. -20%)
        return _interp_score(value, [(-60,-5),(-40,-3),(-25,-1),(-15,0),(-10,1),(-5,3),(0,5)])
    if metric_type == "atr_pct":
        return _interp_score(value, [(0,5),(1,4),(2,3),(3,1),(4,0),(6,-2),(10,-5)])

    # ── Sentiment ─────────────────────────────────────────────────────────
    if metric_type == "short_pct":
        return _interp_score(value, [(0,5),(1,4),(3,2),(5,1),(8,0),(12,-2),(18,-4),(25,-5)])
    if metric_type == "inst_ownership":
        return _interp_score(value, [(0,-1),(10,0),(30,1),(50,2),(70,4),(85,5),(100,4)])
    if metric_type == "insider_ownership":
        return _interp_score(value, [(0,0),(1,1),(5,2),(10,3),(20,4),(30,5),(50,4)])
    if metric_type == "analyst_consensus":
        return _interp_score(value, [(1,5),(1.5,4),(2,3),(2.5,1),(3,0),(3.5,-2),(4,-4),(5,-5)])

    return 0.0


def calculate_seven_factors(
    stock_info:  dict,
    hist:        pd.DataFrame | None,
    factor_data: dict,
) -> dict:
    """
    Compute all 7 factor groups for a single stock.

    Returns a dict:
      {
        "Momentum":   {"score": float, "items": [{"name","value_raw","value_str","score"}]},
        "Value":      {...},
        "Quality":    {...},
        "Growth":     {...},
        "Volatility": {...},
        "Sentiment":  {...},
        "Macro":      {...},
        "composite":  float,   # weighted average of 7 group scores
        "signal":     str,     # "STRONG BUY" / "BUY" / "HOLD" / "SELL" / "STRONG SELL"
      }
    """
    fd = factor_data or {}
    price = fd.get("price") or stock_info.get("price", 0)

    # ── Pre-compute momentum from hist ────────────────────────────────────
    m1 = m3 = m6 = m12 = None
    rsi14 = 50.0
    macd_sig = 0.0
    vol_30d = None
    max_dd = None
    atr_pct = None
    spy_mom = 0.0

    if hist is not None and not hist.empty:
        close = hist["Close"].dropna()
        n = len(close)

        def _pct(days):
            if n >= days:
                return float((close.iloc[-1] / close.iloc[-days] - 1) * 100)
            return None

        m1  = _pct(21)
        m3  = _pct(63)
        m6  = _pct(126)
        m12 = _pct(252) if n >= 252 else _pct(n)

        # RSI(14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_s = 100 - 100 / (1 + rs)
        if not rsi_s.empty:
            rsi14 = float(rsi_s.iloc[-1])

        # MACD signal direction (+1 bullish, -1 bearish)
        try:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line   = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_sig = 1.0 if float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]) else -1.0
        except Exception:
            macd_sig = 0.0

        # 30D realized volatility (annualized %)
        if n >= 30:
            ret30  = close.tail(30).pct_change().dropna()
            vol_30d = float(ret30.std() * np.sqrt(252) * 100)

        # Max drawdown 1Y
        rolling_max = close.cummax()
        dd_series   = (close - rolling_max) / rolling_max * 100
        max_dd = float(dd_series.min())

        # ATR % of price
        if "High" in hist.columns and "Low" in hist.columns and price > 0:
            tr = pd.concat([
                hist["High"] - hist["Low"],
                (hist["High"] - hist["Close"].shift()).abs(),
                (hist["Low"]  - hist["Close"].shift()).abs(),
            ], axis=1).max(axis=1)
            atr14  = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = atr14 / price * 100

    # ── Helper to build item dict ─────────────────────────────────────────
    def _item(name, raw, fmt, metric):
        s = calculate_factor_score(raw, metric) if raw is not None else 0.0
        return {"name": name, "value_raw": raw, "value_str": fmt, "score": s}

    def _mean_score(items):
        scores = [i["score"] for i in items if i["score"] != 0.0]
        return round(sum(scores) / len(scores), 2) if scores else 0.0

    def _fmt(v, suffix="", decimals=1, prefix=""):
        if v is None:
            return "N/A"
        try:
            return f"{prefix}{v:.{decimals}f}{suffix}"
        except Exception:
            return str(v)

    # ── 1. Momentum ───────────────────────────────────────────────────────
    mom_items = [
        _item("1M 動量",   m1,      _fmt(m1, "%"),    "momentum_1m"),
        _item("3M 動量",   m3,      _fmt(m3, "%"),    "momentum_3m"),
        _item("6M 動量",   m6,      _fmt(m6, "%"),    "momentum_6m"),
        _item("12M 動量",  m12,     _fmt(m12, "%"),   "momentum_12m"),
        _item("RSI(14)",   rsi14,   _fmt(rsi14),      "rsi"),
        _item("MACD 信號", macd_sig, "多頭" if macd_sig > 0 else ("空頭" if macd_sig < 0 else "中性"), "macd_signal"),
    ]

    # ── 2. Value ──────────────────────────────────────────────────────────
    val_items = [
        _item("P/E Ratio",      fd.get("pe_ratio"),       _fmt(fd.get("pe_ratio"), "x", 1),   "pe"),
        _item("P/B Ratio",      fd.get("pb_ratio"),       _fmt(fd.get("pb_ratio"), "x", 2),   "pb"),
        _item("EV/EBITDA",      fd.get("ev_ebitda"),      _fmt(fd.get("ev_ebitda"), "x", 1),  "ev_ebitda"),
        _item("P/S Ratio",      fd.get("ps_ratio"),       _fmt(fd.get("ps_ratio"), "x", 2),   "ps"),
        _item("股息殖利率",     fd.get("dividend_yield"), _fmt(fd.get("dividend_yield"), "%", 2), "dividend_yield"),
    ]

    # ── 3. Quality ────────────────────────────────────────────────────────
    qual_items = [
        _item("ROE",         fd.get("roe"),           _fmt(fd.get("roe"), "%", 1),   "ROE"),
        _item("ROA",         fd.get("roa"),           _fmt(fd.get("roa"), "%", 1),   "roa"),
        _item("毛利率",      fd.get("gross_margin"),  _fmt(fd.get("gross_margin"), "%", 1), "gross_margin"),
        _item("營業利益率",  fd.get("op_margin"),     _fmt(fd.get("op_margin"), "%", 1),   "op_margin"),
        _item("債務/股權",   fd.get("debt_equity"),   _fmt(fd.get("debt_equity"), "", 1),   "debt_equity"),
        _item("流動比率",    fd.get("current_ratio"), _fmt(fd.get("current_ratio"), "x", 2), "current_ratio"),
    ]

    # ── 4. Growth ─────────────────────────────────────────────────────────
    fwd_eps  = fd.get("fwd_eps")
    trail_eps = fd.get("trailing_eps")
    fwd_eps_growth = None
    if fwd_eps and trail_eps and trail_eps != 0:
        fwd_eps_growth = (fwd_eps / abs(trail_eps) - 1) * 100

    rec_mean = fd.get("rec_mean")
    grow_items = [
        _item("營收成長",       fd.get("rev_growth"),    _fmt(fd.get("rev_growth"), "%", 1),   "rev_growth"),
        _item("EPS 成長",       fd.get("eps_growth"),    _fmt(fd.get("eps_growth"), "%", 1),   "eps_growth"),
        _item("預期 EPS 成長",  fwd_eps_growth,          _fmt(fwd_eps_growth, "%", 1),         "fwd_eps_growth"),
        _item("分析師評級",     rec_mean,                _fmt(rec_mean, "/5", 2),              "analyst_rating"),
    ]

    # ── 5. Volatility ─────────────────────────────────────────────────────
    vol_items = [
        _item("Beta",            fd.get("beta"), _fmt(fd.get("beta"), "", 2), "beta"),
        _item("30日年化波動率",  vol_30d,        _fmt(vol_30d, "%", 1),       "vol_30d"),
        _item("最大回撤(1Y)",    max_dd,         _fmt(max_dd, "%", 1),        "max_dd"),
        _item("ATR% of Price",   atr_pct,        _fmt(atr_pct, "%", 2),       "atr_pct"),
    ]

    # ── 6. Sentiment ──────────────────────────────────────────────────────
    sent_items = [
        _item("空頭興趣%",    fd.get("short_pct"),       _fmt(fd.get("short_pct"), "%", 1),       "short_pct"),
        _item("機構持股%",    fd.get("inst_ownership"),   _fmt(fd.get("inst_ownership"), "%", 1),  "inst_ownership"),
        _item("內部人持股%",  fd.get("insider_ownership"),_fmt(fd.get("insider_ownership"), "%", 1),"insider_ownership"),
        _item("分析師共識",   fd.get("rec_mean"),         _fmt(fd.get("rec_mean"), "/5", 2),        "analyst_consensus"),
    ]

    # ── 7. Macro ──────────────────────────────────────────────────────────
    # Use hist vs benchmark indicator (simplified — golden cross + 12M momentum signal)
    sma50_val  = None
    sma200_val = None
    golden_cross_score = 0.0
    if hist is not None and not hist.empty:
        c = hist["Close"].dropna()
        if len(c) >= 50:
            sma50_val = float(c.rolling(50).mean().iloc[-1])
        if len(c) >= 200:
            sma200_val = float(c.rolling(200).mean().iloc[-1])
        if sma50_val and sma200_val:
            golden_cross_score = 3.0 if sma50_val > sma200_val else -3.0

    macro_items = [
        _item("SMA50 vs SMA200",
              golden_cross_score,
              "黃金交叉" if golden_cross_score > 0 else ("死亡交叉" if golden_cross_score < 0 else "N/A"),
              "macd_signal"),
        _item("12M 相對表現",  m12, _fmt(m12, "%"), "momentum_12m"),
        _item("Beta 敏感度",   fd.get("beta"), _fmt(fd.get("beta"), "", 2), "beta"),
        _item("機構認可度",    fd.get("inst_ownership"), _fmt(fd.get("inst_ownership"), "%", 1), "inst_ownership"),
    ]

    # ── Assemble groups ───────────────────────────────────────────────────
    groups = {
        "Momentum":   {"label": "Momentum 動量",   "items": mom_items},
        "Value":      {"label": "Value 估值",       "items": val_items},
        "Quality":    {"label": "Quality 質量",     "items": qual_items},
        "Growth":     {"label": "Growth 成長",      "items": grow_items},
        "Volatility": {"label": "Volatility 波動性","items": vol_items},
        "Sentiment":  {"label": "Sentiment 情緒",   "items": sent_items},
        "Macro":      {"label": "Macro 宏觀",       "items": macro_items},
    }
    for k, g in groups.items():
        g["score"] = _mean_score(g["items"])

    # ── Composite score (weighted average) ────────────────────────────────
    weights = {
        "Momentum": 0.20, "Value": 0.15, "Quality": 0.20,
        "Growth":   0.15, "Volatility": 0.10,
        "Sentiment": 0.10, "Macro": 0.10,
    }
    composite = sum(groups[k]["score"] * w for k, w in weights.items())
    composite  = round(composite, 2)

    # ── Signal label ──────────────────────────────────────────────────────
    if composite >= 2.5:
        signal = "STRONG BUY"
    elif composite >= 1.0:
        signal = "BUY"
    elif composite >= -1.0:
        signal = "HOLD"
    elif composite >= -2.5:
        signal = "SELL"
    else:
        signal = "STRONG SELL"

    return {**groups, "composite": composite, "signal": signal}


def plot_factor_radar(factors: dict, ticker: str = "") -> go.Figure:
    """Return a Plotly radar chart for the 7 factor group scores."""
    try:
        factor_keys   = ["Momentum", "Value", "Quality", "Growth", "Volatility", "Sentiment", "Macro"]
        factor_labels = ["Momentum 動量", "Value 估值", "Quality 質量",
                         "Growth 成長", "Volatility 波動", "Sentiment 情緒", "Macro 宏觀"]
        scores = [factors.get(k, {}).get("score", 0) for k in factor_keys]
        scores_closed = scores + [scores[0]]   # close the polygon
        labels_closed = factor_labels + [factor_labels[0]]

        composite = factors.get("composite", 0)
        line_color = "#00FF7F" if composite >= 0 else "#FF4B4B"

        fig = go.Figure()

        # Background reference at ±3 and ±5
        for ref_r, ref_opacity in [(5, 0.05), (3, 0.06), (0, 0)]:
            if ref_r > 0:
                fig.add_trace(go.Scatterpolar(
                    r=[ref_r] * (len(factor_labels) + 1),
                    theta=labels_closed,
                    mode="lines",
                    line=dict(color="#444", width=1, dash="dot"),
                    fill="toself",
                    fillcolor=f"rgba(100,100,100,{ref_opacity})",
                    showlegend=False,
                    hoverinfo="skip",
                ))

        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=labels_closed,
            mode="lines+markers",
            fill="toself",
            fillcolor=f"rgba(0,255,127,0.10)" if composite >= 0 else "rgba(255,75,75,0.10)",
            line=dict(color=line_color, width=2.5),
            marker=dict(size=7, color=line_color,
                        line=dict(color="#fff", width=1.5)),
            name=ticker,
            hovertemplate="<b>%{theta}</b><br>得分：%{r:+.1f}<extra></extra>",
        ))

        fig.update_layout(
            polar=dict(
                bgcolor="#0E1117",
                radialaxis=dict(
                    visible=True, range=[-5, 5],
                    tickvals=[-5, -3, 0, 3, 5],
                    tickfont=dict(size=9, color="#666"),
                    gridcolor="#2a2a3a",
                    linecolor="#333",
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color="#C9D1D9"),
                    gridcolor="#2a2a3a",
                    linecolor="#333",
                ),
            ),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#C9D1D9"),
            height=360,
            margin=dict(t=30, b=30, l=50, r=50),
            showlegend=False,
        )
        return fig
    except Exception as e:
        print(_err("plot_factor_radar", e))
        return go.Figure()


def build_factor_prompt(ticker: str, name: str, factors: dict) -> str:
    """Build the structured Gemini prompt for 7-factor AI analysis."""
    comp = factors.get("composite", 0)
    signal = factors.get("signal", "HOLD")
    lines = [
        f"你是一位資深量化投資分析師。請分析 {name}（{ticker}）的七大因子得分：",
        "",
        f"【綜合評分】{comp:+.2f} / ±5.0　→ 評級：{signal}",
        "",
    ]
    for key in ["Momentum", "Value", "Quality", "Growth", "Volatility", "Sentiment", "Macro"]:
        g = factors.get(key, {})
        g_score = g.get("score", 0)
        items_str = "，".join(
            f"{i['name']} {i['value_str']}（{i['score']:+.1f}）"
            for i in g.get("items", [])
            if i.get("value_str") != "N/A"
        )
        lines.append(f"【{g.get('label', key)}】得分 {g_score:+.2f} — {items_str}")

    lines += [
        "",
        "請用外行人聽得懂的語言，以繁體中文完成以下分析（每節限 2-3 句）：",
        "1. 📈 **核心強項**：哪些因子表現突出，代表什麼投資意義？",
        "2. ⚠️ **主要弱點**：哪些因子拖累評分，需要注意什麼風險？",
        "3. 🎯 **操作建議**：根據綜合評分給出明確建議（",
        f"   目前評級：**{signal}**），並說明進場條件或觀察要點。",
        "4. ⏱️ **投資時框**：適合長期持有還是短期交易？理由為何？",
        "",
        "請以結構化格式輸出，每節使用 **粗體標題**，語言精簡清晰。",
    ]
    return "\n".join(lines)

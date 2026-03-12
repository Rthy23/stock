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
            f' border-radius:4px; padding:2px 8px; font-size:11px;'
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
    alpha:        float,
    stock_ticker: str,
    bm_ticker:    str,
) -> tuple:
    """
    2×2 quadrant scatter.  Returns (fig, q_label, q_action, dot_color).
    """
    try:
        is_pos_r = stock_return >= 0
        is_pos_a = alpha >= 0
        if   is_pos_r and     is_pos_a:
            q_label, q_action, dot_color = "強勢上升", "建議加倉 ✅", "#00FF7F"
        elif not is_pos_r and is_pos_a:
            q_label, q_action, dot_color = "抗跌強勢", "可以持有 💪", "#00D4FF"
        elif not is_pos_r and not is_pos_a:
            q_label, q_action, dot_color = "補跌弱勢", "考慮止損 ⚠️", "#FF4B4B"
        else:
            q_label, q_action, dot_color = "弱勢反彈", "謹慎觀望 🔍", "#FFD700"

        max_r = max(abs(stock_return) * 1.4, 20)
        max_a = max(abs(alpha)        * 1.4, 15)

        fig = go.Figure()
        for (x0, y0, x1, y1, color) in [
            (0,      0,     max_r,  max_a,  "rgba(0,255,127,0.06)"),
            (-max_r, 0,     0,      max_a,  "rgba(0,212,255,0.06)"),
            (-max_r, -max_a, 0,     0,      "rgba(255,75,75,0.06)"),
            (0,      -max_a, max_r, 0,      "rgba(255,215,0,0.06)"),
        ]:
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          fillcolor=color, line_width=0, layer="below")

        ann_kw = dict(showarrow=False, bgcolor="rgba(0,0,0,0.5)")
        fig.add_annotation(x=max_r*0.55,  y=max_a*0.78,
                           text="🚀 強勢上升<br><b>建議加倉</b>",
                           font=dict(color="#00FF7F", size=9), **ann_kw)
        fig.add_annotation(x=-max_r*0.55, y=max_a*0.78,
                           text="💪 抗跌強勢<br><b>可以持有</b>",
                           font=dict(color="#00D4FF", size=9), **ann_kw)
        fig.add_annotation(x=-max_r*0.55, y=-max_a*0.78,
                           text="⚠️ 補跌弱勢<br><b>考慮止損</b>",
                           font=dict(color="#FF4B4B", size=9), **ann_kw)
        fig.add_annotation(x=max_r*0.55,  y=-max_a*0.78,
                           text="🔍 弱勢反彈<br><b>謹慎觀望</b>",
                           font=dict(color="#FFD700", size=9), **ann_kw)

        fig.add_hline(y=0, line_color="#444", line_width=1.5)
        fig.add_vline(x=0, line_color="#444", line_width=1.5)

        dot_x = max(-max_r*0.93, min(max_r*0.93, stock_return))
        dot_y = max(-max_a*0.93, min(max_a*0.93, alpha))
        fig.add_trace(go.Scatter(
            x=[dot_x], y=[dot_y],
            mode="markers+text", text=[f"  {stock_ticker}"],
            textposition="middle right", textfont=dict(color="#fff", size=11),
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
                title=dict(text=f"{stock_ticker} 期間報酬 (%)",
                           font=dict(color="#aaa", size=10)),
                range=[-max_r, max_r], showgrid=False, zeroline=False,
                tickfont=dict(color="#aaa", size=9), ticksuffix="%",
            ),
            yaxis=dict(
                title=dict(text="超額 Alpha (%)", font=dict(color="#aaa", size=10)),
                range=[-max_a, max_a], showgrid=False, zeroline=False,
                tickfont=dict(color="#aaa", size=9), ticksuffix="%",
            ),
            paper_bgcolor="#0E1117", plot_bgcolor="#12151F",
            font=dict(color="#fff"), height=340,
            margin=dict(t=38, b=50, l=60, r=20), showlegend=False,
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

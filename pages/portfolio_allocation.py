"""Macro-aware defensive portfolio allocation page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data_fetcher import beijing_timestamp
from navigation import navigate_to_ticker


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_allocation_data() -> dict[str, Any]:
    """Fetch all allocation inputs in one hourly-cached yfinance call."""
    treasury = yf.Ticker("^TNX").history(period="1mo", auto_adjust=False)
    vix = yf.Ticker("^VIX").history(period="1mo", auto_adjust=False)
    spy_info = yf.Ticker("SPY").info

    if treasury is None or treasury.empty:
        raise RuntimeError("10Y Treasury Yield 暫時沒有資料。")
    if vix is None or vix.empty:
        raise RuntimeError("VIX 暫時沒有資料。")

    pe = spy_info.get("trailingPE") or spy_info.get("forwardPE")
    return {
        "treasury_10y": float(treasury["Close"].dropna().iloc[-1]),
        "vix": float(vix["Close"].dropna().iloc[-1]),
        "sp500_pe": float(pe) if pe is not None else None,
        "loaded_at": beijing_timestamp(),
    }


def classify_market_risk(macro: dict[str, Any]) -> tuple[str, int, list[str]]:
    """Classify risk using transparent VIX, yield, and valuation signals."""
    score = 0
    reasons: list[str] = []
    vix = macro.get("vix")
    yield_10y = macro.get("treasury_10y")
    pe = macro.get("sp500_pe")

    if vix is not None:
        if vix < 18:
            score += 1
            reasons.append("VIX 低於 18，市場波動偏低")
        elif vix > 25:
            score -= 1
            reasons.append("VIX 高於 25，避險需求升溫")
        else:
            reasons.append("VIX 位於中性區間")
    if yield_10y is not None:
        if yield_10y < 4.5:
            score += 1
            reasons.append("10Y 殖利率低於 4.5%，估值壓力相對溫和")
        elif yield_10y > 5.0:
            score -= 1
            reasons.append("10Y 殖利率高於 5%，折現率壓力偏高")
        else:
            reasons.append("10Y 殖利率位於觀察區間")
    if pe is not None:
        if pe < 22:
            score += 1
            reasons.append("S&P 500 P/E 低於 22，估值較有緩衝")
        elif pe > 28:
            score -= 1
            reasons.append("S&P 500 P/E 高於 28，估值偏緊")
        else:
            reasons.append("S&P 500 P/E 位於中性區間")

    level = "Risk-on" if score >= 2 else "Risk-off" if score <= -2 else "Neutral"
    return level, score, reasons


ALLOCATION_PLANS: dict[str, list[dict[str, Any]]] = {
    "Risk-on": [
        {"資產類別": "美股", "比例": 0.70, "建議標的": "VOO / QQQ", "說明": "核心指數與成長動能"},
        {"資產類別": "基金", "比例": 0.20, "建議標的": "VIG / SCHD", "說明": "股息成長與品質因子"},
        {"資產類別": "定存現金", "比例": 0.10, "建議標的": "SGOV / BIL", "說明": "保留流動性與再平衡資金"},
    ],
    "Neutral": [
        {"資產類別": "美股", "比例": 0.50, "建議標的": "VOO / SPLG", "說明": "降低集中於高波動成長股"},
        {"資產類別": "基金", "比例": 0.30, "建議標的": "VIG / SCHD", "說明": "以品質與股息降低波動"},
        {"資產類別": "定存現金", "比例": 0.20, "建議標的": "SGOV / BIL", "說明": "等待更好估值與防守"},
    ],
    "Risk-off": [
        {"資產類別": "美股", "比例": 0.30, "建議標的": "VOO / SPLV", "說明": "維持市場參與但降低風險"},
        {"資產類別": "基金", "比例": 0.30, "建議標的": "VIG / USMV", "說明": "品質、低波動與股息因子"},
        {"資產類別": "定存現金", "比例": 0.40, "建議標的": "SGOV / BIL", "說明": "提高短債與現金緩衝"},
    ],
}


def _allocation_table(level: str, amount: float) -> pd.DataFrame:
    rows = []
    for item in ALLOCATION_PLANS[level]:
        rows.append(
            {
                **item,
                "比例": f"{item['比例']:.0%}",
                "建議金額 (USD)": amount * item["比例"],
            }
        )
    return pd.DataFrame(rows)


def render_portfolio_allocation_page() -> None:
    st.title("🧭 穩健資產配置策略")
    st.caption(
        "以 10Y Treasury Yield、VIX 與 S&P 500 P/E 建立透明的風險分層，"
        "再將總額拆分為美股、基金與定存現金。資料每小時更新。"
    )

    try:
        with st.spinner("正在讀取總體市場數據…"):
            macro = fetch_macro_allocation_data()
    except Exception as exc:
        st.error(f"總體市場數據讀取失敗：{exc}")
        st.info("請稍後重新整理；配置建議不會在缺少關鍵數據時自行猜測。")
        return

    st.caption(
        f"⏱ 宏觀資料載入：{macro['loaded_at']}｜分析完成：{beijing_timestamp()}"
    )
    risk_level, score, reasons = classify_market_risk(macro)
    risk_colors = {"Risk-on": "#3FB950", "Neutral": "#D29922", "Risk-off": "#FF7B72"}
    risk_labels = {
        "Risk-on": "🟢 Risk-on｜風險偏好",
        "Neutral": "🟡 Neutral｜中性觀察",
        "Risk-off": "🔴 Risk-off｜防守優先",
    }

    metric_cols = st.columns(3)
    metric_cols[0].metric("10Y Treasury Yield", f"{macro['treasury_10y']:.2f}%")
    metric_cols[1].metric("VIX", f"{macro['vix']:.2f}")
    metric_cols[2].metric(
        "S&P 500 P/E",
        f"{macro['sp500_pe']:.2f}" if macro.get("sp500_pe") is not None else "N/A",
    )
    st.markdown(
        f"<div style='border:1px solid {risk_colors[risk_level]}; "
        f"background:#1C2128; border-radius:8px; padding:14px 18px; "
        f"margin:12px 0;'><b style='color:{risk_colors[risk_level]}; "
        f"font-size:18px;'>{risk_labels[risk_level]}</b>"
        f"<span style='color:#8B949E; margin-left:12px;'>訊號分數 {score:+d}</span></div>",
        unsafe_allow_html=True,
    )
    for reason in reasons:
        st.caption(f"• {reason}")

    st.markdown("---")
    amount = st.number_input(
        "總投資金額（USD）",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
        format="%.2f",
        key="allocation_amount",
    )
    allocation_df = _allocation_table(risk_level, amount)
    st.subheader(f"💰 {risk_level} 動態配置建議")
    st.dataframe(
        allocation_df.style.format({"建議金額 (USD)": "${:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = allocation_df.copy()
    chart_df["比例數值"] = [item["比例"] for item in ALLOCATION_PLANS[risk_level]]
    fig = px.pie(
        chart_df,
        values="比例數值",
        names="資產類別",
        hole=0.55,
        title="建議資產比例",
        color="資產類別",
        color_discrete_map={
            "美股": "#00D4FF",
            "基金": "#FFD700",
            "定存現金": "#3FB950",
        },
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        height=340,
        margin=dict(l=0, r=0, t=55, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔎 建議標的快速診斷")
    st.caption("點擊任一標的會寫入 selected_ticker，並切換至既有個股診斷／7-Factor。")
    seen: set[str] = set()
    for item in ALLOCATION_PLANS[risk_level]:
        tickers = [ticker.strip() for ticker in item["建議標的"].split("/")]
        cols = st.columns(len(tickers))
        for col, ticker in zip(cols, tickers):
            if ticker in seen:
                continue
            seen.add(ticker)
            with col:
                st.markdown(f"**{ticker}**")
                if st.button("帶入個股診斷", key=f"allocation_diag_{ticker}"):
                    navigate_to_ticker(ticker)

    st.caption(
        "⚠️ 以上是規則化研究框架，不是個人化投資建議；實際配置仍需考慮"
        "投資期限、稅務、流動性與風險承受度。"
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="資產配置｜美股選股儀表板",
        page_icon="🧭",
        layout="wide",
    )
    render_portfolio_allocation_page()

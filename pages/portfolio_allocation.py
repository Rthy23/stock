"""Macro-aware defensive portfolio allocation page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data_fetcher import beijing_timestamp
from navigation import navigate_to_ticker


def _extract_close_frame(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalise yfinance's MultiIndex/single-level Close response."""
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        close = None
        for level in range(downloaded.columns.nlevels):
            if "Close" in downloaded.columns.get_level_values(level):
                close = downloaded.xs("Close", level=level, axis=1)
                break
        if close is None:
            return pd.DataFrame()
    elif "Close" in downloaded.columns:
        close = downloaded[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        return pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.reindex(columns=[ticker for ticker in tickers if ticker in close.columns])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_indicators() -> dict[str, Any]:
    """Fetch ^TNX, ^VIX and SPY once per hour for allocation decisions."""
    tickers = ["^TNX", "^VIX", "SPY"]
    downloaded = yf.download(
        tickers=tickers,
        period="1y",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    close = _extract_close_frame(downloaded, tickers)
    required = [ticker for ticker in tickers if ticker not in close.columns]
    if required:
        raise RuntimeError(f"宏觀市場資料不完整：{', '.join(required)}")
    close = close.ffill().dropna(subset=tickers)
    if len(close) < 200:
        missing = ", ".join(required) if required else "SPY 200MA 歷史資料不足"
        raise RuntimeError(f"宏觀市場資料不完整：{missing}")

    latest = close.iloc[-1]
    spy_ma200 = close["SPY"].rolling(200).mean().iloc[-1]
    if pd.isna(spy_ma200):
        raise RuntimeError("SPY 200MA 尚未有足夠的交易日資料。")
    return {
        "tnx": round(float(latest["^TNX"]), 2),
        "vix": round(float(latest["^VIX"]), 2),
        "spy_price": round(float(latest["SPY"]), 2),
        "spy_ma200": round(float(spy_ma200), 2),
        "spy_above_ma200": bool(latest["SPY"] > spy_ma200),
        "loaded_at": beijing_timestamp(),
    }


# Backward-compatible name for callers that used the earlier module API.
fetch_macro_allocation_data = fetch_macro_indicators


def classify_market_risk(macro: dict[str, Any]) -> tuple[str, int, list[str]]:
    """Return a compact risk summary using the current macro inputs."""
    score = 0
    reasons: list[str] = []
    vix = float(macro["vix"])
    spy_bull = bool(macro["spy_above_ma200"])
    if vix < 15:
        score += 1
        reasons.append("VIX 低於 15，波動風險偏低")
    elif vix > 25:
        score -= 1
        reasons.append("VIX 高於 25，避險需求升溫")
    else:
        reasons.append("VIX 位於中性區間")
    if spy_bull:
        score += 1
        reasons.append("SPY 位於 200MA 之上")
    else:
        score -= 1
        reasons.append("SPY 位於 200MA 之下")
    level = "Risk-on" if score >= 2 else "Risk-off" if score <= -1 else "Neutral"
    return level, score, reasons


def calculate_allocation(
    macro: dict[str, Any],
    user_risk_profile: str | None = None,
) -> tuple[str, str, int, int, int, int]:
    """Calculate four-asset percentages from VIX and SPY 200MA.

    ``user_risk_profile`` is retained as an ignored compatibility argument
    for callers from the earlier three-asset version.
    """
    vix = float(macro["vix"])
    spy_bull = bool(macro["spy_above_ma200"])

    if vix > 25:
        regime = "高風險恐慌期 (High Risk / Offense Guard)"
        description = (
            "市場恐慌指數偏高，建議提高美元定存與短債比重；"
            "全球國際市場以高股息與防守型標的為主。"
        )
        cash_pct, us_etf_pct, global_pct, stocks_pct = 45, 30, 15, 10
    elif not spy_bull:
        regime = "空頭／震盪調整期 (Bear / Correction Market)"
        description = (
            "SPY 跌破 200MA，適度分散至估值較低的歐洲與日本市場，"
            "同時降低高波動個股。"
        )
        cash_pct, us_etf_pct, global_pct, stocks_pct = 35, 35, 15, 15
    elif vix < 15 and spy_bull:
        regime = "多頭牛市期 (Bullish Risk-On)"
        description = (
            "全球風險偏好回升，提高美股個股與全球半導體／AI ADR，"
            "例如台積電 TSM 的配置。"
        )
        cash_pct, us_etf_pct, global_pct, stocks_pct = 10, 40, 20, 30
    else:
        regime = "溫和震盪期 (Neutral / Balanced Market)"
        description = (
            "市場處於均衡狀態，維持 15% 國際市場比重，"
            "達成地理與幣別分散投資。"
        )
        cash_pct, us_etf_pct, global_pct, stocks_pct = 20, 40, 15, 25

    return regime, description, cash_pct, us_etf_pct, global_pct, stocks_pct


ALLOCATION_PLANS: dict[str, list[dict[str, Any]]] = {
    "高風險恐慌期": [
        {"資產類別": "美元定存／短債", "比例": 0.45, "建議標的": "SGOV / BIL", "說明": "提高流動性與防守緩衝"},
        {"資產類別": "美股核心 ETF", "比例": 0.30, "建議標的": "VOO / USMV", "說明": "保留美股核心市場參與"},
        {"資產類別": "全球國際市場 (Global ADRs/ETFs)", "比例": 0.15, "建議標的": "ASML / NVO / EWJ / EWT", "說明": "以防守型海外市場分散風險"},
        {"資產類別": "精選美股個股", "比例": 0.10, "建議標的": "MSFT / BRK-B", "說明": "僅保留高品質核心個股"},
    ],
    "空頭／震盪調整期": [
        {"資產類別": "美元定存／短債", "比例": 0.35, "建議標的": "SGOV / BIL", "說明": "等待趨勢重新站回 200MA"},
        {"資產類別": "美股核心 ETF", "比例": 0.35, "建議標的": "VOO / SPLV", "說明": "以分散與低波動為核心"},
        {"資產類別": "全球國際市場 (Global ADRs/ETFs)", "比例": 0.15, "建議標的": "ASML / NVO / EWJ / EWT", "說明": "分散美股單一市場風險"},
        {"資產類別": "精選美股個股", "比例": 0.15, "建議標的": "JNJ / PG", "說明": "降低個股波動與集中風險"},
    ],
    "多頭牛市期": [
        {"資產類別": "美元定存／短債", "比例": 0.10, "建議標的": "SGOV / BIL", "說明": "維持最低流動性緩衝"},
        {"資產類別": "美股核心 ETF", "比例": 0.40, "建議標的": "VOO / QQQ / XLK", "說明": "參與美股指數與成長趨勢"},
        {"資產類別": "全球國際市場 (Global ADRs/ETFs)", "比例": 0.20, "建議標的": "TSM / ASML / NVO / EWJ / EWT", "說明": "提高全球半導體與 AI 供應鏈配置"},
        {"資產類別": "精選美股個股", "比例": 0.30, "建議標的": "NVDA / MSFT", "說明": "搭配 7-Factor 尋找超額報酬"},
    ],
    "溫和震盪期": [
        {"資產類別": "美元定存／短債", "比例": 0.20, "建議標的": "SGOV / BIL", "說明": "保留再平衡資金"},
        {"資產類別": "美股核心 ETF", "比例": 0.40, "建議標的": "VOO / VIG", "說明": "以分散與品質為核心"},
        {"資產類別": "全球國際市場 (Global ADRs/ETFs)", "比例": 0.15, "建議標的": "TSM / ASML / NVO / EWJ / EWT", "說明": "維持地理與幣別分散"},
        {"資產類別": "精選美股個股", "比例": 0.25, "建議標的": "AAPL / GOOGL", "說明": "適度配置高品質美股龍頭"},
    ],
}


def _allocation_table(plan: list[dict[str, Any]], amount: float) -> pd.DataFrame:
    rows = []
    for item in plan:
        rows.append(
            {
                **item,
                "比例": f"{item['比例']:.0%}",
                "比例數值": item["比例"],
                "建議金額 (USD)": amount * item["比例"],
            }
        )
    return pd.DataFrame(rows)


def render_portfolio_allocation_page() -> None:
    st.title("🧭 穩健型資產配置策略（含全球國際市場）")
    st.caption(
        "以 10Y Treasury Yield、VIX 與 SPY 200MA 建立透明的市場判定，"
        "再將資金動態拆分為美元定存／短債、美股核心 ETF、全球國際市場與精選美股個股。"
        "資料每小時更新。"
    )

    try:
        with st.spinner("正在讀取總體市場數據…"):
            macro = fetch_macro_indicators()
    except Exception as exc:
        st.error(f"總體市場數據讀取失敗：{exc}")
        st.info("請稍後重新整理；配置建議不會在缺少關鍵數據時自行猜測。")
        return

    st.caption(
        f"⏱ 宏觀資料載入：{macro['loaded_at']}｜分析完成：{beijing_timestamp()}"
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("10年期美債收益率", f"{macro['tnx']:.2f}%")
    metric_cols[1].metric("VIX 恐慌指數", f"{macro['vix']:.2f}")
    metric_cols[2].metric("SPY 現價", f"${macro['spy_price']:.2f}")
    metric_cols[3].metric(
        "SPY 200MA 趨勢",
        "多頭" if macro["spy_above_ma200"] else "空頭／修正",
        delta=f"${macro['spy_price'] - macro['spy_ma200']:+.2f} vs 200MA",
    )

    regime, description, cash_pct, us_etf_pct, global_pct, stocks_pct = (
        calculate_allocation(macro)
    )
    allocation_plan = [
        {
            "資產類別": "美元定存／短債",
            "比例": cash_pct / 100,
            "建議標的": "SGOV / BIL",
            "說明": "保本、流動性與等待再平衡機會。",
        },
        {
            "資產類別": "美股核心 ETF",
            "比例": us_etf_pct / 100,
            "建議標的": "VOO / QQQ / XLK",
            "說明": "以美股大盤、納斯達克與科技板塊建立核心。",
        },
        {
            "資產類別": "全球國際市場 (Global ADRs/ETFs)",
            "比例": global_pct / 100,
            "建議標的": "TSM / ASML / NVO / EWJ / EWT",
            "說明": "依市場狀態配置 15%～20%，分散地理與幣別風險。",
        },
        {
            "資產類別": "精選美股個股",
            "比例": stocks_pct / 100,
            "建議標的": "NVDA / MSFT / GOOGL",
            "說明": "以既有 7-Factor 分析驗證個股品質與動能。",
        },
    ]
    st.markdown(
        f"<div style='border:1px solid #D29922; background:#1C2128; "
        f"border-radius:8px; padding:14px 18px; margin:12px 0;'>"
        f"<b style='color:#FFD700; font-size:18px;'>📊 當前市場判定：{regime}</b>"
        f"<span style='color:#8B949E; margin-left:12px;'>"
        f"{'SPY 位於 200MA 之上' if macro['spy_above_ma200'] else 'SPY 位於 200MA 之下'}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.info(description)

    st.markdown("---")
    amount = st.number_input(
        "預計投入總金額（USD）",
        min_value=1000.0,
        value=50000.0,
        step=5000.0,
        format="%.2f",
        key="allocation_amount",
    )
    allocation_df = _allocation_table(allocation_plan, amount)
    st.subheader("🎯 四大資產類別具體分配")
    st.dataframe(
        allocation_df.drop(columns=["比例數值"]).style.format(
            {"建議金額 (USD)": "${:,.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.pie(
        allocation_df,
        values="建議金額 (USD)",
        names="資產類別",
        hole=0.55,
        title=f"{regime}｜四大資產配置比例",
        color="資產類別",
        color_discrete_map={
            "美元定存／短債": "#3FB950",
            "美股核心 ETF": "#3498DB",
            "全球國際市場 (Global ADRs/ETFs)": "#9B59B6",
            "精選美股個股": "#E74C3C",
        },
    )
    fig.update_traces(
        texttemplate="%{label}<br>%{percent}",
        hovertemplate="%{label}<br>$%{value:,.2f}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        height=380,
        margin=dict(l=0, r=0, t=55, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔎 建議標的快速診斷")
    st.caption(
        "Global 類別包含 ADR 與國際 ETF；點擊任一標的會寫入 selected_ticker，"
        "並切換至既有個股診斷與 7-Factor 分析。"
    )
    seen: set[str] = set()
    for item in allocation_plan:
        st.markdown(
            f"**{item['資產類別']}｜{item['比例']:.0%}｜"
            f"${amount * item['比例']:,.2f} USD**　"
            f"建議：`{item['建議標的']}`　{item['說明']}"
        )
        tickers = [ticker.strip() for ticker in item["建議標的"].split("/")]
        cols = st.columns(len(tickers))
        for col, ticker in zip(cols, tickers):
            if ticker in seen:
                continue
            seen.add(ticker)
            with col:
                if st.button(
                    f"診斷 {ticker}",
                    key=f"allocation_diag_{ticker}",
                    use_container_width=True,
                ):
                    navigate_to_ticker(ticker)

    st.caption(
        "⚠️ 以上是規則化研究框架，不是個人化投資建議；實際配置仍需考慮"
        "投資期限、稅務、流動性與風險承受度。"
    )


# Compatibility entry point matching the uploaded page draft.
def render_portfolio_page() -> None:
    render_portfolio_allocation_page()


if __name__ == "__main__":
    st.set_page_config(
        page_title="資產配置｜美股選股儀表板",
        page_icon="🧭",
        layout="wide",
    )
    render_portfolio_page()

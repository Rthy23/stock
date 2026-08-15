"""Curated-whitelist and Wall Street analyst recommendation consensus page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data_fetcher import beijing_timestamp
from kol_config import ANALYST_DIRECTORY
from kol_whitelist import PICKS_DATA, build_consensus_table
from navigation import navigate_to_ticker


RATING_WEIGHTS: dict[str, int] = {
    "strongBuy": 2,
    "buy": 1,
    "hold": 0,
    "sell": -1,
    "strongSell": -2,
}

DEFAULT_RECOMMENDATION_UNIVERSE: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AVGO",
    "AMD",
    "LLY",
    "JPM",
    "V",
    "TSM",
    "ASML",
    "NVO",
    "TM",
    "SONY",
    "BABA",
    "PDD",
    "CPNG",
    "SPY",
    "QQQ",
    "VOO",
)


def _normalise_rating_column(column: Any) -> str:
    return str(column).replace("_", "").replace(" ", "").lower()


def _latest_rating_counts(recommendations: pd.DataFrame | None) -> dict[str, int]:
    """Extract the latest aggregate rating counts from yfinance."""
    if recommendations is None or recommendations.empty:
        return {}
    table = recommendations.copy()
    if not isinstance(table.index, pd.RangeIndex):
        table = table.reset_index()

    column_map = {
        _normalise_rating_column(column): column for column in table.columns
    }
    required = {
        rating: column_map.get(_normalise_rating_column(rating))
        for rating in RATING_WEIGHTS
    }
    if not all(required.values()):
        return {}

    period_column = next(
        (
            column
            for column in table.columns
            if str(column).lower() in {"period", "date", "datetime"}
        ),
        None,
    )
    if period_column:
        table["_period_sort"] = pd.to_datetime(table[period_column], errors="coerce")
        table = table.sort_values("_period_sort", na_position="first")
    latest = table.iloc[-1]
    counts: dict[str, int] = {}
    for rating, column in required.items():
        value = pd.to_numeric(latest[column], errors="coerce")
        counts[rating] = 0 if pd.isna(value) else max(0, int(float(value)))
    return counts


def _rating_label(score: float) -> str:
    if score >= 1.25:
        return "Strong Buy"
    if score >= 0.5:
        return "Buy"
    if score > -0.5:
        return "Hold"
    if score > -1.25:
        return "Sell"
    return "Strong Sell"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_recommendations(
    tickers: tuple[str, ...],
) -> tuple[pd.DataFrame, str]:
    """Fetch latest all-covered-analyst counts and target prices per ticker."""
    rows: list[dict[str, Any]] = []
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            counts = _latest_rating_counts(recommendations)
            if not counts:
                continue
            total = sum(counts.values())
            if total <= 0:
                continue

            rating_score = sum(
                counts[rating] * weight
                for rating, weight in RATING_WEIGHTS.items()
            ) / total
            info = stock.info or {}
            current_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            target_mean = info.get("targetMeanPrice")
            upside = (
                (float(target_mean) - float(current_price))
                / float(current_price)
                * 100
                if target_mean and current_price and float(current_price) > 0
                else None
            )
            rows.append(
                {
                    "Ticker": ticker,
                    "評級加權分數": round(float(rating_score), 3),
                    "加權結論": _rating_label(float(rating_score)),
                    "分析師數": total,
                    "Strong Buy": counts["strongBuy"],
                    "Buy": counts["buy"],
                    "Hold": counts["hold"],
                    "Sell": counts["sell"],
                    "Strong Sell": counts["strongSell"],
                    "現價": float(current_price) if current_price else None,
                    "目標均價": float(target_mean) if target_mean else None,
                    "潛在漲幅 (%)": round(float(upside), 2) if upside is not None else None,
                }
            )
        except Exception:
            # A single unavailable ticker should not hide other broker coverage.
            continue

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["評級加權分數", "潛在漲幅 (%)"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
    return result, beijing_timestamp()


def _render_diagnosis_buttons(tickers: list[str], key_prefix: str) -> None:
    if not tickers:
        return
    columns = st.columns(min(4, len(tickers)))
    for column, ticker in zip(columns, tickers):
        with column:
            if st.button(
                f"診斷 {ticker}",
                key=f"{key_prefix}_{ticker}",
                use_container_width=True,
            ):
                navigate_to_ticker(ticker)


def _render_curated_consensus() -> None:
    st.subheader("⭐ 精選分析師白名單共識")
    st.caption(
        f"目前依 {len(ANALYST_DIRECTORY)} 位精選分析師的推薦，"
        "使用信譽 × 論點品質 × 時效性加權；資料來源與既有總體市場共識模組同步。"
    )
    ranked = build_consensus_table(whitelist=ANALYST_DIRECTORY)
    if not ranked:
        st.info("目前沒有可用的精選分析師推薦資料。")
        return

    all_rows = [
        {
            "排名": index,
            "Ticker": pick["ticker"],
            "加權分數": round(float(pick["total_score"]), 3),
            "推薦專家數": pick["consensus"],
            "推薦專家": "、".join(dict.fromkeys(pick["experts"])),
        }
        for index, pick in enumerate(ranked, 1)
    ]
    top_rows = all_rows[:15]
    rest_rows = all_rows[15:]

    curated_df = pd.DataFrame(top_rows)
    st.dataframe(curated_df, use_container_width=True, hide_index=True)

    if rest_rows:
        with st.expander(f"顯示其餘 {len(rest_rows)} 支標的（共 {len(all_rows)} 支）"):
            st.dataframe(pd.DataFrame(rest_rows), use_container_width=True, hide_index=True)

    all_tickers = [r["Ticker"] for r in all_rows]
    _render_diagnosis_buttons(
        all_tickers[:8],
        "curated_consensus_diag",
    )


def render_analyst_consensus_page() -> None:
    st.title("🏦 分析師共識與 Recommendations 排行榜")
    st.caption(
        "上方為精選分析師白名單共識；下方為 Yahoo Finance 對每支標的所彙整的"
        "華爾街涵蓋券商 Recommendations，使用 Strong Buy 到 Strong Sell 的加權平均評分。"
    )

    _render_curated_consensus()
    st.markdown("---")
    st.subheader("🌎 全市場券商 Recommendations")
    st.caption(
        "評級權重：Strong Buy = +2、Buy = +1、Hold = 0、Sell = -1、"
        "Strong Sell = -2。每支標的使用 Yahoo Finance 最新彙總期的全部可用評級。"
    )

    custom_tickers = st.text_input(
        "加入要查詢的 Ticker（可用逗號或空格分隔）",
        placeholder="例如：TSM, ASML, NVO, NVDA",
        key="consensus_custom_tickers",
    )
    extra = tuple(
        ticker.strip().upper()
        for ticker in custom_tickers.replace(",", " ").split()
        if ticker.strip()
    )
    tickers = tuple(dict.fromkeys(DEFAULT_RECOMMENDATION_UNIVERSE + extra))
    st.caption(f"目前查詢 {len(tickers)} 支美股／ADR／ETF；評級資料每小時更新。")

    with st.spinner("正在讀取全市場分析師 Recommendations…"):
        recommendations_df, loaded_at = fetch_market_recommendations(tickers)
    st.caption(f"Recommendations 資料載入：{loaded_at}｜分析完成：{beijing_timestamp()}")

    if recommendations_df.empty:
        st.warning("目前沒有可用的 Recommendations 彙總資料，請稍後重試或加入其他 Ticker。")
        return

    score_chart = px.bar(
        recommendations_df.head(15).sort_values("評級加權分數"),
        x="評級加權分數",
        y="Ticker",
        orientation="h",
        color="評級加權分數",
        color_continuous_scale="RdYlGn",
        title="全市場分析師評級加權排行榜",
        labels={"評級加權分數": "加權平均（-2 到 +2）", "Ticker": ""},
    )
    score_chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        height=480,
        margin=dict(l=0, r=10, t=55, b=20),
        coloraxis_colorbar={"title": "評分"},
    )
    st.plotly_chart(score_chart, use_container_width=True)

    upside_df = recommendations_df.dropna(subset=["潛在漲幅 (%)"]).copy()
    if not upside_df.empty:
        upside_chart = px.scatter(
            upside_df,
            x="評級加權分數",
            y="潛在漲幅 (%)",
            text="Ticker",
            color="加權結論",
            hover_data=["分析師數", "目標均價", "現價"],
            title="評級分數與目標價潛在漲幅",
            labels={
                "評級加權分數": "評級加權平均",
                "潛在漲幅 (%)": "潛在漲幅 (%)",
            },
            color_discrete_map={
                "Strong Buy": "#00C853",
                "Buy": "#7CB342",
                "Hold": "#F9A825",
                "Sell": "#FB8C00",
                "Strong Sell": "#E53935",
            },
        )
        upside_chart.update_traces(textposition="top center")
        upside_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            height=430,
            margin=dict(l=0, r=10, t=55, b=20),
        )
        st.plotly_chart(upside_chart, use_container_width=True)

    display_columns = [
        "Ticker",
        "加權結論",
        "評級加權分數",
        "分析師數",
        "Strong Buy",
        "Buy",
        "Hold",
        "Sell",
        "Strong Sell",
        "現價",
        "目標均價",
        "潛在漲幅 (%)",
    ]
    st.dataframe(
        recommendations_df[display_columns].style.format(
            {
                "評級加權分數": "{:+.3f}",
                "現價": "${:,.2f}",
                "目標均價": "${:,.2f}",
                "潛在漲幅 (%)": "{:+.2f}%",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
    )
    _render_diagnosis_buttons(
        recommendations_df["Ticker"].head(12).tolist(),
        "market_consensus_diag",
    )
    st.caption(
        "⚠️ Yahoo Finance 的 Recommendations 為各標的可取得的券商覆蓋彙總，"
        "不代表所有華爾街機構的完整名單，也不構成投資建議。"
    )


# Compatibility entry point for direct Streamlit page execution.
def render_analyst_page() -> None:
    render_analyst_consensus_page()


if __name__ == "__main__":
    st.set_page_config(
        page_title="分析師共識｜美股選股儀表板",
        page_icon="🏦",
        layout="wide",
    )
    render_analyst_consensus_page()
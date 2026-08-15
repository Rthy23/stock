"""US equity sector rotation analysis page."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data_fetcher import beijing_timestamp
from navigation import navigate_to_ticker


SECTOR_DATA: dict[str, dict] = {
    "資訊科技 (Technology)": {
        "etf": "XLK",
        "top_etfs": ["XLK", "SOXX", "SMH"],
        "top_stocks": ["NVDA", "AAPL", "MSFT", "AVGO"],
    },
    "金融 (Financials)": {
        "etf": "XLF",
        "top_etfs": ["XLF", "VFH", "KBE"],
        "top_stocks": ["JPM", "BRK-B", "V", "MA"],
    },
    "醫療保健 (Health Care)": {
        "etf": "XLV",
        "top_etfs": ["XLV", "VHT", "IHI"],
        "top_stocks": ["LLY", "UNH", "JNJ", "ABBV"],
    },
    "能源 (Energy)": {
        "etf": "XLE",
        "top_etfs": ["XLE", "VDE", "IYE"],
        "top_stocks": ["XOM", "CVX", "COP", "SLB"],
    },
    "非必需消費 (Consumer Disc.)": {
        "etf": "XLY",
        "top_etfs": ["XLY", "VCR", "FDIS"],
        "top_stocks": ["AMZN", "TSLA", "HD", "NKE"],
    },
    "必需消費 (Consumer Staples)": {
        "etf": "XLP",
        "top_etfs": ["XLP", "VDC", "FSTA"],
        "top_stocks": ["PG", "COST", "PEP", "KO"],
    },
    "工業 (Industrials)": {
        "etf": "XLI",
        "top_etfs": ["XLI", "VIS", "FIDU"],
        "top_stocks": ["GE", "CAT", "HON", "UNP"],
    },
    "原材料 (Materials)": {
        "etf": "XLB",
        "top_etfs": ["XLB", "VAW", "FMAT"],
        "top_stocks": ["LIN", "APD", "SHW", "FCX"],
    },
    "公用事業 (Utilities)": {
        "etf": "XLU",
        "top_etfs": ["XLU", "VPU", "FUTY"],
        "top_stocks": ["NEE", "SO", "DUK", "CEG"],
    },
    "房地產 (Real Estate)": {
        "etf": "XLRE",
        "top_etfs": ["XLRE", "VNQ", "IYR"],
        "top_stocks": ["PLD", "AMT", "EQIX", "WELL"],
    },
    "通訊服務 (Communication)": {
        "etf": "XLC",
        "top_etfs": ["XLC", "VOX", "FCOM"],
        "top_stocks": ["GOOGL", "META", "NFLX", "TMUS"],
    },
}

SECTOR_ETFS = [data["etf"] for data in SECTOR_DATA.values()]


def _close_frame(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalise yfinance's single- and multi-ticker Close response."""
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        for level in range(downloaded.columns.nlevels):
            if "Close" in downloaded.columns.get_level_values(level):
                close = downloaded.xs("Close", level=level, axis=1)
                break
        else:
            return pd.DataFrame()
    elif "Close" in downloaded.columns:
        close = downloaded[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        return pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.reindex(columns=[t for t in tickers if t in close.columns]).dropna(
        how="all"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_performance() -> tuple[pd.DataFrame, float | None, str]:
    """Fetch sector ETF and SPY prices once per hour and calculate returns."""
    tickers = SECTOR_ETFS + ["SPY"]
    downloaded = yf.download(
        tickers=tickers,
        period="1y",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    prices = _close_frame(downloaded, tickers)
    if prices.empty or len(prices) < 22:
        raise RuntimeError("Yahoo Finance 未返回足夠的板塊歷史價格資料。")

    prices = prices.ffill()
    latest = prices.iloc[-1]

    def _return(days: int) -> pd.Series:
        base = prices.iloc[max(0, len(prices) - 1 - days)]
        return (latest / base - 1) * 100

    m1, m3, y1 = _return(21), _return(63), _return(len(prices) - 1)
    rows = []
    for sector_name, info in SECTOR_DATA.items():
        etf = info["etf"]
        if etf not in prices.columns:
            continue
        rows.append(
            {
                "板塊名稱": sector_name,
                "ETF": etf,
                "近1個月 (%)": round(float(m1.get(etf, float("nan"))), 2),
                "近3個月 (%)": round(float(m3.get(etf, float("nan"))), 2),
                "近1年 (%)": round(float(y1.get(etf, float("nan"))), 2),
            }
        )
    if not rows:
        raise RuntimeError("板塊 ETF 價格資料不完整，暫時無法計算排行榜。")
    spy_m1 = float(m1["SPY"]) if "SPY" in m1 and pd.notna(m1["SPY"]) else None
    return pd.DataFrame(rows), spy_m1, beijing_timestamp()


def _render_diagnosis_button(ticker: str, key: str) -> None:
    if st.button("診斷", key=key, use_container_width=True):
        navigate_to_ticker(ticker)


def render_sector_analysis_page() -> None:
    st.title("📊 美股熱門行業板塊分析")
    st.caption(
        "追蹤 11 大 S&P 500 板塊 ETF，從 1M／3M／1Y 報酬率觀察資金輪動。"
        " 價格資料每小時更新一次。"
    )

    try:
        with st.spinner("正在讀取最新板塊數據…"):
            perf_df, spy_m1, loaded_at = fetch_sector_performance()
    except Exception as exc:
        st.error(f"板塊數據讀取失敗：{exc}")
        st.info("可稍後重新整理；Yahoo Finance 暫時限流時不會寫入錯誤快取。")
        return

    st.caption(
        f"⏱ 板塊資料載入：{loaded_at}｜分析完成：{beijing_timestamp()}"
    )
    periods = {
        "近 1 個月": "近1個月 (%)",
        "近 3 個月": "近3個月 (%)",
        "近 1 年": "近1年 (%)",
    }
    selected_period = st.radio(
        "選擇排行榜時間範圍",
        list(periods),
        horizontal=True,
        key="sector_period",
    )
    metric_col = periods[selected_period]
    sorted_df = perf_df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    fig = px.bar(
        sorted_df,
        x=metric_col,
        y="板塊名稱",
        orientation="h",
        color=metric_col,
        color_continuous_scale="RdYlGn",
        title=f"美股板塊報酬率排行榜（{selected_period}）",
        labels={metric_col: "報酬率 (%)", "板塊名稱": ""},
    )
    fig.update_layout(
        height=460,
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        yaxis={"categoryorder": "total ascending"},
        coloraxis_colorbar={"title": "%"},
        margin=dict(l=0, r=10, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔥 當前熱門板塊與精選標的")
    if spy_m1 is not None:
        st.caption(f"SPY 近 1 個月報酬：{spy_m1:+.2f}%（熱門板塊卡片以此作為短期基準）")

    top_3 = sorted_df.head(3)
    card_cols = st.columns(len(top_3))
    for idx, (_, row) in enumerate(top_3.iterrows()):
        data = SECTOR_DATA[row["板塊名稱"]]
        with card_cols[idx]:
            delta = (
                row["近1個月 (%)"] - spy_m1
                if spy_m1 is not None
                else None
            )
            st.success(f"**第 {idx + 1} 名：{row['板塊名稱']}**")
            st.metric(
                "近 1 個月",
                f"{row['近1個月 (%)']:+.2f}%",
                delta=f"{delta:+.2f}% vs SPY" if delta is not None else None,
            )
            st.caption(
                f"3M {row['近3個月 (%)']:+.2f}%　｜　"
                f"1Y {row['近1年 (%)']:+.2f}%"
            )
            st.markdown("**代表性 ETF**")
            for etf in data["top_etfs"]:
                st.code(etf, language=None)
            st.markdown("**代表性龍頭個股**")
            for stock in data["top_stocks"]:
                stock_col, button_col = st.columns([2, 1])
                stock_col.markdown(f"📈 **{stock}**")
                _render_diagnosis_button(
                    stock, key=f"sector_diag_{row['ETF']}_{stock}"
                )

    st.markdown("---")
    st.subheader("📋 11 大板塊完整數據")
    st.dataframe(
        sorted_df.style.format(
            {
                "近1個月 (%)": "{:+.2f}%",
                "近3個月 (%)": "{:+.2f}%",
                "近1年 (%)": "{:+.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "⚠️ 代表性標的僅供研究起點，不構成投資建議；點擊「診斷」會共用 "
        "selected_ticker 並切換至既有個股診斷與 7-Factor 分析。"
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="板塊分析｜美股選股儀表板",
        page_icon="📊",
        layout="wide",
    )
    render_sector_analysis_page()

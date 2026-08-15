"""US equity sector rotation analysis page."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data_fetcher import (
    beijing_timestamp,
    TIME_RANGE_OPTIONS, PERIOD_LABELS, _yf_period_params,
)
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
def fetch_sector_performance(period: str = "1y") -> tuple[pd.DataFrame, float | None, str]:
    """Fetch sector ETF and SPY prices for *period* and compute total return.

    *period* must be a value from TIME_RANGE_OPTIONS (e.g. "1y", "3mo", "3y").
    Returns (df, spy_return, timestamp) where df has columns:
        板塊名稱 | ETF | 報酬率 (%)
    """
    tickers = SECTOR_ETFS + ["SPY"]
    dl_kwargs = _yf_period_params(period)
    downloaded = yf.download(
        tickers=tickers,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
        **dl_kwargs,
    )
    prices = _close_frame(downloaded, tickers)
    if prices.empty or len(prices) < 2:
        raise RuntimeError("Yahoo Finance 未返回足夠的板塊歷史價格資料。")

    prices = prices.ffill().dropna(how="all")
    base   = prices.iloc[0]
    latest = prices.iloc[-1]

    rows = []
    for sector_name, info in SECTOR_DATA.items():
        etf = info["etf"]
        if etf not in prices.columns:
            continue
        b, l = float(base[etf]), float(latest[etf])
        ret  = round((l / b - 1) * 100, 2) if b != 0 else float("nan")
        rows.append({"板塊名稱": sector_name, "ETF": etf, "報酬率 (%)": ret})

    if not rows:
        raise RuntimeError("板塊 ETF 價格資料不完整，暫時無法計算排行榜。")

    spy_return: float | None = None
    if "SPY" in prices.columns:
        sb, sl = float(base["SPY"]), float(latest["SPY"])
        if sb != 0 and pd.notna(sl):
            spy_return = round((sl / sb - 1) * 100, 2)

    return pd.DataFrame(rows), spy_return, beijing_timestamp()


def _render_diagnosis_button(ticker: str, key: str) -> None:
    if st.button("診斷", key=key, use_container_width=True):
        navigate_to_ticker(ticker)


def render_sector_analysis_page() -> None:
    st.title("📊 美股熱門行業板塊分析")
    st.caption(
        "追蹤 11 大 S&P 500 板塊 ETF，從 1M／3M／1Y 報酬率觀察資金輪動。"
        " 價格資料每小時更新一次。"
    )

    selected_label = st.radio(
        "選擇排行榜時間範圍",
        PERIOD_LABELS,
        index=PERIOD_LABELS.index("1年"),
        horizontal=True,
        key="sector_period",
    )
    yf_period = TIME_RANGE_OPTIONS[selected_label]

    try:
        with st.spinner(f"正在讀取 {selected_label} 板塊數據…"):
            perf_df, spy_return, loaded_at = fetch_sector_performance(yf_period)
    except Exception as exc:
        st.error(f"板塊數據讀取失敗：{exc}")
        st.info("可稍後重新整理；Yahoo Finance 暫時限流時不會寫入錯誤快取。")
        return

    st.caption(f"⏱ 板塊資料載入：{loaded_at}｜分析完成：{beijing_timestamp()}")

    metric_col = "報酬率 (%)"
    sorted_df = perf_df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    fig = px.bar(
        sorted_df,
        x=metric_col,
        y="板塊名稱",
        orientation="h",
        color=metric_col,
        color_continuous_scale="RdYlGn",
        title=f"美股板塊報酬率排行榜（{selected_label}）",
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
    if spy_return is not None:
        st.caption(f"SPY {selected_label}報酬：{spy_return:+.2f}%（熱門板塊卡片以此作為基準）")

    top_3 = sorted_df.head(3)
    card_cols = st.columns(len(top_3))
    for idx, (_, row) in enumerate(top_3.iterrows()):
        data = SECTOR_DATA[row["板塊名稱"]]
        with card_cols[idx]:
            delta = (
                row["報酬率 (%)"] - spy_return
                if spy_return is not None
                else None
            )
            st.success(f"**第 {idx + 1} 名：{row['板塊名稱']}**")
            st.metric(
                f"期間報酬 ({selected_label})",
                f"{row['報酬率 (%)']:+.2f}%",
                delta=f"{delta:+.2f}% vs SPY" if delta is not None else None,
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
        sorted_df.style.format({"報酬率 (%)": "{:+.2f}%"}),
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

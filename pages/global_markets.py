"""Global market ETF tracking and US-listed ADR watchlist."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from analysis import classify_sentiment
from data_fetcher import beijing_timestamp
from navigation import navigate_to_ticker


GLOBAL_MARKETS: dict[str, dict[str, Any]] = {
    "台灣 (Taiwan)": {
        "etf": "EWT",
        "description": "全球半導體與 AI 供應鏈核心，擁有世界級晶圓代工與封測聚落。",
        "top_etfs": ["EWT（iShares MSCI Taiwan ETF）"],
        "adr_stocks": [
            {"ticker": "TSM", "name": "台積電 (TSMC)", "desc": "全球晶圓代工龍頭，AI 晶片供應鏈核心。"},
            {"ticker": "UMC", "name": "聯電 (UMC)", "desc": "成熟製程晶圓代工大廠。"},
            {"ticker": "ASX", "name": "日月光投控 (ASE)", "desc": "全球半導體封裝與測試服務龍頭。"},
        ],
    },
    "日本 (Japan)": {
        "etf": "EWJ",
        "description": "受惠企業治理改革與資本效率提升，製造業與金融業權重高。",
        "top_etfs": ["EWJ（iShares MSCI Japan ETF）", "DXJ（WisdomTree Japan Hedged）"],
        "adr_stocks": [
            {"ticker": "TM", "name": "豐田汽車 (Toyota)", "desc": "全球銷量領先的汽車製造商。"},
            {"ticker": "SONY", "name": "索尼 (Sony)", "desc": "遊戲、娛樂與影像感測器業務多元。"},
            {"ticker": "MUFG", "name": "三菱日聯金融", "desc": "日本大型金融集團，受惠利率正常化。"},
        ],
    },
    "歐洲 (Europe)": {
        "etf": "VGK",
        "description": "聚焦歐元區大型企業，涵蓋半導體設備、生技醫療與消費品牌。",
        "top_etfs": ["FEZ（SPDR Euro STOXX 50 ETF）", "VGK（Vanguard FTSE Europe）"],
        "adr_stocks": [
            {"ticker": "NVO", "name": "諾和諾德 (Novo Nordisk)", "desc": "糖尿病與代謝疾病治療藥物龍頭。"},
            {"ticker": "ASML", "name": "阿斯麥 (ASML)", "desc": "全球極紫外光刻機（EUV）設備龍頭。"},
            {"ticker": "SAP", "name": "SAP SE", "desc": "歐洲大型企業雲端軟體供應商。"},
        ],
    },
    "中國 (China)": {
        "etf": "MCHI",
        "description": "聚焦中國網路科技、消費出海與新能源汽車等大型企業。",
        "top_etfs": ["MCHI（iShares MSCI China ETF）", "KWEB（KraneShares CSI China Internet）"],
        "adr_stocks": [
            {"ticker": "BABA", "name": "阿里巴巴 (Alibaba)", "desc": "中國電商與雲端運算平台龍頭。"},
            {"ticker": "PDD", "name": "拼多多 (PDD)", "desc": "電商平台，旗下 Temu 拓展海外市場。"},
            {"ticker": "BYDDY", "name": "比亞迪 (BYD)", "desc": "新能源汽車與電池製造商。"},
        ],
    },
    "韓國 (South Korea)": {
        "etf": "EWY",
        "description": "全球記憶體與 HBM 供應鏈重鎮，同時擁有大型消費與電商企業。",
        "top_etfs": ["EWY（iShares MSCI South Korea ETF）"],
        "adr_stocks": [
            {"ticker": "CPNG", "name": "Coupang", "desc": "韓國大型電商與物流平台。"},
            {"ticker": "KB", "name": "KB 金融集團", "desc": "韓國大型銀行與金融服務集團。"},
        ],
    },
    "香港 (Hong Kong)": {
        "etf": "EWH",
        "description": "亞洲金融樞紐，市場結構以金融、保險與地產企業為主。",
        "top_etfs": ["EWH（iShares MSCI Hong Kong ETF）"],
        "adr_stocks": [
            {"ticker": "AAGIY", "name": "友邦保險 (AIA)", "desc": "泛亞區大型壽險集團。"},
            {"ticker": "HKXCY", "name": "港交所 (HKEX)", "desc": "香港證券交易所營運商。"},
        ],
    },
    "全球 (Global)": {
        "etf": "ACWI",
        "description": "以全球股票市場為核心，透過單一 ETF 分散已開發與新興市場風險。",
        "top_etfs": ["ACWI（iShares MSCI ACWI ETF）", "VT（Vanguard Total World Stock ETF）"],
        "adr_stocks": [
            {"ticker": "TSM", "name": "台積電 (TSMC)", "desc": "全球先進製程與 AI 晶片代工核心。"},
            {"ticker": "SONY", "name": "索尼 (Sony)", "desc": "遊戲、娛樂與影像感測器業務多元。"},
            {"ticker": "TM", "name": "豐田汽車 (Toyota)", "desc": "全球汽車製造與混合動力車龍頭。"},
            {"ticker": "ASML", "name": "阿斯麥 (ASML)", "desc": "全球 EUV 半導體設備供應商。"},
            {"ticker": "NVO", "name": "諾和諾德 (Novo Nordisk)", "desc": "代謝疾病治療藥物龍頭。"},
        ],
    },
}

GLOBAL_ETFS = [market["etf"] for market in GLOBAL_MARKETS.values()]


def _close_frame(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalise yfinance Close output for one- and multi-ticker downloads."""
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
    return close.reindex(columns=[ticker for ticker in tickers if ticker in close.columns])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_global_performance() -> tuple[pd.DataFrame, float, str]:
    """Fetch international ETFs and SPY once per hour and calculate returns."""
    tickers = GLOBAL_ETFS + ["SPY"]
    downloaded = yf.download(
        tickers=tickers,
        period="1y",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    prices = _close_frame(downloaded, tickers)
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise RuntimeError(f"國際市場 ETF 資料不完整：{', '.join(missing)}")
    prices = prices.ffill().dropna(subset=tickers)
    if len(prices) < 22:
        raise RuntimeError("Yahoo Finance 未返回足夠的國際市場歷史價格資料。")

    latest = prices.iloc[-1]

    def period_return(days: int) -> pd.Series:
        base = prices.iloc[max(0, len(prices) - 1 - days)]
        return (latest / base - 1) * 100

    month = period_return(21)
    quarter = period_return(63)
    year = period_return(len(prices) - 1)
    rows = []
    for region, info in GLOBAL_MARKETS.items():
        etf = info["etf"]
        rows.append(
            {
                "國家／區域": region,
                "代表 ETF": etf,
                "近 1 個月 (%)": round(float(month[etf]), 2),
                "近 3 個月 (%)": round(float(quarter[etf]), 2),
                "近 1 年 (%)": round(float(year[etf]), 2),
            }
        )
    return pd.DataFrame(rows), round(float(month["SPY"]), 2), beijing_timestamp()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch one selected region ETF's historical Close series."""
    downloaded = yf.download(
        tickers=[ticker],
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    prices = _close_frame(downloaded, [ticker])
    if ticker not in prices.columns or prices[ticker].dropna().empty:
        raise RuntimeError(f"{ticker} 沒有可用的歷史價格資料。")
    history = prices[[ticker]].rename(columns={ticker: "收盤價"}).dropna()
    history.index = pd.to_datetime(history.index)
    history.index.name = "日期"
    return history.reset_index()


def _news_field(item: dict[str, Any], field: str) -> Any:
    """Read both legacy and current yfinance news response shapes."""
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    return item.get(field) or content.get(field)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_news(
    tickers: tuple[str, ...],
) -> tuple[pd.DataFrame, float]:
    """Fetch Yahoo Finance headlines and calculate a keyword sentiment score."""
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            for item in (yf.Ticker(ticker).news or [])[:8]:
                title = str(_news_field(item, "title") or "").strip()
                if not title:
                    continue
                sentiment = classify_sentiment(title)
                score = {"positive": 100.0, "neutral": 50.0, "negative": 0.0}[
                    sentiment
                ]
                publisher = _news_field(item, "publisher") or _news_field(
                    item, "provider"
                )
                published = _news_field(item, "pubDate") or _news_field(
                    item, "providerPublishTime"
                )
                if isinstance(published, (int, float)):
                    published = pd.to_datetime(published, unit="s", errors="coerce")
                else:
                    published = pd.to_datetime(published, errors="coerce")
                rows.append(
                    {
                        "Ticker": ticker,
                        "標題": title,
                        "發布者": str(publisher or "Yahoo Finance"),
                        "發布時間": published,
                        "情緒": {"positive": "正面", "neutral": "中性", "negative": "負面"}[
                            sentiment
                        ],
                        "情緒分數": score,
                        "連結": _news_field(item, "canonicalUrl")
                        or _news_field(item, "link"),
                    }
                )
        except Exception:
            # One unavailable ticker should not hide headlines from the rest.
            continue

    news_df = pd.DataFrame(rows)
    if news_df.empty:
        return news_df, 50.0
    news_df = news_df.sort_values("發布時間", ascending=False, na_position="last")
    return news_df.head(30).reset_index(drop=True), round(
        float(news_df["情緒分數"].mean()), 1
    )


def _diagnosis_button(ticker: str, key: str) -> None:
    if st.button("個股診斷", key=key, use_container_width=True):
        navigate_to_ticker(ticker)


def _render_adr_list(stocks: list[dict[str, str]], key_prefix: str) -> None:
    for stock in stocks:
        stock_col, button_col = st.columns([3, 1])
        with stock_col:
            st.markdown(f"**{stock['name']} (`{stock['ticker']}`)**")
            st.caption(stock["desc"])
        with button_col:
            _diagnosis_button(stock["ticker"], f"{key_prefix}_{stock['ticker']}")


def render_global_markets_page() -> None:
    st.title("🌐 全球市場與美股 ADR 精選標的")
    st.caption(
        "追蹤台灣、日本、歐洲、全球與其他主要市場 ETF 的歷史走勢，"
        "並從美股上市的 ADR／跨境企業白名單搭配新聞情緒尋找研究起點。資料每小時更新。"
    )

    try:
        with st.spinner("正在讀取國際市場最新行情…"):
            performance_df, spy_month, loaded_at = fetch_global_performance()
    except Exception as exc:
        st.error(f"國際市場數據讀取失敗：{exc}")
        st.info("請稍後重新整理；Yahoo Finance 限流或資料不完整時不會使用假資料。")
        return

    st.caption(f"⏱ 國際市場資料載入：{loaded_at}｜分析完成：{beijing_timestamp()}")
    periods = {
        "近 1 個月": "近 1 個月 (%)",
        "近 3 個月": "近 3 個月 (%)",
        "近 1 年": "近 1 年 (%)",
    }
    selected_period = st.radio(
        "選擇排行榜時間範圍",
        list(periods),
        horizontal=True,
        key="global_market_period",
    )
    metric = periods[selected_period]
    sorted_df = performance_df.sort_values(metric, ascending=False).reset_index(drop=True)

    fig = px.bar(
        sorted_df,
        x=metric,
        y="國家／區域",
        orientation="h",
        color=metric,
        color_continuous_scale="Viridis",
        title=f"六大國際市場 ETF 報酬率（{selected_period}）",
        labels={metric: "報酬率 (%)", "國家／區域": ""},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        height=420,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_colorbar={"title": "%"},
        margin=dict(l=0, r=10, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"SPY 同期報酬：{spy_month:+.2f}%（美股基準）")

    st.markdown("---")
    st.subheader("🔍 國家／區域探索、歷史走勢與 ADR")
    selected_region = st.selectbox(
        "選擇要探索的國家／地區",
        list(GLOBAL_MARKETS),
        key="global_market_region",
    )
    region = GLOBAL_MARKETS[selected_region]
    st.info(f"**市場簡介：** {region['description']}")
    try:
        history_df = fetch_market_history(region["etf"])
        history_fig = px.line(
            history_df,
            x="日期",
            y="收盤價",
            title=f"{selected_region}｜{region['etf']} 五年歷史走勢",
            labels={"收盤價": "收盤價（USD）", "日期": ""},
        )
        history_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            height=360,
            margin=dict(l=0, r=10, t=55, b=20),
        )
        st.plotly_chart(history_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"{region['etf']} 歷史走勢暫時無法載入：{exc}")

    news_tickers = tuple(
        dict.fromkeys([region["etf"]] + [item["ticker"] for item in region["adr_stocks"]])
    )
    news_df, sentiment_score = fetch_market_news(news_tickers)
    sentiment_label = (
        "偏正面" if sentiment_score >= 60 else "偏負面" if sentiment_score <= 40 else "中性"
    )
    sentiment_color = (
        "#3FB950" if sentiment_score >= 60 else "#FF7B72" if sentiment_score <= 40 else "#D29922"
    )
    sentiment_col, news_count_col = st.columns(2)
    with sentiment_col:
        st.metric(
            "📰 新聞情緒評分",
            f"{sentiment_score:.1f}/100",
            delta=sentiment_label,
            delta_color="normal" if sentiment_score >= 50 else "inverse",
        )
    with news_count_col:
        st.metric("熱點新聞數量", f"{len(news_df)} 則")
    st.markdown(
        f"<div style='border-left:4px solid {sentiment_color}; padding:8px 12px; "
        f"background:#1C2128; color:#E6EDF3;'>"
        f"情緒評分由選定區域 ETF 與 ADR 的 Yahoo Finance 新聞標題計算，"
        f"正面／中性／負面分別映射為 100／50／0。</div>",
        unsafe_allow_html=True,
    )
    if news_df.empty:
        st.info("目前沒有可用的 Yahoo Finance 熱點新聞。")
    else:
        st.markdown("### 📰 熱點新聞列表")
        for _, news in news_df.head(10).iterrows():
            sentiment_icon = {"正面": "🟢", "中性": "🟡", "負面": "🔴"}.get(
                news["情緒"], "⚪"
            )
            published = (
                news["發布時間"].strftime("%Y-%m-%d %H:%M")
                if pd.notna(news["發布時間"])
                else "時間未知"
            )
            link = news.get("連結")
            title = (
                f"[{news['標題']}]({link})"
                if isinstance(link, str) and link.startswith("http")
                else news["標題"]
            )
            st.markdown(
                f"{sentiment_icon} **`{news['Ticker']}`** {title}  \n"
                f"<span style='color:#8B949E;font-size:12px;'>"
                f"{news['發布者']} · {published} · {news['情緒']} {news['情緒分數']:.0f}/100"
                f"</span>",
                unsafe_allow_html=True,
            )

    etf_col, stocks_col = st.columns([1, 2])
    with etf_col:
        st.markdown("### 🧺 國家／區域 ETF")
        for etf in region["top_etfs"]:
            st.code(etf, language=None)
    with stocks_col:
        st.markdown("### 📈 精選美股上市 ADR／跨境標的")
        _render_adr_list(region["adr_stocks"], "global_region_diag")

    st.markdown("---")
    st.subheader("⭐ 精選投資白名單")
    st.caption("以下標的為研究白名單；按下「個股診斷」會共用 selected_ticker 並開啟 7-Factor 分析。")
    whitelist_rows = []
    for region_name, info in GLOBAL_MARKETS.items():
        for stock in info["adr_stocks"]:
            whitelist_rows.append(
                {
                    "市場": region_name,
                    "Ticker": stock["ticker"],
                    "公司": stock["name"],
                    "研究方向": stock["desc"],
                }
            )
    whitelist_df = pd.DataFrame(whitelist_rows)
    st.dataframe(whitelist_df, use_container_width=True, hide_index=True)
    with st.expander("展開白名單快速診斷按鈕"):
        for row_start in range(0, len(whitelist_rows), 3):
            cols = st.columns(3)
            for col, row in zip(cols, whitelist_rows[row_start : row_start + 3]):
                with col:
                    st.markdown(f"**{row['Ticker']}** · {row['公司']}")
                    _diagnosis_button(
                        row["Ticker"],
                        f"global_whitelist_diag_{row['Ticker']}",
                    )


# Compatibility name used by the uploaded draft.
def render_global_page() -> None:
    render_global_markets_page()


if __name__ == "__main__":
    st.set_page_config(
        page_title="全球市場｜美股選股儀表板",
        page_icon="🌐",
        layout="wide",
    )
    render_global_markets_page()
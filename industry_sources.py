"""Publisher-owned research feeds, never fabricated or attributed KOL opinions."""
from datetime import timezone
from email.utils import parsedate_to_datetime
import html
import re
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from public_data_cache import last_success, request_get

SOURCES = {
    "金融／銀行研究": ("紐約聯儲 Liberty Street Economics",
                  "https://libertystreeteconomics.newyorkfed.org/feed/"),
    "能源／供需分析": ("美國能源資訊署 EIA Today in Energy",
                  "https://www.eia.gov/rss/todayinenergy.xml"),
    "宏觀／貨幣政策評論": ("聯準會官員公開演說",
                    "https://www.federalreserve.gov/feeds/speeches.xml"),
}


def fetch_articles(url):
    response = request_get(url, timeout=(5, 20),
                           headers={"User-Agent": "StockResearchDashboard/1.0"})
    root = ET.fromstring(response.content)
    articles = []
    for item in root.findall(".//item"):
        title, link, date = (item.findtext(key, "").strip() for key in ("title", "link", "pubDate"))
        if not title or not date or urlparse(link).scheme not in ("http", "https"):
            continue
        try:
            stamp = parsedate_to_datetime(date).astimezone(timezone.utc).isoformat()
        except (ValueError, TypeError):
            continue
        summary = html.unescape(re.sub("<[^>]+>", " ", item.findtext("description", "")))
        articles.append({"title": html.unescape(title), "url": link,
                         "published_at": stamp, "summary": " ".join(summary.split())[:500]})
    articles.sort(key=lambda row: row["published_at"], reverse=True)
    return articles[:20]


def load_articles(theme, force=False):
    source, url = SOURCES[theme]
    result = last_success("industry:" + url, lambda: fetch_articles(url), 3600, force)
    return {**result, "publisher": source, "theme": theme}


def render_industry_references():
    import streamlit as st
    st.caption("按主題顯示出版機構原始文章／分析及日期；不是未查核的人物推薦，"
               "不參與選股評分。保留原文標題與摘要，未用 AI 改寫或歸屬人物觀點。")
    theme = st.selectbox("產業／研究主題", list(SOURCES), key="industry_theme")
    force = st.button("更新來源文章", key="industry_refresh")
    try:
        data = load_articles(theme, force)
    except Exception as exc:
        st.info(f"此主題目前沒有可讀取的來源或成功快取：{type(exc).__name__}")
        return
    st.caption(f"{data['publisher']} · 最近 {len(data['data'])} 篇 · "
               f"{'快取' if data['cache_hit'] else '本次取得'} · "
               f"截至最近成功取得 {data['fetched_at']}；文章發布日期逐篇列示。")
    if data.get("stale"):
        st.warning(f"來源暫時讀取失敗，以下為最近成功內容：{data['error']}")
    for article in data["data"][:8]:
        st.markdown(f"**[{article['title']}]({article['url']})**")
        st.caption(f"主題：{theme}｜來源：{data['publisher']}｜發布：{article['published_at']}")
        if article["summary"]:
            st.write(article["summary"])

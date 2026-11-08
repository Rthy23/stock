"""Verified-source analyst information, separate from quarantined demo picks."""
import pandas as pd
import streamlit as st

from analyst_sources import load_analyst_bundle, analyst_data

RATINGS = {"strongBuy": "強力買入", "buy": "買入", "hold": "持有",
           "sell": "賣出", "strongSell": "強力賣出"}


def _timing(name, meta):
    st.caption(f"{name} · 截至最近成功取得 {meta.get('fetched_at') or '未知'} UTC"
               f" · {'快取' if meta.get('cache_hit') else '本次取得'}；"
               "這是取得時間，個別資料日期見表格，並非全部分析師於同一時間更新。")
    if meta.get("error"):
        if meta.get("stale"):
            st.warning(f"{name} 本次來源失敗，顯示最近成功快取：{meta['error']}")
        else:
            st.caption(f"{name}：{meta['error']}")


def render_recommendation_details(data, ticker, key_prefix="analyst"):
    from ui_components import plot_analyst_recs
    meta = data.get("source_meta") or {}
    recs = data.get("recs_df")
    _timing("評級分布", meta.get("recommendations", {}))
    if recs is not None and not recs.empty:
        table = recs.copy()
        table = table.drop(columns=["index"], errors="ignore")
        if "period" in table:
            table["period"] = table["period"].map(
                lambda p: "最近一期" if p == "0m" else f"往前 {str(p).replace('m', '')} 個月")
        table = table.rename(columns={**RATINGS, "period": "彙總期間"})
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.plotly_chart(plot_analyst_recs(recs), use_container_width=True,
                        key=f"{key_prefix}_trend_{ticker}")
        st.caption("每期為 Yahoo 彙總評級人數，並非升降級事件筆數；期間標籤為相對月份，"
                   "Yahoo 未提供每位分析師的彙總採樣截止時間。")
    else:
        st.info("評級分布尚無可用來源或成功快取。")
    changes = data.get("changes_df")
    _timing("近期評級調整", meta.get("upgrades_downgrades", {}))
    if changes is not None and not changes.empty:
        rename = {"GradeDate": "日期", "Firm": "券商／研究機構", "ToGrade": "調整後評級",
                  "FromGrade": "調整前評級", "Action": "動作",
                  "currentPriceTarget": "本次目標價", "priorPriceTarget": "先前目標價",
                  "priceTargetAction": "目標價動作"}
        view = changes.rename(columns=rename).drop(columns=["index"], errors="ignore").head(30).copy()
        if "動作" in view:
            view["動作"] = view["動作"].replace(
                {"up": "升級", "down": "降級", "main": "維持", "init": "首次覆蓋",
                 "reit": "重申", "res": "恢復覆蓋"})
        # Yahoo often uses zero to mean no target supplied in an event.
        for col in ("本次目標價", "先前目標價"):
            if col in view:
                view[col] = pd.to_numeric(view[col], errors="coerce").map(
                    lambda v: f"${v:,.2f}" if pd.notna(v) and v > 0 else "—")
        st.dataframe(view.fillna("—"), hide_index=True, use_container_width=True)
        st.caption("來源：Yahoo Finance 券商評級事件；日期為來源事件時間。"
                   "目標價未提供時顯示「—」，不以目前共識目標價補填。")
    else:
        st.info("近期升降級尚無可用來源或成功快取。")


def render_analyst_research(key_prefix="official"):
    st.subheader("⭐ 分析師推薦與券商評級（Yahoo 彙總）")
    st.caption("以下為可核對的第三方券商彙總，不是已隔離的示範人物推薦，亦不形成選股排名。")
    ticker = st.selectbox("查看券商評級標的", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"],
                          key=f"{key_prefix}_analyst_ticker")
    force = st.button("重新取得分析師資料", key=f"{key_prefix}_analyst_refresh")
    data = analyst_data(ticker, force=force)
    st.caption(f"{ticker} · Yahoo 分析師意見數："
               f"{data['num_analysts'] if data['num_analysts'] is not None else '—'}"
               "（不保證等於目標價樣本數）")
    columns = st.columns(4)
    for col, label, field in zip(columns, ["目標低價", "目標中位數", "目標均價", "目標高價"],
                                 ["target_low", "target_median", "target_mean", "target_high"]):
        value = data.get(field)
        col.metric(label, f"${value:.2f}" if value is not None else "—")
    _timing("共識目標價", (data.get("source_meta") or {}).get("targets", {}))
    render_recommendation_details(data, ticker, key_prefix)


def render_issuer_holders():
    st.markdown("#### Yahoo 發行公司機構持有人（與上方申報機構 13F 不同）")
    ticker = st.selectbox("查看發行公司機構持有人", ["AAPL", "MSFT", "NVDA", "TSLA"],
                          key="issuer_holders_ticker")
    bundle = load_analyst_bundle(ticker, include_holders=True)
    part = bundle["institutional_holders"]
    _timing("機構持有人", part)
    if not part.get("data"):
        st.info("目前沒有此公司的持有人來源資料或成功快取。")
        return
    table = pd.DataFrame(part["data"]["rows"]).drop(columns=["index"], errors="ignore")
    for key in ("pctHeld", "pctChange"):
        if key in table:
            table[key] = pd.to_numeric(table[key], errors="coerce").map(
                lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—")
    table = table.rename(columns={"Date Reported": "報告日期", "Holder": "機構名稱",
                                 "pctHeld": "占發行公司股數", "Shares": "持有股數",
                                 "Value": "市值（USD）", "pctChange": "來源所報持股變化"})
    st.caption(f"{ticker} · {len(table)} 個可取得機構持有人；非完整股東名冊，"
               "各列報告日期可能不同，不與上方 13F 機構組合混算。")
    st.dataframe(table.fillna("—"), hide_index=True, use_container_width=True)

"""SEC 13F view: institutional disclosures, deliberately not stock recommendations."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from sec_edgar_fetcher import EdgarError, MANAGERS, fetch_manager, validate_user_agent


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_manager(manager: tuple, _user_agent: str) -> dict:
    return fetch_manager(manager, _user_agent)


def _rows(holdings: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "申報公司名稱": h["issuer"],
            "CUSIP": h["cusip"],
            "證券類別": h["class"],
            "選擇權": h["option"] or "—",
            "申報數量": f'{h["quantity"]:,.0f}',
            "單位": h["unit"],
            "市值（申報原值）": f'{h["value_reported"]:,.0f}',
        }
        for h in holdings
    ])


def _changes(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "變化": h["status"],
            "申報公司名稱": h["issuer"],
            "CUSIP": h["cusip"],
            "證券類別": h["class"],
            "選擇權": h["option"] or "—",
            "上季數量": f'{h["previous_quantity"]:,.0f}',
            "本季數量": f'{h["quantity"]:,.0f}',
            "單位": h["unit"],
        }
        for h in rows if h["status"] != "未變"
    ])


def render_sec_holdings() -> None:
    st.subheader("📑 SEC 13F — 機構季末持倉")
    st.info(
        "資料來源：SEC EDGAR 官方 13F 申報。這是申報機構在報告**季末**的"
        "部分美國證券**多頭**部位，申報通常可延遲最多 45 天；"
        "**非即時交易、非放空部位，也非相關經理人的個人推薦或發言。**"
        " 資料每日最多重新查詢一次。"
    )
    manager = st.selectbox(
        "選擇申報機構（依 SEC CIK 查詢）",
        MANAGERS,
        format_func=lambda m: f"{m[1]} · CIK {int(m[2]):010d}",
        key="sec_13f_manager",
    )
    try:
        agent = validate_user_agent(os.environ.get("SEC_USER_AGENT", ""))
    except EdgarError as exc:
        st.warning(f"SEC 資料暫不可用：{exc} 本區不會以模擬持倉替代。")
        return

    try:
        with st.spinner("正在核對 SEC 申報與持倉資訊…"):
            result = _cached_manager(manager, agent)
    except EdgarError as exc:
        st.warning(f"{manager[1]}：{exc} 不會以模擬資料替代。")
        return
    except Exception as exc:
        st.error(f"SEC 資料格式或連線異常：{type(exc).__name__}。不顯示未經核實的持倉。")
        return

    current = result["current"]
    previous = result["previous"]
    try:
        days_since_report = (date.today() - date.fromisoformat(current["report_date"])).days
        if days_since_report > 180:
            st.warning(
                f"此機構最新可取得的報告季末已距今 {days_since_report} 天；"
                "申報可能已停止或尚未更新。不能將下列內容視為目前持倉。"
            )
    except ValueError:
        st.error("SEC 報告日期無法解析，已停止顯示持倉。")
        return
    st.markdown(
        f"**申報機構：{result['sec_name']}**｜CIK {int(result['cik']):010d}  \n"
        f"最新可取得報告季末：**{current['report_date']}**｜實際申報：**{current['filing_date']}**"
        f"（{current['form']}）｜查詢時間 UTC：{result['fetched_at']}  \n"
        f"[本期原始資訊表]({current['source_url']}) · "
        f"[SEC 原始申報頁]({current['filing_url']})"
    )
    if result["note"]:
        st.warning(result["note"])
    st.caption(
        "無可靠交易所代碼對照時僅顯示 SEC 原始公司名稱與 CUSIP；"
        "市值欄保留申報原值，數量變化不等於實際買賣成交量。"
    )
    st.markdown(f"#### 本期持倉（{len(current['holdings'])} 種證券）")
    st.dataframe(_rows(current["holdings"]), use_container_width=True, hide_index=True)

    if previous:
        st.markdown(
            f"#### 前一期持倉（報告季末 {previous['report_date']}；"
            f"申報 {previous['filing_date']}）"
        )
        st.markdown(
            f"[前一期原始資訊表]({previous['source_url']}) · "
            f"[SEC 原始申報頁]({previous['filing_url']})"
        )
        st.dataframe(_rows(previous["holdings"]), use_container_width=True, hide_index=True)
        st.markdown("#### 兩期持倉數量變化（CUSIP／證券類別／選擇權／數量單位逐一比較）")
        changes = _changes(result["changes"])
        if changes.empty:
            st.info("兩期申報持有數量無增減。")
        else:
            st.dataframe(changes, use_container_width=True, hide_index=True)
    else:
        st.info("沒有可核對的前一期完整 13F，暫不推算加減碼或出清。")
"""SEC 13F view: institutional disclosures, deliberately not stock recommendations."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from sec_edgar_fetcher import EdgarError, MANAGERS, fetch_manager, validate_user_agent
from public_data_cache import last_success


def _cached_manager(manager: tuple, _user_agent: str) -> dict:
    cached = last_success("sec-manager:" + manager[2],
                          lambda: fetch_manager(manager, _user_agent), ttl=86400)
    return {**cached["data"], "cache_meta": {k: cached[k] for k in
                                           ("fetched_at", "cache_hit", "stale", "error")}}


def _rows(holdings: list[dict]) -> pd.DataFrame:
    total = sum(h["value_reported"] for h in holdings)
    return pd.DataFrame([
        {
            "申報公司名稱": h["issuer"],
            "CUSIP": h["cusip"],
            "證券類別": h["class"],
            "選擇權": h["option"] or "—",
            "申報數量": f'{h["quantity"]:,.0f}',
            "單位": h["unit"],
            "市值（申報原值）": f'{h["value_reported"]:,.0f}',
            "占本期申報組合": f'{h["value_reported"]/total*100:.2f}%' if total else "資料不足",
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
        with st.spinner("正在核對 SEC 申報與持倉資訊…"):
            result = _cached_manager(manager, os.environ.get("SEC_USER_AGENT", ""))
    except EdgarError as exc:
        st.warning(f"{manager[1]}：{exc} 不會以模擬資料替代。")
        return
    except Exception as exc:
        st.error(f"SEC 資料格式或連線異常：{type(exc).__name__}。不顯示未經核實的持倉。")
        return

    current = result["current"]
    previous = result["previous"]
    meta = result.get("cache_meta") or {}
    if meta.get("stale"):
        st.warning(f"SEC 本次讀取失敗：{meta['error']}；顯示最近成功快取，"
                   f"截至 {current['report_date']}（取得 {meta['fetched_at']}）。")
    st.caption(f"摘要：{result['sec_name']} · 截至 {current['report_date']} · "
               f"{len(current['holdings'])} 種證券 · "
               f"{'快取' if meta.get('cache_hit') else '本次取得'} · "
               f"成功取得時間 {meta.get('fetched_at', result['fetched_at'])}。"
               "比例為本期 13F 申報組合內占比，並非發行公司流通股占比。")
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
    view = _rows(current["holdings"])
    if previous:
        changes = {(h["cusip"], h["class"], h["option"], h["unit"]): h
                   for h in result["changes"]}
        quantities = []
        for h in current["holdings"]:
            change = changes.get((h["cusip"], h["class"], h["option"], h["unit"]))
            quantities.append(f'{change["quantity"] - change["previous_quantity"]:+,.0f}'
                              if change else "—")
        view["較上期數量變化"] = quantities
    view["申報機構"] = result["sec_name"]
    view["報告季末"] = current["report_date"]
    st.dataframe(view, use_container_width=True, hide_index=True)

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
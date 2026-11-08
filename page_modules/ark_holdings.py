"""Official ARKK fund disclosures, separate from simulated recommendations."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from zoneinfo import ZoneInfo
from datetime import datetime

import pandas as pd
import streamlit as st

from ark_holdings import (
    ArkHoldingsError, ArkStorageError, DOCUMENTS_URL, SOURCE_URL, get_ark_holdings, FileArkStore,
)
from public_data_cache import CACHE_DIR


def load_ark_view():
    """A missing snapshot DB must not prevent displaying the official live CSV."""
    local = FileArkStore(CACHE_DIR / "ark")
    try:
        result = get_ark_holdings()
        # Keep a public last-good copy even if the DB becomes unavailable later.
        for snapshot in (result.get("previous"), result.get("current")):
            if snapshot:
                local.save_snapshot(snapshot)
        if result.get("checked_at"):
            local.save_status({k: result.get(k) for k in
                               ("checked_at", "error", "unchanged_on_refresh")})
        result["storage_note"] = None
    except ArkStorageError as exc:
        result = get_ark_holdings(cache_dir=CACHE_DIR / "ark")
        result["storage_note"] = (
            f"跨重新發布快照儲存暫不可用：{exc} 下方仍顯示官方來源／本機最近成功快取；"
            "本機快取不能保證跨重新發布保留，不代表持久化資料庫已修復。")
    return result


def _number(value: str | None, decimals: int = 0) -> str:
    return f"{Decimal(value):,.{decimals}f}" if value is not None else "—"


def render_ark_holdings() -> None:
    st.subheader("🧬 ARK Invest 官方 ARKK 每日持倉")
    st.info(
        "資料來源：ARK Invest 官方每日持倉揭露。這是 **ARK Innovation ETF（ARKK）**"
        "的基金持倉，不是 Cathie Wood 個人即時推薦或逐筆交易紀錄；"
        "與 SEC 13F 分開展示；未查核示範排名已停用。"
    )
    st.caption(
        "開啟此區塊時查詢官方檔案；成功取得的不同持倉日期保存在持久化資料庫。"
        "只有已收集到至少兩個不同實際日期、且識別碼與股數可比時才顯示變化；"
        "不會用模擬或推算的前一天快照補齊。"
    )
    st.markdown(f"[ARK 官方文件頁]({DOCUMENTS_URL}) · [原始 ARKK 持倉 CSV]({SOURCE_URL})")
    try:
        result = load_ark_view()
    except (ArkHoldingsError, ArkStorageError, OSError) as exc:
        st.error(f"ARK 快照無法讀取或儲存：{exc} 不會以模擬資料替代。")
        return
    current = result["current"]
    if result.get("storage_note"):
        st.warning(result["storage_note"])
    if result["error"]:
        st.warning(
            f"ARK 官方來源問題：{result['error']}"
            + (f" 最近成功持倉日期：{current['date']}；下方為先前保存的官方快照。"
               if current else " 尚無成功快照，不會以模擬資料替代。")
        )
    if not current:
        return
    if result["unchanged_on_refresh"]:
        st.warning(f"官網檔案尚未更新至新交易日；最近成功持倉日期：{current['date']}。")
    ny_today = datetime.now(ZoneInfo("America/New_York")).date()
    last_business = ny_today
    while last_business.weekday() >= 5:
        last_business = date.fromordinal(last_business.toordinal() - 1)
    # Weekday-only signal is advisory; exchange holidays are not inferred.
    if date.fromisoformat(current["date"]) < last_business and not result["unchanged_on_refresh"]:
        st.warning(f"官方檔案尚未更新至紐約最近工作日；最近成功持倉日期：{current['date']}。"
                   "假日或官方發布時間可能造成延遲。")
    st.markdown(
        f"**基金：ARK Innovation ETF（{current['fund']}）**｜"
        f"官方檔案所示持倉日期：**{current['date']}**｜"
        f"最近查詢 UTC：{result['checked_at'] or '—'}"
    )
    st.caption(f"截至 {current['date']} · ARKK 共 {len(current['holdings'])} 筆官方持倉；"
               f"組合權重合計 {sum(Decimal(h['weight_pct']) for h in current['holdings']):.2f}% · "
               f"最近來源檢查 {result['checked_at'] or '未知'}；非即時持倉。")
    st.dataframe(pd.DataFrame([
        {
            "公司": h["company"], "Ticker": h["ticker"] or "—", "CUSIP": h["cusip"] or "—",
            "股數": _number(h["shares"]),
            "市值（USD）": _number(h["market_value"], 2),
            "權重": f'{_number(h["weight_pct"], 2)}%',
        }
        for h in current["holdings"]
    ]), use_container_width=True, hide_index=True)
    previous = result["previous"]
    if result["comparison_note"]:
        st.info(result["comparison_note"])
        return
    st.markdown(f"#### 持倉股數變化（{previous['date']} → {current['date']}）")
    st.caption("以官方 CSV 的相同 CUSIP 與股數比較；基金股數變化不等於逐筆買賣，"
               "亦可能涉及公司行動。不是個人交易訊號。")
    if not result["changes"]:
        st.info("兩期持倉股數沒有變化。")
    else:
        st.dataframe(pd.DataFrame([
            {
                "變化": c["status"], "公司": c["company"], "Ticker": c["ticker"] or "—",
                "CUSIP": c["cusip"], "前期股數": _number(c["previous_shares"]),
                "本期股數": _number(c["current_shares"]),
                "股數差": _number(c["change_shares"]),
            }
            for c in result["changes"]
        ]), use_container_width=True, hide_index=True)
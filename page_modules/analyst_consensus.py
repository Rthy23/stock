"""Curated-whitelist and Wall Street analyst recommendation consensus page."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data_fetcher import beijing_timestamp
from kol_config import ANALYST_DIRECTORY, ARK_FUND_ONLY_IDS, COMMENTARY_IDS
from kol_whitelist import build_consensus_table, render_industry_commentary_reference
from page_modules.ark_holdings import render_ark_holdings
from sec_edgar_fetcher import MANAGER_IDS
from page_modules.sec_holdings import render_sec_holdings
from navigation import navigate_to_ticker
import picks_store


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


def _freshness_badge(days_old: int) -> str:
    """Return an emoji freshness label based on pick age."""
    if days_old < 0:
        return "❓ 日期異常"
    if days_old <= 7:
        return f"🟢 {days_old}天前"
    if days_old <= 14:
        return f"🟡 {days_old}天前"
    if days_old <= 30:
        return f"🟠 {days_old}天前"
    return f"🔴 {days_old}天前（過期）"


def _render_picks_manager() -> None:
    """Expander UI for viewing, adding, and deleting analyst picks."""
    all_picks = picks_store.get_picks_with_status()
    total = len(all_picks)
    expired = sum(1 for p in all_picks if p.get("is_expired"))

    expander_label = (
        f"🛠 管理推薦記錄（共 {total} 筆"
        + (f"，⚠️ {expired} 筆已過期" if expired else "")
        + "）"
    )
    with st.expander(expander_label, expanded=False):
        # ── 新增推薦 ──────────────────────────────────────────────────────────
        st.markdown("#### ➕ 新增推薦記錄")
        analyst_options = {a["name"]: a["id"] for a in ANALYST_DIRECTORY
                           if a["id"] not in MANAGER_IDS | COMMENTARY_IDS | ARK_FUND_ONLY_IDS}
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_name = st.selectbox(
                "分析師",
                list(analyst_options.keys()),
                key="pm_analyst",
            )
        with col2:
            new_ticker = st.text_input(
                "Ticker",
                placeholder="如 AAPL",
                key="pm_ticker",
            ).strip().upper()

        col3, col4 = st.columns([1, 1])
        with col3:
            new_date = st.date_input(
                "推薦日期",
                value=datetime.now().date(),
                key="pm_date",
            )
        with col4:
            quality_map = {
                "3 — 強論點（含財報/護城河/估值）": 3,
                "2 — 中等（觀點有據但不夠詳細）": 2,
                "1 — 薄弱（標題黨/無數據支撐）": 1,
            }
            quality_label = st.selectbox(
                "論點品質",
                list(quality_map.keys()),
                key="pm_quality",
            )

        new_thesis = st.text_area(
            "投資論點（thesis）",
            placeholder="請說明推薦理由，建議包含估值、護城河或財報數據…",
            height=90,
            key="pm_thesis",
        )

        if st.button("✅ 新增此推薦", key="pm_add_btn", use_container_width=True):
            if not new_ticker:
                st.error("請填寫 Ticker 代碼。")
            elif not new_thesis.strip():
                st.error("請填寫投資論點。")
            else:
                try:
                    picks_store.add_pick({
                        "kol_id":           analyst_options[selected_name],
                        "ticker":           new_ticker,
                        "date":             str(new_date),
                        "argument_quality": quality_map[quality_label],
                        "thesis":           new_thesis.strip(),
                    })
                    st.success(f"✅ 已新增 {selected_name} 對 {new_ticker} 的推薦！")
                    st.rerun()
                except Exception as exc:
                    st.error(f"新增失敗：{exc}")

        st.markdown("---")

        # ── 清除過期 ─────────────────────────────────────────────────────────
        col_purge, col_info = st.columns([1, 3])
        with col_purge:
            if st.button(
                f"🗑 清除 {expired} 筆過期推薦（>30天）",
                key="pm_purge_btn",
                disabled=expired == 0,
                use_container_width=True,
            ):
                _, removed = picks_store.purge_expired_picks(days=30)
                st.success(f"已移除 {removed} 筆超過 30 天的過期推薦。")
                st.rerun()
        with col_info:
            st.caption(
                "過期推薦仍會被計入共識排行榜（權重 ×0.1），"
                "清除後將完全從計算中移除。"
            )

        st.markdown("---")

        # ── 現有推薦列表 ───────────────────────────────────────────────────────
        st.markdown("#### 📋 現有推薦記錄")
        if not all_picks:
            st.info("目前沒有推薦記錄。")
            return

        # Build analyst id → name lookup
        id_to_name = {a["id"]: a["name"] for a in ANALYST_DIRECTORY}

        # Group tabs: active vs expired
        tab_active, tab_expired = st.tabs([
            f"✅ 有效推薦（{total - expired}）",
            f"⚠️ 過期推薦（{expired}）",
        ])

        def _render_pick_rows(picks_subset: list, offset: int) -> None:
            for local_i, p in enumerate(picks_subset):
                real_idx = p["_storage_index"]
                analyst_name = id_to_name.get(p["kol_id"], p["kol_id"])
                badge = _freshness_badge(p.get("days_old", -1))
                quality_icons = {3: "💪", 2: "📊", 1: "⚠️"}
                q_icon = quality_icons.get(p.get("argument_quality", 0), "❓")
                row_cols = st.columns([2, 1, 2, 5, 1])
                row_cols[0].markdown(f"**{analyst_name}**")
                row_cols[1].markdown(f"`{p['ticker']}`")
                row_cols[2].markdown(badge)
                row_cols[3].caption(f"{q_icon} {p.get('thesis', '')[:80]}{'…' if len(p.get('thesis','')) > 80 else ''}")
                if row_cols[4].button("🗑", key=f"pm_del_{real_idx}", help="刪除此推薦"):
                    try:
                        picks_store.delete_pick(real_idx)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"刪除失敗：{exc}")

        with tab_active:
            active_with_idx = [
                (p["_storage_index"], p) for p in all_picks if not p.get("is_expired")
            ]
            if active_with_idx:
                st.markdown(
                    "<small style='color:#8B949E'>分析師 ／ Ticker ／ 新鮮度 ／ 論點摘要</small>",
                    unsafe_allow_html=True,
                )
                for real_idx, p in active_with_idx:
                    analyst_name = id_to_name.get(p["kol_id"], p["kol_id"])
                    badge = _freshness_badge(p.get("days_old", -1))
                    quality_icons = {3: "💪", 2: "📊", 1: "⚠️"}
                    q_icon = quality_icons.get(p.get("argument_quality", 0), "❓")
                    row_cols = st.columns([2, 1, 2, 5, 1])
                    row_cols[0].markdown(f"**{analyst_name}**")
                    row_cols[1].markdown(f"`{p['ticker']}`")
                    row_cols[2].markdown(badge)
                    row_cols[3].caption(f"{q_icon} {p.get('thesis', '')[:80]}{'…' if len(p.get('thesis','')) > 80 else ''}")
                    if row_cols[4].button("🗑", key=f"pm_del_{real_idx}", help="刪除此推薦"):
                        try:
                            picks_store.delete_pick(real_idx)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"刪除失敗：{exc}")
            else:
                st.info("目前沒有有效推薦。")

        with tab_expired:
            expired_with_idx = [
                (p["_storage_index"], p) for p in all_picks if p.get("is_expired")
            ]
            if expired_with_idx:
                st.caption("以下推薦超過 30 天，在共識計算中權重已降至 ×0.1。")
                for real_idx, p in expired_with_idx:
                    analyst_name = id_to_name.get(p["kol_id"], p["kol_id"])
                    badge = _freshness_badge(p.get("days_old", -1))
                    quality_icons = {3: "💪", 2: "📊", 1: "⚠️"}
                    q_icon = quality_icons.get(p.get("argument_quality", 0), "❓")
                    row_cols = st.columns([2, 1, 2, 5, 1])
                    row_cols[0].markdown(f"**{analyst_name}**")
                    row_cols[1].markdown(f"`{p['ticker']}`")
                    row_cols[2].markdown(badge)
                    row_cols[3].caption(f"{q_icon} {p.get('thesis', '')[:80]}{'…' if len(p.get('thesis','')) > 80 else ''}")
                    if row_cols[4].button("🗑", key=f"pm_del_{real_idx}", help="刪除此推薦"):
                        try:
                            picks_store.delete_pick(real_idx)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"刪除失敗：{exc}")
            else:
                st.info("目前沒有過期推薦。")


def _render_curated_consensus() -> None:
    st.subheader("⭐ 未查核示範資料共識（非持倉）")
    st.error(
        "⚠️ **模擬資料警告**：以下推薦內容為未查核的示範資料，"
        "並非相關人物或機構真實發表的言論或投資建議。"
        "SEC 13F 機構持倉已另列於上方，**不參與此推薦排名**。"
        "ARKK 官方基金持倉也獨立展示、不參與排名。"
        "產業評論參考目前暫不提供，亦不參與排名。"
        "**請勿將此排行榜作為投資決策依據。**"
    )

    # Load picks freshness summary for the caption
    all_picks = picks_store.get_picks_with_status()
    expired_count = sum(1 for p in all_picks if p.get("is_expired"))
    fresh_count = len(all_picks) - expired_count

    st.caption(
        f"目前有 {len(all_picks)} 筆未驗證示範推薦（"
        f"🟢 {fresh_count} 筆有效 / 🔴 {expired_count} 筆過期），"
        "使用信譽 × 論點品質 × 時效性計算示範分數；"
        "不代表真實持倉或經來源查核的推薦。"
    )
    ranked = build_consensus_table(whitelist=ANALYST_DIRECTORY)
    if not ranked:
        st.info("目前沒有可用的精選分析師推薦資料。")
        _render_picks_manager()
        return

    all_rows = [
        {
            "排名": index,
            "Ticker": pick["ticker"],
            "加權分數": round(float(pick["total_score"]), 3),
            "未查核紀錄數": pick["consensus"],
            "示範來源標示": "、".join(dict.fromkeys(pick["experts"])),
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

    _render_picks_manager()


def render_analyst_consensus_page() -> None:
    st.title("🏦 官方機構與基金持倉")
    st.caption(
        "SEC 13F 為季末持倉申報，最多可能延遲約 45 天；ARKK 是官方每日基金持倉。"
        "兩者彼此獨立，只供背景參考，不是即時選股訊號，也不形成綜合名次。"
    )

    render_sec_holdings()
    st.markdown("---")
    render_ark_holdings()
    st.markdown("---")
    render_industry_commentary_reference()
    st.markdown("---")
    st.subheader("⭐ 分析師推薦")
    st.info("此功能開發中，暫不提供。剩餘示範紀錄沒有可查核的原始來源與發言日期，"
            "因此不展示共識排名、券商彙總排名、論點或 AI 推薦；請勿視為買股依據。")


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
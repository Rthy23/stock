"""Compact additions inside existing Streamlit sections (no new pages)."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from market_data import get_history
from research_context import (
    SECTOR_ETFS, relative_returns, trend_structure, sector_strength,
    valuation_comparison, earnings_event, fetch_breadth,
)


def render_stock_context(ticker, info, displayed_hist, interval="1d"):
    structure = trend_structure(displayed_hist, interval)
    st.caption(f"均線單位：50／200 {structure['unit']}；結構判讀：{structure['trend']}。"
               f"斜率＝最近 5 根 K 線的均線變化率；輸入 K 棒：{structure['time']}。")
    for window in (50, 200):
        item = structure[window]
        slope = f"{item['slope']:+.2f}%" if item["slope"] is not None else "資料不足"
        st.caption(f"SMA{window}（{window} {structure['unit']}）斜率：{slope}")
    try:
        event = earnings_event(ticker)
        if event["start"] is not None:
            start = pd.to_datetime(event["start"], unit="s", utc=True).tz_convert("America/New_York")
            end = pd.to_datetime(event["end"], unit="s", utc=True).tz_convert("America/New_York")
            today = datetime.now(ZoneInfo("America/New_York")).date()
            days = (start.date() - today).days
            text = (f"Yahoo 業績公布{'確認日期' if event['confirmed'] else '預計窗口'}："
                    f"{start.date()} ～ {end.date()} ET；"
                    f"抓取時間（非公告時間）：{event['received_at']}")
            if end.date() >= today and days <= 14:
                st.warning("14 日內業績事件風險 · " + text)
            else:
                st.caption(text)
        else:
            st.caption("業績公布日期：資料不足")
    except Exception:
        st.caption("業績公布日期：資料不足")
    with st.expander("相對強弱（1／3／6 個月）與 Forward PE 比較"):
        st.caption("相對強弱固定使用 adjusted 日線，分別為 21／63／126 個共同交易日區間回報差。"
                   "非風險調整 alpha；不同於上方所選圖表週期。")
        try:
            daily = get_history(ticker)
            rows = []
            benchmarks = ["SPY"]
            etf = SECTOR_ETFS.get(info.get("sector"))
            if etf:
                benchmarks.append(etf)
            else:
                st.caption("所屬板塊 ETF：資料不足，未以其他板塊代替。")
            for benchmark in benchmarks:
                try:
                    base = get_history(benchmark)
                    differences = relative_returns(daily, base)
                    rows.append({"基準": benchmark, **{
                        f"{m}M 回報差（百分點）": f"{v:+.2f}" if v is not None else "資料不足"
                        for m, v in differences.items()},
                        "個股日線時間": str(daily.index[-1]),
                        "基準日線時間": str(base.index[-1])})
                except Exception:
                    rows.append({"基準": benchmark, "狀態": "資料不足"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        except Exception:
            st.caption("相對強弱：日線資料不足")
        st.caption("Forward PE 板塊基準為 S&P 500 同板塊可取得的正值樣本中位數，"
                   "並非全球所有同業；至少 5 個有效樣本。歷史比較只接受 Yahoo 真正的歷史 Forward PE，"
                   "不以 trailing PE 或現時 EPS 回推代替。")
        key = f"valuation_context_{ticker}"
        if st.button("取得／更新 Forward PE 相對比較", key=f"valuation_fetch_{ticker}"):
            try:
                with st.spinner("分批取得同板塊估值及歷史估值…"):
                    st.session_state[key] = valuation_comparison(ticker, info.get("sector"))
            except Exception as exc:
                st.warning(f"估值比較無法取得：{exc}")
        comparison = st.session_state.get(key)
        if comparison:
            current = comparison["current"]
            median = comparison["median"]
            st.caption(f"Forward PE：{current:.2f}x" if current is not None else "Forward PE：資料不足")
            st.caption((f"板塊樣本中位數：{median:.2f}x；相對溢價／折價：{comparison['premium']:+.1f}%"
                        if median is not None and comparison["premium"] is not None else "板塊比較：資料不足")
                       + f" · 有效 {comparison['sample']}/{comparison['pool']} 支（僅正值樣本）")
            st.caption(f"當前估值抓取時間（非分析師預測發布時間）：{comparison['received_at']}")
            history = comparison["history"]
            if len(history) >= 2:
                st.caption(f"自身歷史 Forward PE 區間：{history.min():.2f}～{history.max():.2f}x"
                           f" · {len(history)} 個季度觀察 · {history.index.min().date()}～{history.index.max().date()}")
                if current is not None and current > 0:
                    percentile = ((history < current).sum() + .5*(history == current).sum())/len(history)*100
                    st.caption(f"現值在可取得歷史樣本百分位：{percentile:.1f}%（非完整每日歷史區間）")
                st.dataframe(history.rename("歷史 Forward PE").to_frame(),
                             use_container_width=True)
            else:
                st.caption("自身歷史 Forward PE 區間：資料不足（Yahoo 未提供足夠可核實歷史樣本）。")
            with st.expander("板塊估值樣本與抓取時間"):
                st.dataframe(pd.DataFrame(comparison["samples"]), hide_index=True)


def render_market_context():
    with st.expander("市場廣度與板塊相對 SPY 強弱"):
        st.caption("廣度按 S&P 500 adjusted 日線計算；盤中日線可能未完成，唔係逐筆即時行情。"
                   "只合併相同最新資料日期；缺失及舊日期不計入比例分母。")
        if st.button("更新 S&P 500 市場廣度", key="breadth_refresh"):
            bar = st.progress(0.)
            try:
                st.session_state["market_breadth"] = fetch_breadth(
                    lambda done, total: bar.progress(done / total))
            except Exception as exc:
                st.warning(f"市場廣度無法取得：{exc}")
            finally:
                bar.empty()
        data = st.session_state.get("market_breadth")
        if data:
            s = data["summary"]
            if s["valid"]:
                st.caption(f"升 {s['advances']}／跌 {s['declines']}／持平 {s['unchanged']} 支；"
                           f"同日有效覆蓋 {s['valid']}/{s['total']}（{s['coverage']:.1%}）")
                for window in (50, 200):
                    value = s[f"above{window}"]
                    st.caption((f"企於 {window} 日均線之上：{value:.1%}" if value is not None else
                                f"{window} 日均線比例：資料不足")
                               + f" · 有效分母 {s[f'count{window}']} 支")
            else:
                st.caption("市場廣度：資料不足（同日有效覆蓋 0）")
            st.caption(f"來源日線日期：{s['date'] or '未知'}（非今日時僅代表歷史廣度）；"
                       f"掃描完成時間（非行情時間）：{data['received_at']}")
            if data["errors"]:
                st.caption("抓取失敗：" + "、".join(sorted(data["errors"])))
        else:
            st.caption("市場廣度尚未掃描；按上方按鈕分批取得，避免每次開頁都抓取全池。")
        if st.button("更新板塊相對 SPY 強弱", key="sector_rs_refresh"):
            try:
                st.session_state["sector_rs_context"] = sector_strength()
            except Exception as exc:
                st.warning(f"板塊相對強弱無法取得：{exc}")
        if st.session_state.get("sector_rs_context"):
            st.dataframe(pd.DataFrame(st.session_state["sector_rs_context"]).fillna("資料不足"),
                         hide_index=True, use_container_width=True)
            st.caption("回報差而非資金流向；15 分鐘快取，按更新時重算。")

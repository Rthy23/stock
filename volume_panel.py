"""Intraday volume uses the quote's source clock, not the browser/render clock."""
import streamlit as st
from market_data import get_quote, get_history, market_status, quote_volume_ratio
from market_ui import fmt, quote_label


def render_volume_panel(ticker):
    try:
        hist = get_history(ticker)
        quote = get_quote(ticker)
        today = market_status()["now"].date()
        finished = hist.loc[[stamp.date() < today for stamp in hist.index]]
        avg = float(finished["Volume"].tail(20).mean()) if len(finished) >= 20 else None
        ratio = quote_volume_ratio(quote, avg)
        a, b, c = st.columns(3)
        a.metric("正常時段累積成交量", fmt(quote.get("volume"), 0))
        a.caption(quote_label(quote))
        b.metric("前 20 個完整交易日平均量", fmt(avg, 0))
        c.metric("同時間段量比", fmt(ratio, 2, "×"))
        st.caption("分母＝20 日完整均量 × 從開市到來源報價時刻的交易時間比例。"
                   "僅線性近似，不代表已辨識機構資金；來源日期／時間不足時不計算。")
        from ui_components import plot_volume_chart
        fig, _, _ = plot_volume_chart(ticker, hist)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"adjusted_volume_{ticker}")
    except Exception as exc:
        st.warning(f"成交量資料不足：{exc}")

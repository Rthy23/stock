"""S&P 500 scanner UI; preserves existing diagnosis, watchlist and backtest actions."""
import pandas as pd
import plotly.express as px
import streamlit as st
from market_data import get_sp500
from market_ui import fmt, quote_label
from screener_engine import scan, filter_records


def render_screener(params, navigate, save_watchlist):
    st.markdown("### S&P 500 美股篩選器")
    st.caption("股票池：最新可取得的 S&P 500 成分表（多股別可能超過 500 支），不使用固定 93 支名單。"
               "報價快取 60 秒；基本面與 adjusted 日線快取 15 分鐘。")
    try:
        pool = get_sp500()
        st.caption(f"目前股票池：{len(pool['symbols'])} 支 · {pool['source']} · "
                   f"成分表取得：{pool['received_at']}（取得時間，非成分調整生效日）")
    except Exception as exc:
        st.warning(str(exc))
    with st.expander("進階篩選條件", expanded=True):
        ranges = {}
        fields = [
            ("price", "股價範圍（正常時段原始報價）", 0., 1000.),
            ("change_pct", "今日漲跌幅（%）", -10., 10.),
            ("volume", "成交量", 1000000., 1000000000.),
            ("volume_ratio", "同時間段量比", 1., 100.),
            ("rsi", "Wilder RSI（14，adjusted 日線）", 0., 100.),
            ("forward_pe", "Forward PE", 0., 100.),
        ]
        for key, label, low, high in fields:
            c1, c2, c3 = st.columns([2, 1, 1])
            enabled = c1.checkbox(label, key=f"filter_on_{key}")
            minimum = c2.number_input(f"{label}下限", value=low, key=f"filter_min_{key}")
            maximum = c3.number_input(f"{label}上限", value=high, key=f"filter_max_{key}")
            if enabled:
                if minimum > maximum:
                    st.error(f"{label}：下限不可大於上限")
                    return []
                ranges[key] = (minimum, maximum)
    cap, margin, growth, pe = params
    ranges.update(market_cap=(cap * 1e9, None), net_margin=(margin/100, None),
                  revenue_growth=(growth/100, None), pe_ratio=(0.000001, pe))
    st.caption(f"基本面條件沿用側邊欄：市值 ≥ ${cap}B、淨利率 ≥ {margin}%、"
               f"營收成長 ≥ {growth}%、0 < trailing PE ≤ {pe}。啟用條件缺值者不會通過。")
    sort_options = {"市值": "market_cap", "今日漲跌幅": "change_pct", "量比": "volume_ratio", "RSI": "rsi"}
    c1, c2, c3 = st.columns(3)
    sort_key = sort_options[c1.selectbox("排序", list(sort_options))]
    descending = c2.selectbox("方向", ["由高到低", "由低到高"]) == "由高到低"
    limit = c3.selectbox("顯示筆數", [15, 30, 50, 100, 600])
    if st.button("開始掃描 S&P 500", type="primary", key="sp500_scan"):
        try:
            universe = get_sp500()
            st.session_state["scan_universe"] = universe
            bar = st.progress(0.)
            status = st.empty()
            def progress(done, total, ticker):
                bar.progress(done/total)
                status.write(f"分批掃描 {done}/{total} · {ticker}")
            records, failures = scan(universe["symbols"], progress)
            st.session_state["scan_records"] = records
            st.session_state["scan_failures"] = failures
            st.session_state["quote_request_at"] = None
            bar.empty()
            status.empty()
        except Exception as exc:
            st.error(f"掃描失敗：{exc}")
    records = st.session_state.get("scan_records")
    if records is None:
        st.info("按「開始掃描」讀取成分股；免費 Yahoo 完整掃描可能需要數分鐘。")
        return []
    universe = st.session_state.get("scan_universe") or {}
    failures = st.session_state.get("scan_failures") or {}
    total = len(universe.get("symbols") or []) or len(records) + len(failures)
    coverage = len(records) / total * 100 if total else 0.
    partial = {r["ticker"]: r["partial_errors"] for r in records if r.get("partial_errors")}
    st.info(f"掃描覆蓋率：{coverage:.2f}% · 成功 {len(records)} 支／失敗 {len(failures)} 支，共 {total} 支；"
            f"其中 {len(partial)} 支含缺失欄位。")
    st.caption(f"股票池來源：{universe.get('source', '資料不足')} · "
               "成分表取得時間不等於指數調整生效時間。")
    if failures or partial:
        with st.expander("失敗股票與缺失欄位"):
            st.write({"失敗股票": failures, "部分資料不足": partial})
    selected = filter_records(records, ranges, sort_key, descending)
    results = selected[:limit]
    st.session_state["results"] = results
    st.success(f"符合 {len(selected)} 支；顯示 {len(results)} 支。")
    if not results:
        st.info("沒有符合全部條件的股票；缺失數據不會補 0。")
        return []
    columns = {
        "price": "股價", "change_pct": "今日漲跌幅（%）", "volume": "成交量",
        "volume_ratio": "同時段量比", "rsi": "Wilder RSI", "market_cap": "市值",
        "pe_ratio": "Trailing PE", "forward_pe": "Forward PE",
        "net_margin": "淨利率（%）", "revenue_growth": "營收成長（%）", "eps": "EPS",
    }
    output = []
    for row in results:
        view = {"代碼": row["ticker"], "公司": row.get("name", "資料不足")}
        for key, label in columns.items():
            value = row.get(key)
            if key in ("net_margin", "revenue_growth") and value is not None:
                value *= 100
            view[label] = fmt(value)
        view.update({"報價來源時間／年齡": quote_label(row["quote"]),
                     "技術資料最後K棒": row.get("history_time") or "資料不足",
                     "淨利率期間": row.get("margin_period") or "資料不足",
                     "營收成長期間": row.get("growth_period") or "資料不足",
                     "EPS期間": row.get("eps_period") or "資料不足",
                     "基本面取得時間（非財報時間）": row.get("received_at") or "資料不足",
                     "基本面快取": "快取／掃描快照" if row.get("received_at") else "資料不足"})
        output.append(view)
    st.dataframe(pd.DataFrame(output), hide_index=True, use_container_width=True)
    st.caption("量比＝盤中累積量 ÷（前 20 個完整交易日均量 × 今日交易時段已過比例）；"
               "以來源報價時刻計算；這是線性時間近似，未建模開收盤成交量曲線。"
               "K 棒時間是日期／區間起點，不代表當日最後成交時間。財報期間依 Yahoo 最近季度標示，未逐欄查核。")
    # Preserve the original screener's two comparison charts.
    frame = pd.DataFrame(results)
    left, right = st.columns(2)
    cap_chart = px.bar(frame.head(10), x="ticker", y="market_cap", title="市值比較")
    scatter = px.scatter(frame, x="pe_ratio", y="net_margin", size="market_cap",
                         color="revenue_growth", hover_data=["ticker"], title="P/E vs 淨利率（氣泡＝市值）")
    left.plotly_chart(cap_chart, use_container_width=True, key="sp500_cap")
    right.plotly_chart(scatter, use_container_width=True, key="sp500_margin")
    ticker = st.selectbox("選擇標的以診斷或收藏", [r["ticker"] for r in results])
    row = next(r for r in results if r["ticker"] == ticker)
    a, b, c = st.columns(3)
    if a.button("個股診斷", key="scan_diagnose"):
        navigate(row)
    watchlist = st.session_state.setdefault("watchlist", [])
    if b.button("切換收藏", key="scan_watch"):
        watchlist.remove(ticker) if ticker in watchlist else watchlist.append(ticker)
        save_watchlist(watchlist)
        st.rerun()
    if c.button("送前五支到回測", key="scan_backtest"):
        st.session_state["bt_tickers"] = ", ".join(r["ticker"] for r in results[:5])
        st.session_state["bt_from_screener"] = True
        st.rerun()
    return results

"""Compact additions to existing Streamlit pages; all UI labels are Traditional Chinese."""
import pandas as pd
import streamlit as st
from research_store import ResearchStore
from factor_validation import validation_report, ic_composite
from scientific_factors import FACTORS, calibrate, calibrate_reference, SECTOR_ALIASES
from trade_reference import trade_reference, position_size


def text(value, pattern=",.2f", missing="無法取得"):
    return format(value, pattern) if value is not None else missing


@st.cache_data(ttl=60)
def research_state():
    store = ResearchStore()
    latest = store.latest()
    scores, prices = store.history()
    statistics = store.statistics()
    return dict(latest=latest, status=store.status(), report=validation_report(scores, prices),
                alerts=store.alerts(), settings=store.settings(), count=statistics["count"],
                first=statistics["first"])


def render_scientific_panels():
    try:
        state = research_state()
    except Exception as exc:
        with st.expander("因子效能／警報"):
            st.warning(f"研究持久化儲存未就緒（{type(exc).__name__}）；尚未驗證。"
                       "需要發布研究資料表結構，不會以舊回測冒充新驗證。")
        return
    status, latest = state["status"], state["latest"]
    st.caption(f"最後背景更新：{status.get('completed_at') or '尚未完成'}"
               f" · 本輪開始：{status.get('started_at') or '尚未開始'}"
               f" · 最近背景抓取覆蓋率：{text(status.get('coverage'), '.1%', '未知')}")
    if status.get("last_error"):
        st.warning(status["last_error"])
    with st.expander("因子效能"):
        st.caption(f"新制歷史起點：{state['first'] or '尚未建立'}；"
                   f"已保存 {state['count']} 筆股票評分快照。只由啟用後累積，不回填歷史評分。")
        if latest:
            st.caption(f"最近評分快照：{latest['time']}（計算／保存時間，非報價時間）")
            sentiment = latest.get("sentiment") or {}
            st.caption(f"背景 SPY 情緒：{text(sentiment.get('combined'), '.1f', '無數據')}；"
                       f"VIX 反向百分位：{text(sentiment.get('vix_greed'), '.1f', '無數據')} · "
                       f"VIX 資料時間：{sentiment.get('vix_time') or '未知'}")
            st.caption(f"SPY情緒來源抓取時間：{sentiment.get('combined_fetched_at') or '未知'}"
                       f"{'（快取；本次抓取失敗）' if sentiment.get('combined_stale') else ''}；"
                       f"VIX來源抓取時間：{sentiment.get('vix_fetched_at') or '未知'}"
                       f"{'（快取；本次抓取失敗）' if sentiment.get('vix_stale') else ''}；"
                       f"本輪抓取失敗 {len(latest.get('failures',{}))} 支。")
        names = dict(zip(FACTORS, ("動量","估值","品質","增長","波動","持倉與分析師共識","均線結構")))
        rows = [dict(因子=names[r["factor"]], 週期=f"{r['weeks']} 週",
                     **{"Rank IC":r["ic"], "95%區間下限":r["ci_low"], "95%區間上限":r["ci_high"],
                        "多重檢定q":r["q"], "高減低五分位回報差（百分點）":r["spread"]},
                     股票樣本=r["samples"], 不重疊期數=r["cohorts"], 狀態=r["status"])
                for r in state["report"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("入場基準＝評分快照之後首個完整交易日 adjusted close；"
                   "出場＝1／4／12 個曆週後首個交易日收盤，最多容許延後 7 天。"
                   "每個美東觀察日每股只採首筆有效快照，再按入場日去重；每個週期只用互不重疊的入場群組。"
                   "IC＝同一期因子分數與未來回報之 Spearman 等級相關；"
                   "回報差＝最高20%減最低20%平均回報（同分可能擴大組別），各期等權。")
        st.caption("每期至少30股、至少12個不重疊期；按期重抽4999次的95%區間，"
                   "七因子同週期使用BH多重檢定q≤0.05。未排除零：效能未證實。"
                   "IC加權只用1週正向顯著IC，至少3個有效因子，缺失再正規化；"
                   "採截至計算當時已到期回報，不用未來結果。交易成本、下市缺報與存活偏差未完全建模。")
    with st.expander("警報"):
        st.caption("每小時背景執行一次全池評分與SPY情緒更新，不需要開啟頁面。"
                   "前提是伺服器持續運行：Autoscale 可能休眠；需 Reserved VM 或獨立排程部署才可無人值守保證。"
                   "本研究警報只提供站內紀錄，未接駁外部推送；既有通知功能不受影響。")
        cfg = state["settings"]
        score = st.number_input("總分變動門檻（−5 至 +5 刻度，共用設定）", min_value=.1,
                                max_value=10., value=float(cfg.get("score_threshold",1.)), step=.1)
        swing = st.number_input("情緒急變門檻（0–100刻度）", min_value=1.,
                                max_value=100., value=float(cfg.get("sentiment_threshold",20.)))
        watch = st.text_input("背景收藏清單（逗號分隔；共用，不取代原收藏功能）",
                             value=", ".join(cfg.get("watchlist", [])))
        if st.button("保存背景警報設定"):
            import re
            symbols = sorted({x.strip().upper() for x in watch.split(",") if x.strip()})
            if len(symbols) > 100 or any(not re.fullmatch(r"[A-Z0-9^=-]{1,15}", t) for t in symbols):
                st.error("最多100個有效股票代碼。")
            else:
                try:
                    ResearchStore().save_settings(dict(score_threshold=score,
                                                       sentiment_threshold=swing, watchlist=symbols))
                    research_state.clear()
                    st.success("已保存；下次背景執行採用。")
                except Exception:
                    st.error("設定保存失敗；未變更設定。")
        if state["alerts"]:
            st.dataframe(pd.DataFrame(state["alerts"]), hide_index=True, use_container_width=True)
        else:
            st.info("尚無警報；缺失資料不會觸發虛構分數或情緒變動。")
        st.caption("相對強弱穿越＝對 SPY 的21個共同交易日回報差穿越零；"
                   "業績提醒以來源日期／預估窗口起點未來7天為準，每股每個業績日期只記一次。"
                   "校準分數亦受股票池覆蓋改變影響，變動不一定來自公司資料改變。")


def contribution_table(model):
    rows = [dict(因子=model[k].get("label", k), 分數=model[k].get("score"),
                 有效權重=model[k].get("effective_weight"), 總分貢獻=model[k].get("contribution"),
                 有效細項=f"{model[k].get('valid_count',0)}/{model[k].get('total_count',0)}",
                 輸入數據時間=model[k].get("input_time") or "未知")
            for k in FACTORS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("細項先按GICS板塊70%／全池30%百分位校準，再在組內平均。"
               "板塊少於5個有效樣本則只用全池；全池少於20個有效樣本則該細項無數據。"
               "百分位使用同分平均名次：(rank−0.5)/N；分數＝10×百分位−5。"
               "數據完整度只代表有效欄位覆蓋，並非預測成功概率。")
    st.caption("校準只使用本輪成功取得的S&P500資料；不保證全503股每欄均有效。"
               "Yahoo限流與缺值會改變參照樣本及分位名次。")
    st.caption("倍數估值、負債、波動、空頭比例與分析師評級數值愈低愈優；"
               "非正估值倍數剔除。非單調細項以負絕對距離排序：RSI距60、Beta距0.65、"
               "殖利率距5%、機構持股距85%、內部人持股距30%；其餘細項愈高愈優。"
               "這些偏好仍是研究假設，不代表已驗證的經濟定律。")


def render_score_details(row):
    model = row.get("factors")
    if not model:
        st.caption("尚未建立校準分數；重新掃描或等待背景更新。")
        return
    if model.get("reference_note"):
        st.caption(model["reference_note"])
    contribution_table(model)
    ic = model.get("ic_composite")
    st.caption(f"固定權重校準總分：{text(model.get('composite'), '+.2f', '數據不足')} · "
               f"另列 IC 加權綜合分：{text(ic, '+.2f', '尚未驗證／效能未證實')} · "
               f"IC權重計算截至：{model.get('ic_weight_asof') or '尚未建立'} · "
               f"完整度 {model.get('completeness',0):.0%} · "
               f"技術輸入：{model.get('input_time') or '未知'} · "
               f"基本面取得：{row.get('factor_fetched_at') or '未知'}（非發布時間）")


def render_trade_block(ticker, plan, capital, risk):
    st.caption("僅為風險參考計算，非投資建議或委託；所有價格使用 adjusted 日線口徑。")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("入場計算價", text(plan.get("entry")))
    c2.metric("Wilder ATR(14)", text(plan.get("atr14")))
    c3.metric("止損參考（入場−2×ATR）", text(plan.get("stop")))
    c4.metric("建議股數（不槓桿）", text(position_size(plan, capital, risk), ",.0f"))
    st.caption(f"支持／20根K線低點區間：{text(plan.get('entry_low'))}–{text(plan.get('entry_high'))} · "
               f"20個完整交易日平均成交額：${text(plan.get('avg_dollar_volume20'))} · "
               f"估算買賣價差：{text(plan.get('spread_pct'))}"
               f"{'%' if plan.get('spread_pct') is not None else ''} · K線資料：{plan.get('time') or '未知'}")
    if plan.get("low_liquidity"):
        st.warning(f"{ticker} 低流動性：20日平均成交額低於US$10M；滑價風險較高。")
    st.caption("入場區間＝[20根最低低點, min(目前adjusted收盤, 低點＋0.5×ATR)]；"
               "入場計算價為最新K線收盤，非保證成交價。股數＝"
               "min(⌊資金×風險%÷(入場−止損)⌋,⌊資金÷入場⌋)。"
               "價差＝(ask−bid)/中間價；來源報價時間未知，非即時可成交保證。"
               "未計手續費、跳空或滑價，實際虧損可超過設定風險。")


def render_individual_trade(ticker, hist, quote):
    with st.expander("交易計劃（參考）"):
        if hist is not None and len(hist) > 1 and hist.index.to_series().diff().median() < pd.Timedelta(hours=12):
            try:
                from market_data import get_history
                hist = get_history(ticker)
            except Exception:
                hist = None
                st.warning("日線資料不足；不會以小時線冒充20日成交額。")
        capital = st.number_input("參考資金（美元）", min_value=1., value=100000., key=f"capital_{ticker}")
        risk = st.number_input("每筆風險（%）", min_value=.01, max_value=100., value=1., key=f"risk_{ticker}")
        plan = trade_reference(hist, (quote or {}).get("bid"), (quote or {}).get("ask"))
        # Reuse complete-session dollar-volume calculation from the pipeline.
        if hist is not None:
            from research_pipeline import closed_prices
            observations = closed_prices(ticker, hist)
            dates = {t.astimezone(hist.index.tz).date() if hist.index.tz else t.date() for _,t,_ in observations}
            complete = hist.loc[[stamp.date() in dates for stamp in hist.index]]
            avg = trade_reference(complete).get("avg_dollar_volume20")
            plan.update(avg_dollar_volume20=avg, low_liquidity=avg < 10_000_000 if avg is not None else None)
        render_trade_block(ticker, plan, capital, risk)


def apply_reference_calibration(ticker, model, sector=None):
    """Single-name diagnosis uses latest scanned cohort, never a one-stock percentile."""
    try:
        latest = research_state()["latest"]
        references = latest["records"] if latest else []
        if not references:
            return model
        rows = [dict(ticker=r["ticker"], gics_sector=r.get("gics_sector"), factors=r["factors"])
                for r in references if "factors" in r and r["ticker"] in latest["universe"]["symbols"]]
        if any(r["ticker"] == ticker for r in rows):
            for row in rows:
                if row["ticker"] == ticker:
                    row["factors"] = model
            updated = next(r["factors"] for r in calibrate(rows) if r["ticker"] == ticker)
        else:
            updated = calibrate_reference(dict(ticker=ticker,factors=model,
                                               gics_sector=SECTOR_ALIASES.get(sector,sector)),rows)["factors"]
            updated["reference_note"] = "非S&P500成分股；以Yahoo板塊對應／S&P500參照分布評估，未納入因子效能樣本"
        updated["reference_time"] = latest["time"]
        from factor_validation import attach_ic_weighting
        from datetime import datetime, timezone
        attach_ic_weighting([{"factors":updated}], research_state()["report"],
                            datetime.now(timezone.utc).isoformat())
        return updated
    except Exception:
        # Original absolute score remains explicitly labelled, not passed off as calibrated.
        return model


def persist_individual_score(ticker, model, fetched_at):
    if model.get("method") != "sector-percentiles-v1":
        return
    latest = research_state()["latest"]
    row = dict(ticker=ticker, factors=model, factor_fetched_at=fetched_at)
    stamp = ResearchStore().append([row], latest["universe"], {}, {}, [], [], scope="individual")
    research_state.clear()
    st.caption(f"已保存此股前瞻評分快照：{stamp}（不改寫全池掃描結果）")

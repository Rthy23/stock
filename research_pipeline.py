"""The same scoring pipeline is used by foreground scans and the server worker."""
from datetime import datetime, timezone
import pandas as pd
import exchange_calendars as xcals
from analysis import calculate_seven_factors
from data_fetcher import get_factor_data, get_combined_sentiment, get_vix_history
from market_data import get_history, get_fundamentals
from research_context import relative_returns
from research_store import ResearchStore
from scientific_factors import calibrate, calibrate_reference, SECTOR_ALIASES
from trade_reference import trade_reference
from research_alerts import generate_alerts
from factor_validation import validation_report, attach_ic_weighting


def closed_prices(ticker, hist, now=None):
    """Calendar-derived close times, not bar-midnight timestamps or fetch times."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    calendar = xcals.get_calendar("XNYS")
    result = []
    for index, value in hist["Close"].items():
        date = pd.Timestamp(index.date())
        if calendar.is_session(date):
            close = calendar.session_close(date)
            if close <= now and pd.notna(value) and value > 0:
                result.append((ticker, close.to_pydatetime(), float(value)))
    return result


def score_records(records, universe, progress=None, references=None):
    enriched, prices = [], []
    try:
        spy = get_history("SPY")
    except Exception:
        spy = None
    for index, original in enumerate(records):
        row = dict(original)
        ticker = row["ticker"]
        row["gics_sector"] = universe.get("sectors", {}).get(ticker)
        try:
            hist = get_history(ticker)
        except Exception:
            hist = None
        fd = get_factor_data(ticker)
        if not row["gics_sector"]:
            sector = fd.get("sector")
            row["gics_sector"] = SECTOR_ALIASES.get(sector, sector if sector not in (None,"N/A","資料不足") else None)
            row["sector_note"] = "Yahoo板塊對應，非官方成分表GICS分類"
        row["factors"] = calculate_seven_factors(row, hist, fd)
        row["factor_fetched_at"] = fd.get("received_at")
        row["rs1m"] = relative_returns(hist, spy)[1]
        try:
            facts = get_fundamentals(ticker)
            raw = facts.get("raw") or {}
        except Exception:
            facts, raw = {}, {}
        row["earnings"] = dict(start=raw.get("earningsTimestampStart") or raw.get("earningsTimestamp"),
                               confirmed=raw.get("isEarningsDateEstimate") is False)
        row["trade_reference"] = trade_reference(hist, raw.get("bid"), raw.get("ask"))
        row["trade_reference"]["spread_fetched_at"] = facts.get("received_at")
        if hist is not None:
            observations = closed_prices(ticker, hist)
            prices.extend(observations)
            # Complete-session average, excluding today's unfinished daily volume.
            dates = {stamp.astimezone(hist.index.tz).date() if hist.index.tz else stamp.date()
                     for _,stamp,_ in observations}
            complete = hist.loc[[stamp.date() in dates for stamp in hist.index]]
            if len(complete) >= 20 and complete[["Close", "Volume"]].tail(20).notna().all().all():
                dollar = float((complete["Close"].tail(20)*complete["Volume"].tail(20)).mean())
                row["trade_reference"].update(avg_dollar_volume20=dollar, low_liquidity=dollar < 10_000_000)
            else:
                row["trade_reference"].update(avg_dollar_volume20=None, low_liquidity=None)
        enriched.append(row)
        if progress:
            progress(index+1, len(records), ticker)
    members = set(universe.get("symbols", []))
    calibrated = calibrate([r for r in enriched if r["ticker"] in members])
    for row in calibrated:
        row["factors"]["universe_member"] = True
    for row in enriched:
        if row["ticker"] not in members:
            calibrated.append(calibrate_reference(row, references or calibrated))
    try:
        prior_scores, prior_prices = ResearchStore().history()
        now = datetime.now(timezone.utc).isoformat()
        attach_ic_weighting(calibrated, validation_report(prior_scores, prior_prices, now), now)
    except Exception:
        # No evidence must never produce a learned score or a confidence claim.
        for row in calibrated:
            row["factors"].update(ic_composite=None, ic_weights={}, ic_weight_asof=None)
    return calibrated, prices


def market_sentiment(previous=None):
    previous = previous or {}
    result = get_combined_sentiment("SPY")
    now = datetime.now(timezone.utc).isoformat()
    result["combined_stale"] = False
    source_times = [(result.get(key) or {}).get("received_at") for key in ("news_data","social_data")]
    result["combined_fetched_at"] = (min(source_times) if all(isinstance(x,str) for x in source_times)
                                    and result.get("combined") is not None else None)
    if result.get("combined") is None and previous.get("combined") is not None:
        for key in ("combined","news_score","news_data","social_score","social_data","combined_fetched_at"):
            result[key] = previous.get(key)
        result["combined_stale"] = True
    result["vix_greed"] = None
    result["vix_stale"] = False
    try:
        vix = get_vix_history()
        if not vix.empty and pd.notna(vix["greed"].iloc[-1]):
            result.update(vix_greed=float(vix["greed"].iloc[-1]), vix_time=str(vix.index[-1]),
                          vix_fetched_at=vix.attrs.get("received_at"))
    except Exception:
        pass
    if result["vix_greed"] is None and previous.get("vix_greed") is not None:
        result.update(vix_greed=previous["vix_greed"],vix_time=previous.get("vix_time"),
                      vix_fetched_at=previous.get("vix_fetched_at"),vix_stale=True)
    result["attempted_at"] = now
    return result


def persist_scan(records, prices, universe, failures, background=False, store=None, sentiment=None):
    if not records:
        raise ValueError("全池抓取失敗；不覆蓋最近成功的持久化結果")
    store = store or ResearchStore()
    previous = store.latest()
    settings = store.settings()
    sentiment = sentiment if sentiment is not None else market_sentiment((previous or {}).get("sentiment"))
    alerts = generate_alerts(previous, records, sentiment, settings)
    # No prices before prospective history starts are needed or written.
    start = pd.Timestamp(store.earliest() or datetime.now(timezone.utc))
    prices = [(t,d,p) for t,d,p in prices if pd.Timestamp(d) >= start]
    stamp = store.append(records, universe, failures, sentiment, prices, alerts, background)
    return stamp

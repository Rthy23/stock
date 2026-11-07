"""Batched S&P 500 research scan, with explicit partial/missing-data coverage."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from indicators import wilder_rsi
from market_data import get_quote, get_fundamentals, get_history, quote_volume_ratio


def build_record(ticker, force=False):
    quote = get_quote(ticker, force)
    errors = []
    try:
        facts = get_fundamentals(ticker, force)
        facts.pop("raw", None)
    except Exception as exc:
        facts = {"ticker": ticker, "name": ticker}
        errors.append(f"基本面：{exc}")
    try:
        hist = get_history(ticker, force=force)
        # Exclude today's incomplete daily volume from the full-day baseline.
        from market_data import market_status
        today = market_status()["now"].date()
        finished = hist.loc[[stamp.date() < today for stamp in hist.index]]
        avg = float(finished["Volume"].tail(20).mean()) if len(finished) >= 20 else None
        rsi = wilder_rsi(hist["Close"]).iloc[-1]
        rsi = float(rsi) if pd.notna(rsi) else None
        hist_time = hist.index[-1].isoformat()
    except Exception as exc:
        avg = rsi = hist_time = None
        errors.append(f"技術面：{exc}")
    return {**facts, "price": quote["regular"]["price"], "quote": quote,
            "change_pct": quote["change_pct"], "volume": quote["volume"],
            "avg_volume20": avg, "volume_ratio": quote_volume_ratio(quote, avg),
            "rsi": rsi, "history_time": hist_time, "partial_errors": errors}


def scan(symbols, progress=None, force=False, batch_size=6):
    records, failures = [], {}
    for start in range(0, len(symbols), batch_size):
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {pool.submit(build_record, t, force): t for t in symbols[start:start+batch_size]}
            for job in as_completed(jobs):
                ticker = jobs[job]
                try:
                    records.append(job.result())
                except Exception as exc:
                    failures[ticker] = str(exc)
                if progress:
                    progress(len(records) + len(failures), len(symbols), ticker)
    return records, failures


def filter_records(records, ranges, sort_key="market_cap", descending=True):
    selected = []
    for row in records:
        valid = True
        for key, (minimum, maximum) in ranges.items():
            if minimum is None and maximum is None:
                continue
            value = row.get(key)
            if value is None or pd.isna(value):
                valid = False
                break
            if minimum is not None and value < minimum or maximum is not None and value > maximum:
                valid = False
                break
        if valid:
            selected.append(row)
    available = [r for r in selected if r.get(sort_key) is not None]
    missing = [r for r in selected if r.get(sort_key) is None]
    return sorted(available, key=lambda r: r[sort_key], reverse=descending) + missing

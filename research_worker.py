"""Unattended worker: no Streamlit sessions, browser requests, or startup migrations."""
import argparse
import logging
import time
from datetime import datetime, timezone

from market_data import get_sp500, clear_market_cache
from screener_engine import scan
from research_store import ResearchStore
from research_pipeline import score_records, persist_scan, market_sentiment

LOG = logging.getLogger("research_worker")
INTERVAL = 3600


def run_once(store=None):
    store = store or ResearchStore()
    # Session advisory lock is released automatically on exit/crash.
    with store.connect() as lock:
        if not lock.execute("SELECT pg_try_advisory_lock(72190431)").fetchone()[0]:
            return {"skipped": True, "reason": "另一個背景工作正在執行"}
        completed = store.status().get("completed_at")
        if isinstance(completed, str) and (datetime.now(timezone.utc)-datetime.fromisoformat(completed)).total_seconds() < INTERVAL:
            return {"skipped": True, "reason": "未到下一輪全域更新時間"}
        store.mark_worker()
        try:
            # Fetch sentiment before a large quote scan can consume Yahoo's request allowance.
            previous = store.latest() or {}
            sentiment = market_sentiment(previous.get("sentiment"))
            clear_market_cache()
            universe = get_sp500()
            watched = store.settings().get("watchlist", [])
            # S&P constituents establish the distribution; outside names never alter it.
            symbols = universe["symbols"]
            LOG.info("Background scan started: %d S&P constituents", len(symbols))
            def progress(done, total, ticker):
                if done % 50 == 0 or done == total:
                    LOG.info("Background progress: %d/%d (%s)", done, total, ticker)
            records, failures = scan(symbols, progress=progress)
            if not records:
                raise RuntimeError("全部股票抓取失敗；保留上一輪結果")
            records, prices = score_records(records, universe, progress=progress)
            # Non-S&P watchlist earnings and RS alerts use the same sources but no fabricated scores.
            outside = [t for t in watched if t not in symbols]
            if outside:
                extras, extra_failures = scan(outside)
                extras, extra_prices = score_records(extras, universe, references=records)
                records.extend(extras)
                prices.extend(extra_prices)
                failures.update(extra_failures)
            stamp = persist_scan(records, prices, universe, failures, background=True, store=store,
                                 sentiment=sentiment)
            LOG.info("Background scan persisted: %d records, %d failures", len(records), len(failures))
            return {"time": stamp, "records": len(records), "failures": len(failures)}
        except Exception as exc:
            # Never log a driver exception's credential-bearing connection string.
            store.mark_worker(error=f"{type(exc).__name__}：背景更新失敗，保留最近成功結果")
            LOG.error("Background update failed (%s)", type(exc).__name__)
            raise


def serve(stop=None, interval=INTERVAL):
    while stop is None or not stop.is_set():
        started = time.monotonic()
        try:
            run_once()
        except Exception as exc:
            LOG.error("Worker attempt unsuccessful (%s); next scheduled attempt will retry", type(exc).__name__)
        wait = min(60, max(1, interval-(time.monotonic()-started)))
        if stop is None:
            time.sleep(wait)
        elif stop.wait(wait):
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.once:
        print(run_once())
    else:
        serve()

"""Independent last-good caches for Yahoo targets, ratings, changes and holders."""
import json

import pandas as pd
import yfinance as yf

from market_data import get_fundamentals
from public_data_cache import last_success, retry_call


def _table(ticker, field):
    table = getattr(yf.Ticker(ticker), field)
    if table is None or table.empty:
        raise ValueError("Yahoo 沒有提供此項資料")
    if field == "upgrades_downgrades":
        table = table.sort_index(ascending=False).head(100)
    elif field == "recommendations":
        table = table.tail(6)
    return {"rows": json.loads(table.reset_index().to_json(
        orient="records", date_format="iso"))}


def _targets(ticker):
    info = get_fundamentals(ticker)["raw"]
    fields = ("targetMeanPrice", "targetMedianPrice", "targetHighPrice", "targetLowPrice",
              "numberOfAnalystOpinions", "recommendationKey", "recommendationMean")
    result = {field: info.get(field) for field in fields}
    if not any(value is not None for value in result.values()):
        raise ValueError("Yahoo 尚無目標價／共識資料")
    return result


def load_analyst_bundle(ticker, force=False, include_holders=False):
    fields = ["targets", "recommendations", "upgrades_downgrades"]
    if include_holders:
        fields.append("institutional_holders")
    result = {}
    for field in fields:
        try:
            loader = (lambda: _targets(ticker)) if field == "targets" else (
                lambda f=field: _table(ticker, f))
            result[field] = last_success(
                f"yahoo:{ticker}:{field}", lambda: retry_call(loader), 900, force)
        except Exception as exc:
            result[field] = {"data": None, "error": f"{type(exc).__name__}：{exc}",
                             "cache_hit": False, "stale": False, "fetched_at": None}
    return result


def analyst_data(ticker, force=False):
    bundle = load_analyst_bundle(ticker, force=force)
    targets = bundle["targets"]["data"] or {}
    recommendations = bundle["recommendations"]["data"] or {}
    changes = bundle["upgrades_downgrades"]["data"] or {}
    return {
        "target_mean": targets.get("targetMeanPrice"),
        "target_median": targets.get("targetMedianPrice"),
        "target_low": targets.get("targetLowPrice"),
        "target_high": targets.get("targetHighPrice"),
        "num_analysts": targets.get("numberOfAnalystOpinions"),
        "recommendation": targets.get("recommendationKey"),
        "rec_mean": targets.get("recommendationMean"),
        "recs_df": pd.DataFrame(recommendations.get("rows", [])),
        "changes_df": pd.DataFrame(changes.get("rows", [])),
        "received_at": bundle["targets"]["fetched_at"],
        "source_meta": {key: {k: value.get(k) for k in
                              ("fetched_at", "cache_hit", "stale", "error")}
                        for key, value in bundle.items()},
    }

"""Cross-sectional calibration. No historical score snapshots are synthesized."""
from copy import deepcopy
import math

import pandas as pd

FACTORS = ("Momentum", "Value", "Quality", "Growth", "Volatility", "Sentiment", "Macro")
WEIGHTS = dict(zip(FACTORS, (.10, .15, .20, .20, .10, .15, .10)))
METHOD = "sector-percentiles-v1"
SECTOR_ALIASES = {"Technology":"Information Technology", "Healthcare":"Health Care",
                  "Financial Services":"Financials", "Consumer Cyclical":"Consumer Discretionary",
                  "Consumer Defensive":"Consumer Staples", "Basic Materials":"Materials"}
LOWER = {"pe", "pb", "ev_ebitda", "ps", "debt_equity", "vol_30d",
         "atr_pct", "short_pct", "analyst_consensus", "analyst_rating"}


def finite(value):
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def benefit(item):
    """Orient raw inputs, without saturating them at old absolute score cutoffs."""
    value, metric = finite(item.get("value_raw")), item.get("metric")
    if value is None:
        return None
    if metric in {"pe", "pb", "ev_ebitda", "ps"} and value <= 0:
        return None  # Nonpositive valuation multiples aren't 'cheap'.
    targets = {"rsi": 60, "beta": .65, "dividend_yield": 5,
               "inst_ownership": 85, "insider_ownership": 30}
    if metric in targets:
        return -abs(value - targets[metric])
    return -value if metric in LOWER else value


def signal(score):
    if score is None:
        return "數據不足"
    return ("STRONG BUY" if score >= 2.5 else "BUY" if score >= 1 else
            "HOLD" if score >= -1 else "SELL" if score >= -2.5 else "STRONG SELL")


def recombine(model):
    available = [k for k in FACTORS if model[k]["score"] is not None]
    weight = sum(WEIGHTS[k] for k in available)
    for key in FACTORS:
        group = model[key]
        group["effective_weight"] = WEIGHTS[key] / weight if key in available else 0.
        group["contribution"] = (group["score"] * group["effective_weight"]
                                 if key in available else None)
    model["valid_count"] = sum(model[k]["valid_count"] for k in FACTORS)
    model["total_count"] = sum(model[k]["total_count"] for k in FACTORS)
    model["valid_groups"] = len(available)
    model["completeness"] = model["valid_count"] / model["total_count"] if model["total_count"] else 0.
    sufficient = model["completeness"] >= .5 and len(available) >= 4
    model["composite"] = (sum(model[k]["contribution"] for k in available) if sufficient else None)
    model["signal"] = signal(model["composite"])
    return model


def calibrate(rows, min_universe=20, min_sector=5):
    """70% GICS-sector + 30% universe midrank; small/unknown sector uses universe."""
    result = deepcopy(rows)
    for key in FACTORS:
        names = {i["name"] for r in result for i in r["factors"][key]["items"]}
        for name in names:
            items = [(r, i) for r in result for i in r["factors"][key]["items"]
                     if i["name"] == name]
            values = pd.Series({r["ticker"]: benefit(i) for r, i in items}, dtype=float).dropna()
            whole = (values.rank(method="average") - .5) / len(values) if len(values) else values
            sectors = {}
            for r, _ in items:
                sector = r.get("gics_sector")
                if sector and r["ticker"] in values:
                    sectors.setdefault(sector, []).append(r["ticker"])
            ranked = {}
            for sector, tickers in sectors.items():
                subset = values.loc[tickers]
                if len(subset) >= min_sector:
                    ranked[sector] = (subset.rank(method="average") - .5) / len(subset)
            for row, item in items:
                ticker = row["ticker"]
                item["absolute_score"] = item.get("score")
                item["score"] = None
                item["universe_n"] = len(values)
                item["sector_n"] = len(sectors.get(row.get("gics_sector"), []))
                item["calibration"] = "無數據"
                if len(values) >= min_universe and ticker in values:
                    local = ranked.get(row.get("gics_sector"))
                    percentile = (.7 * local[ticker] + .3 * whole[ticker]
                                  if local is not None else whole[ticker])
                    item["score"] = float(10 * percentile - 5)
                    item["calibration"] = "板塊70%＋全池30%" if local is not None else "全池（板塊樣本不足）"
        for row in result:
            group = row["factors"][key]
            valid = [i["score"] for i in group["items"] if i["score"] is not None]
            group.update(score=sum(valid)/len(valid) if valid else None,
                         valid_count=len(valid), total_count=len(group["items"]),
                         status="有效" if valid else "無數據")
    for row in result:
        row["factors"]["method"] = METHOD
        recombine(row["factors"])
    return result


def calibrate_reference(row, references, min_universe=20, min_sector=5):
    """Out-of-pool watchlist name: empirical CDF against S&P peers, not its own pool."""
    result = deepcopy(row)
    for key in FACTORS:
        group = result["factors"][key]
        for item in group["items"]:
            peers = [(r.get("gics_sector"), benefit(i)) for r in references
                     for i in r["factors"][key]["items"] if i["name"] == item["name"]]
            whole = [v for _,v in peers if v is not None]
            local = [v for s,v in peers if v is not None and s == result.get("gics_sector")]
            value = benefit(item)
            item["absolute_score"] = item.get("score")
            item.update(score=None, universe_n=len(whole),sector_n=len(local),calibration="無數據")
            if value is not None and len(whole) >= min_universe:
                def percentile(values):
                    return (sum(v < value for v in values)+.5*sum(v == value for v in values))/len(values)
                p = (.7*percentile(local)+.3*percentile(whole)
                     if result.get("gics_sector") and len(local) >= min_sector else percentile(whole))
                item.update(score=10*p-5,calibration="非成分股；以S&P500參照分布評估")
        valid = [i["score"] for i in group["items"] if i["score"] is not None]
        group.update(score=sum(valid)/len(valid) if valid else None,
                     valid_count=len(valid),total_count=len(group["items"]),status="有效" if valid else "無數據")
    result["factors"].update(method=METHOD,universe_member=False)
    recombine(result["factors"])
    return result

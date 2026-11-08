"""Prospective, date-blocked factor validation. Never creates historical scores."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scientific_factors import FACTORS, finite

MIN_STOCKS = 30
MIN_COHORTS = 12


def forward_pairs(scores, prices, weeks, as_of=None):
    """Enter first completed close AFTER score; exit first close >= entry+7*w days."""
    now = pd.Timestamp(as_of or datetime.now(timezone.utc))
    prices_by_ticker = {}
    for price in prices:
        stamp = pd.Timestamp(price["time"])
        if stamp <= now and finite(price["close"]) is not None and price["close"] > 0:
            prices_by_ticker.setdefault(price["ticker"], []).append((stamp, price["close"]))
    for ticker in prices_by_ticker:
        prices_by_ticker[ticker] = sorted(dict(prices_by_ticker[ticker]).items())
    result, seen, observed_days = [], set(), set()
    for score in sorted(scores, key=lambda r: r["time"]):
        if (pd.Timestamp(score["time"]) > now or score["factors"].get("composite") is None
                or score["factors"].get("universe_member") is False):
            continue
        observed = (score["ticker"], pd.Timestamp(score["time"]).tz_convert("America/New_York").date())
        if observed in observed_days:
            continue
        observed_days.add(observed)
        series = prices_by_ticker.get(score["ticker"], [])
        entry = next(((t,p) for t,p in series if t > pd.Timestamp(score["time"])), None)
        if entry is None:
            continue
        key = (score["ticker"], entry[0].date())
        if key in seen:
            continue
        target = (entry[0].tz_convert("America/New_York") + pd.Timedelta(weeks=weeks)).date()
        last = target + pd.Timedelta(days=7)
        exit_ = next(((t,p) for t,p in series
                      if target <= t.tz_convert("America/New_York").date() <= last), None)
        if exit_ is None:
            continue
        seen.add(key)
        result.append(dict(ticker=score["ticker"], cohort=entry[0].date().isoformat(),
                           time=score["time"], return_pct=100*(exit_[1]/entry[1]-1),
                           factors=score["factors"]))
    return result


def validation_report(scores, prices, as_of=None, min_stocks=MIN_STOCKS, min_cohorts=MIN_COHORTS):
    report = []
    for weeks in (1, 4, 12):
        pairs = forward_pairs(scores, prices, weeks, as_of)
        all_days = sorted({r["cohort"] for r in pairs})
        # Non-overlapping cohorts, so intraday and overlapping daily scores cannot inflate N.
        days = []
        for day in all_days:
            if not days or pd.Timestamp(day) >= pd.Timestamp(days[-1])+pd.Timedelta(weeks=weeks):
                days.append(day)
        horizon = []
        for factor in FACTORS:
            ics, spreads, sizes = [], [], []
            for day in days:
                rows = [r for r in pairs if r["cohort"] == day and
                        finite(r["factors"].get(factor, {}).get("score")) is not None]
                if len(rows) < min_stocks:
                    continue
                values = pd.Series([r["factors"][factor]["score"] for r in rows])
                returns = pd.Series([r["return_pct"] for r in rows])
                if values.nunique() < 2 or returns.nunique() < 2:
                    continue
                ic = values.rank(method="average").corr(returns.rank(method="average"))
                lower, upper = values.quantile(.2), values.quantile(.8)
                if not lower < upper or pd.isna(ic):
                    continue
                ics.append(float(ic))
                spreads.append(float(returns[values >= upper].mean()-returns[values <= lower].mean()))
                sizes.append(len(rows))
            entry = dict(factor=factor, weeks=weeks, cohorts=len(ics), samples=sum(sizes),
                         ic=float(np.mean(ics)) if ics else None,
                         spread=float(np.mean(spreads)) if spreads else None,
                         ci_low=None, ci_high=None, p=None, q=None,
                         status="尚未驗證（樣本不足）", eligible=False)
            if len(ics) >= min_cohorts:
                rng = np.random.default_rng(20261108)
                boot = rng.choice(ics, size=(4999, len(ics)), replace=True).mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                entry.update(ci_low=float(lo), ci_high=float(hi),
                             p=float((1+np.sum(np.abs(boot-entry["ic"]) >= abs(entry["ic"])))/5000))
            horizon.append(entry)
        # Benjamini–Hochberg across all seven factors, separately for each horizon.
        valid = sorted((r for r in horizon if r["p"] is not None), key=lambda r: r["p"])
        q = 1.
        for rank in range(len(valid), 0, -1):
            row = valid[rank-1]
            q = min(q, row["p"]*len(FACTORS)/rank)
            row["q"] = q
            row["eligible"] = q <= .05 and row["ci_low"] > 0
            row["status"] = ("已觀察正向關聯（非預測保證）" if row["eligible"] else
                             "負向關聯（不採正向加權）" if q <= .05 and row["ci_high"] < 0 else
                             "效能未證實（IC 未能排除零）")
        report.extend(horizon)
    return report


def ic_composite(model, report):
    eligible = {r["factor"]: r["ic"] for r in report if r["weeks"] == 1 and r["eligible"]}
    available = {k:w for k,w in eligible.items() if finite(model.get(k, {}).get("score")) is not None}
    if model.get("composite") is None or len(available) < 3:
        return None
    return sum(model[k]["score"]*w for k,w in available.items()) / sum(available.values())


def attach_ic_weighting(rows, report, as_of):
    """Freeze weights with the newly computed score, not retrospectively on old rows."""
    eligible = {r["factor"]:r["ic"] for r in report if r["weeks"] == 1 and r["eligible"]}
    for row in rows:
        model = row["factors"]
        score = ic_composite(model, report)
        available = {k:w for k,w in eligible.items() if finite(model.get(k, {}).get("score")) is not None}
        model["ic_composite"] = score
        model["ic_weights"] = ({k:w/sum(available.values()) for k,w in available.items()}
                               if score is not None else {})
        model["ic_weight_asof"] = as_of
    return rows

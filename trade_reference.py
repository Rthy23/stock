"""Adjusted daily-bar reference calculations, not execution or investment advice."""
import math
import pandas as pd
from scientific_factors import finite


def wilder_atr(hist, period=14):
    if hist is None or not {"High", "Low", "Close"} <= set(hist.columns):
        return None
    previous = hist["Close"].shift()
    tr = pd.concat([hist["High"]-hist["Low"], (hist["High"]-previous).abs(),
                    (hist["Low"]-previous).abs()], axis=1).max(axis=1)
    tr[hist[["High", "Low", "Close"]].isna().any(axis=1)] = float("nan")
    atr = None
    for index in range(period-1, len(tr)):
        if atr is None or pd.isna(tr.iloc[index]):
            window = tr.iloc[index-period+1:index+1]
            atr = float(window.mean()) if window.notna().all() else None
        else:
            atr = ((period-1)*atr + float(tr.iloc[index])) / period
    return finite(atr)


def trade_reference(hist, bid=None, ask=None):
    result = dict(atr14=None, entry=None, entry_low=None, entry_high=None, stop=None,
                  avg_dollar_volume20=None, spread_pct=None, low_liquidity=None, time=None)
    if hist is None or hist.empty:
        return result
    result["time"] = str(hist.index[-1])
    entry = finite(hist["Close"].iloc[-1])
    atr = wilder_atr(hist)
    result.update(entry=entry, atr14=atr)
    if entry is not None and entry > 0 and atr is not None:
        low = finite(hist["Low"].tail(20).min()) if len(hist) >= 20 else None
        result["stop"] = entry-2*atr if entry > 2*atr else None
        if low is not None and low > 0:
            result.update(entry_low=min(low, entry), entry_high=min(entry, low+.5*atr))
    if {"Close", "Volume"} <= set(hist.columns) and len(hist) >= 20:
        dollar = hist["Close"].tail(20) * hist["Volume"].tail(20)
        if dollar.notna().all() and (dollar >= 0).all():
            result["avg_dollar_volume20"] = float(dollar.mean())
            result["low_liquidity"] = result["avg_dollar_volume20"] < 10_000_000
    bid, ask = finite(bid), finite(ask)
    if bid is not None and ask is not None and 0 < bid <= ask:
        result["spread_pct"] = 100*(ask-bid)/((ask+bid)/2)
    return result


def position_size(plan, capital=100000., risk_pct=1.):
    if not (capital > 0 and 0 < risk_pct <= 100):
        raise ValueError("資金與每筆風險百分比須為正值")
    entry, stop = plan.get("entry"), plan.get("stop")
    if entry is None or stop is None or not 0 < stop < entry:
        return None
    return max(0, min(math.floor(capital*risk_pct/100/(entry-stop)),
                      math.floor(capital/entry)))

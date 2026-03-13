# ═══════════════════════════════════════════════════════════════════════════════
# mpf_strategy.py  —  MPF Quantitative Strategy Engine
# RS (Relative Strength) + SMA trend signals for each fund's proxy ETF
# Outputs monthly add/reduce/switch-to-defensive recommendations
# ═══════════════════════════════════════════════════════════════════════════════
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

_MODULE = "mpf_strategy"


def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── Data fetching ──────────────────────────────────────────────────────────────
def _fetch_close(ticker: str, years: int = 1) -> pd.Series:
    """Fetch adjusted close prices for a ticker."""
    try:
        end   = datetime.today()
        start = end - timedelta(days=int(years * 365.25))
        hist  = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist["Close"].dropna()
    except Exception as e:
        print(_err(f"_fetch_close[{ticker}]", e))
        return pd.Series(dtype=float)


# ── RS (Relative Strength) calculation ────────────────────────────────────────
def calc_rs(etf_series: pd.Series, base_series: pd.Series,
            period: int = 20) -> dict:
    """
    Relative Strength of ETF vs base (SPY).

    Returns dict:
      rs_raw      - latest RS ratio
      rs_sma20    - 20-day SMA of RS ratio
      rs_trend    - 'rising' | 'falling' | 'flat'
      rs_signal   - 'strong' | 'weak' | 'neutral'
    """
    try:
        common = etf_series.index.intersection(base_series.index)
        if len(common) < period + 5:
            return {"rs_raw": None, "rs_sma20": None, "rs_trend": "flat", "rs_signal": "neutral"}
        etf  = etf_series.reindex(common)
        base = base_series.reindex(common)
        rs   = (etf / base).replace([np.inf, -np.inf], np.nan).dropna()
        if rs.empty:
            return {"rs_raw": None, "rs_sma20": None, "rs_trend": "flat", "rs_signal": "neutral"}
        sma  = rs.rolling(period).mean()
        rs_now   = float(rs.iloc[-1])
        sma_now  = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else rs_now
        sma_prev = float(sma.iloc[-period // 2]) if len(sma) > period // 2 else sma_now
        trend = ("rising" if sma_now > sma_prev * 1.005 else
                 "falling" if sma_now < sma_prev * 0.995 else "flat")
        signal = ("strong" if rs_now > sma_now * 1.02 else
                  "weak"   if rs_now < sma_now * 0.98 else "neutral")
        return {
            "rs_raw":   round(rs_now, 4),
            "rs_sma20": round(sma_now, 4),
            "rs_trend": trend,
            "rs_signal": signal,
        }
    except Exception as e:
        print(_err("calc_rs", e))
        return {"rs_raw": None, "rs_sma20": None, "rs_trend": "flat", "rs_signal": "neutral"}


# ── SMA trend analysis ─────────────────────────────────────────────────────────
def calc_sma_signals(prices: pd.Series) -> dict:
    """
    Compute SMA20 and SMA50; detect crosses.

    Returns:
      price      - latest price
      sma20      - latest 20-day SMA
      sma50      - latest 50-day SMA
      above_sma20 - bool
      above_sma50 - bool
      golden     - bool (sma20 > sma50)
      trend      - 'uptrend' | 'downtrend' | 'sideways'
      momentum   - % change vs 20 sessions ago
    """
    try:
        if len(prices) < 55:
            return {
                "price": None, "sma20": None, "sma50": None,
                "above_sma20": None, "above_sma50": None,
                "golden": None, "trend": "sideways", "momentum": 0.0,
            }
        sma20 = prices.rolling(20).mean()
        sma50 = prices.rolling(50).mean()
        p     = float(prices.iloc[-1])
        s20   = float(sma20.iloc[-1])
        s50   = float(sma50.iloc[-1])
        mom   = (p / float(prices.iloc[-20]) - 1) * 100 if len(prices) >= 20 else 0.0
        golden = s20 > s50
        trend = ("uptrend"   if p > s20 > s50 else
                 "downtrend" if p < s20 < s50 else
                 "sideways")
        return {
            "price":       round(p, 2),
            "sma20":       round(s20, 2),
            "sma50":       round(s50, 2),
            "above_sma20": p > s20,
            "above_sma50": p > s50,
            "golden":      golden,
            "trend":       trend,
            "momentum":    round(mom, 2),
        }
    except Exception as e:
        print(_err("calc_sma_signals", e))
        return {
            "price": None, "sma20": None, "sma50": None,
            "above_sma20": None, "above_sma50": None,
            "golden": None, "trend": "sideways", "momentum": 0.0,
        }


# ── Recommendation engine ──────────────────────────────────────────────────────
def _make_recommendation(sma: dict, rs: dict, category: str) -> dict:
    """
    Combine SMA and RS signals into a monthly action recommendation.

    Returns:
      action     - 'add' | 'hold' | 'reduce' | 'switch_defensive'
      label      - Chinese label
      color      - hex color for UI
      reason     - short explanation (Chinese)
      confidence - 'high' | 'medium' | 'low'
    """
    trend   = sma.get("trend", "sideways")
    rs_sig  = rs.get("rs_signal", "neutral")
    rs_trnd = rs.get("rs_trend", "flat")
    mom     = sma.get("momentum", 0.0) or 0.0
    is_def  = category in ("固定收益", "保本")

    # ── Pure defensive / bond funds: always hold unless RS collapsing ──────
    if is_def:
        if rs_sig == "weak" and rs_trnd == "falling":
            return {
                "action":     "switch_defensive",
                "label":      "考慮轉出防禦",
                "color":      "#FF8C00",
                "reason":     "防禦型基金相對強度下滑，可考慮縮短久期",
                "confidence": "medium",
            }
        return {
            "action":     "hold",
            "label":      "持平觀察",
            "color":      "#888888",
            "reason":     "防禦型配置保持不變",
            "confidence": "high",
        }

    # ── Strong add signal ──────────────────────────────────────────────────
    if trend == "uptrend" and rs_sig == "strong" and rs_trnd == "rising" and mom > 3:
        return {
            "action":     "add",
            "label":      "強烈建議加碼",
            "color":      "#00CC44",
            "reason":     "ETF 代理處於上升趨勢，相對強度高且仍在上升",
            "confidence": "high",
        }

    # ── Moderate add ──────────────────────────────────────────────────────
    if trend in ("uptrend", "sideways") and rs_sig != "weak" and mom > 0:
        return {
            "action":     "add",
            "label":      "可適度加碼",
            "color":      "#66DD66",
            "reason":     "ETF 代理走勢偏正面，相對市場尚可",
            "confidence": "medium",
        }

    # ── Defensive switch ──────────────────────────────────────────────────
    if trend == "downtrend" and rs_sig == "weak" and rs_trnd == "falling" and mom < -5:
        return {
            "action":     "switch_defensive",
            "label":      "建議轉入防禦配置",
            "color":      "#FF4B4B",
            "reason":     "ETF 代理跌破 SMA20/50，相對強度弱且下降。建議部分轉為固定收益基金",
            "confidence": "high",
        }

    # ── Reduce ────────────────────────────────────────────────────────────
    if trend == "downtrend" or (rs_sig == "weak" and mom < -2):
        return {
            "action":     "reduce",
            "label":      "建議減持",
            "color":      "#FF8C00",
            "reason":     "ETF 代理走勢偏弱，可考慮降低此基金比重",
            "confidence": "medium",
        }

    # ── Hold ──────────────────────────────────────────────────────────────
    return {
        "action":     "hold",
        "label":      "持平觀察",
        "color":      "#888888",
        "reason":     "訊號不明確，維持現有配置，等待更清晰方向",
        "confidence": "low",
    }


# ── Defensive allocation calculator ───────────────────────────────────────────
def calc_defensive_allocation(portfolio: list[dict],
                              signals: list[dict]) -> dict:
    """
    When overall market is bearish, calculate how much to shift to defensive.

    Returns:
      suggested_defensive_pct  - total % to hold in defensive assets
      current_defensive_pct    - current % in defensive/bond funds
      shift_amount_pct         - how much to add to defensive
      defensive_target_funds   - list of defensive fund names in portfolio
    """
    try:
        total_pct       = sum(p.get("pct", 0) for p in portfolio)
        defensive_cats  = ("固定收益", "保本")

        current_def = sum(
            p.get("pct", 0) for p in portfolio
            if p.get("category", "") in defensive_cats
        )
        current_def_pct = (current_def / total_pct * 100) if total_pct else 0

        # Count how many funds have switch_defensive signal
        n_switch = sum(1 for s in signals if s.get("action") == "switch_defensive")
        n_reduce = sum(1 for s in signals if s.get("action") == "reduce")
        n_total  = len(signals) or 1

        # Target: scale up defensive allocation proportionally to bear signals
        bear_ratio = (n_switch * 2 + n_reduce) / (n_total * 2)
        target_def_pct = min(60.0, current_def_pct + bear_ratio * 30.0)

        def_funds = [p["fund_name"] for p in portfolio
                     if p.get("category", "") in defensive_cats]

        return {
            "current_defensive_pct":  round(current_def_pct, 1),
            "suggested_defensive_pct": round(target_def_pct, 1),
            "shift_amount_pct":       round(max(0, target_def_pct - current_def_pct), 1),
            "defensive_target_funds": def_funds,
        }
    except Exception as e:
        print(_err("calc_defensive_allocation", e))
        return {
            "current_defensive_pct":   0.0,
            "suggested_defensive_pct": 0.0,
            "shift_amount_pct":        0.0,
            "defensive_target_funds":  [],
        }


# ── Main strategy entry point ──────────────────────────────────────────────────
def get_strategy_signals(portfolio: list[dict]) -> dict:
    """
    For each MPF fund in portfolio, analyze its proxy ETF.
    Returns full strategy analysis.

    Output structure:
    {
      "signals": [
        {
          "fund_name":    str,
          "etf":          str,
          "category":     str,
          "pct":          float,
          "sma":          dict,
          "rs":           dict,
          "recommendation": dict,
        }, ...
      ],
      "market_condition": "bullish" | "bearish" | "neutral",
      "defensive_advice": dict,
      "error": None | str,
    }
    """
    try:
        if not portfolio:
            return {"signals": [], "market_condition": "neutral",
                    "defensive_advice": {}, "error": None}

        # ── Fetch SPY as RS base ───────────────────────────────────────────
        spy = _fetch_close("SPY", years=1)

        signals = []
        for fund in portfolio:
            etf      = fund.get("etf", "N/A")
            category = fund.get("category", "")
            fname    = fund.get("fund_name", "")
            pct      = fund.get("pct", 0)

            if etf == "N/A":
                signals.append({
                    "fund_name":      fname,
                    "etf":            etf,
                    "category":       category,
                    "pct":            pct,
                    "sma":            {},
                    "rs":             {},
                    "recommendation": {
                        "action": "hold", "label": "持平觀察",
                        "color": "#888", "reason": "無對標 ETF，無法分析",
                        "confidence": "low",
                    },
                })
                continue

            prices = _fetch_close(etf, years=1)
            sma    = calc_sma_signals(prices)
            rs     = calc_rs(prices, spy) if not prices.empty and not spy.empty else {
                "rs_raw": None, "rs_sma20": None,
                "rs_trend": "flat", "rs_signal": "neutral",
            }
            rec = _make_recommendation(sma, rs, category)

            signals.append({
                "fund_name":      fname,
                "etf":            etf,
                "category":       category,
                "pct":            pct,
                "sma":            sma,
                "rs":             rs,
                "recommendation": rec,
            })

        # ── Overall market condition ───────────────────────────────────────
        n_bear = sum(1 for s in signals
                     if s["recommendation"]["action"] in ("reduce", "switch_defensive"))
        n_bull = sum(1 for s in signals
                     if s["recommendation"]["action"] == "add")
        n_tot  = len(signals) or 1

        if n_bear / n_tot >= 0.5:
            market_cond = "bearish"
        elif n_bull / n_tot >= 0.5:
            market_cond = "bullish"
        else:
            market_cond = "neutral"

        def_advice = calc_defensive_allocation(portfolio, [s["recommendation"] for s in signals])

        return {
            "signals":          signals,
            "market_condition": market_cond,
            "defensive_advice": def_advice,
            "error":            None,
        }

    except Exception as e:
        return {"signals": [], "market_condition": "neutral",
                "defensive_advice": {}, "error": str(e)}


# ── ETF historical comparison ──────────────────────────────────────────────────
def get_etf_vs_spy_history(etfs: list[str], years: int = 1) -> pd.DataFrame:
    """
    Return normalized (base=100) price history for ETFs + SPY.
    Useful for the MPF historical comparison chart.
    """
    try:
        tickers = list(dict.fromkeys(etfs + ["SPY"]))
        end     = datetime.today()
        start   = end - timedelta(days=int(years * 365.25))
        frames  = {}
        for t in tickers:
            s = _fetch_close(t, years)
            if not s.empty:
                frames[t] = (s / s.iloc[0] * 100).rename(t)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna(how="all")
    except Exception as e:
        print(_err("get_etf_vs_spy_history", e))
        return pd.DataFrame()

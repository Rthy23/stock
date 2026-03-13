# ═══════════════════════════════════════════════════════════════════════════════
# backtest_engine.py  —  Pure quantitative backtest logic (no Streamlit)
# ═══════════════════════════════════════════════════════════════════════════════
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

_MODULE = "backtest_engine"


def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── Data fetching ──────────────────────────────────────────────────────────────
def fetch_price_history(tickers: list, years: int) -> pd.DataFrame:
    """Return DataFrame of adjusted Close prices for each ticker."""
    try:
        end   = datetime.today()
        start = end - timedelta(days=int(years * 365.25))
        frames = {}
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                )
                if not hist.empty:
                    frames[t] = hist["Close"].rename(t)
            except Exception as _inner:
                print(_err(f"fetch_price_history[{t}]", _inner))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna(how="all")
    except Exception as e:
        print(_err("fetch_price_history", e))
        return pd.DataFrame()


# ── Portfolio construction ─────────────────────────────────────────────────────
def calc_portfolio_series(prices_df: pd.DataFrame) -> pd.Series:
    """Equal-weight portfolio, normalised to 100 at first valid date."""
    try:
        clean = prices_df.dropna(how="any")
        if clean.empty:
            return pd.Series(dtype=float)
        normed = clean.div(clean.iloc[0]) * 100
        return normed.mean(axis=1).rename("Portfolio")
    except Exception as e:
        print(_err("calc_portfolio_series", e))
        return pd.Series(dtype=float)


# ── Performance metrics ────────────────────────────────────────────────────────
def calc_metrics(series: pd.Series, rf: float = 0.05) -> dict:
    """
    Compute CAGR, annualised Sharpe (rf = 5%), Max Drawdown, annualised vol.
    Returns dict with keys: cagr, sharpe, max_dd, vol  (all as floats or None).
    """
    try:
        if series is None or series.empty or len(series) < 5:
            return {"cagr": None, "sharpe": None, "max_dd": None, "vol": None}
        years = (series.index[-1] - series.index[0]).days / 365.25
        if years <= 0:
            return {"cagr": None, "sharpe": None, "max_dd": None, "vol": None}
        total_ret  = (series.iloc[-1] / series.iloc[0]) - 1
        cagr       = (1 + total_ret) ** (1 / years) - 1
        daily_ret  = series.pct_change().dropna()
        vol_annual = daily_ret.std() * np.sqrt(252)
        sharpe     = (cagr - rf) / vol_annual if vol_annual > 0 else 0.0
        rolling_max = series.cummax()
        drawdown    = (series - rolling_max) / rolling_max
        max_dd      = float(drawdown.min())
        return {
            "cagr":   float(cagr),
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
            "vol":    float(vol_annual),
        }
    except Exception as e:
        print(_err("calc_metrics", e))
        return {"cagr": None, "sharpe": None, "max_dd": None, "vol": None}


# ── Main backtest entry point ──────────────────────────────────────────────────
def run_backtest(
    ticker_list:      list,
    window_years:     int,
    benchmark_ticker: str = "SPY",
) -> dict:
    """
    Run equal-weight backtest for ticker_list vs benchmark.

    Returns dict with:
      portfolio_series   pd.Series  (daily, normalised to 100)
      benchmark_series   pd.Series  (daily, normalised to 100)
      portfolio_metrics  dict       CAGR / Sharpe / MaxDD / Vol
      benchmark_metrics  dict
      alpha              float | None   portfolio CAGR − benchmark CAGR
      high_vol_flags     dict  {ticker: bool}  vol > 35 % annually
      per_ticker_metrics dict  {ticker: metrics_dict}
      pf_tickers         list   tickers successfully loaded
      benchmark_ticker   str
      window_years       int
      error              str | None
    """
    try:
        all_tickers = list(dict.fromkeys(ticker_list + [benchmark_ticker]))
        prices      = fetch_price_history(all_tickers, window_years)

        if prices.empty:
            return {"error": "無法獲取任何價格數據，請確認股票代碼是否正確。"}

        pf_tickers = [t for t in ticker_list if t in prices.columns]
        if not pf_tickers:
            return {"error": "所有指定股票均無法獲取資料，請檢查代碼。"}

        # ── Portfolio ──────────────────────────────────────────────────────
        pf_prices       = prices[pf_tickers]
        portfolio_series = calc_portfolio_series(pf_prices)

        # ── Benchmark ─────────────────────────────────────────────────────
        bm_series = None
        if benchmark_ticker in prices.columns:
            bm_raw    = prices[[benchmark_ticker]].dropna()
            bm_normed = bm_raw.div(bm_raw.iloc[0]) * 100
            bm_series = bm_normed[benchmark_ticker].rename(benchmark_ticker)

        # ── Align on common dates ──────────────────────────────────────────
        if bm_series is not None and not portfolio_series.empty:
            common = portfolio_series.index.intersection(bm_series.index)
            portfolio_series = portfolio_series.loc[common]
            bm_series        = bm_series.loc[common]

        # ── Metrics ───────────────────────────────────────────────────────
        pf_metrics = calc_metrics(portfolio_series)
        bm_metrics = calc_metrics(bm_series) if bm_series is not None else {}

        alpha = None
        if pf_metrics["cagr"] is not None and bm_metrics.get("cagr") is not None:
            alpha = pf_metrics["cagr"] - bm_metrics["cagr"]

        # ── Per-ticker metrics + high-vol flags ───────────────────────────
        per_ticker_metrics = {}
        high_vol_flags     = {}
        for t in pf_tickers:
            col = prices[t].dropna()
            if len(col) < 5:
                continue
            s = (col / col.iloc[0] * 100).rename(t)
            m = calc_metrics(s)
            per_ticker_metrics[t] = m
            high_vol_flags[t]     = (m["vol"] is not None and m["vol"] > 0.35)

        return {
            "portfolio_series":   portfolio_series,
            "benchmark_series":   bm_series,
            "portfolio_metrics":  pf_metrics,
            "benchmark_metrics":  bm_metrics,
            "alpha":              alpha,
            "high_vol_flags":     high_vol_flags,
            "per_ticker_metrics": per_ticker_metrics,
            "pf_tickers":         pf_tickers,
            "benchmark_ticker":   benchmark_ticker,
            "window_years":       window_years,
            "error":              None,
        }
    except Exception as e:
        print(_err("run_backtest", e))
        return {"error": str(e)}


# ── Rebalancing helper (used by MPF module too) ───────────────────────────────
def calc_rebalance(
    current_pct:    dict,
    target_pct:     dict,
    monthly_budget: float = 2400.0,
) -> dict:
    """
    Given current portfolio weights and target weights, calculate how to
    allocate monthly_budget to bring the portfolio closer to target.

    Parameters
    ----------
    current_pct : {label: float}  current allocation in %  (should sum ≈ 100)
    target_pct  : {label: float}  target  allocation in %  (should sum ≈ 100)
    monthly_budget : float        HKD or USD monthly contribution

    Returns
    -------
    {label: {"allocation": float, "deviation": float, "action": str}}
    """
    try:
        labels     = list(target_pct.keys())
        deviations = {k: target_pct[k] - current_pct.get(k, 0.0) for k in labels}
        underweight = {k: v for k, v in deviations.items() if v > 0}

        if not underweight:
            total_target = sum(target_pct.values()) or 1.0
            allocation = {k: monthly_budget * target_pct[k] / total_target
                          for k in labels}
        else:
            total_gap  = sum(underweight.values())
            allocation = {k: monthly_budget * underweight.get(k, 0.0) / total_gap
                          for k in labels}

        result = {}
        for k in labels:
            dev = deviations[k]
            result[k] = {
                "allocation": round(allocation.get(k, 0.0), 1),
                "deviation":  round(dev, 1),
                "action":     "加碼 ▲" if dev > 2 else ("減持 ▼" if dev < -2 else "持平 —"),
            }
        return result
    except Exception as e:
        print(_err("calc_rebalance", e))
        return {}

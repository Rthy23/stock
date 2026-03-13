# ═══════════════════════════════════════════════════════════════════════════════
# backtest_engine.py  —  Quantitative backtest engine (no Streamlit)
# Strategies: Buy-Hold · RSI Mean-Reversion (with SMA200 filter + stop-loss)
# ═══════════════════════════════════════════════════════════════════════════════
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

_MODULE = "backtest_engine"


def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── RSI preset thresholds by asset class ──────────────────────────────────────
_RSI_PRESETS_DATA: dict[str, dict] = {
    "科技股":    {"buy": 30, "sell": 70, "sma_period": 200, "stop_loss": 0.12},
    "高波動股":  {"buy": 25, "sell": 75, "sma_period": 200, "stop_loss": 0.15},
    "防禦型股票": {"buy": 40, "sell": 60, "sma_period": 200, "stop_loss": 0.08},
    "自定義":    {"buy": 30, "sell": 70, "sma_period": 200, "stop_loss": 0.10},
}


def get_rsi_presets() -> dict[str, dict]:
    """Return a copy of all RSI preset configurations keyed by asset class."""
    return dict(_RSI_PRESETS_DATA)


# ── Data fetching ─────────────────────────────────────────────────────────────
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


def fetch_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    """Return OHLCV DataFrame for a single ticker (for technical analysis)."""
    try:
        end   = datetime.today()
        start = end - timedelta(days=int(years * 365.25))
        hist  = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        print(_err(f"fetch_ohlcv[{ticker}]", e))
        return pd.DataFrame()


# ── Technical indicators ──────────────────────────────────────────────────────
def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    try:
        delta  = prices.diff()
        gain   = delta.clip(lower=0)
        loss   = (-delta).clip(lower=0)
        avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
        rs     = avg_g / avg_l.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).rename("RSI")
    except Exception as e:
        print(_err("calc_rsi", e))
        return pd.Series(dtype=float)


def calc_macd(prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (MACD line, signal line, histogram)."""
    try:
        ema12  = prices.ewm(span=12, adjust=False).mean()
        ema26  = prices.ewm(span=26, adjust=False).mean()
        macd   = (ema12 - ema26).rename("MACD")
        signal = macd.ewm(span=9, adjust=False).mean().rename("Signal")
        hist   = (macd - signal).rename("Histogram")
        return macd, signal, hist
    except Exception as e:
        print(_err("calc_macd", e))
        empty = pd.Series(dtype=float)
        return empty, empty, empty


def calc_bollinger(prices: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper band, middle SMA, lower band)."""
    try:
        mid   = prices.rolling(period).mean().rename("BB_mid")
        std   = prices.rolling(period).std()
        upper = (mid + 2 * std).rename("BB_upper")
        lower = (mid - 2 * std).rename("BB_lower")
        return upper, mid, lower
    except Exception as e:
        print(_err("calc_bollinger", e))
        empty = pd.Series(dtype=float)
        return empty, empty, empty


def calc_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    try:
        direction = np.sign(prices.diff())
        obv = (direction * volume).fillna(0).cumsum().rename("OBV")
        return obv
    except Exception as e:
        print(_err("calc_obv", e))
        return pd.Series(dtype=float)


def calc_mfi(high, low, close, volume, period: int = 14) -> pd.Series:
    """Money Flow Index (0-100)."""
    try:
        typical = (high + low + close) / 3
        raw_mf  = typical * volume
        pos_mf  = raw_mf.where(typical > typical.shift(1), 0)
        neg_mf  = raw_mf.where(typical < typical.shift(1), 0)
        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum()
        mfr     = pos_sum / neg_sum.replace(0, np.nan)
        return (100 - 100 / (1 + mfr)).rename("MFI")
    except Exception as e:
        print(_err("calc_mfi", e))
        return pd.Series(dtype=float)


# ── RSI Mean-Reversion strategy ───────────────────────────────────────────────
def run_rsi_strategy(
    prices:       pd.Series,
    rsi_buy:      int   = 30,
    rsi_sell:     int   = 70,
    sma_period:   int   = 200,
    stop_loss_pct: float = 0.10,
) -> pd.Series:
    """
    Event-driven RSI mean-reversion backtest on a single price series.

    Rules
    -----
    BUY  : RSI crosses below rsi_buy AND close > SMA(sma_period)
    SELL : RSI crosses above rsi_sell
           OR drawdown from peak buy entry exceeds stop_loss_pct

    Returns daily equity curve normalised to 100 at start.
    """
    try:
        rsi  = calc_rsi(prices, 14)
        sma  = prices.rolling(sma_period, min_periods=sma_period // 2).mean()

        equity   = [100.0]
        in_trade = False
        buy_price = None
        cash_val  = 100.0
        shares    = 0.0

        dates  = prices.index.tolist()
        closes = prices.values.tolist()
        rsi_v  = rsi.reindex(prices.index).values.tolist()
        sma_v  = sma.reindex(prices.index).values.tolist()

        for i in range(1, len(closes)):
            p   = closes[i]
            r   = rsi_v[i]
            r0  = rsi_v[i - 1]
            sm  = sma_v[i]
            if np.isnan(p):
                equity.append(equity[-1])
                continue

            if not in_trade:
                if (r is not None and not np.isnan(r) and
                        r0 is not None and not np.isnan(r0) and
                        r0 >= rsi_buy > r and
                        sm is not None and not np.isnan(sm) and
                        p > sm):
                    shares    = cash_val / p
                    buy_price = p
                    in_trade  = True
                equity.append(cash_val if not in_trade else shares * p)
            else:
                cur_val   = shares * p
                drawdown  = (p - buy_price) / buy_price

                sell = (
                    (r is not None and not np.isnan(r) and
                     r0 is not None and not np.isnan(r0) and
                     r0 <= rsi_sell < r)
                    or drawdown < -stop_loss_pct
                )
                if sell:
                    cash_val  = shares * p
                    shares    = 0.0
                    buy_price = None
                    in_trade  = False
                    equity.append(cash_val)
                else:
                    equity.append(cur_val)

        return pd.Series(equity, index=prices.index, name="RSI Strategy")
    except Exception as e:
        print(_err("run_rsi_strategy", e))
        return pd.Series(dtype=float)


# ── Portfolio construction ────────────────────────────────────────────────────
def calc_portfolio_series(prices_df: pd.DataFrame) -> pd.Series:
    """Equal-weight buy-and-hold portfolio, normalised to 100."""
    try:
        clean  = prices_df.dropna(how="any")
        if clean.empty:
            return pd.Series(dtype=float)
        normed = clean.div(clean.iloc[0]) * 100
        return normed.mean(axis=1).rename("Portfolio")
    except Exception as e:
        print(_err("calc_portfolio_series", e))
        return pd.Series(dtype=float)


def calc_rsi_portfolio(
    prices_df: pd.DataFrame,
    rsi_buy:   int   = 30,
    rsi_sell:  int   = 70,
    sma_period: int  = 200,
    stop_loss: float = 0.10,
) -> pd.Series:
    """Equal-weight RSI-strategy portfolio across multiple tickers."""
    try:
        series_list = []
        for col in prices_df.columns:
            s = run_rsi_strategy(
                prices_df[col].dropna(),
                rsi_buy, rsi_sell, sma_period, stop_loss,
            )
            if not s.empty:
                series_list.append(s.reindex(prices_df.index))
        if not series_list:
            return pd.Series(dtype=float)
        combined = pd.concat(series_list, axis=1).mean(axis=1)
        return combined.rename("RSI Portfolio")
    except Exception as e:
        print(_err("calc_rsi_portfolio", e))
        return pd.Series(dtype=float)


# ── Performance metrics ───────────────────────────────────────────────────────
def calc_metrics(series: pd.Series, rf: float = 0.05) -> dict:
    """CAGR, Sharpe (rf=5%), Max Drawdown, annualised Vol."""
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


# ── Drawdown period analysis ──────────────────────────────────────────────────
def analyze_drawdown_periods(series: pd.Series, threshold: float = 0.05) -> list[dict]:
    """
    Identify contiguous drawdown periods > threshold.

    Returns list of dicts, sorted by severity:
      start_date, end_date, duration_days, max_loss_pct, recovered
    """
    try:
        if series is None or series.empty:
            return []
        roll_max  = series.cummax()
        dd_series = (series - roll_max) / roll_max

        periods   = []
        in_dd     = False
        dd_start  = None
        dd_peak   = None

        for date, val in dd_series.items():
            if not in_dd and val < -threshold:
                in_dd    = True
                dd_start = date
                dd_peak  = val
            elif in_dd:
                dd_peak = min(dd_peak, val)
                if val >= -0.01:
                    periods.append({
                        "start_date":    dd_start,
                        "end_date":      date,
                        "duration_days": (date - dd_start).days,
                        "max_loss_pct":  round(dd_peak * 100, 2),
                        "recovered":     True,
                    })
                    in_dd = False
                    dd_start = None
                    dd_peak  = None

        if in_dd:
            last_date = series.index[-1]
            periods.append({
                "start_date":    dd_start,
                "end_date":      last_date,
                "duration_days": (last_date - dd_start).days,
                "max_loss_pct":  round(dd_peak * 100, 2),
                "recovered":     False,
            })

        return sorted(periods, key=lambda x: x["max_loss_pct"])
    except Exception as e:
        print(_err("analyze_drawdown_periods", e))
        return []


# ── Contribution analysis ─────────────────────────────────────────────────────
def calc_contribution(prices_df: pd.DataFrame) -> dict:
    """
    Per-ticker total return and contribution to equal-weight portfolio.
    Returns {ticker: {"total_return_pct": float, "contribution_pct": float}}
    """
    try:
        clean = prices_df.dropna(how="any")
        if clean.empty:
            return {}
        returns = {}
        for col in clean.columns:
            ret = (clean[col].iloc[-1] / clean[col].iloc[0] - 1) * 100
            returns[col] = ret
        n    = len(returns)
        result = {}
        for t, r in returns.items():
            result[t] = {
                "total_return_pct":  round(r, 2),
                "contribution_pct":  round(r / n, 2),
            }
        return result
    except Exception as e:
        print(_err("calc_contribution", e))
        return {}


# ── Main backtest entry point ─────────────────────────────────────────────────
def run_backtest(
    ticker_list:      list,
    window_years:     int,
    benchmark_ticker: str   = "SPY",
    strategy_mode:    str   = "買入持有",
    rsi_buy:          int   = 30,
    rsi_sell:         int   = 70,
    sma_period:       int   = 200,
    stop_loss_pct:    float = 0.10,
) -> dict:
    """
    Run backtest.  strategy_mode: '買入持有' | 'RSI均值回歸'
    """
    try:
        all_tickers = list(dict.fromkeys(ticker_list + [benchmark_ticker]))
        prices      = fetch_price_history(all_tickers, window_years)

        if prices.empty:
            return {"error": "無法獲取任何價格數據，請確認股票代碼是否正確。"}

        pf_tickers = [t for t in ticker_list if t in prices.columns]
        if not pf_tickers:
            return {"error": "所有指定股票均無法獲取資料，請檢查代碼。"}

        pf_prices = prices[pf_tickers]

        if strategy_mode == "RSI均值回歸":
            portfolio_series = calc_rsi_portfolio(
                pf_prices, rsi_buy, rsi_sell, sma_period, stop_loss_pct
            )
        else:
            portfolio_series = calc_portfolio_series(pf_prices)

        bm_series = None
        if benchmark_ticker in prices.columns:
            bm_raw    = prices[[benchmark_ticker]].dropna()
            bm_normed = bm_raw.div(bm_raw.iloc[0]) * 100
            bm_series = bm_normed[benchmark_ticker].rename(benchmark_ticker)

        if bm_series is not None and not portfolio_series.empty:
            common           = portfolio_series.index.intersection(bm_series.index)
            portfolio_series = portfolio_series.loc[common]
            bm_series        = bm_series.loc[common]

        pf_metrics = calc_metrics(portfolio_series)
        bm_metrics = calc_metrics(bm_series) if bm_series is not None else {}

        alpha = None
        if pf_metrics["cagr"] is not None and bm_metrics.get("cagr") is not None:
            alpha = pf_metrics["cagr"] - bm_metrics["cagr"]

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

        drawdown_periods = analyze_drawdown_periods(portfolio_series)
        contribution     = calc_contribution(pf_prices)

        return {
            "portfolio_series":   portfolio_series,
            "benchmark_series":   bm_series,
            "portfolio_metrics":  pf_metrics,
            "benchmark_metrics":  bm_metrics,
            "alpha":              alpha,
            "high_vol_flags":     high_vol_flags,
            "per_ticker_metrics": per_ticker_metrics,
            "drawdown_periods":   drawdown_periods,
            "contribution":       contribution,
            "pf_tickers":         pf_tickers,
            "benchmark_ticker":   benchmark_ticker,
            "window_years":       window_years,
            "strategy_mode":      strategy_mode,
            "error":              None,
        }
    except Exception as e:
        print(_err("run_backtest", e))
        return {"error": str(e)}


# ── Rebalancing helper ────────────────────────────────────────────────────────
def calc_rebalance(
    current_pct:    dict,
    target_pct:     dict,
    monthly_budget: float = 2400.0,
) -> dict:
    """Allocate monthly contribution to minimise deviation from target."""
    try:
        labels      = list(target_pct.keys())
        deviations  = {k: target_pct[k] - current_pct.get(k, 0.0) for k in labels}
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

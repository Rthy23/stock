"""One Wilder RSI implementation for every analysis and backtest."""
import numpy as np
import pandas as pd


def wilder_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    prices = prices.astype(float)
    result = pd.Series(np.nan, index=prices.index, name="RSI", dtype=float)
    changes = prices.diff()
    gain = changes.clip(lower=0)
    loss = -changes.clip(upper=0)
    avg_gain = avg_loss = None
    for i in range(period, len(prices)):
        if avg_gain is None or not np.isfinite(changes.iloc[i]):
            window = changes.iloc[i-period+1:i+1]
            if window.isna().any():
                avg_gain = avg_loss = None
                continue
            avg_gain = gain.iloc[i-period+1:i+1].mean()
            avg_loss = loss.iloc[i-period+1:i+1].mean()
        else:
            avg_gain = (avg_gain * (period-1) + gain.iloc[i]) / period
            avg_loss = (avg_loss * (period-1) + loss.iloc[i]) / period
        result.iloc[i] = (50. if avg_loss == 0 and avg_gain == 0 else
                          100. if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss))
    return result

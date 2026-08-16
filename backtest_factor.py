"""
backtest_factor.py
══════════════════════════════════════════════════════════════════════════════
七因子 Composite Score 快速回測快照（健檢版）

設計原則（節省 API 用量）：
  • 每支股票只拉一次完整歷史（fetch_start → fetch_end），
    再在 Python 切片，不重複呼叫 yfinance。
  • 基本面數據（PE/ROE 等）使用當前 yfinance .info 近似代替歷史值，
    這是已知 look-ahead bias，報告中已標注警語。
  • 20 檔股票 × 8 季度 = 160 筆資料點。

已知限制（請在 UI 呈現警語）：
  1. 基本面因子使用「現在」的 .info，而非當時的真實值（前瞻偏差）。
  2. 樣本量 160 筆（有效筆數可能更少），統計顯著性弱，只能看方向。
  3. 沒有做交易成本、滑點、持有調整等處理。
  4. 歷史結果不代表未來績效。
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from analysis import calculate_seven_factors

# ── 20 檔分散龍頭股（各產業 2-4 檔）────────────────────────────────────────
BACKTEST_STOCKS: list[str] = [
    # 科技
    "AAPL", "MSFT", "GOOGL", "NVDA", "AVGO",
    # 消費 / 零售
    "AMZN", "COST", "HD", "MCD", "NKE",
    # 金融
    "JPM", "V",
    # 醫療
    "UNH", "JNJ", "LLY", "ABBV",
    # 能源
    "CVX",
    # 公用 / 半導體 / 必需消費
    "NEE", "TXN", "PG",
]

# ── 8 個季度快照日期（約 2 年）─────────────────────────────────────────────
_SNAPSHOT_DATES: list[date] = [
    date(2022, 11, 1),
    date(2023, 2,  1),
    date(2023, 5,  1),
    date(2023, 8,  1),
    date(2023, 11, 1),
    date(2024, 2,  1),
    date(2024, 5,  1),
    date(2024, 8,  1),
]

# 拉取資料的邊界（最早快照 - 15 個月 的歷史，讓 252-day 動量有足夠資料）
_FETCH_START = date(2022, 1, 1)
# 最晚快照 + 3 個月（2024-08-01 + 91 days ≈ 2024-11-01）
_FETCH_END   = date(2024, 11, 30)


# ── 從 yfinance info 組裝 factor_data ───────────────────────────────────────
def _build_factor_data(info: dict) -> dict:
    def _pct(key):  # convert fraction to %
        v = info.get(key)
        return round(v * 100, 2) if v is not None else None

    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    return {
        "price":              price,
        "pe_ratio":           info.get("trailingPE"),
        "pb_ratio":           info.get("priceToBook"),
        "ev_ebitda":          info.get("enterpriseToEbitda"),
        "ps_ratio":           info.get("priceToSalesTrailing12Months"),
        "dividend_yield":     _pct("dividendYield"),
        "roe":                _pct("returnOnEquity"),
        "roa":                _pct("returnOnAssets"),
        "gross_margin":       _pct("grossMargins"),
        "op_margin":          _pct("operatingMargins"),
        "debt_equity":        info.get("debtToEquity"),
        "current_ratio":      info.get("currentRatio"),
        "fwd_eps":            info.get("forwardEps"),
        "trailing_eps":       info.get("trailingEps"),
        "rev_growth":         _pct("revenueGrowth"),
        "eps_growth":         _pct("earningsGrowth"),
        "rec_mean":           info.get("recommendationMean"),
        "beta":               info.get("beta"),
        "short_pct":          _pct("shortPercentOfFloat"),
        "inst_ownership":     _pct("heldPercentInstitutions"),
        "insider_ownership":  _pct("heldPercentInsiders"),
    }


def run_factor_backtest(
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """
    執行 7-Factor 快速回測。

    Parameters
    ----------
    progress_cb : callable(ticker, idx, total) | None
        可選的進度回呼，供 Streamlit progress bar 使用。

    Returns
    -------
    dict:
        records  : list[dict] — 每筆 (ticker, snapshot, composite, signal, fwd_3m_pct)
        summary  : dict       — 高/低分組統計
        warnings : list[str]  — 錯誤 / 跳過記錄
    """
    records:  list[dict] = []
    warnings: list[str]  = []
    total = len(BACKTEST_STOCKS)

    for idx, ticker in enumerate(BACKTEST_STOCKS):
        if progress_cb:
            progress_cb(ticker, idx, total)

        try:
            tk   = yf.Ticker(ticker)
            info = {}
            try:
                info = tk.info or {}
            except Exception:
                pass  # fallback to empty; factor_data will have None fields

            fd = _build_factor_data(info)

            # ── 拉一次完整歷史 ───────────────────────────────────────────────
            try:
                hist_full = tk.history(
                    start=_FETCH_START.isoformat(),
                    end=_FETCH_END.isoformat(),
                    auto_adjust=True,
                )
            except Exception as e:
                warnings.append(f"{ticker}: 歷史資料抓取失敗 — {e}")
                continue

            if hist_full is None or hist_full.empty:
                warnings.append(f"{ticker}: 歷史資料為空，略過")
                continue

            # 標準化時區
            hist_full.index = pd.to_datetime(hist_full.index).tz_localize(None)

            # ── 逐季計算 ─────────────────────────────────────────────────────
            for snap_date in _SNAPSHOT_DATES:
                snap_ts = pd.Timestamp(snap_date)
                fwd_ts  = pd.Timestamp(snap_date + timedelta(days=91))

                # 截至快照日期的歷史（最多 252 個交易日 ≈ 1 年）
                hist_slice = hist_full[hist_full.index <= snap_ts].tail(300).copy()
                if len(hist_slice) < 30:
                    warnings.append(
                        f"{ticker}@{snap_date}: 歷史不足 30 列，略過"
                    )
                    continue

                # 計算 composite score
                try:
                    result    = calculate_seven_factors(
                        {"price": fd.get("price", 0)},
                        hist_slice,
                        fd,
                    )
                    composite = float(result.get("composite", 0.0))
                    signal    = result.get("signal", "HOLD")
                except Exception as e:
                    warnings.append(f"{ticker}@{snap_date}: 因子計算失敗 — {e}")
                    continue

                # 計算後 3 個月實際報酬
                fwd_slice = hist_full[
                    (hist_full.index > snap_ts) & (hist_full.index <= fwd_ts)
                ]
                if len(fwd_slice) < 5:
                    warnings.append(
                        f"{ticker}@{snap_date}: 後 3 個月資料不足，略過"
                    )
                    continue

                entry_price = float(hist_slice["Close"].iloc[-1])
                exit_price  = float(fwd_slice["Close"].iloc[-1])
                if entry_price <= 0:
                    continue
                fwd_return = round((exit_price / entry_price - 1) * 100, 2)

                records.append({
                    "ticker":    ticker,
                    "snapshot":  snap_date.isoformat(),
                    "composite": round(composite, 3),
                    "signal":    signal,
                    "fwd_3m_pct": fwd_return,
                })

            time.sleep(0.25)  # 禮貌性限速

        except Exception as e:
            warnings.append(f"{ticker}: 未預期錯誤 — {e}")

    # ── 彙整統計 ─────────────────────────────────────────────────────────────
    if not records:
        return {"records": [], "summary": None, "warnings": warnings}

    df = pd.DataFrame(records)

    high  = df[df["composite"] >  1.0]["fwd_3m_pct"]
    low   = df[df["composite"] < -1.0]["fwd_3m_pct"]
    mid   = df[(df["composite"] >= -1.0) & (df["composite"] <= 1.0)]["fwd_3m_pct"]

    def _stats(s: pd.Series) -> dict:
        if s.empty:
            return {"mean": None, "median": None, "win_rate": None, "n": 0}
        return {
            "mean":     round(float(s.mean()),   2),
            "median":   round(float(s.median()), 2),
            "win_rate": round(float((s > 0).mean() * 100), 1),
            "n":        len(s),
        }

    high_stats = _stats(high)
    low_stats  = _stats(low)
    mid_stats  = _stats(mid)

    spread = None
    if high_stats["mean"] is not None and low_stats["mean"] is not None:
        spread = round(high_stats["mean"] - low_stats["mean"], 2)

    summary = {
        "n_total":   len(df),
        "high":      high_stats,
        "low":       low_stats,
        "hold":      mid_stats,
        "spread":    spread,
        "direction_ok": (
            spread is not None and spread > 0
        ),
    }

    return {
        "records":  df.to_dict("records"),
        "summary":  summary,
        "warnings": warnings,
    }


def get_cached_result() -> Optional[dict]:
    """Try to load a previously cached result from session (caller passes via st.session_state)."""
    return None  # caller manages the cache via st.session_state


if __name__ == "__main__":
    # 直接執行腳本時的 CLI 輸出
    import json

    print("開始七因子快速回測…（約需 2-3 分鐘）")

    def cli_progress(ticker, idx, total):
        print(f"  [{idx+1:02d}/{total}] {ticker}")

    result = run_factor_backtest(progress_cb=cli_progress)
    s = result.get("summary")
    if s:
        print("\n═══ 回測快照結果 ═══")
        print(f"總資料點：{s['n_total']} 筆")
        print(f"高分組 (score > 1.0)：{s['high']['n']} 筆，平均 3M 報酬 {s['high']['mean']}%，"
              f"勝率 {s['high']['win_rate']}%")
        print(f"觀望組 (-1~1)         {s['hold']['n']} 筆，平均 3M 報酬 {s['hold']['mean']}%")
        print(f"低分組 (score < -1.0)：{s['low']['n']} 筆，平均 3M 報酬 {s['low']['mean']}%，"
              f"勝率 {s['low']['win_rate']}%")
        print(f"高低分組報酬差距：{s['spread']}%")
        print(f"方向符合預期：{'✅ 是' if s['direction_ok'] else '❌ 否'}")
    if result["warnings"]:
        print(f"\n警告 ({len(result['warnings'])} 筆)：")
        for w in result["warnings"][:10]:
            print(f"  {w}")

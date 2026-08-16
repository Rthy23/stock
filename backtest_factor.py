"""
backtest_factor.py
══════════════════════════════════════════════════════════════════════════════
七因子 Composite Score 快速回測快照（擴大版）

規格：
  • 50 檔產業分散龍頭股 × 18 個雙月快照 ≈ 900 筆資料
  • 時間範圍：近 3 年（2021-Q4 → 2024-Q3）
  • 每支股票只拉一次完整歷史再切片，不重複呼叫 yfinance
  • 同步計算 7 個子因子各自與未來 3M 報酬的 Pearson 相關係數
  • 結果快取至 /tmp/factor_bt_cache.csv（避免重複抓資料）

已知限制（在 UI 呈現警語）：
  1. 基本面因子使用「當前」.info（前瞻偏差）
  2. 倖存者偏差（只含現存龍頭股）
  3. 無交易成本/滑點
  4. 歷史結果不代表未來
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from analysis import calculate_seven_factors

# ── 50 檔產業分散龍頭股 ───────────────────────────────────────────────────────
BACKTEST_STOCKS: list[str] = [
    # ── 科技（10）
    "AAPL", "MSFT", "GOOGL", "NVDA", "AVGO",
    "META", "ADBE", "CRM",  "ORCL", "AMD",
    # ── 消費 / 零售（10）
    "AMZN", "COST", "HD",   "MCD",  "NKE",
    "WMT",  "SBUX", "TGT",  "LOW",  "BKNG",
    # ── 金融（7）
    "JPM",  "V",    "BAC",  "GS",   "AXP",  "BRK-B", "MS",
    # ── 醫療（8）
    "UNH",  "JNJ",  "LLY",  "ABBV",
    "TMO",  "BMY",  "GILD", "MDT",
    # ── 工業（5）
    "HON",  "CAT",  "GE",   "BA",   "UNP",
    # ── 能源（3）
    "CVX",  "XOM",  "OXY",
    # ── 公用 / 材料 / 必需消費（7）
    "NEE",  "SO",   "TXN",  "PG",   "LIN",  "APD",  "CI",
]

assert len(BACKTEST_STOCKS) == 50, f"BACKTEST_STOCKS 應為 50 檔，現有 {len(BACKTEST_STOCKS)}"

# ── 18 個雙月快照日期（3 年，從 2021-10 → 2024-08）─────────────────────────
_SNAPSHOT_DATES: list[date] = [
    date(2021, 10, 1),
    date(2021, 12, 1),
    date(2022,  2, 1),
    date(2022,  4, 1),
    date(2022,  6, 1),
    date(2022,  8, 1),
    date(2022, 10, 1),
    date(2022, 12, 1),
    date(2023,  2, 1),
    date(2023,  4, 1),
    date(2023,  6, 1),
    date(2023,  8, 1),
    date(2023, 10, 1),
    date(2023, 12, 1),
    date(2024,  2, 1),
    date(2024,  4, 1),
    date(2024,  6, 1),
    date(2024,  8, 1),
]
assert len(_SNAPSHOT_DATES) == 18

# 最早快照 2021-10-01 往前 15 個月 → 2020-07；最晚快照 2024-08-01 + 91 天
_FETCH_START = date(2020, 6, 1)
_FETCH_END   = date(2024, 11, 30)

# 快取路徑
_CACHE_CSV  = "/tmp/factor_bt_cache.csv"
_CACHE_META = "/tmp/factor_bt_meta.json"

# 子因子名稱對應（用於相關係數表）
_FACTOR_LABELS: dict[str, str] = {
    "Momentum":   "Momentum 動量",
    "Value":      "Value 估值",
    "Quality":    "Quality 質量",
    "Growth":     "Growth 成長",
    "Volatility": "Volatility 波動性",
    "Sentiment":  "Sentiment 情緒",
    "Macro":      "Macro 宏觀",
}


# ── 從 yfinance info 組裝 factor_data ────────────────────────────────────────
def _build_factor_data(info: dict) -> dict:
    def _pct(key):
        v = info.get(key)
        return round(v * 100, 2) if v is not None else None

    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    return {
        "price":             price,
        "pe_ratio":          info.get("trailingPE"),
        "pb_ratio":          info.get("priceToBook"),
        "ev_ebitda":         info.get("enterpriseToEbitda"),
        "ps_ratio":          info.get("priceToSalesTrailing12Months"),
        "dividend_yield":    _pct("dividendYield"),
        "roe":               _pct("returnOnEquity"),
        "roa":               _pct("returnOnAssets"),
        "gross_margin":      _pct("grossMargins"),
        "op_margin":         _pct("operatingMargins"),
        "debt_equity":       info.get("debtToEquity"),
        "current_ratio":     info.get("currentRatio"),
        "fwd_eps":           info.get("forwardEps"),
        "trailing_eps":      info.get("trailingEps"),
        "rev_growth":        _pct("revenueGrowth"),
        "eps_growth":        _pct("earningsGrowth"),
        "rec_mean":          info.get("recommendationMean"),
        "beta":              info.get("beta"),
        "short_pct":         _pct("shortPercentOfFloat"),
        "inst_ownership":    _pct("heldPercentInstitutions"),
        "insider_ownership": _pct("heldPercentInsiders"),
    }


# ── 子因子相關係數計算 ────────────────────────────────────────────────────────
def compute_factor_correlations(records: list[dict]) -> list[dict]:
    """
    對 records 中 7 個子因子各自計算與 fwd_3m_pct 的 Pearson 相關係數。
    回傳按相關係數降序排列的清單：
      [{"factor": str, "label": str, "corr": float, "n": int}, ...]
    """
    if not records:
        return []

    df = pd.DataFrame(records)
    results = []
    for key, label in _FACTOR_LABELS.items():
        col = f"score_{key.lower()}"
        if col not in df.columns:
            continue
        sub = df[[col, "fwd_3m_pct"]].dropna()
        if len(sub) < 10:
            corr = None
        else:
            corr = round(float(sub[col].corr(sub["fwd_3m_pct"])), 4)
        results.append({
            "factor": key,
            "label":  label,
            "corr":   corr,
            "n":      len(sub),
        })

    # 降序排列（None 排最後）
    results.sort(key=lambda x: (x["corr"] is None, -(x["corr"] or 0)))
    return results


# ── 組別統計 ─────────────────────────────────────────────────────────────────
def _stats(s: pd.Series) -> dict:
    if s.empty:
        return {"mean": None, "median": None, "win_rate": None, "n": 0}
    return {
        "mean":     round(float(s.mean()),              2),
        "median":   round(float(s.median()),            2),
        "win_rate": round(float((s > 0).mean() * 100),  1),
        "n":        len(s),
    }


# ── CSV 快取 I/O ──────────────────────────────────────────────────────────────
def save_cache(result: dict) -> None:
    """將回測結果快取至 /tmp CSV + meta JSON。"""
    try:
        df = pd.DataFrame(result.get("records", []))
        df.to_csv(_CACHE_CSV, index=False, encoding="utf-8")
        meta = {
            "saved_at": datetime.utcnow().isoformat(),
            "summary":  result.get("summary"),
            "warnings": result.get("warnings", []),
            "n_stocks": len(BACKTEST_STOCKS),
            "n_snapshots": len(_SNAPSHOT_DATES),
        }
        with open(_CACHE_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass  # 快取失敗不影響主流程


def load_cache(max_age_hours: float = 12.0) -> Optional[dict]:
    """
    嘗試讀取快取。
    若快取存在且不超過 max_age_hours，回傳 result dict；否則回傳 None。
    """
    try:
        if not os.path.exists(_CACHE_CSV) or not os.path.exists(_CACHE_META):
            return None
        with open(_CACHE_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        saved_at = datetime.fromisoformat(meta["saved_at"])
        age_hours = (datetime.utcnow() - saved_at).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None
        df = pd.read_csv(_CACHE_CSV)
        records = df.to_dict("records")
        return {
            "records":    records,
            "summary":    meta.get("summary"),
            "warnings":   meta.get("warnings", []),
            "cached_at":  meta["saved_at"],
            "from_cache": True,
        }
    except Exception:
        return None


def cache_info() -> Optional[dict]:
    """回傳快取 meta（不載入 records），用於 UI 顯示快取狀態。"""
    try:
        if not os.path.exists(_CACHE_META):
            return None
        with open(_CACHE_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_cache() -> None:
    for p in [_CACHE_CSV, _CACHE_META]:
        try:
            os.remove(p)
        except Exception:
            pass


# ── 主回測函式 ────────────────────────────────────────────────────────────────
def run_factor_backtest(
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """
    執行 7-Factor 快速回測（50 檔 × 18 時間點）。

    回傳 dict：
      records   : list[dict]  每筆含 ticker / snapshot / composite / signal /
                               fwd_3m_pct / score_momentum … score_macro
      summary   : dict        高/觀望/低分組統計 + factor_correlations
      warnings  : list[str]
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
                pass

            fd = _build_factor_data(info)

            # ── 一次性拉取完整歷史 ───────────────────────────────────────────
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

            hist_full.index = pd.to_datetime(hist_full.index).tz_localize(None)

            # ── 逐雙月快照計算 ───────────────────────────────────────────────
            for snap_date in _SNAPSHOT_DATES:
                snap_ts = pd.Timestamp(snap_date)
                fwd_ts  = pd.Timestamp(snap_date + timedelta(days=91))

                hist_slice = hist_full[hist_full.index <= snap_ts].tail(300).copy()
                if len(hist_slice) < 30:
                    continue  # 歷史不足，靜默跳過

                # 計算 composite + 7 個子因子分數
                try:
                    factors   = calculate_seven_factors(
                        {"price": fd.get("price", 0)},
                        hist_slice,
                        fd,
                    )
                    composite = float(factors.get("composite", 0.0))
                    signal    = factors.get("signal", "HOLD")
                    sub_scores = {
                        f"score_{k.lower()}": round(float(factors[k]["score"]), 3)
                        for k in _FACTOR_LABELS
                        if k in factors and isinstance(factors[k], dict)
                    }
                except Exception as e:
                    warnings.append(f"{ticker}@{snap_date}: 因子計算失敗 — {e}")
                    continue

                # 後 3 個月實際報酬
                fwd_slice = hist_full[
                    (hist_full.index > snap_ts) & (hist_full.index <= fwd_ts)
                ]
                if len(fwd_slice) < 5:
                    continue

                entry = float(hist_slice["Close"].iloc[-1])
                exit_ = float(fwd_slice["Close"].iloc[-1])
                if entry <= 0:
                    continue
                fwd_return = round((exit_ / entry - 1) * 100, 2)

                record = {
                    "ticker":     ticker,
                    "snapshot":   snap_date.isoformat(),
                    "composite":  round(composite, 3),
                    "signal":     signal,
                    "fwd_3m_pct": fwd_return,
                }
                record.update(sub_scores)
                records.append(record)

            time.sleep(0.25)  # 禮貌性限速

        except Exception as e:
            warnings.append(f"{ticker}: 未預期錯誤 — {e}")

    # ── 彙整統計 ─────────────────────────────────────────────────────────────
    if not records:
        return {"records": [], "summary": None, "warnings": warnings}

    df = pd.DataFrame(records)

    high_ret = df[df["composite"] >  1.0]["fwd_3m_pct"]
    low_ret  = df[df["composite"] < -1.0]["fwd_3m_pct"]
    mid_ret  = df[(df["composite"] >= -1.0) & (df["composite"] <= 1.0)]["fwd_3m_pct"]

    high_stats = _stats(high_ret)
    low_stats  = _stats(low_ret)
    mid_stats  = _stats(mid_ret)

    spread = None
    if high_stats["mean"] is not None and low_stats["mean"] is not None:
        spread = round(high_stats["mean"] - low_stats["mean"], 2)

    factor_corrs = compute_factor_correlations(records)

    summary = {
        "n_total":           len(df),
        "n_stocks":          len(BACKTEST_STOCKS),
        "n_snapshots":       len(_SNAPSHOT_DATES),
        "high":              high_stats,
        "low":               low_stats,
        "hold":              mid_stats,
        "spread":            spread,
        "direction_ok":      spread is not None and spread > 0,
        "factor_corrs":      factor_corrs,
    }

    result = {
        "records":    df.to_dict("records"),
        "summary":    summary,
        "warnings":   warnings,
        "from_cache": False,
    }
    save_cache(result)
    return result


if __name__ == "__main__":
    print("開始七因子快速回測（50 檔 × 18 時間點，約需 4-6 分鐘）…")

    def _cli_prog(ticker, idx, total):
        print(f"  [{idx+1:02d}/{total}] {ticker}")

    result = run_factor_backtest(progress_cb=_cli_prog)
    s = result.get("summary")
    if s:
        print(f"\n═══ 回測快照結果（{s['n_stocks']} 檔 × {s['n_snapshots']} 時間點）═══")
        print(f"總有效資料點：{s['n_total']} 筆")
        print(f"高分組 (>1.0)  : n={s['high']['n']:3d}  平均 {s['high']['mean']}%  勝率 {s['high']['win_rate']}%")
        print(f"觀望組 (-1~1)  : n={s['hold']['n']:3d}  平均 {s['hold']['mean']}%")
        print(f"低分組 (<-1.0) : n={s['low']['n']:3d}  平均 {s['low']['mean']}%  勝率 {s['low']['win_rate']}%")
        print(f"報酬差距（高−低）：{s['spread']}%  ({'符合預期 ✅' if s['direction_ok'] else '不符合預期 ❌'})")
        print("\n── 子因子相關係數排序 ──")
        for fc in s.get("factor_corrs", []):
            bar = "█" * int(abs(fc["corr"] or 0) * 20)
            sign = "+" if (fc["corr"] or 0) >= 0 else ""
            print(f"  {fc['label']:22s}  r={sign}{fc['corr']:.4f}  {bar}")
    if result.get("warnings"):
        print(f"\n⚠️ 警告 ({len(result['warnings'])} 筆)")
        for w in result["warnings"][:5]:
            print(f"  {w}")

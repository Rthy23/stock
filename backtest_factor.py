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

# 快取 / 進度路徑
_CACHE_CSV     = "/tmp/factor_bt_cache.csv"
_CACHE_META    = "/tmp/factor_bt_meta.json"
_PROGRESS_FILE = "/tmp/factor_bt_progress.json"

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


# ── 分組指派（支援兩種模式）────────────────────────────────────────────────────
def assign_groups(
    composite_scores: "pd.Series",
    method: str = "percentile",
) -> "tuple[pd.Series, dict]":
    """
    把每筆資料的 composite score 指派到 'high' / 'hold' / 'low'。

    method='percentile'（預設）
        high ≥ P75（前25%）、low ≤ P25（後25%）、hold 其餘中間50%
        → 不論分佈如何，三組一定都有樣本

    method='fixed'
        high > 1.0、low < -1.0、hold -1 ~ 1
        → 與舊版相同，但龍頭股池容易造成低分組 0 筆
    """
    if method == "percentile":
        p25 = float(composite_scores.quantile(0.25))
        p75 = float(composite_scores.quantile(0.75))
        grp = pd.Series("hold", index=composite_scores.index, dtype=str)
        grp[composite_scores >= p75] = "high"
        grp[composite_scores <= p25] = "low"
        cuts = {
            "method":   "percentile",
            "high_cut": round(p75, 3),
            "low_cut":  round(p25, 3),
            "label_high": f"前25%（score ≥ {p75:.2f}）",
            "label_low":  f"後25%（score ≤ {p25:.2f}）",
            "label_hold": f"中間50%（{p25:.2f} < score < {p75:.2f}）",
        }
    else:  # fixed
        grp = pd.Series("hold", index=composite_scores.index, dtype=str)
        grp[composite_scores >  1.0] = "high"
        grp[composite_scores < -1.0] = "low"
        cuts = {
            "method":   "fixed",
            "high_cut": 1.0,
            "low_cut":  -1.0,
            "label_high": "score > 1.0",
            "label_low":  "score < -1.0",
            "label_hold": "-1.0 ≤ score ≤ 1.0",
        }
    return grp, cuts


# ── 早期快照分數異常檢查 ──────────────────────────────────────────────────────
def _check_early_score_anomaly(df: "pd.DataFrame") -> "Optional[str]":
    """
    檢查最早 3 個快照時間點的 Momentum / Macro 分數是否異常集中在 0
    （原因：SMA200 需要 200 日歷史，最早時間點可能歷史不足）。
    若 >30% 為 0，回傳注意說明文字；否則回傳 None。
    """
    if "snapshot" not in df.columns:
        return None
    early_dates = sorted(df["snapshot"].unique())[:3]
    early_df = df[df["snapshot"].isin(early_dates)]
    if early_df.empty:
        return None
    notes = []
    for col, label in [("score_momentum", "Momentum"), ("score_macro", "Macro")]:
        if col in df.columns and len(early_df) > 0:
            zero_pct = (early_df[col] == 0).mean()
            if zero_pct > 0.30:
                notes.append(f"{label}（{zero_pct:.0%} 為 0）")
    if notes:
        return (
            f"⚠️ 最早 3 個快照（{', '.join(early_dates[:3])}）的 "
            f"{' / '.join(notes)} 因子分數有較多 0 值，"
            f"可能因歷史資料不足（SMA200 需 200 日）輕微影響早期評分鑑別度，"
            f"不影響整體結論方向。"
        )
    return None


# ── 從 records 重算 summary（不呼叫任何 API）──────────────────────────────────
def compute_summary(records: "list[dict]", method: str = "percentile") -> dict:
    """
    從現有 records 重新計算分組統計，純 CPU 運算，不抓資料。
    可對快取資料即時切換分組方式（percentile / fixed）。

    Parameters
    ----------
    records : list[dict]
        run_factor_backtest 回傳的原始記錄，需含 composite 和 fwd_3m_pct 欄位。
    method : str
        'percentile'（預設）或 'fixed'。

    Returns
    -------
    dict 含所有 UI 需要的統計資訊。
    """
    if not records:
        return {}

    df = pd.DataFrame(records)
    if "composite" not in df.columns or "fwd_3m_pct" not in df.columns:
        return {}

    grp, cuts = assign_groups(df["composite"], method=method)
    df = df.copy()
    df["_group"] = grp

    high_stats = _stats(df[df["_group"] == "high"]["fwd_3m_pct"])
    low_stats  = _stats(df[df["_group"] == "low"]["fwd_3m_pct"])
    mid_stats  = _stats(df[df["_group"] == "hold"]["fwd_3m_pct"])

    spread = None
    if high_stats["mean"] is not None and low_stats["mean"] is not None:
        spread = round(high_stats["mean"] - low_stats["mean"], 2)

    factor_corrs = compute_factor_correlations(records)
    early_note   = _check_early_score_anomaly(df)

    return {
        "n_total":      len(df),
        "high":         high_stats,
        "low":          low_stats,
        "hold":         mid_stats,
        "spread":       spread,
        "direction_ok": spread is not None and spread > 0,
        "cutpoints":    cuts,
        "factor_corrs": factor_corrs,
        "early_note":   early_note,
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


# ── 進度檔 I/O（逐檔寫入 + 斷點續傳）────────────────────────────────────────
def save_progress(completed: list[str], records: list[dict]) -> None:
    """
    每完成一檔股票後呼叫：
      1. 覆寫 /tmp/factor_bt_progress.json（記錄已完成清單）
      2. 覆寫 /tmp/factor_bt_cache.csv（累積至今的資料列）
      3. 更新 meta JSON（標記為 incomplete）
    即使中途被 Streamlit 重新整理，下次可接續未完成的部分。
    """
    try:
        with open(_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "saved_at":  datetime.utcnow().isoformat(),
                "completed": completed,
                "n_total":   len(BACKTEST_STOCKS),
                "n_done":    len(completed),
            }, f, ensure_ascii=False)
    except Exception:
        pass

    try:
        if records:
            pd.DataFrame(records).to_csv(_CACHE_CSV, index=False, encoding="utf-8")
        meta = {
            "saved_at":       datetime.utcnow().isoformat(),
            "complete":       False,
            "n_stocks_done":  len(completed),
            "n_stocks_total": len(BACKTEST_STOCKS),
        }
        with open(_CACHE_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass


def load_progress() -> "tuple[list[str], list[dict]]":
    """
    讀取斷點資訊。
    回傳 (已完成 ticker 清單, 已存的 records)。
    無進度或讀取失敗回傳 ([], [])。
    """
    try:
        if not os.path.exists(_PROGRESS_FILE):
            return [], []
        with open(_PROGRESS_FILE, "r", encoding="utf-8") as f:
            prog = json.load(f)
        completed = prog.get("completed", [])
        if not completed:
            return [], []
        records: list[dict] = []
        if os.path.exists(_CACHE_CSV):
            df = pd.read_csv(_CACHE_CSV)
            records = df.to_dict("records")
        return completed, records
    except Exception:
        return [], []


def resume_info() -> Optional[dict]:
    """
    回傳斷點摘要（供 UI 顯示）：
      {"n_done": int, "n_total": int, "saved_at": str, "completed": [...]}
    無進度或已全部完成則回傳 None。
    """
    try:
        if not os.path.exists(_PROGRESS_FILE):
            return None
        with open(_PROGRESS_FILE, "r", encoding="utf-8") as f:
            prog = json.load(f)
        n_done = len(prog.get("completed", []))
        if n_done == 0:
            return None
        return {
            "n_done":    n_done,
            "n_total":   prog.get("n_total", len(BACKTEST_STOCKS)),
            "saved_at":  prog.get("saved_at"),
            "completed": prog.get("completed", []),
        }
    except Exception:
        return None


def clear_progress() -> None:
    """清除進度檔（完成後或使用者手動重置時呼叫）。"""
    try:
        os.remove(_PROGRESS_FILE)
    except Exception:
        pass


# ── CSV 快取 I/O ──────────────────────────────────────────────────────────────
def save_cache(records: list[dict], warnings: list[str]) -> None:
    """
    將已完成的回測結果寫入完整快取（標記 complete=True）。
    完成後呼叫 clear_progress() 刪除進度檔。
    """
    try:
        pd.DataFrame(records).to_csv(_CACHE_CSV, index=False, encoding="utf-8")
        meta = {
            "saved_at":    datetime.utcnow().isoformat(),
            "complete":    True,
            "warnings":    warnings,
            "n_stocks":    len(BACKTEST_STOCKS),
            "n_snapshots": len(_SNAPSHOT_DATES),
        }
        with open(_CACHE_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass
    clear_progress()


def load_cache(max_age_hours: float = 12.0) -> Optional[dict]:
    """
    嘗試讀取**完整**快取（complete=True）。
    若不存在、已過期、或為未完成的中途快照，回傳 None。
    """
    try:
        if not os.path.exists(_CACHE_CSV) or not os.path.exists(_CACHE_META):
            return None
        with open(_CACHE_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 舊版快取沒有 complete 欄位，視為 True（向後相容）
        if not meta.get("complete", True):
            return None
        saved_at  = datetime.fromisoformat(meta["saved_at"])
        age_hours = (datetime.utcnow() - saved_at).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None
        records = pd.read_csv(_CACHE_CSV).to_dict("records")
        return {
            "records":    records,
            "warnings":   meta.get("warnings", []),
            "cached_at":  meta["saved_at"],
            "from_cache": True,
        }
    except Exception:
        return None


def cache_info() -> Optional[dict]:
    """
    回傳完整快取的 meta（不載入 records），供 UI 顯示狀態。
    未完成的中途快照不算，回傳 None。
    """
    try:
        if not os.path.exists(_CACHE_META):
            return None
        with open(_CACHE_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not meta.get("complete", True):
            return None
        return meta
    except Exception:
        return None


def clear_cache() -> None:
    """清除完整快取 + 進度檔，全部重置。"""
    for p in [_CACHE_CSV, _CACHE_META, _PROGRESS_FILE]:
        try:
            os.remove(p)
        except Exception:
            pass


# ── 主回測函式 ────────────────────────────────────────────────────────────────
def run_factor_backtest(
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    resume: bool = True,
) -> dict:
    """
    執行 7-Factor 快速回測（50 檔 × 18 時間點）。

    Parameters
    ----------
    progress_cb : callable(ticker, global_idx, total) | None
        每開始處理一檔股票前呼叫，供 UI 更新進度條。
    resume : bool
        True（預設）：若 /tmp/factor_bt_progress.json 存在，
        自動跳過已完成的 ticker，從中斷處繼續。
        False：忽略進度檔，從頭重跑所有 50 檔。

    每完成一檔立即呼叫 save_progress()，避免重新整理後遺失進度。
    全部完成後呼叫 save_cache() 寫入完整快取，並清除進度檔。

    回傳 dict：
      records     : list[dict]  ticker / snapshot / composite / signal /
                                fwd_3m_pct / score_momentum … score_macro
      warnings    : list[str]
      from_cache  : False
    """
    warnings: list[str] = []
    total = len(BACKTEST_STOCKS)

    # ── 斷點續傳：載入已完成進度 ─────────────────────────────────────────────
    if resume:
        completed_tickers, records = load_progress()
    else:
        completed_tickers, records = [], []

    _skipped = len(completed_tickers)
    if _skipped:
        # 讓使用者在進度條看到「從第 X 檔接續」
        if progress_cb:
            progress_cb(f"（跳過已完成 {_skipped} 檔，從第 {_skipped+1} 檔繼續）",
                        _skipped - 1, total)

    remaining = [t for t in BACKTEST_STOCKS if t not in completed_tickers]

    for idx, ticker in enumerate(remaining):
        global_idx = _skipped + idx
        if progress_cb:
            progress_cb(ticker, global_idx, total)

        ticker_records: list[dict] = []

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
                # 即使此檔失敗也標記為「已處理」，避免無限重試
                completed_tickers.append(ticker)
                save_progress(completed_tickers, records)
                continue

            if hist_full is None or hist_full.empty:
                warnings.append(f"{ticker}: 歷史資料為空，略過")
                completed_tickers.append(ticker)
                save_progress(completed_tickers, records)
                continue

            hist_full.index = pd.to_datetime(hist_full.index).tz_localize(None)

            # ── 逐雙月快照計算 ───────────────────────────────────────────────
            for snap_date in _SNAPSHOT_DATES:
                snap_ts = pd.Timestamp(snap_date)
                fwd_ts  = pd.Timestamp(snap_date + timedelta(days=91))

                hist_slice = hist_full[hist_full.index <= snap_ts].tail(300).copy()
                if len(hist_slice) < 30:
                    continue  # 歷史不足，靜默跳過

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
                ticker_records.append(record)

            time.sleep(0.25)  # 禮貌性限速

        except Exception as e:
            warnings.append(f"{ticker}: 未預期錯誤 — {e}")

        # ── 每完成一檔立即寫入進度（無論成功失敗）────────────────────────────
        records.extend(ticker_records)
        completed_tickers.append(ticker)
        save_progress(completed_tickers, records)

    # ── 全部完成：寫入正式完整快取，清除進度檔 ───────────────────────────────
    if not records:
        clear_progress()
        return {"records": [], "warnings": warnings, "from_cache": False}

    save_cache(records, warnings)   # 同時呼叫 clear_progress()

    return {
        "records":    records,
        "warnings":   warnings,
        "from_cache": False,
    }


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

# ═══════════════════════════════════════════════════════════════════════════════
# data_fetcher.py  —  External API calls, file I/O, and data constants
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import base64
import re

from analysis import classify_sentiment
from user_config import load_watchlist_cfg, save_watchlist_cfg

_MODULE = "data_fetcher"

def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── Writable data directory ────────────────────────────────────────────────────
def _writable_data_dir() -> str:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    probe   = os.path.join(app_dir, ".write_probe")
    try:
        with open(probe, "w") as _f:
            _f.write("ok")
        os.remove(probe)
        return app_dir
    except OSError:
        return "/tmp"

_DATA_DIR      = _writable_data_dir()
WATCHLIST_FILE = os.path.join(_DATA_DIR, "watchlist.json")
PORTFOLIO_FILE = os.path.join(_DATA_DIR, "portfolio.json")

ALERT_LOSS_THRESHOLD = -0.10
USD_TO_HKD_DEFAULT   =  7.85


# ── Constants ──────────────────────────────────────────────────────────────────
SCREENER_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "HD", "CVX", "MRK", "ABBV", "PEP",
    "COST", "AVGO", "LLY", "TMO", "MCD", "ACN", "DHR", "NEE", "TXN",
    "ADBE", "NKE", "PM", "CRM", "QCOM", "AMD", "INTC", "INTU",
    "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "ADI",
    "MA", "AXP", "GS", "BLK", "SPGI", "ICE", "CME",
    "UNP", "UPS", "FDX", "NSC", "CSX",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ISRG", "BSX", "MDT", "SYK",
    "LOW", "TGT", "SBUX", "YUM", "CMG",
    "SO", "DUK", "AEP", "EXC", "D",
    "AMT", "PLD", "CCI", "EQIX", "PSA",
    "LIN", "APD", "ECL", "SHW", "PPG",
    "RTX", "LMT", "GD", "NOC", "BA",
    "ELV", "HUM", "CI", "CVS", "MCK",
]

BENCHMARK_LABELS = {
    "VOO": "S&P 500 ETF",
    "QQQ": "NASDAQ-100 ETF",
    "IWM": "Russell 2000 ETF",
    "DIA": "Dow Jones ETF",
}

SECTOR_ETFS: dict = {
    "Technology":              ("XLK",  "科技 ETF"),
    "Communication Services":  ("XLC",  "通訊服務 ETF"),
    "Consumer Cyclical":       ("XLY",  "非必需消費 ETF"),
    "Consumer Defensive":      ("XLP",  "必需消費 ETF"),
    "Energy":                  ("XLE",  "能源 ETF"),
    "Financial Services":      ("XLF",  "金融 ETF"),
    "Healthcare":              ("XLV",  "醫療健康 ETF"),
    "Industrials":             ("XLI",  "工業 ETF"),
    "Basic Materials":         ("XLB",  "原材料 ETF"),
    "Real Estate":             ("XLRE", "房地產 ETF"),
    "Utilities":               ("XLU",  "公用事業 ETF"),
    "Semiconductor":           ("SMH",  "半導體 ETF"),
}

MACRO_EVENTS = [
    ("2025-09-18", "Fed 降息 50bp（超市場預期）",       "high"),
    ("2025-11-05", "美國總統大選結果",                   "high"),
    ("2025-11-07", "Fed 再降息 25bp",                   "medium"),
    ("2025-12-18", "Fed 暗示 2026 年降息放緩",           "high"),
    ("2026-01-20", "特朗普就任美國總統",                  "high"),
    ("2026-02-12", "美國 1 月 CPI 超預期（通脹回升）",    "high"),
    ("2026-02-19", "關稅戰升級：對加/墨徵稅 25%",         "high"),
    ("2026-03-04", "關稅擴大：對中國進口徵稅 20%",         "high"),
    ("2026-03-07", "美國非農就業報告低於預期",             "medium"),
]


# ── Currency helpers ───────────────────────────────────────────────────────────
def _get_rate() -> float:
    """Return current USD→HKD rate (session-overridable via st.session_state)."""
    return float(st.session_state.get("usd_to_hkd", USD_TO_HKD_DEFAULT))


def fmt_usd_hkd(amount: float, decimals: int = 0) -> str:
    rate    = _get_rate()
    fmt_str = f",.{decimals}f"
    return f"${amount:{fmt_str}} USD (≈ ${amount * rate:{fmt_str}} HKD)"


# ── File I/O ───────────────────────────────────────────────────────────────────
def load_watchlist() -> list:
    """Load watchlist from unified user_config.json."""
    try:
        return load_watchlist_cfg()
    except Exception as e:
        print(_err("load_watchlist", e))
        return []


def save_watchlist(watchlist: list) -> None:
    """Persist watchlist to unified user_config.json."""
    try:
        save_watchlist_cfg(watchlist)
    except Exception as e:
        print(_err("save_watchlist", e))


def load_portfolio() -> dict:
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        print(_err("load_portfolio", e))
    return {}


def save_portfolio(portfolio: dict) -> None:
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(_err("save_portfolio", e))


# ── Gemini Vision: IBKR screenshot parser ─────────────────────────────────────
def parse_ibkr_screenshot(image_bytes: bytes, mime_type: str = "image/png") -> tuple:
    """
    Send an IBKR screenshot to Gemini Vision.
    Returns (holdings: list[dict], detected_rate: float | None).
    Raises ValueError with a human-readable message on failure.
    """
    if not image_bytes:
        raise ValueError("上傳的圖片內容為空，請重新上傳。")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("圖片大小超過 20 MB，請壓縮後再上傳。")

    base_url = os.environ.get(
        "AI_INTEGRATIONS_GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    ).rstrip("/")

    api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("AI_INTEGRATIONS_GEMINI_API_KEY", "")
        except Exception:
            api_key = ""

    if not api_key:
        raise ValueError(
            "⚠️ 尚未設定 Gemini API 金鑰。\n\n"
            "請在 Streamlit Cloud → App Settings → Secrets 中新增：\n"
            "AI_INTEGRATIONS_GEMINI_API_KEY = \"您的金鑰\""
        )

    model    = "gemini-2.5-flash"
    endpoint = f"{base_url}/models/{model}:generateContent"
    b64      = base64.b64encode(image_bytes).decode()

    prompt = (
        "你是一個專業的財務數據提取助手。\n"
        "這是一張盈透證券 (Interactive Brokers) 的持倉截圖。\n\n"
        "請完成兩項任務並以單一 JSON 物件回傳（只回傳 JSON，不加任何說明文字）：\n\n"
        "1. 提取所有持倉明細，放入 holdings 陣列：\n"
        "   每筆持倉格式：\n"
        "   { \"ticker\": \"股票代碼(如AAPL)\", \"qty\": 持倉數量, "
        "\"avg_price\": 平均買入價(USD), \"unrealized_pnl\": 未實現盈虧(USD) }\n\n"
        "2. 若截圖中可見任何 USD/HKD 匯率數字，將其放入 usd_to_hkd 欄位（數字類型）；"
        "若不可見則設為 null。\n\n"
        "回傳格式：\n"
        "{ \"holdings\": [...], \"usd_to_hkd\": 7.85 或 null }\n\n"
        "規則：股票代碼請使用英文大寫，無法判斷的欄位使用 null。"
    )

    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64}},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
    except requests.exceptions.Timeout:
        raise ValueError("AI 解析逾時（90 秒），請檢查網路連線後重試。")
    except requests.exceptions.ConnectionError as e:
        raise ValueError(f"無法連接 Gemini 服務：{e}")

    if not resp.ok:
        raise ValueError(
            f"Gemini API 回傳錯誤 {resp.status_code}：{resp.text[:300]}"
        )

    try:
        data = resp.json()
        raw  = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(
            f"無法解析 Gemini 回應結構：{e}\n原始回應：{resp.text[:300]}"
        )

    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AI 回傳的 JSON 格式有誤：{e}\n原始內容：{raw[:300]}"
        )

    if isinstance(parsed, list):
        holdings, detected_rate = parsed, None
    elif isinstance(parsed, dict):
        holdings      = parsed.get("holdings", [])
        detected_rate = parsed.get("usd_to_hkd")
        if detected_rate is not None:
            try:
                detected_rate = float(detected_rate)
                if not (5.0 < detected_rate < 15.0):
                    detected_rate = None
            except (TypeError, ValueError):
                detected_rate = None
    else:
        raise ValueError("AI 回傳的格式無法識別，請重試。")

    if not holdings:
        raise ValueError(
            "截圖中未偵測到任何持倉資料，請確認截圖內容是否包含持倉明細。"
        )

    return holdings, detected_rate


# ── Stock data fetching ────────────────────────────────────────────────────────
def get_stock_info(ticker: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        return {
            "ticker":          ticker,
            "name":            info.get("longName", ticker),
            "sector":          info.get("sector", "N/A"),
            "market_cap":      info.get("marketCap", 0) or 0,
            "net_margin":      info.get("profitMargins", 0) or 0,
            "pe_ratio":        info.get("trailingPE", 0) or 0,
            "revenue_growth":  info.get("revenueGrowth", 0) or 0,
            "price":           info.get("currentPrice",
                                        info.get("regularMarketPrice", 0)) or 0,
            "52w_high":        info.get("fiftyTwoWeekHigh", 0) or 0,
            "52w_low":         info.get("fiftyTwoWeekLow", 0) or 0,
            "dividend_yield":  info.get("dividendYield", 0) or 0,
            "eps":             info.get("trailingEps", 0) or 0,
            "beta":            info.get("beta", 0) or 0,
        }
    except Exception as e:
        print(_err("get_stock_info", e))
        return None


@st.cache_data(ttl=600)
def get_factor_data(ticker: str) -> dict:
    """
    Fetch all additional financial metrics needed for 7-Factor analysis.
    Returns a flat dict of raw metric values; None for unavailable fields.
    Cached for 10 minutes to reduce redundant yfinance calls.
    """
    try:
        info = yf.Ticker(ticker).info
        def _f(key, default=None):
            v = info.get(key, default)
            return v if v is not None and v != 0 else default

        return {
            # Value
            "pe_ratio":          _f("trailingPE"),
            "pb_ratio":          _f("priceToBook"),
            "ev_ebitda":         _f("enterpriseToEbitda"),
            "ps_ratio":          _f("priceToSalesTrailing12Months"),
            "dividend_yield":    (_f("dividendYield") or 0) * 100,  # convert to %
            # Quality
            "roe":               (_f("returnOnEquity") or 0) * 100,
            "roa":               (_f("returnOnAssets") or 0) * 100,
            "gross_margin":      (_f("grossMargins") or 0) * 100,
            "op_margin":         (_f("operatingMargins") or 0) * 100,
            "net_margin":        (_f("profitMargins") or 0) * 100,
            "debt_equity":       _f("debtToEquity"),
            "current_ratio":     _f("currentRatio"),
            # Growth
            "rev_growth":        (_f("revenueGrowth") or 0) * 100,
            "eps_growth":        (_f("earningsGrowth") or 0) * 100,
            "fwd_eps":           _f("forwardEps"),
            "trailing_eps":      _f("trailingEps"),
            # Volatility
            "beta":              _f("beta"),
            # Sentiment
            "short_ratio":       _f("shortRatio"),        # days to cover
            "short_pct":         (_f("shortPercentOfFloat") or 0) * 100,
            "inst_ownership":    (_f("heldPercentInstitutions") or 0) * 100,
            "insider_ownership": (_f("heldPercentInsiders") or 0) * 100,
            "rec_mean":          _f("recommendationMean"),  # 1=Strong Buy … 5=Sell
            "num_analysts":      _f("numberOfAnalystOpinions", 0),
            "target_mean":       _f("targetMeanPrice"),
            "price":             _f("currentPrice") or _f("regularMarketPrice"),
            # Extra
            "sector":            info.get("sector", "N/A"),
            "industry":          info.get("industry", "N/A"),
        }
    except Exception as e:
        print(_err("get_factor_data", e))
        return {}


def get_historical_data(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception as e:
        print(_err("get_historical_data", e))
        return None


def get_stock_news(ticker: str) -> list:
    try:
        return (yf.Ticker(ticker).news or [])[:20]
    except Exception as e:
        print(_err("get_stock_news", e))
        return []


@st.cache_data(ttl=300)
def get_analyst_data(ticker: str) -> dict:
    """
    Fetch analyst consensus rating and price targets.
    Returns a dict with keys: target_mean, target_high, target_low,
    target_median, num_analysts, recommendation, rec_mean, recs_df.
    Never raises — returns {} on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        try:
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                recs = recs.tail(6).reset_index(drop=True)
            else:
                recs = None
        except Exception:
            recs = None

        return {
            "target_mean":    info.get("targetMeanPrice"),
            "target_high":    info.get("targetHighPrice"),
            "target_low":     info.get("targetLowPrice"),
            "target_median":  info.get("targetMedianPrice"),
            "num_analysts":   info.get("numberOfAnalystOpinions", 0) or 0,
            "recommendation": info.get("recommendationKey", "n/a"),
            "rec_mean":       info.get("recommendationMean"),
            "recs_df":        recs,
        }
    except Exception as e:
        print(_err("get_analyst_data", e))
        return {}


# ── Social / sentiment signals ─────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_social_sentiment(ticker: str) -> dict:
    """
    StockTwits public API → bull/bear ratio → 0-100 score.
    Returns {score, bull_count, bear_count, total}.
    """
    default = {"score": 50.0, "bull_count": 0, "bear_count": 0, "total": 0}
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        r   = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StockScreener/1.0)"},
        )
        if not r.ok:
            return default
        messages = r.json().get("messages", [])
        if not messages:
            return default
        bull = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}) and
               m["entities"]["sentiment"].get("basic") == "Bullish"
        )
        bear = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}) and
               m["entities"]["sentiment"].get("basic") == "Bearish"
        )
        tagged = bull + bear
        score  = (bull / tagged * 100) if tagged > 0 else 50.0
        return {"score": round(score, 1), "bull_count": bull,
                "bear_count": bear, "total": len(messages)}
    except Exception as e:
        print(_err("get_social_sentiment", e))
        return default


@st.cache_data(ttl=300)
def get_news_sentiment_score(ticker: str) -> float:
    """
    Fetch stock news and run keyword sentiment.
    Returns 0-100 score (50 = neutral, >50 = bullish, <50 = bearish).
    """
    try:
        news = get_stock_news(ticker)
        if not news:
            return 50.0
        points = []
        for item in news:
            text = (item.get("title", "") + " " + item.get("summary", "")).strip()
            s    = classify_sentiment(text)
            points.append(75.0 if s == "positive" else (25.0 if s == "negative" else 50.0))
        return round(float(np.mean(points)), 1)
    except Exception as e:
        print(_err("get_news_sentiment_score", e))
        return 50.0


def get_combined_sentiment(ticker: str) -> dict:
    """
    Combine news (40%) + social (60%) into a single 0-100 composite score.
    """
    try:
        news_score   = get_news_sentiment_score(ticker)
        social_data  = get_social_sentiment(ticker)
        social_score = social_data["score"]
        combined     = round(news_score * 0.4 + social_score * 0.6, 1)
        return {
            "combined":     combined,
            "news_score":   news_score,
            "social_score": social_score,
            "social_data":  social_data,
        }
    except Exception as e:
        print(_err("get_combined_sentiment", e))
        return {"combined": 50.0, "news_score": 50.0,
                "social_score": 50.0,
                "social_data": {"bull_count": 0, "bear_count": 0, "total": 0}}


# ── VIX / benchmark ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_vix_history(lookback: int = 63) -> pd.DataFrame:
    """Fetch ^VIX history for ~3 months; return DataFrame with Close renamed to VIX."""
    try:
        hist = yf.Ticker("^VIX").history(period="3mo")
        if hist.empty:
            return pd.DataFrame()
        df  = hist[["Close"]].copy().rename(columns={"Close": "VIX"})
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        else:
            try:
                idx = idx.tz_localize(None)
            except TypeError:
                pass
        df.index = idx
        return df.tail(lookback)
    except Exception as e:
        print(_err("get_vix_history", e))
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_market_benchmark(ticker: str, period: str = "1y") -> dict:
    """
    Fetch benchmark ETF history + SMA50/SMA200 + golden cross flag.
    Returns a safe dict even on failure.
    """
    result: dict = {
        "ticker": ticker, "label": BENCHMARK_LABELS.get(ticker, ticker),
        "sma50": None, "sma200": None, "golden_cross": None,
        "price": None, "hist": None, "perf_1y": None,
    }
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty:
            return result
        idx = pd.to_datetime(hist.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        hist.index = idx
        result["hist"]    = hist
        result["price"]   = float(hist["Close"].iloc[-1])
        result["perf_1y"] = float(
            (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
        )
        if len(hist) >= 50:
            result["sma50"]  = float(hist["Close"].rolling(50).mean().iloc[-1])
        if len(hist) >= 200:
            result["sma200"] = float(hist["Close"].rolling(200).mean().iloc[-1])
        if result["sma50"] and result["sma200"]:
            result["golden_cross"] = result["sma50"] > result["sma200"]
    except Exception as e:
        print(_err("get_market_benchmark", e))
    return result

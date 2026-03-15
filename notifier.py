# ═══════════════════════════════════════════════════════════════════════════════
# notifier.py — Telegram real-time alert module
# Sends contextualized Gemini AI notifications for portfolio, watchlist,
# MPF rebalancing, and macro market events.
# ═══════════════════════════════════════════════════════════════════════════════
import json
import os
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def _tz_strip(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from DataFrame index to avoid mixed-tz comparison errors."""
    if df is not None and not df.empty:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


_MODULE  = "notifier"
_CACHE_F = "telegram_notif_cache.json"
_LOG_F   = "telegram_notif_log.json"
_COOLDOWN_HOURS  = 4          # default cooldown between identical alerts
_MAX_LOG_ENTRIES = 100        # cap the in-file log


# ── Helpers ───────────────────────────────────────────────────────────────────

def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


def _load_cache() -> dict:
    try:
        if Path(_CACHE_F).exists():
            with open(_CACHE_F, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_F, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(_err("_save_cache", e))


def _is_duplicate(key: str, cooldown_hours: int = _COOLDOWN_HOURS) -> bool:
    """Return True if this alert key was already sent within cooldown window."""
    cache = _load_cache()
    if key in cache:
        try:
            last = datetime.fromisoformat(cache[key])
            if datetime.now() - last < timedelta(hours=cooldown_hours):
                return True
        except Exception:
            pass
    return False


def _mark_sent(key: str) -> None:
    cache = _load_cache()
    cache[key] = datetime.now().isoformat()
    _save_cache(cache)


def _append_log(entry: dict) -> None:
    """Append a sent-notification entry to the local log file."""
    try:
        log = []
        if Path(_LOG_F).exists():
            with open(_LOG_F, encoding="utf-8") as f:
                log = json.load(f)
        log.append(entry)
        log = log[-_MAX_LOG_ENTRIES:]   # keep last N
        with open(_LOG_F, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(_err("_append_log", e))


def load_notification_log() -> list:
    """Load and return the notification history log."""
    try:
        if Path(_LOG_F).exists():
            with open(_LOG_F, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


# ── Core Telegram sender ──────────────────────────────────────────────────────

def send_telegram_notification(bot_token: str, chat_id: str, message: str) -> bool:
    """
    Send a message via Telegram Bot API.
    Uses MarkdownV2 with automatic fallback to plain text on parse error.
    Returns True on success.
    """
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    for parse_mode in ("Markdown", None):
        try:
            payload: dict = {"chat_id": chat_id, "text": message}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            # If markdown parse fails, retry without formatting
            if resp.status_code == 400 and parse_mode:
                continue
            print(_err("send_telegram_notification",
                       Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")))
            return False
        except Exception as e:
            print(_err("send_telegram_notification", e))
            return False
    return False


# ── AI message generator ──────────────────────────────────────────────────────

_ALERT_PROMPTS = {
    "take_profit": (
        "你是一位投資顧問助手，請用繁體中文撰寫一則 Telegram 投資通知訊息（限 150 字以內）。\n"
        "場景：持倉股票已觸發止盈信號。\n"
        "數據：{data}\n"
        "要求：包含 emoji，說明漲幅、建議操作（部份獲利了結 or 移動止損），結尾附上提醒語。"
    ),
    "stop_loss": (
        "你是一位投資顧問助手，請用繁體中文撰寫一則 Telegram 投資預警訊息（限 150 字以內）。\n"
        "場景：持倉股票已觸發止損信號。\n"
        "數據：{data}\n"
        "要求：包含 emoji，說明跌幅，建議停損操作，語氣穩健不恐慌，結尾提醒風險控制。"
    ),
    "watchlist_rsi": (
        "你是一位投資顧問助手，請用繁體中文撰寫一則 Telegram 投資機會通知（限 150 字以內）。\n"
        "場景：觀察名單股票出現技術面買入信號（RSI 超賣或均線突破）。\n"
        "數據：{data}\n"
        "要求：包含 emoji，說明技術信號，建議可小量觀察佈局，附上謹慎投資提醒。"
    ),
    "watchlist_breakout": (
        "你是一位投資顧問助手，請用繁體中文撰寫一則 Telegram 突破機會通知（限 150 字以內）。\n"
        "場景：觀察名單股票突破重要均線，出現看漲信號。\n"
        "數據：{data}\n"
        "要求：包含 emoji，說明均線突破詳情，建議關注並設定入場條件，加入風險提示。"
    ),
    "mpf_rebalance": (
        "你是一位 MPF（強積金）投資顧問，請用繁體中文撰寫一則 Telegram 再平衡提醒（限 150 字以內）。\n"
        "場景：MPF 策略引擎分析顯示基金組合應進行再平衡。\n"
        "數據：{data}\n"
        "要求：包含 emoji，說明調整方向，建議操作步驟，附上長線投資提醒。"
    ),
    "macro_fear": (
        "你是一位市場分析師，請用繁體中文撰寫一則 Telegram 市場預警訊息（限 150 字以內）。\n"
        "場景：宏觀市場出現極度恐慌或貪婪信號。\n"
        "數據：{data}\n"
        "要求：包含 emoji，說明市場情緒狀態，提供操作建議（恐慌=考慮逐步建倉，貪婪=適當獲利了結），附上理性投資提醒。"
    ),
    "macro_crash": (
        "你是一位市場分析師，請用繁體中文撰寫一則 Telegram 市場崩跌預警（限 150 字以內）。\n"
        "場景：大盤檢測到死亡交叉且市場情緒極度悲觀。\n"
        "數據：{data}\n"
        "要求：包含 emoji，語氣穩健，建議降低倉位或持有現金，提醒不要恐慌性拋售，附上長期視角。"
    ),
}


def _build_ai_message(alert_type: str, data: dict, gemini_key: str,
                      fallback: str = "") -> str:
    """
    Generate a Gemini AI contextualized notification message.
    Uses exponential-backoff retry on 429/timeout; falls back to template on failure.
    """
    if not gemini_key:
        return fallback or _default_message(alert_type, data)
    try:
        from gemini_helper import call_gemini_raw
        prompt_tmpl = _ALERT_PROMPTS.get(alert_type, "")
        if not prompt_tmpl:
            return fallback or _default_message(alert_type, data)
        data_str = "\n".join(f"  {k}: {v}" for k, v in data.items())
        prompt   = prompt_tmpl.format(data=data_str)
        return call_gemini_raw(prompt, gemini_key).strip()
    except Exception as e:
        print(_err("_build_ai_message", e))
        return fallback or _default_message(alert_type, data)


def _default_message(alert_type: str, data: dict) -> str:
    """Simple fallback message when Gemini is unavailable."""
    templates = {
        "take_profit": "💰 *止盈提醒*\n{ticker} 漲幅達 {pct}%，已觸發止盈閾值，建議考慮部份獲利。",
        "stop_loss":   "⚠️ *止損預警*\n{ticker} 跌幅達 {pct}%，已觸發止損閾值，請留意風險控制。",
        "watchlist_rsi":      "📉 *超賣機會*\n{ticker} RSI={rsi:.1f}，出現超賣信號，可關注低接機會。",
        "watchlist_breakout": "📈 *均線突破*\n{ticker} 收盤 ${price:.2f} 突破 SMA{sma}，看漲訊號出現。",
        "mpf_rebalance": "🛡️ *MPF 再平衡提醒*\n策略引擎建議調整基金配置。",
        "macro_fear":   "😱 *市場情緒預警*\n恐懼貪婪指數 {index}，市場出現{state}信號。",
        "macro_crash":  "📉 *宏觀市場預警*\n大盤出現死亡交叉且情緒極度悲觀，建議審視持倉風險。",
    }
    tmpl = templates.get(alert_type, "📢 *系統通知*\n{alert_type} 觸發。")
    try:
        return tmpl.format(alert_type=alert_type, **data)
    except Exception:
        return tmpl.format(alert_type=alert_type, **{k: str(v) for k, v in data.items()})


# ── Alert checkers ────────────────────────────────────────────────────────────

def check_portfolio_alerts(
    portfolio: dict,
    prices: dict,
    bot_token: str,
    chat_id: str,
    gemini_key: str,
    threshold_pct: float = 10.0,
    app_url: str = "",
) -> list[dict]:
    """
    Check each portfolio position for stop-loss / take-profit triggers.
    Returns list of sent alert records.
    """
    sent = []
    for ticker, entry in portfolio.items():
        buy_price = entry.get("buy_price", 0)
        qty       = entry.get("qty", 0)
        cur_price = prices.get(ticker)
        if not cur_price or not buy_price:
            continue
        pct_chg = (cur_price - buy_price) / buy_price * 100

        # Take-profit trigger (positive threshold)
        if pct_chg >= threshold_pct:
            key = f"take_profit_{ticker}_{int(pct_chg // threshold_pct) * int(threshold_pct)}"
            if not _is_duplicate(key):
                data = {
                    "ticker":    ticker,
                    "pct":       f"{pct_chg:+.1f}",
                    "買入價":    f"${buy_price:.2f}",
                    "現價":      f"${cur_price:.2f}",
                    "持股數量":  qty,
                    "浮動盈虧":  f"${(cur_price - buy_price) * qty:+,.0f}",
                    "止盈閾值":  f"{threshold_pct}%",
                }
                msg = _build_ai_message("take_profit", data, gemini_key,
                                        _default_message("take_profit", data))
                if app_url:
                    msg += f"\n\n🔗 [查看詳情]({app_url})"
                ok = send_telegram_notification(bot_token, chat_id, msg)
                if ok:
                    _mark_sent(key)
                    rec = {"time": datetime.now().isoformat(), "type": "take_profit",
                           "ticker": ticker, "pct": round(pct_chg, 2), "message": msg}
                    _append_log(rec)
                    sent.append(rec)

        # Stop-loss trigger (negative threshold)
        elif pct_chg <= -threshold_pct:
            key = f"stop_loss_{ticker}_{int(abs(pct_chg) // threshold_pct) * int(threshold_pct)}"
            if not _is_duplicate(key):
                data = {
                    "ticker":    ticker,
                    "pct":       f"{pct_chg:+.1f}",
                    "買入價":    f"${buy_price:.2f}",
                    "現價":      f"${cur_price:.2f}",
                    "持股數量":  qty,
                    "浮動虧損":  f"${(cur_price - buy_price) * qty:+,.0f}",
                    "止損閾值":  f"{threshold_pct}%",
                }
                msg = _build_ai_message("stop_loss", data, gemini_key,
                                        _default_message("stop_loss", data))
                if app_url:
                    msg += f"\n\n🔗 [查看詳情]({app_url})"
                ok = send_telegram_notification(bot_token, chat_id, msg)
                if ok:
                    _mark_sent(key)
                    rec = {"time": datetime.now().isoformat(), "type": "stop_loss",
                           "ticker": ticker, "pct": round(pct_chg, 2), "message": msg}
                    _append_log(rec)
                    sent.append(rec)
    return sent


def check_watchlist_alerts(
    watchlist: list[str],
    bot_token: str,
    chat_id: str,
    gemini_key: str,
    app_url: str = "",
) -> list[dict]:
    """
    Check watchlist stocks for RSI oversold (<30) or SMA50 breakout.
    Returns list of sent alert records.
    """
    sent = []
    try:
        import yfinance as yf
        import numpy as np
    except ImportError:
        return sent

    for ticker in watchlist[:10]:   # cap at 10 to avoid rate limits
        try:
            hist = _tz_strip(yf.Ticker(ticker).history(period="3mo"))
            if hist.empty or len(hist) < 20:
                continue
            close   = hist["Close"]
            sma50   = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            sma200  = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
            prev_close = float(close.iloc[-2]) if len(close) >= 2 else None
            cur_close  = float(close.iloc[-1])

            # RSI calculation
            delta   = close.diff()
            gain    = delta.clip(lower=0).rolling(14).mean()
            loss    = (-delta.clip(upper=0)).rolling(14).mean()
            rs      = gain / loss.replace(0, float("nan"))
            rsi_s   = 100 - 100 / (1 + rs)
            cur_rsi = float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0

            # RSI oversold (<30) signal
            if cur_rsi <= 30:
                key = f"watchlist_rsi_{ticker}_{datetime.now().strftime('%Y-%m-%d')}"
                if not _is_duplicate(key, cooldown_hours=24):
                    data = {
                        "ticker":   ticker,
                        "rsi":      cur_rsi,
                        "現價":     f"${cur_close:.2f}",
                        "RSI":      f"{cur_rsi:.1f}（超賣區間 <30）",
                        "SMA50":    f"${sma50:.2f}" if sma50 else "N/A",
                        "信號":     "RSI 超賣，可能即將反彈",
                    }
                    msg = _build_ai_message("watchlist_rsi", data, gemini_key,
                                            _default_message("watchlist_rsi", data))
                    if app_url:
                        msg += f"\n\n🔗 [查看詳情]({app_url})"
                    ok = send_telegram_notification(bot_token, chat_id, msg)
                    if ok:
                        _mark_sent(key)
                        rec = {"time": datetime.now().isoformat(),
                               "type": "watchlist_rsi",
                               "ticker": ticker, "rsi": round(cur_rsi, 1),
                               "message": msg}
                        _append_log(rec)
                        sent.append(rec)

            # SMA50 breakout (price crosses above SMA50)
            if sma50 and prev_close and prev_close < sma50 <= cur_close:
                key = f"watchlist_breakout_{ticker}_{datetime.now().strftime('%Y-%m-%d')}"
                if not _is_duplicate(key, cooldown_hours=24):
                    data = {
                        "ticker":  ticker,
                        "price":   cur_close,
                        "sma":     50,
                        "現價":    f"${cur_close:.2f}",
                        "SMA50":   f"${sma50:.2f}",
                        "SMA200":  f"${sma200:.2f}" if sma200 else "N/A",
                        "RSI":     f"{cur_rsi:.1f}",
                        "信號":    "突破 SMA50，多頭趨勢確認",
                    }
                    msg = _build_ai_message("watchlist_breakout", data, gemini_key,
                                            _default_message("watchlist_breakout", data))
                    if app_url:
                        msg += f"\n\n🔗 [查看詳情]({app_url})"
                    ok = send_telegram_notification(bot_token, chat_id, msg)
                    if ok:
                        _mark_sent(key)
                        rec = {"time": datetime.now().isoformat(),
                               "type": "watchlist_breakout",
                               "ticker": ticker, "price": round(cur_close, 2),
                               "message": msg}
                        _append_log(rec)
                        sent.append(rec)
        except Exception as e:
            print(_err(f"check_watchlist_alerts/{ticker}", e))
            continue

    return sent


def check_macro_alerts(
    fear_index: float,
    golden_cross: bool | None,
    bot_token: str,
    chat_id: str,
    gemini_key: str,
    app_url: str = "",
) -> list[dict]:
    """
    Check macro conditions: extreme fear/greed index or death-cross + high fear.
    Returns list of sent alert records.
    """
    sent = []
    today = datetime.now().strftime("%Y-%m-%d")

    # Extreme fear (<20) or greed (>80)
    if fear_index is not None and (fear_index <= 20 or fear_index >= 80):
        state   = "極度恐慌" if fear_index <= 20 else "極度貪婪"
        key     = f"macro_fear_{state}_{today}"
        if not _is_duplicate(key, cooldown_hours=8):
            data = {
                "恐懼貪婪指數": f"{fear_index:.0f}/100",
                "state":       state,
                "index":       fear_index,
                "大盤狀態":    "黃金交叉" if golden_cross else "死亡交叉" if golden_cross is False else "不明",
                "日期":        today,
            }
            msg = _build_ai_message("macro_fear", data, gemini_key,
                                    _default_message("macro_fear", data))
            if app_url:
                msg += f"\n\n🔗 [查看詳情]({app_url})"
            ok = send_telegram_notification(bot_token, chat_id, msg)
            if ok:
                _mark_sent(key)
                rec = {"time": datetime.now().isoformat(),
                       "type": "macro_fear",
                       "fear_index": fear_index, "state": state,
                       "message": msg}
                _append_log(rec)
                sent.append(rec)

    # Death-cross + extreme fear (double warning)
    if golden_cross is False and fear_index is not None and fear_index <= 30:
        key = f"macro_crash_{today}"
        if not _is_duplicate(key, cooldown_hours=12):
            data = {
                "大盤趨勢":     "死亡交叉（SMA50 < SMA200）",
                "恐懼貪婪指數": f"{fear_index:.0f}/100",
                "市場情緒":     "極度悲觀",
                "建議":         "審視持倉，降低風險敞口",
                "日期":         today,
            }
            msg = _build_ai_message("macro_crash", data, gemini_key,
                                    _default_message("macro_crash", data))
            if app_url:
                msg += f"\n\n🔗 [查看詳情]({app_url})"
            ok = send_telegram_notification(bot_token, chat_id, msg)
            if ok:
                _mark_sent(key)
                rec = {"time": datetime.now().isoformat(),
                       "type": "macro_crash",
                       "fear_index": fear_index,
                       "message": msg}
                _append_log(rec)
                sent.append(rec)

    return sent


def check_mpf_alerts(
    signals: list[dict],
    bot_token: str,
    chat_id: str,
    gemini_key: str,
    app_url: str = "",
) -> list[dict]:
    """
    Check MPF strategy signals for rebalancing recommendations.
    `signals` is a list of dicts with keys: ticker, action, reason.
    Returns list of sent alert records.
    """
    sent = []
    today = datetime.now().strftime("%Y-%m-%d")
    if not signals:
        return sent

    key = f"mpf_rebalance_{today}"
    if _is_duplicate(key, cooldown_hours=24):
        return sent

    action_list = "\n".join(
        f"  • {s.get('ticker','?')}: {s.get('action','?')} — {s.get('reason','')}"
        for s in signals[:5]
    )
    data = {
        "調整信號數量": len(signals),
        "調整明細":    action_list,
        "策略基礎":    "移動平均線 + 相對強度",
        "觸發日期":    today,
    }
    msg = _build_ai_message("mpf_rebalance", data, gemini_key,
                            _default_message("mpf_rebalance", data))
    if app_url:
        msg += f"\n\n🔗 [查看詳情]({app_url})"
    ok = send_telegram_notification(bot_token, chat_id, msg)
    if ok:
        _mark_sent(key)
        rec = {"time": datetime.now().isoformat(),
               "type": "mpf_rebalance",
               "signal_count": len(signals),
               "message": msg}
        _append_log(rec)
        sent.append(rec)

    return sent


# ── Master orchestrator ───────────────────────────────────────────────────────

def run_all_checks(
    portfolio:    dict,
    watchlist:    list[str],
    prices:       dict,
    fear_index:   float | None,
    golden_cross: bool | None,
    mpf_signals:  list[dict],
    bot_token:    str,
    chat_id:      str,
    gemini_key:   str,
    threshold_pct: float = 10.0,
    app_url:      str = "",
) -> list[dict]:
    """
    Run all alert checks and return aggregated list of sent notifications.
    Call this from the Streamlit UI (button or auto-trigger).
    """
    all_sent: list[dict] = []

    all_sent += check_portfolio_alerts(
        portfolio, prices, bot_token, chat_id, gemini_key, threshold_pct, app_url
    )
    all_sent += check_watchlist_alerts(
        watchlist, bot_token, chat_id, gemini_key, app_url
    )
    if fear_index is not None:
        all_sent += check_macro_alerts(
            fear_index, golden_cross, bot_token, chat_id, gemini_key, app_url
        )
    if mpf_signals:
        all_sent += check_mpf_alerts(
            mpf_signals, bot_token, chat_id, gemini_key, app_url
        )

    return all_sent


def get_current_prices(tickers: list[str]) -> dict:
    """Fetch latest close prices for a list of tickers via yfinance."""
    prices = {}
    try:
        import yfinance as yf
        for t in tickers:
            try:
                h = _tz_strip(yf.Ticker(t).history(period="1d"))
                if not h.empty:
                    prices[t] = float(h["Close"].iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    return prices

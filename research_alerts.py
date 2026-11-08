"""Deterministic events. A feed, not a simulated push-notification integration."""
from datetime import datetime, timezone
from scientific_factors import finite


def generate_alerts(previous, records, sentiment, settings, now=None):
    now = now or datetime.now(timezone.utc)
    previous = previous or {}
    old = {r["ticker"]:r for r in previous.get("records", [])}
    alerts = []
    token = now.isoformat()
    for row in records:
        ticker = row["ticker"]
        before = old.get(ticker, {})
        a = finite(before.get("factors", {}).get("composite"))
        b = finite(row.get("factors", {}).get("composite"))
        if a is not None and b is not None and abs(b-a) >= settings.get("score_threshold", 1.):
            alerts.append(dict(key=f"score:{ticker}:{token}", ticker=ticker, kind="分數變動",
                               message=f"{ticker} 校準總分 {a:+.2f} → {b:+.2f}；變動 {b-a:+.2f}"))
        a, b = finite(before.get("rs1m")), finite(row.get("rs1m"))
        if a is not None and b is not None and ((a <= 0 < b) or (a >= 0 > b)):
            alerts.append(dict(key=f"rs:{ticker}:{token}", ticker=ticker, kind="相對強弱穿越",
                               message=f"{ticker} 對 SPY 的 21 交易日回報差 {'升穿' if b > 0 else '跌穿'}零：{b:+.2f} 個百分點"))
        event = row.get("earnings") or {}
        start = finite(event.get("start"))
        if start is not None:
            date = datetime.fromtimestamp(start, timezone.utc)
            if 0 <= (date-now).total_seconds() <= 7*86400:
                alerts.append(dict(key=f"earnings:{ticker}:{date.date()}", ticker=ticker, kind="業績臨近",
                                   message=f"{ticker} 業績日期／預估窗口起點 {date.isoformat()}；"
                                   f"{'來源標示確定' if event.get('confirmed') else '日期可能調整'}"))
    for key, label in (("combined", "StockTwits／Yahoo 綜合情緒"), ("vix_greed", "VIX 百分位反向指標")):
        a = finite(previous.get("sentiment", {}).get(key))
        b = finite(sentiment.get(key))
        if a is not None and b is not None and abs(b-a) >= settings.get("sentiment_threshold", 20.):
            alerts.append(dict(key=f"sentiment:{key}:{token}", ticker=None, kind="情緒急變",
                               message=f"{label} {a:.1f} → {b:.1f}（0–100）；兩者不是同一指標"))
    return alerts

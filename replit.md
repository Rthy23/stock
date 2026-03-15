# 美股交易儀表板系統 (US Stock Trading Dashboard)

## Overview

A Streamlit-based US stock screening and trading dashboard with technical analysis, portfolio management, and market data visualization.

## Architecture

- **Framework**: Streamlit (Python)
- **Port**: 5000
- **Entry Point**: `app.py`

## Key Files

- `app.py` — Main Streamlit app entry point, page routing, sidebar navigation
- `analysis.py` — Technical indicators, stock screening, buy zone/exit strategy calculations, charts; includes full **7-Factor engine** (`calculate_factor_score`, `calculate_seven_factors`, `plot_factor_radar`, `build_factor_prompt`)
- `data_fetcher.py` — yfinance data fetching, watchlist/portfolio persistence, market benchmarks; includes `get_factor_data()` (10-min cached, fetches ROE/ROA/PB/EV-EBITDA/short-pct/inst-ownership etc.)
- `ui_components.py` — Reusable Streamlit UI components and session state management
- `backtest_engine.py` — RSI mean-reversion backtest engine with SMA200 filter, dynamic stop-loss, individual_series
- `mpf_assistant.py` — MPF (積金) intelligent advisor: OCR, portfolio, rebalance, strategy signals, AI report
- `mpf_db.py` — SQLite persistence for MPF holdings (mpf_holdings.db)
- `mpf_strategy.py` — RS+SMA strategy engine, dual pie charts, ETF history comparison, defensive allocation
- `ocr_module.py` — Gemini AI OCR for MPF statement parsing + quant report generation
- `kol_whitelist.py` — KOL whitelist scoring engine: 8 curated analysts, 3-dimensional scoring (consistency × argument quality × recency), Gemini AI structured consensus report
- `.streamlit/config.toml` — Streamlit server config (host: 0.0.0.0, port: 5000)
- `requirements.txt` — Python dependencies

## Dependencies

- `streamlit` — Web UI framework
- `yfinance` — Yahoo Finance market data
- `pandas` — Data manipulation
- `plotly` — Interactive charts
- `numpy` — Numerical computations
- `requests` — HTTP requests

## Running the App

```bash
streamlit run app.py --server.port 5000
```

## Recent Changes

- **Telegram real-time notifications** (`notifier.py`): Full alert module with 4-hour deduplication cache, Gemini AI message generator, and 4 checkers: portfolio stop-loss/take-profit, watchlist RSI oversold + SMA50 breakout, macro fear/greed extremes, MPF rebalancing signals. Integrated into `app.py` sidebar ("📱 Telegram 即時通知" expander) with manual check button, test message, auto-trigger toggle, and notification log panel on 我的持倉 page. Requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_USER_ID` secrets.
- **Tabs → Scrollable layout**: 美股回測 page replaced tabs 1-4 (策略設定/績效報告/技術指標對比/資金流入分析) with scrollable `st.container()` sections; kept only the 🔍 股票篩選 tab.
- **Gemini AI for 個股診斷**: Section 7 added — "🤖 Gemini AI 個股深度分析" button with `_call_gemini_stock_ai()`.
- **Gemini AI for 我的持倉**: "🤖 Gemini AI 持倉組合分析" section added after `render_portfolio_dashboard()`.
- **Error handling**: `generate_quant_report()` in `ocr_module.py` now returns readable error strings for 403 (leaked key) and 429 (quota exceeded) instead of None.
- **Dark theme fix**: All `background:#E8F8EE / #F4F6F8 / #EEF6FF / #F8FAFB / #F0F8FF / #FFF5E6 / #FEE8E8 / #F0F7FF` HTML divs replaced with dark equivalents (#0D2E1A / #1C2128 / #1B2A3D / #2D1B00 / #2D1B1B). Text colors updated from #555/#666/#1A1A2E to #8B949E/#E6EDF3.
- **Screener charts**: `paper_bgcolor="#FFFFFF"` and `plot_bgcolor="#F8F9FA"` changed to #0d1117 / #161B22 with dark axis colors.

## Deployment

Configured for autoscale deployment on Replit.

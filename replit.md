# 美股交易儀表板系統 (US Stock Trading Dashboard)

## Overview

A Streamlit-based US stock screening and trading dashboard with technical analysis, portfolio management, and market data visualization.

## Architecture

- **Framework**: Streamlit (Python)
- **Port**: 5000
- **Entry Point**: `app.py`

## Key Files

- `app.py` — Main Streamlit app entry point, page routing, sidebar navigation
- `analysis.py` — Technical indicators, stock screening, buy zone/exit strategy calculations, charts
- `data_fetcher.py` — yfinance data fetching, watchlist/portfolio persistence, market benchmarks
- `ui_components.py` — Reusable Streamlit UI components and session state management
- `.streamlit/config.toml` — Streamlit server config (host: 0.0.0.0, port: 5000, dark theme)
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

## Deployment

Configured for autoscale deployment on Replit.

# 美股交易儀表板系統

美股選股與技術分析儀表板，整合基本面篩選、K 線圖、止損風控、情緒指標與分析師共識評級。

---

## 功能總覽

- **總體市場 (Macro)**：大盤黃金/死亡交叉偵測、市場恐慌指標（Fear & Greed，3 個月）
- **個股診斷 (Micro)**：技術指標、買入區間、止損/目標計算、1週/1個月/3個月對比分析
- **我的持倉**：持倉管理、盈虧追蹤、盈透證券截圖自動同步（Gemini Vision）

---

## 快速啟動

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 啟動應用

```bash
streamlit run app.py --server.port 5000
```

本地預覽：http://localhost:5000

---

## API Secrets 設定

### Streamlit Cloud 部署

在 Streamlit Cloud → App Settings → **Secrets** 新增以下金鑰：

```toml
# 盈透證券截圖 AI 解析（Gemini Vision）
AI_INTEGRATIONS_GEMINI_API_KEY = "your_gemini_api_key_here"
```

> **注意**：若未設定 Gemini API 金鑰，IBKR 截圖解析功能將無法使用，其他功能不受影響。

### 本地開發

建立 `.streamlit/secrets.toml` 檔案：

```toml
AI_INTEGRATIONS_GEMINI_API_KEY = "your_gemini_api_key_here"
```

---

## 依賴套件

| 套件 | 版本需求 | 用途 |
|------|----------|------|
| streamlit | ≥ 1.32.0 | Web UI 框架 |
| yfinance | ≥ 0.2.36 | Yahoo Finance 市場數據 |
| pandas | ≥ 2.0.0 | 數據處理 |
| plotly | ≥ 5.18.0 | 互動圖表 |
| numpy | ≥ 1.24.0 | 數值計算 |
| requests | ≥ 2.31.0 | HTTP 請求 |

---

## 技術架構

```
app.py            — 主入口、頁面路由、側邊欄
analysis.py       — 技術指標計算（無網路、無 Streamlit）
data_fetcher.py   — 外部 API 呼叫、檔案 I/O、資料常數
ui_components.py  — Plotly 圖表、Streamlit 元件
```

---

## 術語說明

| 術語 | 說明 |
|------|------|
| RSI | 股票的體力值，>70 易回調，<30 易反彈 |
| MACD | 趨勢指南針，金叉看漲，死叉看跌 |
| 乖離率 | 股價與均線的橡皮筋，拉太遠會彈回 |
| 黃金交叉 | SMA50 向上穿越 SMA200，多頭訊號 |
| 做 T | 當天低買高賣，降低持倉成本 |
| 做空 | 先借券賣出再低價買回，從跌勢獲利 |

---

## 資料來源

- 股票數據：[Yahoo Finance (yfinance)](https://github.com/ranaroussi/yfinance)
- 社交情緒：StockTwits 公開 API
- 恐慌指數：^VIX（芝加哥期權交易所波動率指數）

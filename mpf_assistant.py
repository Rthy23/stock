# ═══════════════════════════════════════════════════════════════════════════════
# mpf_assistant.py  —  MPF 智投診斷室 UI + OCR + Rebalancing
# ═══════════════════════════════════════════════════════════════════════════════
import os
import re
import json
import base64
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

from backtest_engine import calc_rebalance
from ocr_module import ocr_with_gemini, render_manual_correction_form
from data_fetcher import standardize_timezone
import mpf_db
import mpf_strategy

_MODULE       = "mpf_assistant"
_MAPPING_PATH = "fund_mapping.json"

_FALLBACK_HOLDINGS: dict = {
    "SPY":  [("AAPL","Apple Inc.",7.1),("MSFT","Microsoft Corp.",6.8),
             ("NVDA","NVIDIA Corp.",6.1),("AMZN","Amazon.com",3.8),
             ("META","Meta Platforms",2.5),("GOOGL","Alphabet A",2.2),
             ("GOOG","Alphabet C",1.9),("BRK-B","Berkshire Hathaway",1.7),
             ("LLY","Eli Lilly",1.6),("AVGO","Broadcom",1.5)],
    "VOO":  [("AAPL","Apple Inc.",7.1),("MSFT","Microsoft Corp.",6.8),
             ("NVDA","NVIDIA Corp.",6.1),("AMZN","Amazon.com",3.8),
             ("META","Meta Platforms",2.5),("GOOGL","Alphabet A",2.2),
             ("GOOG","Alphabet C",1.9),("BRK-B","Berkshire Hathaway",1.7),
             ("LLY","Eli Lilly",1.6),("AVGO","Broadcom",1.5)],
    "QQQ":  [("MSFT","Microsoft",8.9),("AAPL","Apple",8.2),
             ("NVDA","NVIDIA",8.0),("AMZN","Amazon",5.1),
             ("META","Meta",4.9),("GOOGL","Alphabet A",4.3),
             ("GOOG","Alphabet C",4.0),("TSLA","Tesla",3.0),
             ("AVGO","Broadcom",2.7),("COST","Costco",2.2)],
    "ACWI": [("AAPL","Apple Inc.",4.2),("MSFT","Microsoft",3.9),
             ("NVDA","NVIDIA",3.5),("AMZN","Amazon",2.1),
             ("META","Meta",1.6),("GOOGL","Alphabet A",1.4),
             ("TSLA","Tesla",1.1),("BRK-B","Berkshire",0.9),
             ("LLY","Eli Lilly",0.9),("JPM","JPMorgan",0.8)],
    "VT":   [("AAPL","Apple",3.8),("MSFT","Microsoft",3.4),
             ("NVDA","NVIDIA",3.1),("AMZN","Amazon",1.9),
             ("META","Meta",1.4),("GOOGL","Alphabet",1.2),
             ("TSLA","Tesla",1.0),("BRK-B","Berkshire",0.8),
             ("LLY","Eli Lilly",0.8),("JPM","JPMorgan",0.7)],
    "VPL":  [("7203.T","Toyota",3.2),("Samsung","Samsung Elec.",3.0),
             ("6758.T","Sony Group",1.8),("6861.T","Keyence",1.5),
             ("8306.T","MUFG",1.4),("9984.T","SoftBank",1.3),
             ("2330.TW","TSMC",5.1),("005930.KS","Samsung KR",3.2),
             ("EWH","Hang Seng ETF",2.0),("VPL","Asia Pacific",1.0)],
    "EWJ":  [("7203.T","Toyota",5.1),("6758.T","Sony",4.2),
             ("6861.T","Keyence",3.8),("8306.T","MUFG",3.5),
             ("9984.T","SoftBank",3.1),("7974.T","Nintendo",2.8),
             ("6501.T","Hitachi",2.5),("4063.T","Shin-Etsu Chem",2.3),
             ("8035.T","Tokyo Electron",2.2),("9433.T","KDDI",2.0)],
    "EWH":  [("0005.HK","HSBC Holdings",10.2),("0700.HK","Tencent",8.9),
             ("0941.HK","China Mobile",6.1),("1299.HK","AIA Group",5.8),
             ("0883.HK","CNOOC",4.2),("2318.HK","Ping An",3.9),
             ("1398.HK","ICBC",3.7),("0388.HK","HK Exchange",3.5),
             ("2628.HK","China Life",3.2),("3988.HK","Bank of China",3.0)],
    "MCHI": [("0700.HK","Tencent",16.2),("BABA","Alibaba",8.9),
             ("PDD","Pinduoduo",6.1),("NIO","NIO Inc.",2.1),
             ("BIDU","Baidu",3.9),("JD","JD.com",3.7),
             ("NTES","NetEase",3.5),("XPEV","XPeng",1.2),
             ("LI","Li Auto",1.5),("0939.HK","CCB",2.8)],
    "XLV":  [("LLY","Eli Lilly",12.1),("UNH","UnitedHealth",11.8),
             ("JNJ","Johnson & Johnson",7.2),("ABBV","AbbVie",6.8),
             ("MRK","Merck",5.9),("TMO","Thermo Fisher",4.1),
             ("ABT","Abbott Labs",3.8),("DHR","Danaher",3.5),
             ("AMGN","Amgen",3.2),("MDT","Medtronic",2.9)],
    "XLK":  [("NVDA","NVIDIA",20.1),("AAPL","Apple",19.8),
             ("MSFT","Microsoft",18.9),("AVGO","Broadcom",4.8),
             ("ORCL","Oracle",3.9),("CRM","Salesforce",3.5),
             ("ACN","Accenture",3.1),("AMD","AMD",2.8),
             ("ADBE","Adobe",2.5),("TXN","Texas Instruments",2.1)],
    "VGK":  [("ASML","ASML Holding",4.2),("NESN","Nestlé",3.8),
             ("NOVO-B","Novo Nordisk",3.6),("LVMH","LVMH",2.9),
             ("ROG","Roche",2.7),("SAP","SAP SE",2.5),
             ("MC","LVMH PA",2.3),("SIE","Siemens",2.1),
             ("OR","L'Oréal",2.0),("BAS","BASF",1.8)],
    "EEM":  [("TSMC","Taiwan Semiconductor",6.1),("Samsung","Samsung Elec.",4.2),
             ("0700.HK","Tencent",3.8),("BABA","Alibaba",2.9),
             ("RELIANCE","Reliance Ind.",2.5),("PDD","Pinduoduo",2.2),
             ("INFOSYS","Infosys",1.9),("HDFC","HDFC Bank",1.8),
             ("NIO","NIO Inc.",1.5),("JD","JD.com",1.4)],
    "AGG":  [("US 10Y","US Treasury 10Y",28.1),("US 7Y","US Treasury 7Y",12.3),
             ("MBS","Mortgage-Backed Sec.",22.5),("IG Corp","Corp Bond IG",18.7),
             ("TIPS","TIPS",6.2),("Agency","Agency MBS",5.8),
             ("Muni","Municipal Bond",2.1),("ABS","Asset-Backed Sec.",1.8),
             ("CMBS","Comm. Mortgage",1.5),("Other","Other Bonds",1.0)],
    "SHY":  [("1-3Y UST","1-3Y US Treasury",98.5),("Cash","Cash & Equiv.",1.5),
             ("T-Bill","T-Bills 3M",0.0),("N/A","—",0.0),
             ("N/A","—",0.0),("N/A","—",0.0),
             ("N/A","—",0.0),("N/A","—",0.0),
             ("N/A","—",0.0),("N/A","—",0.0)],
    "CBON": [("CGB 10Y","China Gov Bond 10Y",35.1),("CGB 5Y","China Gov Bond 5Y",22.3),
             ("CGB 3Y","China Gov Bond 3Y",18.5),("Policy Bank","Policy Bank Bond",12.7),
             ("Corp CNY","Corp Bond CNY",7.2),("Cash","RMB Cash",4.2),
             ("N/A","—",0.0),("N/A","—",0.0),("N/A","—",0.0),("N/A","—",0.0)],
}


def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── Fund mapping ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_fund_mapping() -> dict:
    try:
        with open(_MAPPING_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(_err("load_fund_mapping", e))
        return {}


def lookup_etf(fund_name: str) -> dict | None:
    mapping = load_fund_mapping()
    if fund_name in mapping:
        return mapping[fund_name]
    lower = fund_name.lower()
    for k, v in mapping.items():
        if k.lower() in lower or lower in k.lower():
            return v
    return None


# ── Gemini OCR (delegates to ocr_module) ──────────────────────────────────────
def _enrich_ocr_items(raw_items: list[dict]) -> list[dict]:
    """Add ETF mapping to raw OCR items."""
    result = []
    for item in raw_items:
        name     = str(item.get("fund_name", "")).strip()
        pct      = float(item.get("percentage", 0))
        etf_info = lookup_etf(name)
        result.append({
            "fund_name":        name,
            "pct":              pct,
            "etf":              etf_info["etf"]      if etf_info else "N/A",
            "category":         etf_info["category"] if etf_info else "未知",
            "desc":             etf_info["desc"]     if etf_info else "—",
            "market_value_hkd": item.get("market_value_hkd"),
            "pnl_hkd":          item.get("pnl_hkd"),
            "units":            item.get("units"),
            "unit_price_hkd":   item.get("unit_price_hkd"),
        })
    return result


# ── Holdings lookup ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_etf_holdings(etf_ticker: str) -> pd.DataFrame:
    """Return top-10 holdings DataFrame for an ETF via yfinance."""
    try:
        ticker = yf.Ticker(etf_ticker)
        df     = ticker.funds_data.top_holdings
        if df is not None and not df.empty:
            df = df.reset_index()
            # ── FIX: remove duplicate columns before rename ────────────────
            df = df.loc[:, ~df.columns.duplicated()]
            cols   = df.columns.tolist()
            rename = {}
            used   = set()
            for c in cols:
                cl = c.lower()
                if ("symbol" in cl or "ticker" in cl) and "代碼" not in used:
                    rename[c] = "代碼"; used.add("代碼")
                elif ("name" in cl or "holding" in cl) and "名稱" not in used:
                    rename[c] = "名稱"; used.add("名稱")
                elif ("percent" in cl or "weight" in cl or "%" in cl) and "佔比%" not in used:
                    rename[c] = "佔比%"; used.add("佔比%")
            df = df.rename(columns=rename)
            # ── FIX: second dedup pass after rename ───────────────────────
            df = df.loc[:, ~df.columns.duplicated()]
            keep = [c for c in ["代碼", "名稱", "佔比%"] if c in df.columns]
            return df[keep].head(10)
    except Exception as e:
        print(_err(f"get_etf_holdings[{etf_ticker}]", e))

    fallback = _FALLBACK_HOLDINGS.get(etf_ticker)
    if fallback:
        rows = [(sym, name, f"{pct:.1f}%") for sym, name, pct in fallback
                if sym != "N/A"]
        return pd.DataFrame(rows, columns=["代碼", "名稱", "佔比%"])
    return pd.DataFrame(columns=["代碼", "名稱", "佔比%"])


# ── Market alert engine ────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _get_market_signal() -> dict:
    """Simple market condition check: SMA50 vs SMA200 of SPY."""
    try:
        hist   = standardize_timezone(yf.Ticker("SPY").history(period="1y", auto_adjust=True))
        close  = hist["Close"]
        sma50  = float(close.tail(50).mean())
        sma200 = float(close.tail(200).mean())
        price  = float(close.iloc[-1])
        chg5d  = (price / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0

        if sma50 > sma200 and chg5d > 0:
            return {"signal": "bullish", "label": "市場偏多", "chg5d": chg5d,
                    "sma50": sma50, "sma200": sma200}
        elif chg5d < -3:
            return {"signal": "pullback", "label": "技術性回撤", "chg5d": chg5d,
                    "sma50": sma50, "sma200": sma200}
        elif sma50 < sma200:
            return {"signal": "bearish", "label": "趨勢偏弱", "chg5d": chg5d,
                    "sma50": sma50, "sma200": sma200}
        else:
            return {"signal": "neutral", "label": "市場中性", "chg5d": chg5d,
                    "sma50": sma50, "sma200": sma200}
    except Exception as e:
        print(_err("_get_market_signal", e))
        return {"signal": "neutral", "label": "無法獲取市場數據", "chg5d": 0,
                "sma50": None, "sma200": None}


# ── DB helpers ─────────────────────────────────────────────────────────────────
def _load_from_db_if_empty() -> None:
    """On first page load, seed session state from SQLite."""
    if "mpf_portfolio_loaded" not in st.session_state:
        rows = mpf_db.load_portfolio()
        if rows:
            st.session_state["mpf_portfolio"] = rows
        st.session_state["mpf_portfolio_loaded"] = True


def _persist(portfolio: list[dict]) -> None:
    """Save current portfolio to both session_state and SQLite."""
    st.session_state["mpf_portfolio"] = portfolio
    mpf_db.save_portfolio(portfolio)


# ── UI rendering ───────────────────────────────────────────────────────────────
def _card(bg: str, border: str, title: str, value: str, sub: str = "") -> str:
    return (
        f"<div style='background:{bg}; border-left:4px solid {border}; "
        f"border-radius:8px; padding:14px 16px; margin-bottom:6px;'>"
        f"<div style='font-size:13px; color:#666; margin-bottom:3px;'>{title}</div>"
        f"<div style='font-size:18px; font-weight:700; color:{border};'>{value}</div>"
        f"<div style='font-size:13px; color:#888;'>{sub}</div>"
        f"</div>"
    )


def render_market_alert() -> None:
    """Top-of-page dynamic alert banner based on SPY market signal."""
    try:
        sig = _get_market_signal()
        s   = sig["signal"]
        chg = sig["chg5d"]

        if s == "bullish":
            st.success(
                f"📈 **市場偏多訊號**：SPY SMA50 ({sig['sma50']:.1f}) > SMA200 ({sig['sma200']:.1f})，"
                f"近5日上漲 {chg:+.1f}%。MPF **增長型**基金可考慮適度加碼。"
            )
        elif s == "pullback":
            st.warning(
                f"⚠️ **技術性回撤警示**：SPY 近5日下跌 {chg:+.1f}%。"
                f"建議暫緩加碼高風險基金，可考慮增持**保守/穩定**基金作避險。"
            )
        elif s == "bearish":
            st.error(
                f"🛡️ **市場趨勢偏弱**：SPY SMA50 ({sig['sma50']:.1f}) < SMA200 ({sig['sma200']:.1f})。"
                f"建議增持**固定收益**基金，降低整體組合波動。"
            )
        else:
            st.info(
                f"📊 **市場中性**：SPY 近5日變動 {chg:+.1f}%。按既定計劃執行月供即可。"
            )
    except Exception as e:
        print(_err("render_market_alert", e))


def render_upload_section() -> None:
    """OCR upload section: grayscale preprocessing + auto correction form fallback."""
    try:
        st.markdown("### 📸 上傳 eMPF 截圖（OCR 自動識別）")
        st.caption("支援多張圖片同時上傳。自動進行灰階預處理以提升識別率。OCR 失敗時自動彈出手動修正表單。")

        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "")

        if not api_key:
            st.warning(
                "⚠️ OCR 功能需要 Gemini API Key。"
                "請於 Replit Secrets 中新增 **GEMINI_API_KEY**。"
            )

        uploaded = st.file_uploader(
            "選擇截圖檔案",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="mpf_uploader",
        )

        if uploaded and st.button("🔍 開始 OCR 識別", type="primary", key="mpf_ocr_btn"):
            if not api_key:
                st.error("請先在 Replit Secrets 中設定 GEMINI_API_KEY 才能使用 OCR 識別功能。")
                return

            all_raw_items: list[dict] = []
            has_failure = False

            for up in uploaded:
                with st.spinner(f"正在預處理並識別 {up.name}…"):
                    raw_items, err = ocr_with_gemini(up.read(), api_key)
                if err:
                    st.warning(f"⚠️ {up.name} 識別失敗：{err}")
                    has_failure = True
                else:
                    all_raw_items.extend(raw_items)
                    st.success(f"✅ {up.name}：識別到 {len(raw_items)} 個基金")

            if has_failure or not all_raw_items:
                st.session_state["mpf_ocr_pending_correction"] = all_raw_items
            elif all_raw_items:
                _commit_ocr_items(all_raw_items)

        # ── Auto-popup correction form when OCR had issues ──────────────────
        if "mpf_ocr_pending_correction" in st.session_state:
            pending   = st.session_state["mpf_ocr_pending_correction"]
            corrected = render_manual_correction_form(pending, form_key="mpf_ocr_fix")
            if corrected is not None:
                _commit_ocr_items(corrected)
                del st.session_state["mpf_ocr_pending_correction"]
                st.rerun()

        st.markdown("---")
        render_manual_input()

    except Exception as e:
        st.error(f"⚠️ 上傳識別異常：{e}")


def _commit_ocr_items(raw_items: list[dict]) -> None:
    """Enrich OCR items, merge into portfolio, save to DB."""
    enriched = _enrich_ocr_items(raw_items)
    fund_map: dict[str, dict] = {
        p["fund_name"]: dict(p)
        for p in st.session_state.get("mpf_portfolio", [])
    }
    for item in enriched:
        k = item["fund_name"]
        if k in fund_map:
            fund_map[k]["pct"] = round(fund_map[k]["pct"] + item["pct"], 1)
            for field in ("market_value_hkd", "pnl_hkd", "units", "unit_price_hkd"):
                if item.get(field) is not None:
                    fund_map[k][field] = item[field]
        else:
            fund_map[k] = dict(item)
    _persist(list(fund_map.values()))


def render_manual_input() -> None:
    """Manual fund entry with full financial detail fields, bound to SQLite."""
    try:
        mapping  = load_fund_mapping()
        all_keys = ["（手動輸入基金名稱）"] + list(mapping.keys())

        with st.expander("✏️ 手動新增 / 編輯基金持倉", expanded=False):
            st.caption(
                "填入基金名稱及財務數據後點擊「➕ 新增」。"
                "數據會自動儲存，頁面刷新後仍保留。"
            )

            # ── Row 1: fund name + pct ─────────────────────────────────────
            col_name, col_pct = st.columns([4, 2])
            with col_name:
                choice = st.selectbox("選擇基金", all_keys, key="mpf_manual_fund")
                if choice == "（手動輸入基金名稱）":
                    choice = st.text_input("輸入基金名稱", key="mpf_manual_fund_txt")
            with col_pct:
                pct = st.number_input(
                    "投資組合佔比 %",
                    min_value=0.0, max_value=100.0,
                    value=10.0, step=0.5, key="mpf_manual_pct",
                )

            # ── Row 2: financial details ───────────────────────────────────
            col_mv, col_pnl, col_units, col_price = st.columns(4)
            with col_mv:
                market_value = st.number_input(
                    "市場價值 (港幣 HKD)",
                    min_value=0.0, value=0.0, step=100.0,
                    key="mpf_manual_mv",
                    help="目前此基金的市場總值（港幣）",
                )
            with col_pnl:
                pnl = st.number_input(
                    "投資收益(虧損) HKD",
                    value=0.0, step=100.0, format="%.2f",
                    key="mpf_manual_pnl",
                    help="正數=盈利，負數=虧損（港幣）",
                )
            with col_units:
                units = st.number_input(
                    "單位數量",
                    min_value=0.0, value=0.0, step=1.0, format="%.4f",
                    key="mpf_manual_units",
                    help="持有的基金單位數",
                )
            with col_price:
                unit_price = st.number_input(
                    "單位價格 (港幣 HKD)",
                    min_value=0.0, value=0.0, step=0.01, format="%.4f",
                    key="mpf_manual_uprice",
                    help="每個基金單位的現行淨值（港幣）",
                )

            # ── Auto-derive missing fields ─────────────────────────────────
            # If user filled market_value + units but not unit_price → derive
            derived_price = unit_price
            if unit_price == 0.0 and units > 0 and market_value > 0:
                derived_price = round(market_value / units, 4)

            col_add, col_clr = st.columns([1, 1])
            with col_add:
                if st.button("➕ 新增 / 更新", key="mpf_manual_add", type="primary"):
                    name = choice.strip()
                    if name:
                        ei = lookup_etf(name)
                        entry: dict = {
                            "fund_name":        name,
                            "pct":              round(pct, 1),
                            "etf":              ei["etf"]      if ei else "N/A",
                            "category":         ei["category"] if ei else "未知",
                            "desc":             ei["desc"]     if ei else "—",
                            "market_value_hkd": market_value  if market_value > 0 else None,
                            "pnl_hkd":          pnl           if pnl != 0.0    else None,
                            "units":            units         if units > 0      else None,
                            "unit_price_hkd":   derived_price if derived_price > 0 else None,
                        }
                        portfolio = list(st.session_state.get("mpf_portfolio", []))
                        idx = next(
                            (i for i, x in enumerate(portfolio)
                             if x["fund_name"] == name), None
                        )
                        if idx is not None:
                            portfolio[idx] = entry
                        else:
                            portfolio.append(entry)
                        _persist(portfolio)
                        st.success(f"✅ 已儲存：{name}")
                        st.rerun()

            with col_clr:
                if st.button("🗑️ 清除所有持倉", key="mpf_clear"):
                    mpf_db.clear_portfolio()
                    st.session_state["mpf_portfolio"] = []
                    st.rerun()

    except Exception as e:
        st.error(f"⚠️ 手動輸入異常：{e}")


def render_portfolio_table() -> None:
    """Display current MPF holdings table with full financial details."""
    try:
        portfolio = st.session_state.get("mpf_portfolio", [])
        if not portfolio:
            st.info("尚未載入 MPF 持倉。請上傳截圖或手動新增。")
            return

        total_pct = sum(p.get("pct", 0) for p in portfolio)
        total_mv  = sum(p.get("market_value_hkd") or 0 for p in portfolio)
        total_pnl = sum(p.get("pnl_hkd") or 0 for p in portfolio)

        # ── Summary metrics ────────────────────────────────────────────────
        st.markdown(f"### 📊 目前 MPF 持倉  ·  合計 {total_pct:.1f}%")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "總市場價值 (HKD)",
            f"${total_mv:,.0f}" if total_mv else "—",
        )
        mc2.metric(
            "總投資收益 (HKD)",
            f"${total_pnl:+,.0f}" if total_pnl else "—",
            delta=f"{(total_pnl/(total_mv - total_pnl)*100):+.1f}%" if total_mv and (total_mv - total_pnl) > 0 else None,
        )
        mc3.metric("基金數目", len(portfolio))

        st.markdown("---")

        # ── Holdings rows ──────────────────────────────────────────────────
        for i, p in enumerate(portfolio):
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 2, 1])
                c1.markdown(
                    f"**{p['fund_name']}**  \n"
                    f"<span style='font-size:12px; color:#666;'>"
                    f"ETF: {p.get('etf','N/A')} ｜ {p.get('category','—')}</span>",
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style='font-size:18px; font-weight:700; color:#0066CC;'>"
                    f"{p.get('pct',0):.1f}%</div>"
                    f"<div style='font-size:11px; color:#888;'>佔比</div></div>",
                    unsafe_allow_html=True,
                )
                mv  = p.get("market_value_hkd")
                pnl = p.get("pnl_hkd")
                uni = p.get("units")
                upr = p.get("unit_price_hkd")
                c3.markdown(
                    f"<div style='font-size:12px;'>"
                    f"{'$'+f'{mv:,.0f}' if mv else '—'}<br>"
                    f"<span style='color:{'#2ecc71' if (pnl or 0) >= 0 else '#e74c3c'};'>"
                    f"{'%+,.0f' % pnl + ' PnL' if pnl is not None else '—'}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                c4.markdown(
                    f"<div style='font-size:12px;'>"
                    f"單位: {f'{uni:,.4f}' if uni else '—'}<br>"
                    f"單價: {f'HK${upr:,.4f}' if upr else '—'}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if p.get("etf") and p["etf"] != "N/A":
                    if c5.button("🔍", key=f"mpf_h_{i}_{p['etf']}",
                                 help=f"查看 {p['etf']} 前十大持倉"):
                        st.session_state["mpf_show_holdings"] = p["etf"]

                # delete button per row
                if c5.button("🗑", key=f"mpf_del_{i}", help="刪除此筆"):
                    new_pf = [x for x in portfolio if x["fund_name"] != p["fund_name"]]
                    _persist(new_pf)
                    st.rerun()

            st.markdown("<hr style='margin:4px 0; border-color:#eee;'>",
                        unsafe_allow_html=True)

        shown_etf = st.session_state.get("mpf_show_holdings")
        if shown_etf:
            render_fund_holdings(shown_etf)

    except Exception as e:
        st.error(f"⚠️ 持倉表格渲染異常：{e}")


def render_fund_holdings(etf_ticker: str) -> None:
    """Show top-10 holdings for an ETF with a Plotly bar chart."""
    try:
        st.markdown(f"#### 🏦 {etf_ticker} 前十大持倉")
        df = get_etf_holdings(etf_ticker)
        if df.empty:
            st.warning(f"無法獲取 {etf_ticker} 持倉資料。")
            return

        pct_col  = next((c for c in df.columns if "%" in c or "佔" in c), None)
        name_col = next((c for c in df.columns if "名稱" in c or "name" in c.lower()), None)

        if pct_col and name_col:
            raw_pct  = df[pct_col].astype(str).str.replace("%", "").str.strip()
            pct_vals = pd.to_numeric(raw_pct, errors="coerce").fillna(0)
            labels   = df[name_col].astype(str)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pct_vals, y=labels, orientation="h",
                marker=dict(color=pct_vals, colorscale="Blues",
                            line=dict(color="#0066CC22", width=0.5)),
                text=[f"{v:.1f}%" for v in pct_vals],
                textposition="outside",
                textfont=dict(size=12, color="#333"),
                hovertemplate="<b>%{y}</b><br>佔比: %{x:.2f}%<extra></extra>",
            ))
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=60, t=20, b=20),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8F9FA",
                xaxis=dict(showgrid=True, gridcolor="#E0E0E0",
                           tickfont=dict(size=12, color="#555"),
                           title=dict(text="佔比 (%)", font=dict(size=12, color="#555"))),
                yaxis=dict(tickfont=dict(size=12, color="#333"), autorange="reversed"),
                font=dict(size=12, color="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df.rename(columns={pct_col: "佔比%"} if pct_col else {}),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("關閉", key=f"mpf_close_{etf_ticker}"):
            st.session_state["mpf_show_holdings"] = None
            st.rerun()

    except Exception as e:
        st.error(f"⚠️ 持倉圖表渲染異常：{e}")


def render_rebalance_section(monthly_budget: float = 2400.0) -> None:
    """Rebalancing calculator: allocate monthly contribution."""
    try:
        portfolio = st.session_state.get("mpf_portfolio", [])
        if not portfolio:
            st.info("請先在「截圖識別」或「持倉透視」頁籤載入 MPF 持倉。")
            return

        st.markdown("### ⚖️ 再平衡計算器")

        col_bud, col_mode = st.columns([2, 2])
        with col_bud:
            monthly_budget = st.number_input(
                "每月供款金額 (HKD)",
                min_value=100.0, max_value=50000.0,
                value=2400.0, step=100.0,
                key="mpf_monthly_budget",
            )
        with col_mode:
            mode = st.selectbox(
                "目標分配策略",
                ["等權重（均等分配）", "自定義目標比例"],
                key="mpf_rebal_mode",
            )

        names       = [p["fund_name"] for p in portfolio]
        current_pct = {p["fund_name"]: p.get("pct", 0) for p in portfolio}

        if mode == "自定義目標比例":
            st.caption("請輸入每個基金的目標佔比（合計應為 100%）：")
            target_pct = {}
            eq_default = round(100.0 / len(portfolio), 1)
            cols = st.columns(min(4, len(portfolio)))
            for i, name in enumerate(names):
                with cols[i % 4]:
                    target_pct[name] = st.number_input(
                        name[:12] + "…" if len(name) > 12 else name,
                        min_value=0.0, max_value=100.0,
                        value=eq_default, step=0.5,
                        key=f"mpf_tgt_{i}",
                    )
        else:
            eq = round(100.0 / len(portfolio), 1)
            target_pct = {p["fund_name"]: eq for p in portfolio}

        result = calc_rebalance(current_pct, target_pct, monthly_budget)
        if not result:
            return

        total_alloc = sum(v["allocation"] for v in result.values())
        st.markdown(
            f"<div style='font-size:13px; color:#555; margin-bottom:8px;'>"
            f"本月建議總分配：<b style='color:#0066CC;'>HKD {total_alloc:,.0f}</b>"
            f"　（月供 {monthly_budget:,.0f}）</div>",
            unsafe_allow_html=True,
        )

        rows = []
        for name in names:
            r     = result.get(name, {})
            alloc = r.get("allocation", 0)
            dev   = r.get("deviation", 0)
            act   = r.get("action", "—")
            cur   = current_pct.get(name, 0)
            tgt   = target_pct.get(name, 0)
            rows.append({
                "基金名稱":       name,
                "現持佔比":       f"{cur:.1f}%",
                "目標佔比":       f"{tgt:.1f}%",
                "偏離度":         f"{dev:+.1f}%",
                "建議操作":       act,
                "本月供款 (HKD)": f"$ {alloc:,.0f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        _render_rebalance_chart(result, target_pct, current_pct)

    except Exception as e:
        st.error(f"⚠️ 再平衡計算異常：{e}")


def _render_rebalance_chart(result: dict, target_pct: dict, current_pct: dict) -> None:
    try:
        labels   = list(result.keys())
        allocs   = [result[k]["allocation"] for k in labels]
        cur_vals = [current_pct.get(k, 0) for k in labels]
        tgt_vals = [target_pct.get(k, 0) for k in labels]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="現持佔比 %", x=labels, y=cur_vals,
            marker=dict(color="#93C6E7", line=dict(color="#0066CC44", width=1)),
            text=[f"{v:.1f}%" for v in cur_vals],
            textposition="auto", textfont=dict(size=12, color="#333"),
        ))
        fig.add_trace(go.Bar(
            name="目標佔比 %", x=labels, y=tgt_vals,
            marker=dict(color="#FBBF24", line=dict(color="#D97706aa", width=1)),
            text=[f"{v:.1f}%" for v in tgt_vals],
            textposition="auto", textfont=dict(size=12, color="#333"),
        ))
        fig.add_trace(go.Bar(
            name="本月供款 HKD", x=labels, y=allocs, yaxis="y2",
            marker=dict(color="#86EFAC", line=dict(color="#16A34Aaa", width=1)),
            text=[f"${v:,.0f}" for v in allocs],
            textposition="auto", textfont=dict(size=12, color="#166534"),
        ))
        fig.update_layout(
            barmode="group", height=320,
            margin=dict(l=0, r=0, t=30, b=80),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FA",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=12, color="#333")),
            xaxis=dict(tickfont=dict(size=12, color="#555"), tickangle=-30),
            yaxis=dict(title=dict(text="佔比 %", font=dict(size=12, color="#555")),
                       tickfont=dict(size=12, color="#555"), gridcolor="#E0E0E0"),
            yaxis2=dict(title=dict(text="供款 HKD", font=dict(size=12, color="#0a6640")),
                        tickfont=dict(size=12, color="#0a6640"),
                        overlaying="y", side="right", gridcolor="#E8F5E9"),
            font=dict(size=12, color="#333"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        print(_err("_render_rebalance_chart", e))


# ── Main page entry point ──────────────────────────────────────────────────────
def _render_strategy_tab() -> None:
    """RS + SMA strategy signals for each fund with dual pie charts."""
    portfolio = st.session_state.get("mpf_portfolio", [])
    if not portfolio:
        st.info("📂 請先在「截圖識別」或「持倉透視」頁籤輸入您的 MPF 持倉，然後回到此頁籤查看策略建議。")
        return

    if st.button("🔄 分析最新市場訊號", type="primary", key="mpf_run_strategy"):
        with st.spinner("正在連接市場數據，分析各基金 ETF 代理的 RS + SMA 趨勢…"):
            result = mpf_strategy.get_strategy_signals(portfolio)
            st.session_state["mpf_strategy_result"] = result

    result = st.session_state.get("mpf_strategy_result")
    if not result:
        st.info("👆 點擊按鈕以分析當前持倉的量化訊號")
        return

    if result.get("error"):
        st.error(f"⚠️ 分析失敗：{result['error']}")
        return

    signals  = result["signals"]
    mkt_cond = result["market_condition"]
    def_adv  = result["defensive_advice"]

    # ── Market condition banner ────────────────────────────────────────────
    mkt_colors = {"bullish": "#00CC44", "bearish": "#FF4B4B", "neutral": "#888"}
    mkt_labels = {"bullish": "📈 宏觀多頭訊號", "bearish": "📉 宏觀空頭警示", "neutral": "🔄 市場中性"}
    mkt_color  = mkt_colors.get(mkt_cond, "#888")
    mkt_label  = mkt_labels.get(mkt_cond, "🔄 市場中性")
    st.markdown(
        f"<div style='background:#F0F8FF; border-left:5px solid {mkt_color}; "
        f"border-radius:6px; padding:12px 18px; margin-bottom:12px;'>"
        f"<b style='font-size:16px; color:{mkt_color};'>{mkt_label}</b>"
        f"<span style='color:#555; font-size:13px; margin-left:16px;'>"
        f"當前防禦配置：{def_adv.get('current_defensive_pct', 0):.1f}%  ·  "
        f"建議目標：{def_adv.get('suggested_defensive_pct', 0):.1f}%  ·  "
        f"需轉移：{def_adv.get('shift_amount_pct', 0):.1f}%</span></div>",
        unsafe_allow_html=True,
    )

    # ── Signal cards per fund ──────────────────────────────────────────────
    st.markdown("### 📋 各基金訊號明細")
    for sig in signals:
        rec   = sig["recommendation"]
        sma   = sig.get("sma", {})
        rs    = sig.get("rs", {})
        color = rec.get("color", "#888")
        with st.expander(
            f"{rec['label']}  ·  {sig['fund_name']}  （ETF代理：{sig['etf']}）",
            expanded=(rec["action"] in ("add", "switch_defensive")),
        ):
            sc1, sc2, sc3 = st.columns(3)
            sc1.markdown(
                f"**建議操作**  \n"
                f"<span style='color:{color}; font-size:18px; font-weight:700;'>"
                f"{rec['label']}</span>",
                unsafe_allow_html=True,
            )
            sc1.caption(f"信心度：{rec.get('confidence', 'low')}")
            sc2.markdown(
                f"**SMA 趨勢**  \n"
                f"趨勢：`{sma.get('trend', 'N/A')}`  \n"
                f"SMA20：`{sma.get('sma20', 'N/A')}`  \n"
                f"SMA50：`{sma.get('sma50', 'N/A')}`  \n"
                f"20日動量：`{sma.get('momentum', 0):.1f}%`"
            )
            sc3.markdown(
                f"**相對強度 (RS)**  \n"
                f"RS vs SPY：`{rs.get('rs_raw', 'N/A')}`  \n"
                f"RS SMA20：`{rs.get('rs_sma20', 'N/A')}`  \n"
                f"RS趨勢：`{rs.get('rs_trend', 'N/A')}`  \n"
                f"RS訊號：`{rs.get('rs_signal', 'N/A')}`"
            )
            st.info(f"💡 {rec['reason']}")

    # ── Dual pie charts: Current vs Suggested ─────────────────────────────
    st.markdown("---")
    st.markdown("### 🥧 現有 vs 建議配置比較")

    fund_names = [s["fund_name"] for s in signals]
    cur_pcts   = [s["pct"] for s in signals]
    total_pct  = sum(cur_pcts) or 1.0
    cur_norm   = [p / total_pct * 100 for p in cur_pcts]

    # Build suggested: reduce funds with "reduce"/"switch_defensive" by 20%, add to defensive
    sug_pcts = list(cur_norm)
    reduce_indices  = [i for i, s in enumerate(signals)
                       if s["recommendation"]["action"] in ("reduce", "switch_defensive")]
    add_indices     = [i for i, s in enumerate(signals)
                       if s["recommendation"]["action"] == "add"]
    def_indices     = [i for i, s in enumerate(signals)
                       if s.get("category", "") in ("固定收益", "保本")]

    released = 0.0
    for i in reduce_indices:
        cut      = sug_pcts[i] * 0.20
        sug_pcts[i] -= cut
        released += cut

    if released > 0:
        if def_indices:
            per_def = released / len(def_indices)
            for i in def_indices:
                sug_pcts[i] += per_def
        elif add_indices:
            per_add = released / len(add_indices)
            for i in add_indices:
                sug_pcts[i] += per_add

    pie_colors = [
        "#00CC44", "#4DB6E6", "#FFD700", "#FF8C00", "#CC66FF",
        "#FF6B6B", "#00CCAA", "#FF99CC", "#99CCFF", "#FFCC66",
    ]

    c1, c2 = st.columns(2)
    with c1:
        fig_cur = go.Figure(go.Pie(
            labels=fund_names, values=cur_norm,
            hole=0.4, marker_colors=pie_colors[:len(fund_names)],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ))
        fig_cur.update_layout(
            title="現有配置", height=360,
            paper_bgcolor="#FFFFFF",
            margin=dict(t=40, b=20, l=10, r=10),
            legend=dict(font=dict(size=10)),
            showlegend=False,
        )
        st.plotly_chart(fig_cur, use_container_width=True)
    with c2:
        fig_sug = go.Figure(go.Pie(
            labels=fund_names, values=sug_pcts,
            hole=0.4, marker_colors=pie_colors[:len(fund_names)],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ))
        fig_sug.update_layout(
            title="量化建議配置（基於 RS+SMA）", height=360,
            paper_bgcolor="#FFFFFF",
            margin=dict(t=40, b=20, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_sug, use_container_width=True)

    # ── ETF historical comparison chart ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 ETF 代理歷史對比（標準化，基準=100）")
    etf_list = list(dict.fromkeys(
        [s["etf"] for s in signals if s["etf"] != "N/A"]
    ))
    if etf_list:
        period_yr = st.selectbox("選擇時間段", [1, 2, 3], index=0,
                                 key="mpf_etf_period", format_func=lambda x: f"{x}年")
        with st.spinner("加載 ETF 歷史數據…"):
            hist_df = mpf_strategy.get_etf_vs_spy_history(etf_list, years=period_yr)
        if not hist_df.empty:
            fig_hist = go.Figure()
            _hist_colors = ["#00CC44","#4DB6E6","#FFD700","#FF8C00","#CC66FF",
                            "#FF6B6B","#00CCAA","#FF99CC"]
            for hi, col in enumerate(hist_df.columns):
                lw = 2.5 if col == "SPY" else 1.5
                dash = "dot" if col == "SPY" else "solid"
                fig_hist.add_trace(go.Scatter(
                    x=hist_df.index, y=hist_df[col].values,
                    mode="lines", name=col,
                    line=dict(color=_hist_colors[hi % len(_hist_colors)], width=lw, dash=dash),
                    hovertemplate=f"{col} %{{x|%Y-%m-%d}}: %{{y:.1f}}<extra></extra>",
                ))
            fig_hist.update_layout(
                height=380, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(size=11, color="#ddd")),
                xaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                           tickfont=dict(size=11, color="#aaa")),
                yaxis=dict(showgrid=True, gridcolor="#2a2a3a",
                           tickfont=dict(size=11, color="#aaa"),
                           title=dict(text="標準化淨值（初始=100）",
                                      font=dict(size=11, color="#aaa"))),
                margin=dict(l=0, r=0, t=20, b=20),
                font=dict(size=11, color="#ddd"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("⚠️ 無法獲取 ETF 歷史數據")
    else:
        st.info("持倉中無有效 ETF 代理，無法顯示歷史對比")


def _render_ai_report_tab() -> None:
    """Gemini AI investment analysis report for MPF portfolio."""
    try:
        from ocr_module import generate_quant_report
    except ImportError:
        st.error("⚠️ AI 報告模組無法載入")
        return

    portfolio = st.session_state.get("mpf_portfolio", [])
    if not portfolio:
        st.info("📂 請先輸入您的 MPF 持倉，再生成 AI 分析報告。")
        return

    st.caption("根據您的 MPF 持倉配置與當前宏觀環境，由 Gemini AI 生成個性化投資建議")

    strategy_result = st.session_state.get("mpf_strategy_result")

    if st.button("📝 生成 AI 分析報告", type="primary", key="mpf_ai_report_btn"):
        with st.spinner("Gemini AI 正在分析您的 MPF 持倉…（約15-30秒）"):
            try:
                total_mv  = sum(p.get("market_value_hkd", 0) or 0 for p in portfolio)
                total_pnl = sum(p.get("pnl_hkd", 0) or 0 for p in portfolio)
                pnl_pct   = (total_pnl / (total_mv - total_pnl) * 100) if (total_mv - total_pnl) > 0 else 0

                # Build prompt about holdings
                holdings_summary = "\n".join(
                    f"- {p['fund_name']} ({p.get('etf','N/A')}): "
                    f"{p.get('pct',0):.1f}% 配置, "
                    f"市值 HKD {p.get('market_value_hkd',0):,.0f}, "
                    f"損益 HKD {p.get('pnl_hkd',0):+,.0f}"
                    for p in portfolio
                )

                strategy_summary = ""
                if strategy_result and not strategy_result.get("error"):
                    sigs = strategy_result["signals"]
                    strategy_summary = "\n量化訊號摘要：\n" + "\n".join(
                        f"- {s['fund_name']}: {s['recommendation']['label']} "
                        f"（RS {s['rs'].get('rs_signal','N/A')}, SMA趨勢 {s['sma'].get('trend','N/A')}）"
                        for s in sigs
                    )

                prompt = f"""
你是一位專業的香港強積金（MPF）投資顧問，請用繁體中文分析以下持倉並給出具體建議：

## 持倉摘要
總市值：HKD {total_mv:,.0f}
總損益：HKD {total_pnl:+,.0f}（{pnl_pct:+.1f}%）

## 各基金配置
{holdings_summary}
{strategy_summary}

請提供：
1. **整體配置評估**（多元化程度、風險集中度）
2. **各基金具體建議**（加碼/持平/減持，附理由）
3. **宏觀市場風險因素**（2026年值得關注的事件）
4. **行動計劃**（本月最優先的1-2個調整動作）
5. **長期優化建議**（3-5年視角下的理想配置方向）

請保持簡潔、實用，避免空泛的投資警告。
"""
                report = generate_quant_report(prompt)
                st.session_state["mpf_ai_report"] = report
            except Exception as _e:
                st.error(f"⚠️ AI 報告生成失敗：{_e}")

    report = st.session_state.get("mpf_ai_report")
    if report:
        st.markdown("---")
        st.markdown(report)
        st.download_button(
            "⬇️ 下載報告 (.txt)",
            data=report,
            file_name=f"MPF_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            key="mpf_ai_dl",
        )


def render_mpf_page() -> None:
    """Full MPF 智投診斷室 page."""
    try:
        # ── Init DB and auto-load persisted holdings ───────────────────────
        mpf_db.init_db()
        _load_from_db_if_empty()

        st.markdown("## 🛡️ MPF 智投診斷室")
        st.caption(
            "上傳 eMPF 截圖 → 自動識別持倉 → 透視底層成分股 → 智能再平衡建議  ·  "
            "**持倉數據自動持久化，頁面刷新後仍保留**"
        )
        st.markdown("---")

        render_market_alert()
        st.markdown("---")

        # ── Section 1: OCR Upload ──────────────────────────────────────────
        with st.container():
            st.markdown("## 📸 截圖識別")
            st.caption("上傳 eMPF / 月結單截圖，自動識別持倉資料")
            render_upload_section()

        st.markdown("---")

        # ── Section 2: Portfolio Table ─────────────────────────────────────
        with st.container():
            st.markdown("## 📊 持倉透視")
            st.caption("查看底層 ETF 成分股，手動輸入或修改各基金持倉")
            render_portfolio_table()

        st.markdown("---")

        # ── Section 3: Rebalance ───────────────────────────────────────────
        with st.container():
            st.markdown("## ⚖️ 再平衡建議")
            st.caption("根據目標配置計算每月供款方向")
            render_rebalance_section(
                monthly_budget=st.session_state.get("mpf_monthly_budget", 2400.0)
            )

        st.markdown("---")

        # ── Section 4: Strategy Signals ────────────────────────────────────
        with st.container():
            st.markdown("## 🎯 智投建議")
            st.caption("RS（相對強度）+ SMA 趨勢分析 · 每隻基金獨立量化訊號 · 雙圓餅圖配置對比")
            _render_strategy_tab()

        st.markdown("---")

        # ── Section 5: AI Report ───────────────────────────────────────────
        with st.container():
            st.markdown("## 🤖 AI 分析報告")
            st.caption("由 Gemini AI 根據您的持倉與量化訊號生成個性化投資建議")
            _render_ai_report_tab()

    except Exception as e:
        st.error(f"❌ MPF 智投頁面異常：{e}")

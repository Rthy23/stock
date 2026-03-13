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

_MODULE     = "mpf_assistant"
_MAPPING_PATH = "fund_mapping.json"

_FALLBACK_HOLDINGS: dict = {
    "SPY":  [("AAPL","Apple Inc.",7.1),("MSFT","Microsoft Corp.",6.8),
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
}


def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── Fund mapping ──────────────────────────────────────────────────────────────
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


# ── Gemini OCR (delegates to ocr_module) ─────────────────────────────────────
def _enrich_ocr_items(raw_items: list[dict]) -> list[dict]:
    """Add ETF mapping to raw OCR items."""
    result = []
    for item in raw_items:
        name     = str(item.get("fund_name", "")).strip()
        pct      = float(item.get("percentage", 0))
        etf_info = lookup_etf(name)
        result.append({
            "fund_name": name,
            "pct":       pct,
            "etf":       etf_info["etf"]       if etf_info else "N/A",
            "category":  etf_info["category"]  if etf_info else "未知",
            "desc":      etf_info["desc"]       if etf_info else "—",
        })
    return result


# ── Holdings lookup ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_etf_holdings(etf_ticker: str) -> pd.DataFrame:
    """Return top-10 holdings DataFrame for an ETF via yfinance."""
    try:
        ticker = yf.Ticker(etf_ticker)
        df     = ticker.funds_data.top_holdings
        if df is not None and not df.empty:
            df = df.reset_index()
            cols = df.columns.tolist()
            rename = {}
            for c in cols:
                cl = c.lower()
                if "symbol" in cl or "ticker" in cl:
                    rename[c] = "代碼"
                elif "name" in cl or "holding" in cl:
                    rename[c] = "名稱"
                elif "percent" in cl or "weight" in cl or "%" in cl:
                    rename[c] = "佔比%"
            df = df.rename(columns=rename)
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


# ── Market alert engine ───────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _get_market_signal() -> dict:
    """Simple market condition check: SMA50 vs SMA200 of SPY."""
    try:
        hist   = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
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


# ── UI rendering ──────────────────────────────────────────────────────────────

def _card(bg: str, border: str, title: str, value: str, sub: str = "") -> str:
    return (
        f"<div style='background:{bg}; border-left:4px solid {border}; "
        f"border-radius:8px; padding:14px 16px; margin-bottom:6px;'>"
        f"<div style='font-size:13px; color:#aaa; margin-bottom:3px;'>{title}</div>"
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
                    all_raw_items.extend([])
                else:
                    all_raw_items.extend(raw_items)
                    st.success(f"✅ {up.name}：識別到 {len(raw_items)} 個基金")

            if has_failure or not all_raw_items:
                st.session_state["mpf_ocr_pending_correction"] = all_raw_items
            elif all_raw_items:
                _commit_ocr_items(all_raw_items)

        # ── Auto-popup correction form when OCR had issues ──────────────────
        if "mpf_ocr_pending_correction" in st.session_state:
            pending = st.session_state["mpf_ocr_pending_correction"]
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
    """Enrich OCR items and save to session state portfolio."""
    enriched = _enrich_ocr_items(raw_items)
    fund_map: dict[str, dict] = {}
    for item in enriched:
        k = item["fund_name"]
        if k in fund_map:
            fund_map[k]["pct"] = round(fund_map[k]["pct"] + item["pct"], 1)
        else:
            fund_map[k] = dict(item)
    st.session_state["mpf_portfolio"] = list(fund_map.values())


def render_manual_input() -> None:
    """Manual fund entry as fallback for OCR."""
    try:
        mapping  = load_fund_mapping()
        all_keys = ["（手動輸入基金名稱）"] + list(mapping.keys())

        with st.expander("✏️ 手動新增 / 編輯基金持倉", expanded=False):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                choice = st.selectbox("選擇基金", all_keys, key="mpf_manual_fund")
                if choice == "（手動輸入基金名稱）":
                    choice = st.text_input("輸入基金名稱", key="mpf_manual_fund_txt")
            with col2:
                pct = st.number_input(
                    "佔比 %", min_value=0.0, max_value=100.0,
                    value=10.0, step=0.5, key="mpf_manual_pct",
                )
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕ 新增", key="mpf_manual_add"):
                    name = choice.strip()
                    if name:
                        ei = lookup_etf(name)
                        entry = {
                            "fund_name": name,
                            "pct":       round(pct, 1),
                            "etf":       ei["etf"]      if ei else "N/A",
                            "category":  ei["category"] if ei else "未知",
                            "desc":      ei["desc"]      if ei else "—",
                        }
                        portfolio = st.session_state.get("mpf_portfolio", [])
                        existing  = next(
                            (i for i, x in enumerate(portfolio)
                             if x["fund_name"] == name), None
                        )
                        if existing is not None:
                            portfolio[existing] = entry
                        else:
                            portfolio.append(entry)
                        st.session_state["mpf_portfolio"] = portfolio
                        st.rerun()

            if st.button("🗑️ 清除所有持倉", key="mpf_clear"):
                st.session_state["mpf_portfolio"] = []
                st.rerun()
    except Exception as e:
        st.error(f"⚠️ 手動輸入異常：{e}")


def render_portfolio_table() -> None:
    """Display current MPF holdings table with holdings lookup button."""
    try:
        portfolio = st.session_state.get("mpf_portfolio", [])
        if not portfolio:
            st.info("尚未載入 MPF 持倉。請上傳截圖或手動新增。")
            return

        total_pct = sum(p["pct"] for p in portfolio)
        st.markdown(f"### 📊 目前 MPF 持倉  ·  合計 {total_pct:.1f}%")

        header = st.columns([3, 1, 1, 2, 1])
        for col, txt in zip(header, ["基金名稱", "佔比%", "ETF代理", "類別", "成分股"]):
            col.markdown(f"<span style='font-size:13px; color:#aaa;'>{txt}</span>",
                         unsafe_allow_html=True)
        st.markdown("<hr style='margin:4px 0; border-color:#333;'>",
                    unsafe_allow_html=True)

        for i, p in enumerate(portfolio):
            cols = st.columns([3, 1, 1, 2, 1])
            cols[0].markdown(
                f"<span style='font-size:13px;'>{p['fund_name']}</span>",
                unsafe_allow_html=True,
            )
            cols[1].markdown(
                f"<span style='font-size:13px; color:#00FF7F; font-weight:600;'>"
                f"{p['pct']:.1f}%</span>",
                unsafe_allow_html=True,
            )
            cols[2].markdown(
                f"<span style='font-size:13px; color:#00D4FF;'>{p['etf']}</span>",
                unsafe_allow_html=True,
            )
            cols[3].markdown(
                f"<span style='font-size:13px; color:#aaa;'>{p['category']}</span>",
                unsafe_allow_html=True,
            )
            if p["etf"] != "N/A":
                if cols[4].button("🔍", key=f"mpf_holdings_{i}_{p['etf']}",
                                  help=f"查看 {p['etf']} 前十大持倉"):
                    st.session_state["mpf_show_holdings"] = p["etf"]

        st.markdown("---")

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

        pct_col = next((c for c in df.columns if "%" in c or "佔" in c), None)
        name_col = next((c for c in df.columns if "名稱" in c or "name" in c.lower()), None)
        sym_col  = next((c for c in df.columns if "代碼" in c or "sym" in c.lower()), None)

        if pct_col and name_col:
            raw_pct = df[pct_col].astype(str).str.replace("%", "").str.strip()
            pct_vals = pd.to_numeric(raw_pct, errors="coerce").fillna(0)
            labels   = df[name_col].astype(str)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pct_vals,
                y=labels,
                orientation="h",
                marker=dict(
                    color=pct_vals,
                    colorscale="Teal",
                    line=dict(color="#1a1a2e", width=0.5),
                ),
                text=[f"{v:.1f}%" for v in pct_vals],
                textposition="outside",
                textfont=dict(size=12, color="#ccc"),
                hovertemplate=(
                    "<b>%{y}</b><br>佔比: %{x:.2f}%<extra></extra>"
                ),
            ))
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=60, t=20, b=20),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#2a2a3a",
                    tickfont=dict(size=12, color="#aaa"),
                    title=dict(text="佔比 (%)", font=dict(size=12, color="#aaa")),
                ),
                yaxis=dict(
                    tickfont=dict(size=12, color="#ddd"),
                    autorange="reversed",
                ),
                font=dict(size=12, color="#ddd"),
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

        names = [p["fund_name"] for p in portfolio]
        current_pct = {p["fund_name"]: p["pct"] for p in portfolio}

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
            f"<div style='font-size:13px; color:#aaa; margin-bottom:8px;'>"
            f"本月建議總分配：<b style='color:#FFD700;'>HKD {total_alloc:,.0f}</b>"
            f"　（月供 {monthly_budget:,.0f}）</div>",
            unsafe_allow_html=True,
        )

        rows = []
        for name in names:
            r   = result.get(name, {})
            alloc = r.get("allocation", 0)
            dev   = r.get("deviation", 0)
            act   = r.get("action", "—")
            cur   = current_pct.get(name, 0)
            tgt   = target_pct.get(name, 0)
            rows.append({
                "基金名稱":   name,
                "現持佔比":   f"{cur:.1f}%",
                "目標佔比":   f"{tgt:.1f}%",
                "偏離度":     f"{dev:+.1f}%",
                "建議操作":   act,
                "本月供款 (HKD)": f"$ {alloc:,.0f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        _render_rebalance_chart(result, target_pct, current_pct)

    except Exception as e:
        st.error(f"⚠️ 再平衡計算異常：{e}")


def _render_rebalance_chart(
    result: dict,
    target_pct: dict,
    current_pct: dict,
) -> None:
    try:
        labels     = list(result.keys())
        allocs     = [result[k]["allocation"] for k in labels]
        cur_vals   = [current_pct.get(k, 0) for k in labels]
        tgt_vals   = [target_pct.get(k, 0) for k in labels]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="現持佔比 %",
            x=labels,
            y=cur_vals,
            marker=dict(color="#3a5a3a", line=dict(color="#00FF7F44", width=1)),
            text=[f"{v:.1f}%" for v in cur_vals],
            textposition="auto",
            textfont=dict(size=12, color="#ddd"),
        ))
        fig.add_trace(go.Bar(
            name="目標佔比 %",
            x=labels,
            y=tgt_vals,
            marker=dict(color="#1a3a5a", line=dict(color="#00D4FF44", width=1)),
            text=[f"{v:.1f}%" for v in tgt_vals],
            textposition="auto",
            textfont=dict(size=12, color="#ddd"),
        ))
        fig.add_trace(go.Bar(
            name="本月供款 HKD",
            x=labels,
            y=allocs,
            yaxis="y2",
            marker=dict(color="#5a4a00", line=dict(color="#FFD70044", width=1)),
            text=[f"${v:,.0f}" for v in allocs],
            textposition="auto",
            textfont=dict(size=12, color="#FFD700"),
        ))
        fig.update_layout(
            barmode="group",
            height=320,
            margin=dict(l=0, r=0, t=30, b=80),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color="#ddd"),
            ),
            xaxis=dict(
                tickfont=dict(size=12, color="#aaa"),
                tickangle=-30,
            ),
            yaxis=dict(
                title=dict(text="佔比 %", font=dict(size=12, color="#aaa")),
                tickfont=dict(size=12, color="#aaa"),
                gridcolor="#2a2a3a",
            ),
            yaxis2=dict(
                title=dict(text="供款 HKD", font=dict(size=12, color="#FFD700")),
                tickfont=dict(size=12, color="#FFD700"),
                overlaying="y",
                side="right",
                gridcolor="#2a2a2a",
            ),
            font=dict(size=12, color="#ddd"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        print(_err("_render_rebalance_chart", e))


# ── Main page entry point ─────────────────────────────────────────────────────
def render_mpf_page() -> None:
    """Full MPF 智投診斷室 page."""
    try:
        st.markdown("## 🛡️ MPF 智投診斷室")
        st.caption(
            "上傳 eMPF 截圖 → 自動識別持倉 → 透視底層成分股 → 智能再平衡建議"
        )
        st.markdown("---")

        render_market_alert()
        st.markdown("---")

        tab_ocr, tab_portfolio, tab_rebalance = st.tabs(
            ["📸 截圖識別", "📊 持倉透視", "⚖️ 再平衡建議"]
        )

        with tab_ocr:
            render_upload_section()

        with tab_portfolio:
            render_portfolio_table()

        with tab_rebalance:
            render_rebalance_section(
                monthly_budget=st.session_state.get("mpf_monthly_budget", 2400.0)
            )

    except Exception as e:
        st.error(f"❌ MPF 智投頁面異常：{e}")

"""
picks_store.py — 分析師推薦持久化模組

職責:
  - 從 picks_data.json 載入推薦記錄，首次啟動時自動以 PICKS_DATA 種子資料初始化
  - 提供 CRUD 操作：新增、更新、刪除推薦
  - 超過 EXPIRY_DAYS (預設 30) 天的推薦自動標記為過期；可選擇清除過期記錄
  - load_picks() 回傳的列表始終是副本，避免外部修改污染持久化狀態
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────────────────────────────────────
PICKS_FILE = "picks_data.json"
EXPIRY_DAYS = 30          # 超過此天數視為過期（score 仍計入 0.1 權重，但可選擇移除）
STALE_DAYS  = 30          # purge_expired_picks() 的預設閾值


# ──────────────────────────────────────────────────────────────────────────────
# 種子資料（首次啟動或檔案遺失時使用）
# ──────────────────────────────────────────────────────────────────────────────
def _build_seed_picks() -> List[Dict]:
    """以當天為基準動態產生種子推薦，避免硬編碼過期日期。"""
    today = datetime.now()

    def d(days_ago: int) -> str:
        return (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    return [
        # Howard Marks
        {"kol_id": "howard_marks", "ticker": "HYG",  "date": d(2),  "argument_quality": 3, "thesis": "高收益債利差擴大，風險補償提升，適合防禦性配置"},
        {"kol_id": "howard_marks", "ticker": "LQD",  "date": d(2),  "argument_quality": 3, "thesis": "投資等級公司債在升息尾聲具備良好風險報酬"},
        {"kol_id": "howard_marks", "ticker": "BIL",  "date": d(8),  "argument_quality": 3, "thesis": "短期國債作為現金替代，保本優先於追求報酬"},
        # Cathie Wood
        {"kol_id": "cathie_wood",  "ticker": "NVDA", "date": d(1),  "argument_quality": 3, "thesis": "AI 算力基礎設施仍處早期，資料中心資本支出持續高速增長"},
        {"kol_id": "cathie_wood",  "ticker": "TSLA", "date": d(3),  "argument_quality": 3, "thesis": "FSD 商業化落地 + Robotaxi 潛力，長期 TAM 遠超傳統車企"},
        {"kol_id": "cathie_wood",  "ticker": "COIN", "date": d(5),  "argument_quality": 2, "thesis": "加密監管明朗化利好交易所盈利模型"},
        {"kol_id": "cathie_wood",  "ticker": "MSFT", "date": d(6),  "argument_quality": 3, "thesis": "Azure AI 服務滲透率提升，企業軟體訂閱黏性強"},
        # Adam Khoo
        {"kol_id": "adam_khoo",    "ticker": "AAPL", "date": d(4),  "argument_quality": 3, "thesis": "服務業務毛利率持續提升，生態系鎖定效應強，PE 合理"},
        {"kol_id": "adam_khoo",    "ticker": "MSFT", "date": d(4),  "argument_quality": 3, "thesis": "企業 AI 採用進入加速期，Azure 收入指引上調"},
        {"kol_id": "adam_khoo",    "ticker": "SPY",  "date": d(7),  "argument_quality": 3, "thesis": "SMA200 多頭排列，分批定投優質指數 ETF"},
        {"kol_id": "adam_khoo",    "ticker": "NVDA", "date": d(3),  "argument_quality": 3, "thesis": "Blackwell 出貨加速，AI 訓練推論需求未見頂"},
        # Jeremy Siegel
        {"kol_id": "jeremy_siegel", "ticker": "VT",  "date": d(5),  "argument_quality": 3, "thesis": "全球分散配置，長期複利效應超越擇時操作"},
        {"kol_id": "jeremy_siegel", "ticker": "VIG", "date": d(5),  "argument_quality": 3, "thesis": "股息成長股歷史風險調整後報酬優秀，防禦性佳"},
        {"kol_id": "jeremy_siegel", "ticker": "SPY", "date": d(12), "argument_quality": 3, "thesis": "歷史數據：S&P500 長期年化 7% 實質報酬不變，持有就是策略"},
        # Michael Burry
        {"kol_id": "michael_burry", "ticker": "SQQQ","date": d(3),  "argument_quality": 3, "thesis": "科技股 P/E 過高，利率維持高位，做空 QQQ 作為組合避險"},
        {"kol_id": "michael_burry", "ticker": "GEO", "date": d(9),  "argument_quality": 2, "thesis": "低估值監獄營運商，政策逆風已反映股價"},
        {"kol_id": "michael_burry", "ticker": "SHV", "date": d(3),  "argument_quality": 3, "thesis": "6個月短債持有到期，規避市場系統性風險"},
        # Joseph Carlson
        {"kol_id": "joseph_carlson", "ticker": "MSFT","date": d(2), "argument_quality": 3, "thesis": "核心持倉，自由現金流 YoY 成長 25%+，AI Copilot 訂閱收入加速"},
        {"kol_id": "joseph_carlson", "ticker": "AAPL","date": d(2), "argument_quality": 3, "thesis": "服務收入佔比提升至 25%，毛利率擴張，持續回購股票"},
        {"kol_id": "joseph_carlson", "ticker": "V",   "date": d(6), "argument_quality": 3, "thesis": "支付網路護城河，每年穩定回購 2-3%，跨境支付量回升"},
        {"kol_id": "joseph_carlson", "ticker": "AMZN","date": d(6), "argument_quality": 3, "thesis": "AWS 毛利擴張，廣告業務高速增長，整體自由現金流爆發"},
        # Seeking Alpha Quant
        {"kol_id": "seeking_alpha_quant", "ticker": "NVDA", "date": d(1), "argument_quality": 3, "thesis": "量化因子：估值A/成長A+/獲利A+/動能A — 四維全優，罕見高分"},
        {"kol_id": "seeking_alpha_quant", "ticker": "AAPL", "date": d(1), "argument_quality": 3, "thesis": "量化因子：估值B/成長B+/獲利A/動能A — 穩健複合評分"},
        {"kol_id": "seeking_alpha_quant", "ticker": "META", "date": d(2), "argument_quality": 3, "thesis": "量化因子：廣告ARPU創歷史新高，AI推薦引擎推動用量 +20%"},
        {"kol_id": "seeking_alpha_quant", "ticker": "MSFT", "date": d(2), "argument_quality": 3, "thesis": "量化因子：訂閱黏性A+，自由現金流殖利率 2.8%，ROE 35%+"},
        {"kol_id": "seeking_alpha_quant", "ticker": "GOOGL","date": d(3), "argument_quality": 3, "thesis": "量化因子：搜索護城河依然穩固，Gemini 廣告整合初見成效"},
        # WSJ Markets
        {"kol_id": "wsj_markets", "ticker": "MSFT",  "date": d(1), "argument_quality": 3, "thesis": "報導：企業 AI 軟體採用進入主流，Copilot 付費席次季增 40%"},
        {"kol_id": "wsj_markets", "ticker": "GOOGL", "date": d(4), "argument_quality": 3, "thesis": "報導：Gemini 整合 Workspace 後廣告 CTR 提升，廣告主預算回流"},
        {"kol_id": "wsj_markets", "ticker": "META",  "date": d(4), "argument_quality": 3, "thesis": "報導：Llama AI 模型開源策略吸引企業用戶，廣告算法精準度再提升"},
        # Warren Buffett
        {"kol_id": "warren_buffett", "ticker": "AAPL", "date": d(5),  "argument_quality": 3, "thesis": "蘋果是消費者行為最佳護城河之一，服務收入持續成長，現金回購力道強勁"},
        {"kol_id": "warren_buffett", "ticker": "KO",   "date": d(10), "argument_quality": 3, "thesis": "定價權穩固，全球分銷網絡無可複製，股息增長超過 60 年"},
        {"kol_id": "warren_buffett", "ticker": "BRK-B","date": d(14), "argument_quality": 3, "thesis": "多元業務組合提供穩定現金流，帳面保守低槓桿，長期複利機器"},
        # Charlie Munger
        {"kol_id": "charlie_munger", "ticker": "COST", "date": d(6),  "argument_quality": 3, "thesis": "會員制商業模式黏性極強，倉儲零售護城河可持續複利增長"},
        {"kol_id": "charlie_munger", "ticker": "AAPL", "date": d(9),  "argument_quality": 3, "thesis": "品質企業應長期持有，蘋果生態系統鎖定效應為最佳商業模式範本"},
        {"kol_id": "charlie_munger", "ticker": "BRK-B","date": d(20), "argument_quality": 3, "thesis": "避免愚蠢決策勝過追求聰明操作，持有優質資產等待時間複利"},
        # Stanley Druckenmiller
        {"kol_id": "stanley_druckenmiller", "ticker": "NVDA", "date": d(2),  "argument_quality": 3, "thesis": "AI 算力需求正處於史上最大資本支出週期，流動性驅動動能顯著"},
        {"kol_id": "stanley_druckenmiller", "ticker": "MSFT", "date": d(7),  "argument_quality": 3, "thesis": "企業雲端與 AI 整合加速，Azure 收入週期確定性高，值得集中高確信"},
        {"kol_id": "stanley_druckenmiller", "ticker": "QQQ",  "date": d(12), "argument_quality": 3, "thesis": "科技龍頭盈利動能與流動性環境共振，趨勢跟隨策略持有科技 ETF"},
        # David Tepper
        {"kol_id": "david_tepper", "ticker": "AMZN", "date": d(3),  "argument_quality": 3, "thesis": "AWS 與廣告雙引擎驅動自由現金流爆發，政策寬鬆環境利好估值修復"},
        {"kol_id": "david_tepper", "ticker": "META", "date": d(5),  "argument_quality": 3, "thesis": "AI 廣告精準化提升 ARPU，宏觀消費回暖利好廣告支出週期"},
        {"kol_id": "david_tepper", "ticker": "QQQ",  "date": d(11), "argument_quality": 2, "thesis": "聯準會政策轉向訊號明確，流動性驅動科技股估值擴張"},
        # Bill Ackman
        {"kol_id": "bill_ackman", "ticker": "CMG",  "date": d(4),  "argument_quality": 3, "thesis": "品牌定價權強，同店銷售穩健，數位點餐滲透率持續提升"},
        {"kol_id": "bill_ackman", "ticker": "HLT",  "date": d(8),  "argument_quality": 3, "thesis": "輕資產加盟模式現金流穩定，旅遊復甦長期趨勢支撐入住率"},
        {"kol_id": "bill_ackman", "ticker": "GOOG", "date": d(13), "argument_quality": 3, "thesis": "搜索廣告護城河穩固，雲端業務加速，AI 整合提升貨幣化效率"},
        # Seth Klarman
        {"kol_id": "seth_klarman", "ticker": "GOOG",  "date": d(7),  "argument_quality": 3, "thesis": "估值相對科技同業具備安全邊際，廣告業務現金流充足支撐下行保護"},
        {"kol_id": "seth_klarman", "ticker": "EBAY",  "date": d(20), "argument_quality": 2, "thesis": "低估值平台業務，廣告變現尚未完全定價，自由現金流穩定"},
        {"kol_id": "seth_klarman", "ticker": "WBD",   "date": d(25), "argument_quality": 2, "thesis": "串流整合陣痛期已過，內容資產帳面值顯著低於重置成本"},
        # Joel Greenblatt
        {"kol_id": "joel_greenblatt", "ticker": "GILD","date": d(4), "argument_quality": 3, "thesis": "Magic Formula: 高資本回報率 + 低盈利殖利率，生技低估值機會"},
        {"kol_id": "joel_greenblatt", "ticker": "VTV", "date": d(8), "argument_quality": 3, "thesis": "價值因子週期回歸，VTV 持倉以高ROE傳統企業為主，估值安全邊際充足"},
        # Tom Lee
        {"kol_id": "tom_lee", "ticker": "SPY",  "date": d(3), "argument_quality": 3, "thesis": "市場廣度回升，新高家數擴散，S&P500 年底目標上調"},
        {"kol_id": "tom_lee", "ticker": "NVDA", "date": d(6), "argument_quality": 3, "thesis": "AI 超級週期下 Nvidia 為科技股多頭核心持倉"},
        # Dan Ives
        {"kol_id": "dan_ives", "ticker": "TSLA", "date": d(2), "argument_quality": 3, "thesis": "FSD v12 里程碑驅動 Robotaxi 故事，自動駕駛 TAM 破兆美元"},
        {"kol_id": "dan_ives", "ticker": "AAPL", "date": d(5), "argument_quality": 3, "thesis": "Apple Intelligence 觸發換機超級週期，服務 ARR 持續攀升"},
        # Goldman Sachs
        {"kol_id": "goldman_global_research", "ticker": "SPY",  "date": d(3), "argument_quality": 3, "thesis": "盈利預期上調，EPS 增速重回雙位數，S&P500 年度目標維持正向"},
        {"kol_id": "goldman_global_research", "ticker": "NVDA", "date": d(5), "argument_quality": 3, "thesis": "AI 基礎設施支出週期不可逆，GPU 供不應求至少延續至 2026"},
        # BlackRock
        {"kol_id": "blackrock_institute", "ticker": "IVV",  "date": d(4), "argument_quality": 3, "thesis": "核心配置首選：美股大盤寬基指數，長期複利優於主動選股"},
        {"kol_id": "blackrock_institute", "ticker": "SGOV", "date": d(7), "argument_quality": 3, "thesis": "短期國債殖利率具吸引力，流動性儲備倉位最佳替代品"},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 核心 I/O
# ──────────────────────────────────────────────────────────────────────────────
def _read_file() -> List[Dict]:
    """從 JSON 檔案讀取推薦記錄；檔案不存在時回傳空列表（非種子）。"""
    try:
        with open(PICKS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def _write_file(picks: List[Dict]) -> None:
    with open(PICKS_FILE, "w", encoding="utf-8") as fh:
        json.dump(picks, fh, ensure_ascii=False, indent=2)


def _ensure_initialized() -> None:
    """首次啟動：若 picks_data.json 不存在，以種子資料初始化。"""
    if not os.path.exists(PICKS_FILE):
        seed = _build_seed_picks()
        _write_file(seed)


# ──────────────────────────────────────────────────────────────────────────────
# 公開 API
# ──────────────────────────────────────────────────────────────────────────────
def load_picks() -> List[Dict]:
    """
    回傳目前所有推薦記錄（副本）。
    首次呼叫若檔案不存在，自動以種子資料初始化。
    """
    _ensure_initialized()
    return list(_read_file())


def save_picks(picks: List[Dict]) -> None:
    """覆寫整個推薦記錄列表（完整替換，用於批次操作）。"""
    _write_file(list(picks))


def add_pick(pick: Dict) -> List[Dict]:
    """
    新增一筆推薦。必填欄位：kol_id, ticker, date, argument_quality, thesis。
    回傳更新後的完整列表。
    """
    required = {"kol_id", "ticker", "date", "argument_quality", "thesis"}
    missing = required - set(pick.keys())
    if missing:
        raise ValueError(f"缺少必填欄位：{missing}")

    picks = load_picks()
    entry = {
        "kol_id":           str(pick["kol_id"]).strip(),
        "ticker":           str(pick["ticker"]).strip().upper(),
        "date":             str(pick["date"]),
        "argument_quality": int(pick["argument_quality"]),
        "thesis":           str(pick["thesis"]).strip(),
        "added_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    picks.append(entry)
    _write_file(picks)
    return picks


def delete_pick(index: int) -> List[Dict]:
    """
    依索引刪除推薦（0-based）。
    回傳更新後的完整列表。
    """
    picks = load_picks()
    if index < 0 or index >= len(picks):
        raise IndexError(f"索引 {index} 超出範圍（共 {len(picks)} 筆）")
    picks.pop(index)
    _write_file(picks)
    return picks


def update_pick(index: int, updates: Dict) -> List[Dict]:
    """
    更新指定索引的推薦欄位。
    回傳更新後的完整列表。
    """
    picks = load_picks()
    if index < 0 or index >= len(picks):
        raise IndexError(f"索引 {index} 超出範圍（共 {len(picks)} 筆）")
    picks[index].update(updates)
    if "ticker" in updates:
        picks[index]["ticker"] = picks[index]["ticker"].strip().upper()
    picks[index]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_file(picks)
    return picks


def purge_expired_picks(days: int = STALE_DAYS) -> tuple[List[Dict], int]:
    """
    移除超過 `days` 天的推薦。
    回傳 (更新後列表, 移除筆數)。
    """
    picks = load_picks()
    cutoff = datetime.now() - timedelta(days=days)
    active = []
    removed = 0
    for p in picks:
        try:
            pick_date = datetime.strptime(p["date"], "%Y-%m-%d")
            if pick_date >= cutoff:
                active.append(p)
            else:
                removed += 1
        except Exception:
            active.append(p)   # 無法解析日期的記錄保留
    _write_file(active)
    return active, removed


def get_picks_with_status(days: int = EXPIRY_DAYS) -> List[Dict]:
    """
    回傳所有推薦並加上 `is_expired` 與 `days_old` 欄位，供 UI 顯示用。
    """
    picks = load_picks()
    now = datetime.now()
    result = []
    for p in picks:
        entry = dict(p)
        try:
            pick_date = datetime.strptime(p["date"], "%Y-%m-%d")
            days_old = (now - pick_date).days
            entry["days_old"]   = days_old
            entry["is_expired"] = days_old > days
        except Exception:
            entry["days_old"]   = -1
            entry["is_expired"] = False
        result.append(entry)
    return result

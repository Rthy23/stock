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
from kol_config import ARK_FUND_ONLY_IDS, COMMENTARY_IDS, is_ark_identity, is_commentary_identity
from sec_edgar_fetcher import MANAGER_IDS

# ──────────────────────────────────────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────────────────────────────────────
PICKS_FILE = "picks_data.json"
EXPIRY_DAYS = 30          # 超過此天數視為過期（score 仍計入 0.1 權重，但可選擇移除）
STALE_DAYS  = 30          # purge_expired_picks() 的預設閾值
QUARANTINED_IDS = MANAGER_IDS | COMMENTARY_IDS | ARK_FUND_ONLY_IDS


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
        # Charlie Munger
        {"kol_id": "charlie_munger", "ticker": "COST", "date": d(6),  "argument_quality": 3, "thesis": "會員制商業模式黏性極強，倉儲零售護城河可持續複利增長"},
        {"kol_id": "charlie_munger", "ticker": "AAPL", "date": d(9),  "argument_quality": 3, "thesis": "品質企業應長期持有，蘋果生態系統鎖定效應為最佳商業模式範本"},
        {"kol_id": "charlie_munger", "ticker": "BRK-B","date": d(20), "argument_quality": 3, "thesis": "避免愚蠢決策勝過追求聰明操作，持有優質資產等待時間複利"},
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
    """Never seed invented recommendations on a new installation."""
    if not os.path.exists(PICKS_FILE):
        _write_file([])


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
    """Only a future human-verified source workflow may reopen this editor."""
    raise ValueError("分析師推薦缺乏人工查證的原始來源；暫不提供新增。")


def delete_pick(index: int) -> List[Dict]:
    """Archived records are preserved, not managed as public recommendations."""
    raise ValueError("未查核歷史紀錄已隔離，暫不提供刪除。")


def update_pick(index: int, updates: Dict) -> List[Dict]:
    """Archived records cannot be re-attributed without verification."""
    raise ValueError("分析師推薦缺乏人工查證的原始來源；暫不提供編輯。")


def purge_expired_picks(days: int = STALE_DAYS) -> tuple[List[Dict], int]:
    """Preserve all archived examples, regardless of their old dates."""
    return load_picks(), 0


def get_picks_with_status(days: int = EXPIRY_DAYS) -> List[Dict]:
    """No archived example is publicly available as a verified pick."""
    return []

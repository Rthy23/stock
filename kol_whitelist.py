"""
kol_whitelist.py — 白名單 KOL 精選分析師系統

Architecture:
  WHITELIST      — 手工篩選的高信譽分析師清單
  PICKS_DATA     — 各 KOL 近期推薦 (模擬爬蟲輸出，可替換為真實 API 結果)
  score_picks()  — 多維度評分：一致性 × 論點品質 × 時效性
  build_consensus_table() — 整合後的共識排行榜
  call_gemini_consensus() — Gemini AI 輸出結構化報告
  render_kol_section()    — Streamlit 渲染入口
"""

from __future__ import annotations
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import streamlit as st
from user_config import load_kol_whitelist
from kol_config import ANALYST_DIRECTORY
from navigation import navigate_to_ticker

# ──────────────────────────────────────────────────────────────────────────────
# 1. WHITELIST — 手工篩選高信譽分析師
# ──────────────────────────────────────────────────────────────────────────────
WHITELIST: List[Dict] = [
    {
        "id":       "howard_marks",
        "name":     "Howard Marks",
        "org":      "Oaktree Capital",
        "focus":    "信用周期、風險評估、宏觀市場備忘錄",
        "platform": "oaktreecapital.com / Bloomberg",
        "stance":   "謹慎",
        "stance_color": "#FF8C00",
        "rep":      5,
        "years_active": 30,
        "type":     "機構級別分析師",
        "rationale": "全球頂級信用市場投資人，Memo 以嚴謹量化分析著稱，無標題黨",
    },
    {
        "id":       "cathie_wood",
        "name":     "Cathie Wood",
        "org":      "ARK Invest",
        "focus":    "顛覆性創新：AI、基因組學、加密貨幣",
        "platform": "ARK ETFs / X @CathieDWood",
        "stance":   "積極做多",
        "stance_color": "#00CC44",
        "rep":      4,
        "years_active": 15,
        "type":     "機構級別分析師",
        "rationale": "創新主題 ETF 先驅，每日公開持倉透明，研究框架具結構性",
    },
    {
        "id":       "adam_khoo",
        "name":     "Adam Khoo",
        "org":      "Adam Khoo Learning Technologies",
        "focus":    "價值投資、技術分析結合、波段操作",
        "platform": "YouTube / 課程平台",
        "stance":   "中性偏多",
        "stance_color": "#0066CC",
        "rep":      4,
        "years_active": 12,
        "type":     "長期價值投資專家",
        "rationale": "結合基本面+技術面，系統化教學，有清晰推薦理由與進出場邏輯",
    },
    {
        "id":       "jeremy_siegel",
        "name":     "Jeremy Siegel",
        "org":      "Wharton School / WisdomTree",
        "focus":    "長期股票市場、退休投資組合",
        "platform": "CNBC / Bloomberg TV",
        "stance":   "長期樂觀",
        "stance_color": "#00CC44",
        "rep":      5,
        "years_active": 35,
        "type":     "機構級別分析師",
        "rationale": "《股市長線法寶》作者，嚴謹學術研究背景，CNBC 常駐分析師",
    },
    {
        "id":       "michael_burry",
        "name":     "Michael Burry",
        "org":      "Scion Asset Management",
        "focus":    "逆向投資、系統性風險、做空策略",
        "platform": "X (間歇性) / SEC 13F",
        "stance":   "偏空",
        "stance_color": "#FF4B4B",
        "rep":      4,
        "years_active": 20,
        "type":     "長期價值投資專家",
        "rationale": "2008 金融危機預測者，13F 持倉公開透明，邏輯嚴謹非噱頭",
    },
    {
        "id":       "joseph_carlson",
        "name":     "Joseph Carlson",
        "org":      "個人投資者 / YouTube",
        "focus":    "成長股 + 股息再投資、個人持倉透明化",
        "platform": "YouTube @josephcarlsonshow",
        "stance":   "長期持有",
        "stance_color": "#0066CC",
        "rep":      4,
        "years_active": 8,
        "type":     "長期價值投資專家",
        "rationale": "月更持倉，數據驅動分析，公開績效記錄，無誇大標題",
    },
    {
        "id":       "seeking_alpha_quant",
        "name":     "Seeking Alpha Quant",
        "org":      "Seeking Alpha",
        "focus":    "量化篩股、因子投資、財報分析",
        "platform": "seekingalpha.com",
        "stance":   "數據驅動",
        "stance_color": "#6600CC",
        "rep":      5,
        "years_active": 15,
        "type":     "專業財經媒體/數據平台",
        "rationale": "系統化因子模型，涵蓋估值/成長/獲利/動能四大維度，純量化無主觀偏誤",
    },
    {
        "id":       "wsj_markets",
        "name":     "WSJ Markets Desk",
        "org":      "Wall Street Journal",
        "focus":    "宏觀政策、企業盈利、市場結構分析",
        "platform": "wsj.com / Twitter @WSJ",
        "stance":   "中性報導",
        "stance_color": "#555555",
        "rep":      5,
        "years_active": 40,
        "type":     "專業財經媒體/數據平台",
        "rationale": "權威財經媒體，記者具備深度財務培訓，報導包含數據來源引用",
    },
]

# Keep the historical scoring/picks schema while sourcing the expanded
# directory from the data-only config module.
WHITELIST = ANALYST_DIRECTORY

# ──────────────────────────────────────────────────────────────────────────────
# 2. PICKS_DATA — 模擬爬蟲輸出 (可替換為真實 API/爬蟲結果)
#    argument_quality: 3=強論點(含財報/護城河/估值), 2=中等, 1=薄弱/標題黨
# ──────────────────────────────────────────────────────────────────────────────
_today = datetime.now()

PICKS_DATA: List[Dict] = [
    # Howard Marks
    {"kol_id": "howard_marks",    "ticker": "HYG",  "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "高收益債利差擴大，風險補償提升，適合防禦性配置"},
    {"kol_id": "howard_marks",    "ticker": "LQD",  "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "投資等級公司債在升息尾聲具備良好風險報酬"},
    {"kol_id": "howard_marks",    "ticker": "BIL",  "date": (_today - timedelta(days=8)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "短期國債作為現金替代，保本優先於追求報酬"},

    # Cathie Wood
    {"kol_id": "cathie_wood",     "ticker": "NVDA", "date": (_today - timedelta(days=1)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AI 算力基礎設施仍處早期，資料中心資本支出持續高速增長"},
    {"kol_id": "cathie_wood",     "ticker": "TSLA", "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "FSD 商業化落地 + Robotaxi 潛力，長期 TAM 遠超傳統車企"},
    {"kol_id": "cathie_wood",     "ticker": "COIN", "date": (_today - timedelta(days=5)).strftime("%Y-%m-%d"),  "argument_quality": 2, "thesis": "加密監管明朗化利好交易所盈利模型"},
    {"kol_id": "cathie_wood",     "ticker": "MSFT", "date": (_today - timedelta(days=6)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "Azure AI 服務滲透率提升，企業軟體訂閱黏性強"},

    # Adam Khoo
    {"kol_id": "adam_khoo",       "ticker": "AAPL", "date": (_today - timedelta(days=4)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "服務業務毛利率持續提升，生態系鎖定效應強，PE 合理"},
    {"kol_id": "adam_khoo",       "ticker": "MSFT", "date": (_today - timedelta(days=4)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "企業 AI 採用進入加速期，Azure 收入指引上調"},
    {"kol_id": "adam_khoo",       "ticker": "SPY",  "date": (_today - timedelta(days=7)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "SMA200 多頭排列，分批定投優質指數 ETF"},
    {"kol_id": "adam_khoo",       "ticker": "NVDA", "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "Blackwell 出貨加速，AI 訓練推論需求未見頂"},

    # Jeremy Siegel
    {"kol_id": "jeremy_siegel",   "ticker": "VT",   "date": (_today - timedelta(days=5)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "全球分散配置，長期複利效應超越擇時操作"},
    {"kol_id": "jeremy_siegel",   "ticker": "VIG",  "date": (_today - timedelta(days=5)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "股息成長股歷史風險調整後報酬優秀，防禦性佳"},
    {"kol_id": "jeremy_siegel",   "ticker": "SPY",  "date": (_today - timedelta(days=12)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "歷史數據：S&P500 長期年化 7% 實質報酬不變，持有就是策略"},

    # Michael Burry
    {"kol_id": "michael_burry",   "ticker": "SQQQ", "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "科技股 P/E 過高，利率維持高位，做空 QQQ 作為組合避險"},
    {"kol_id": "michael_burry",   "ticker": "GEO",  "date": (_today - timedelta(days=9)).strftime("%Y-%m-%d"),  "argument_quality": 2, "thesis": "低估值監獄營運商，政策逆風已反映股價"},
    {"kol_id": "michael_burry",   "ticker": "SHV",  "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "6個月短債持有到期，規避市場系統性風險"},

    # Joseph Carlson
    {"kol_id": "joseph_carlson",  "ticker": "MSFT", "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "核心持倉，自由現金流 year over year 成長 25%+，AI Copilot 訂閱收入加速"},
    {"kol_id": "joseph_carlson",  "ticker": "AAPL", "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "服務收入佔比提升至 25%，毛利率擴張，持續回購股票"},
    {"kol_id": "joseph_carlson",  "ticker": "V",    "date": (_today - timedelta(days=6)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "支付網路護城河，每年穩定回購 2-3%，跨境支付量回升"},
    {"kol_id": "joseph_carlson",  "ticker": "AMZN", "date": (_today - timedelta(days=6)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AWS 毛利擴張，廣告業務高速增長，整體自由現金流爆發"},

    # Seeking Alpha Quant
    {"kol_id": "seeking_alpha_quant", "ticker": "NVDA", "date": (_today - timedelta(days=1)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "量化因子：估值A/成長A+/獲利A+/動能A — 四維全優，罕見高分"},
    {"kol_id": "seeking_alpha_quant", "ticker": "AAPL", "date": (_today - timedelta(days=1)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "量化因子：估值B/成長B+/獲利A/動能A — 穩健複合評分"},
    {"kol_id": "seeking_alpha_quant", "ticker": "META", "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "量化因子：廣告ARPU創歷史新高，AI推薦引擎推動用量 +20%"},
    {"kol_id": "seeking_alpha_quant", "ticker": "MSFT", "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "量化因子：訂閱黏性A+，自由現金流殖利率 2.8%，ROE 35%+"},
    {"kol_id": "seeking_alpha_quant", "ticker": "GOOGL","date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "量化因子：搜索護城河依然穩固，Gemini 廣告整合初見成效"},

    # WSJ Markets
    {"kol_id": "wsj_markets",     "ticker": "MSFT", "date": (_today - timedelta(days=1)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "報導：企業 AI 軟體採用進入主流，Copilot 付費席次季增 40%"},
    {"kol_id": "wsj_markets",     "ticker": "GOOGL","date": (_today - timedelta(days=4)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "報導：Gemini 整合 Workspace 後廣告 CTR 提升，廣告主預算回流"},
    {"kol_id": "wsj_markets",     "ticker": "META", "date": (_today - timedelta(days=4)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "報導：Llama AI 模型開源策略吸引企業用戶，廣告算法精準度再提升"},

    # Warren Buffett
    {"kol_id": "warren_buffett",  "ticker": "AAPL", "date": (_today - timedelta(days=5)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "蘋果是消費者行為最佳護城河之一，服務收入持續成長，現金回購力道強勁"},
    {"kol_id": "warren_buffett",  "ticker": "KO",   "date": (_today - timedelta(days=10)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "定價權穩固，全球分銷網絡無可複製，股息增長超過 60 年"},
    {"kol_id": "warren_buffett",  "ticker": "BRK-B","date": (_today - timedelta(days=14)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "多元業務組合提供穩定現金流，帳面保守低槓桿，長期複利機器"},

    # Charlie Munger
    {"kol_id": "charlie_munger",  "ticker": "COST", "date": (_today - timedelta(days=6)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "會員制商業模式黏性極強，倉儲零售護城河可持續複利增長"},
    {"kol_id": "charlie_munger",  "ticker": "AAPL", "date": (_today - timedelta(days=9)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "品質企業應長期持有，蘋果生態系統鎖定效應為最佳商業模式範本"},
    {"kol_id": "charlie_munger",  "ticker": "BRK-B","date": (_today - timedelta(days=20)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "避免愚蠢決策勝過追求聰明操作，持有優質資產等待時間複利"},

    # Stanley Druckenmiller
    {"kol_id": "stanley_druckenmiller", "ticker": "NVDA", "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AI 算力需求正處於史上最大資本支出週期，流動性驅動動能顯著"},
    {"kol_id": "stanley_druckenmiller", "ticker": "MSFT", "date": (_today - timedelta(days=7)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "企業雲端與 AI 整合加速，Azure 收入週期確定性高，值得集中高確信"},
    {"kol_id": "stanley_druckenmiller", "ticker": "QQQ",  "date": (_today - timedelta(days=12)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "科技龍頭盈利動能與流動性環境共振，趨勢跟隨策略持有科技 ETF"},

    # David Tepper
    {"kol_id": "david_tepper",    "ticker": "AMZN", "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AWS 與廣告雙引擎驅動自由現金流爆發，政策寬鬆環境利好估值修復"},
    {"kol_id": "david_tepper",    "ticker": "META", "date": (_today - timedelta(days=5)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AI 廣告精準化提升 ARPU，宏觀消費回暖利好廣告支出週期"},
    {"kol_id": "david_tepper",    "ticker": "QQQ",  "date": (_today - timedelta(days=11)).strftime("%Y-%m-%d"), "argument_quality": 2, "thesis": "聯準會政策轉向訊號明確，流動性驅動科技股估值擴張"},

    # Bill Ackman
    {"kol_id": "bill_ackman",     "ticker": "CMG",  "date": (_today - timedelta(days=4)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "品牌定價權強，同店銷售穩健，數位點餐滲透率持續提升"},
    {"kol_id": "bill_ackman",     "ticker": "HLT",  "date": (_today - timedelta(days=8)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "輕資產加盟模式現金流穩定，旅遊復甦長期趨勢支撐入住率"},
    {"kol_id": "bill_ackman",     "ticker": "GOOG", "date": (_today - timedelta(days=13)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "搜索廣告護城河穩固，雲端業務加速，AI 整合提升貨幣化效率"},

    # Seth Klarman
    {"kol_id": "seth_klarman",    "ticker": "GOOG", "date": (_today - timedelta(days=7)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "估值相對科技同業具備安全邊際，廣告業務現金流充足支撐下行保護"},
    {"kol_id": "seth_klarman",    "ticker": "EBAY", "date": (_today - timedelta(days=15)).strftime("%Y-%m-%d"), "argument_quality": 2, "thesis": "市場忽視的二手電商平台，估值偏低，自由現金流持續回購支撐"},
    {"kol_id": "seth_klarman",    "ticker": "WBD",  "date": (_today - timedelta(days=20)).strftime("%Y-%m-%d"), "argument_quality": 2, "thesis": "困境資產估值已大幅折讓，串流整合進展超預期可帶來困境反轉"},

    # Joel Greenblatt
    {"kol_id": "joel_greenblatt", "ticker": "SPY",  "date": (_today - timedelta(days=6)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "Magic Formula 篩選後，S&P500 整體盈利殖利率仍具吸引力"},
    {"kol_id": "joel_greenblatt", "ticker": "VTV",  "date": (_today - timedelta(days=10)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "價值因子輪動訊號浮現，高資本回報率 + 低估值的組合歷史表現優異"},
    {"kol_id": "joel_greenblatt", "ticker": "GILD", "date": (_today - timedelta(days=18)).strftime("%Y-%m-%d"), "argument_quality": 2, "thesis": "生技股盈利殖利率偏高，研發管線具備上行期權，符合 Magic Formula 篩選"},

    # Mohnish Pabrai
    {"kol_id": "mohnish_pabrai",  "ticker": "GOOG", "date": (_today - timedelta(days=5)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "複製 Buffett 邏輯：護城河深、現金流豐沛，長期複利特質明顯"},
    {"kol_id": "mohnish_pabrai",  "ticker": "MU",   "date": (_today - timedelta(days=9)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "記憶體週期底部已過，HBM AI 需求爆發，勝率高且下行有限"},
    {"kol_id": "mohnish_pabrai",  "ticker": "AMR",  "date": (_today - timedelta(days=17)).strftime("%Y-%m-%d"), "argument_quality": 2, "thesis": "冶金煤供應緊縮，鋼鐵產業鏈需求穩定，低估值高確信反向佈局"},

    # Aswath Damodaran
    {"kol_id": "aswath_damodaran","ticker": "AAPL", "date": (_today - timedelta(days=4)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "DCF 模型顯示服務業務成長支撐合理估值，風險溢價已充分反映"},
    {"kol_id": "aswath_damodaran","ticker": "MSFT", "date": (_today - timedelta(days=8)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "終端成長率假設保守下仍具內在價值，雲端業務可見度高降低估值不確定性"},
    {"kol_id": "aswath_damodaran","ticker": "SPY",  "date": (_today - timedelta(days=14)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "權益風險溢價維持歷史均值，整體市場估值中性，指數持有具合理報酬期望"},

    # Tom Lee
    {"kol_id": "tom_lee",         "ticker": "SPY",  "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "盈利動能強、市場廣度擴張，Fundstrat 年度目標價上調，偏多策略"},
    {"kol_id": "tom_lee",         "ticker": "QQQ",  "date": (_today - timedelta(days=5)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "科技股通常在升息尾聲反彈最猛，流動性改善有利成長股估值修復"},
    {"kol_id": "tom_lee",         "ticker": "NVDA", "date": (_today - timedelta(days=7)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AI 資本支出超預期，NVDA 作為核心受益者動能仍強，中線目標持續上調"},

    # Dan Ives
    {"kol_id": "dan_ives",        "ticker": "TSLA", "date": (_today - timedelta(days=1)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "Robotaxi 時間表明確，FSD 商業化打開新 TAM，Wedbush 維持 Buy 評級"},
    {"kol_id": "dan_ives",        "ticker": "AAPL", "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "Apple Intelligence 帶動 iPhone 升級超級週期，AI 服務貨幣化進入加速期"},
    {"kol_id": "dan_ives",        "ticker": "MSFT", "date": (_today - timedelta(days=6)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "Copilot 企業滲透率每季加速，Azure AI 收入指引上調，科技大多頭的核心"},

    # Gene Munster
    {"kol_id": "gene_munster",    "ticker": "AAPL", "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "Apple Intelligence 推動 AI 手機升級浪潮，Deepwater 看好五年 AI 生態收入"},
    {"kol_id": "gene_munster",    "ticker": "NVDA", "date": (_today - timedelta(days=4)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AI 訓練與推論算力需求長期飽和，Blackwell 架構確立下一代主流地位"},
    {"kol_id": "gene_munster",    "ticker": "TSLA", "date": (_today - timedelta(days=8)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "自動駕駛技術領先優勢擴大，Optimus 機器人可能成為下一個十億級產品"},

    # Liz Ann Sonders
    {"kol_id": "liz_ann_sonders", "ticker": "SPY",  "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "市場廣度持續改善，景氣擴張中期，Schwab 策略建議均衡持有核心 ETF"},
    {"kol_id": "liz_ann_sonders", "ticker": "SCHD", "date": (_today - timedelta(days=9)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "股息成長因子在景氣轉換期提供防禦性，現金流質量篩選降低波動風險"},
    {"kol_id": "liz_ann_sonders", "ticker": "IWM",  "date": (_today - timedelta(days=16)).strftime("%Y-%m-%d"), "argument_quality": 2, "thesis": "小型股估值折讓歷史高位，若景氣軟著陸成功將受益均值回歸輪動"},

    # Ed Yardeni
    {"kol_id": "ed_yardeni",      "ticker": "SPY",  "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "盈利週期依然強勁，Yardeni Research 維持牛市基本情境，S&P 500 目標上調"},
    {"kol_id": "ed_yardeni",      "ticker": "XLI",  "date": (_today - timedelta(days=10)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "製造業回流 + 基建支出帶動工業板塊盈利，景氣擴張中期工業股具上升空間"},
    {"kol_id": "ed_yardeni",      "ticker": "XLF",  "date": (_today - timedelta(days=14)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "利率高原期有利銀行淨息差，金融板塊估值合理且股息率具吸引力"},

    # Goldman Sachs Global Research
    {"kol_id": "goldman_global_research", "ticker": "SPY",  "date": (_today - timedelta(days=1)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "GS 2025 S&P 500 目標上調，盈利預期上修，機構研究維持超配美股"},
    {"kol_id": "goldman_global_research", "ticker": "QQQ",  "date": (_today - timedelta(days=5)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AI Capex 超預期帶動科技股盈利修正向上，Goldman 因子模型看好科技配置"},
    {"kol_id": "goldman_global_research", "ticker": "XLF",  "date": (_today - timedelta(days=11)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "利率環境支撐銀行盈利，GS 量化模型顯示金融板塊估值因子與動能共振"},

    # J.P. Morgan Global Research
    {"kol_id": "jpmorgan_global_research","ticker": "SPY",  "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "JPM 策略師認為美股盈利韌性支撐估值，軟著陸情境下維持股票標配"},
    {"kol_id": "jpmorgan_global_research","ticker": "JPM",  "date": (_today - timedelta(days=7)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "淨息差擴張與信貸品質改善，J.P. Morgan 自身盈利動能強，銀行股首選"},
    {"kol_id": "jpmorgan_global_research","ticker": "TLT",  "date": (_today - timedelta(days=13)).strftime("%Y-%m-%d"), "argument_quality": 2, "thesis": "長債殖利率接近頂部，若降息週期啟動長天期國債提供資本利得機會"},

    # BlackRock Investment Institute
    {"kol_id": "blackrock_institute",     "ticker": "IVV",  "date": (_today - timedelta(days=4)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "BlackRock 維持美股核心配置，AI 主題結構性成長支撐大型股長期回報"},
    {"kol_id": "blackrock_institute",     "ticker": "AGG",  "date": (_today - timedelta(days=8)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "投資等級債提供穩定票息，降息環境下資本增值空間開啟，防禦性配置首選"},
    {"kol_id": "blackrock_institute",     "ticker": "SGOV", "date": (_today - timedelta(days=12)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "超短期國庫券利率具吸引力，BlackRock 建議以現金等價物作為戰術緩衝部位"},

    # Morgan Stanley Research
    {"kol_id": "morgan_stanley_research", "ticker": "SPY",  "date": (_today - timedelta(days=2)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "MS 策略師看好盈利週期延長，產業鏈數據顯示需求持續優於預期"},
    {"kol_id": "morgan_stanley_research", "ticker": "XLK",  "date": (_today - timedelta(days=6)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "AI 算力建設與軟體採用同步加速，科技板塊收入能見度高，維持超配"},
    {"kol_id": "morgan_stanley_research", "ticker": "MSFT", "date": (_today - timedelta(days=10)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "MS 研究：Azure 季增速再加速，Copilot 付費席次倍增，上調目標價"},

    # Morningstar Quantitative Research
    {"kol_id": "morningstar_quant",       "ticker": "VOO",  "date": (_today - timedelta(days=3)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "Morningstar 量化評估 VOO 費用率最低、因子分布均衡，長期核心配置首選"},
    {"kol_id": "morningstar_quant",       "ticker": "VTV",  "date": (_today - timedelta(days=7)).strftime("%Y-%m-%d"),  "argument_quality": 3, "thesis": "價值因子估值折讓擴大，Morningstar 星級評分顯示潛在漲幅高於歷史中位數"},
    {"kol_id": "morningstar_quant",       "ticker": "VIG",  "date": (_today - timedelta(days=11)).strftime("%Y-%m-%d"), "argument_quality": 3, "thesis": "股息成長篩選確保盈利品質，Morningstar 量化模型顯示長期風險調整報酬優異"},
]

# ──────────────────────────────────────────────────────────────────────────────
# 3. 評分引擎
# ──────────────────────────────────────────────────────────────────────────────
# Kept as a compatibility export.  The scoring engine rebuilds this mapping
# from the active whitelist on every call so directory changes are reflected
# immediately in the Macro page.
WHITELIST_MAP: Dict[str, Dict] = {k["id"]: k for k in WHITELIST}

def _recency_weight(date_str: str) -> float:
    """時效性加權：7天內=1.0，14天=0.6，30天=0.3，更舊=0.1"""
    try:
        pick_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_old  = (datetime.now() - pick_date).days
        if   days_old <= 7:  return 1.0
        elif days_old <= 14: return 0.6
        elif days_old <= 30: return 0.3
        else:                return 0.1
    except Exception:
        return 0.5


def _argument_weight(quality: int) -> float:
    """論點品質加權：3=完整論述, 2=中等, 1=薄弱/標題黨"""
    return {3: 1.0, 2: 0.5, 1: 0.1}.get(quality, 0.3)


def score_picks(
    picks: List[Dict] | None = None,
    whitelist: List[Dict] | None = None,
) -> List[Dict]:
    """
    多維評分並回傳排行榜。

    每筆 pick 分數 = 信譽分(rep/5) × 時效性加權 × 論點品質加權
    同一 ticker 的多筆合計，並記錄推薦專家清單。
    """
    if picks is None:
        picks = PICKS_DATA
    active_whitelist = whitelist if whitelist is not None else WHITELIST
    active_whitelist_map = {k["id"]: k for k in active_whitelist}

    ticker_scores: Dict[str, Dict] = {}
    for p in picks:
        kol = active_whitelist_map.get(p["kol_id"])
        if not kol:
            continue

        rep_score    = kol["rep"] / 5.0
        recency_w    = _recency_weight(p["date"])
        argument_w   = _argument_weight(p["argument_quality"])
        pick_score   = rep_score * recency_w * argument_w

        t = p["ticker"]
        if t not in ticker_scores:
            ticker_scores[t] = {
                "ticker":      t,
                "total_score": 0.0,
                "experts":     [],
                "theses":      [],
                "dates":       [],
                "consensus":   0,
            }
        ticker_scores[t]["total_score"] += pick_score
        ticker_scores[t]["consensus"]   += 1
        ticker_scores[t]["experts"].append(kol["name"])
        ticker_scores[t]["theses"].append(p["thesis"])
        ticker_scores[t]["dates"].append(p["date"])

    ranked = sorted(ticker_scores.values(), key=lambda x: x["total_score"], reverse=True)
    return ranked


def build_consensus_table(
    picks: List[Dict] | None = None,
    whitelist: List[Dict] | None = None,
) -> List[Dict]:
    """Build the Macro whitelist consensus table from the active directory."""
    return score_picks(picks=picks, whitelist=whitelist)


def _stars(score: float, max_score: float) -> Tuple[int, str]:
    """將分數轉為 1-5 星（整體最高分=5星，比例映射）"""
    if max_score == 0:
        return 1, "⭐"
    ratio = score / max_score
    if   ratio >= 0.85: stars = 5
    elif ratio >= 0.65: stars = 4
    elif ratio >= 0.45: stars = 3
    elif ratio >= 0.25: stars = 2
    else:               stars = 1
    return stars, "⭐" * stars


# ──────────────────────────────────────────────────────────────────────────────
# 4. Gemini AI 結構化共識報告
# ──────────────────────────────────────────────────────────────────────────────
def call_gemini_consensus(top_picks: List[Dict], api_key: str) -> List[Dict]:
    """
    對前 N 名共識股票呼叫 Gemini，輸出：
      - summary   : 一句話推薦摘要
      - confidence: 1-5 星信心評分 (整數)
      - reason    : AI 整合多方論點後的說明
    回傳 enhanced list (加上 ai_summary / ai_confidence / ai_reason)
    Uses 1-hour cache + exponential-backoff retry on 429/timeout.
    """
    from gemini_helper import call_gemini_cached, is_quota_error, is_auth_error

    results = []
    for pick in top_picks:
        theses_text = "\n".join(
            f"- [{pick['experts'][i]}]: {pick['theses'][i]}"
            for i in range(len(pick["experts"]))
        )
        prompt = f"""你是一位量化基本面分析師。以下是來自頂級投資機構與分析師對 {pick['ticker']} 的近期觀點：

{theses_text}

共有 {pick['consensus']} 位白名單專家推薦此標的（白名單目錄共 {len(WHITELIST)} 位）。

請輸出 JSON 格式（只回傳 JSON，不要其他文字）：
{{
  "summary": "一句話總結（繁體中文，15-30字，說明為何此標的受多位專家關注）",
  "confidence": <整數1到5，根據論點強烈程度與專家一致性評估信心>,
  "reason": "2-3句說明AI整合各方論點後的投資邏輯（繁體中文）"
}}"""
        try:
            raw  = call_gemini_cached(prompt, api_key)
            raw  = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            pick["ai_summary"]    = data.get("summary", "")
            pick["ai_confidence"] = int(data.get("confidence", 3))
            pick["ai_reason"]     = data.get("reason", "")
        except Exception as e:
            if is_quota_error(e):
                pick["ai_summary"]    = "⏳ AI 配額暫時達上限，請1小時後重試"
                pick["ai_confidence"] = pick.get("_star_n", 3)
                pick["ai_reason"]     = "系統將自動於下次刷新後重新生成分析。"
            elif is_auth_error(e):
                pick["ai_summary"]    = "🔑 API Key 錯誤，請更新 Replit Secrets"
                pick["ai_confidence"] = pick.get("_star_n", 3)
                pick["ai_reason"]     = "請確認 GEMINI_API_KEY 已正確設定。"
            else:
                pick["ai_summary"]    = "AI 分析暫時不可用"
                pick["ai_confidence"] = pick.get("_star_n", 3)
                pick["ai_reason"]     = f"（錯誤：{e}）"
        results.append(pick)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5. Streamlit 渲染入口
# ──────────────────────────────────────────────────────────────────────────────
def render_analyst_directory() -> None:
    """Render the searchable, style-filtered 20+ analyst directory."""
    st.markdown("### 🧭 精選美股分析師目錄")
    st.caption(
        f"共 {len(ANALYST_DIRECTORY)} 位公開研究者與機構。"
        "代表標的是研究主題或公開資料中的代表性標的，不代表即時持倉。"
    )
    search = st.text_input(
        "搜尋姓名、機構或代表 Ticker",
        placeholder="例如：Buffett、Macro、NVDA、BlackRock",
        key="analyst_directory_search",
    ).strip().lower()
    core_styles = ["全部", "Value", "Tech", "Macro", "Quant", "Income"]
    tabs = st.tabs(core_styles)

    for tab, style in zip(tabs, core_styles):
        with tab:
            matches = [
                analyst
                for analyst in ANALYST_DIRECTORY
                if (
                    style == "全部" or style in analyst.get("style_tags", [])
                )
                and (
                    not search
                    or search
                    in " ".join(
                        [
                            analyst["name"],
                            analyst["org"],
                            " ".join(analyst.get("style_tags", [])),
                            " ".join(analyst.get("representative_tickers", [])),
                            analyst.get("focus", ""),
                        ]
                    ).lower()
                )
            ]
            if not matches:
                st.info("找不到符合條件的分析師，請調整搜尋文字或分類。")
                continue
            st.caption(f"顯示 {len(matches)} 位")
            for start in range(0, len(matches), 3):
                cols = st.columns(3)
                for col, analyst in zip(cols, matches[start : start + 3]):
                    with col:
                        tags = " ".join(
                            f"`{tag}`" for tag in analyst.get("style_tags", [])
                        )
                        st.markdown(
                            f"**{analyst['name']}**  \n"
                            f"<span style='color:#8B949E'>{analyst['org']}</span>  \n"
                            f"{tags}  \n"
                            f"<span style='font-size:12px;color:#8B949E'>"
                            f"{analyst['focus']}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption("代表研究標的")
                        ticker_cols = st.columns(
                            len(analyst.get("representative_tickers", []))
                        )
                        for ticker_col, ticker in zip(
                            ticker_cols, analyst.get("representative_tickers", [])
                        ):
                            with ticker_col:
                                if st.button(
                                    ticker,
                                    key=f"analyst_diag_{style}_{analyst['id']}_{ticker}",
                                    use_container_width=True,
                                    help="帶入既有個股診斷與 7-Factor 分析",
                                ):
                                    navigate_to_ticker(ticker)
                st.markdown("---")


def render_kol_section(api_key: str = "") -> None:
    """主渲染函數，插入 Macro 頁面的 KOL 區塊。"""

    st.markdown("### 🧠 精選分析師白名單 (KOL Whitelist)")
    st.caption(
        "以下為手工篩選的高信譽分析師，須符合：①5年以上活躍記錄、"
        "②推薦包含結構化論點（非標題黨）、③機構或平台背書。"
        "觀點一致性、時效性、論點品質三維加權計分。"
    )
    # ── 合併用戶自定義 KOL ──────────────────────────────────────────────────────
    _user_handles = load_kol_whitelist()
    _user_kols: List[Dict] = []
    for _h in _user_handles:
        _user_kols.append({
            "id":           _h.lstrip("@").lower().replace(" ", "_"),
            "name":         _h,
            "org":          "用戶自定義",
            "focus":        "（未設定）",
            "platform":     _h,
            "stance":       "中性",
            "stance_color": "#8B949E",
            "rep":          3,
            "years_active": 0,
            "type":         "用戶手動加入",
            "rationale":    "由用戶透過白名單管理面板手動加入。",
        })
    _all_kols: List[Dict] = list(WHITELIST) + _user_kols

    with st.expander(
        f"📋 展開精選分析師白名單目錄（共 {len(WHITELIST)} 位）",
        expanded=False,
    ):
        render_analyst_directory()

    # ── 白名單卡片 ──
    with st.expander(
        f"查看白名單評分資料（共 {len(_all_kols)} 位）",
        expanded=False,
    ):
        for kol in _all_kols:
            star_str = "⭐" * kol["rep"]
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(
                    f"**{kol['name']}** &nbsp;·&nbsp; {kol['org']}  \n"
                    f"<span style='font-size:12px;color:#8B949E;'>{kol['type']}</span>  \n"
                    f"**研究方向：** {kol['focus']}  \n"
                    f"**平台：** {kol['platform']}  \n"
                    f"**入選理由：** {kol['rationale']}",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(
                    f"<div style='text-align:center; background:#1C2128; "
                    f"border:1px solid #30363D; border-radius:8px; padding:10px;'>"
                    f"<div style='font-size:11px; color:#8B949E;'>信譽評分</div>"
                    f"<div style='font-size:16px;'>{star_str}</div>"
                    f"<div style='font-size:11px; margin-top:4px;'>"
                    f"<span style='color:{kol['stance_color']}; font-weight:700;'>{kol['stance']}</span>"
                    f"</div>"
                    f"<div style='font-size:11px; color:#8B949E;'>活躍 {kol['years_active']}+ 年</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")

    st.markdown("### 📊 白名單共識選股分析")
    st.caption(
        "觀點一致性加權：多位專家同時推薦 → 分數提升 ｜ "
        "論點維度驗證：含財報/護城河/估值論述 → 高權重 ｜ "
        f"時效性加權：7天內=滿分，逾月遞減至 10% ｜ "
        f"目前依 {len(_all_kols)} 位白名單分析師動態計算"
    )

    # ── 評分計算 ──
    ranked = build_consensus_table(whitelist=_all_kols)
    if not ranked:
        st.warning("暫無可用的分析師觀點資料。")
        return

    max_score = ranked[0]["total_score"] if ranked else 1.0
    top_n     = min(10, len(ranked))
    top_picks = ranked[:top_n]
    for p in top_picks:
        n, s          = _stars(p["total_score"], max_score)
        p["_star_n"]  = n
        p["_star_str"] = s

    # ── 共識排行表 (無需 AI) ──
    _render_consensus_table(top_picks, max_score)

    # ── AI 深度報告 ──
    st.markdown("#### 🤖 Gemini AI 深度解析")
    if not api_key:
        st.info("💡 設定 GEMINI_API_KEY 後可啟用 AI 結構化報告（推薦摘要、信心評分、整合論點）。")
        return

    btn_col, _ = st.columns([2, 3])
    with btn_col:
        run_ai = st.button(
            "🚀 執行 AI 共識分析（Top 5）",
            type="primary",
            use_container_width=True,
            key="kol_run_ai",
        )

    if run_ai:
        with st.spinner("Gemini 正在整合各方觀點並評估信心指數…"):
            enhanced = call_gemini_consensus(top_picks[:5], api_key)
        st.session_state["kol_ai_result"] = enhanced
        st.session_state["kol_ai_done"]   = True
        st.rerun()

    if st.session_state.get("kol_ai_done"):
        enhanced = st.session_state.get("kol_ai_result", [])
        if enhanced:
            _render_ai_cards(enhanced)


def _render_consensus_table(picks: List[Dict], max_score: float) -> None:
    """渲染共識排行榜（靜態，無 Gemini）"""
    st.markdown("#### 🏆 白名單共識排行榜（評分加權）")

    header_html = """
    <div style='display:grid; grid-template-columns:60px 1fr 120px 80px 90px;
                gap:8px; background:#21262D; border-radius:6px;
                padding:8px 12px; font-weight:700; font-size:13px; margin-bottom:4px; color:#E6EDF3;'>
      <div>#</div><div>標的</div><div>推薦專家</div><div>一致性</div><div>信心</div>
    </div>"""
    st.markdown(header_html, unsafe_allow_html=True)

    for i, p in enumerate(picks, 1):
        experts_str  = "、".join(set(p["experts"]))
        consensus    = p["consensus"]
        star_str     = p["_star_str"]
        score_pct    = p["total_score"] / max_score * 100
        bar_color    = "#28A745" if score_pct >= 70 else ("#FFA500" if score_pct >= 40 else "#DC3545")

        row_html = f"""
        <div style='display:grid; grid-template-columns:60px 1fr 120px 80px 90px;
                    gap:8px; border-bottom:1px solid #30363D; color:#E6EDF3;
                    padding:8px 12px; font-size:13px; align-items:center;'>
          <div style='font-weight:700; color:#8B949E;'>#{i}</div>
          <div>
            <b style='font-size:15px; color:#E6EDF3;'>{p['ticker']}</b>
            <div style='height:6px; background:#30363D; border-radius:3px; margin-top:4px;'>
              <div style='height:6px; background:{bar_color};
                          border-radius:3px; width:{score_pct:.0f}%;'></div>
            </div>
          </div>
          <div style='font-size:11px; color:#8B949E;'>{experts_str[:30]}{"…" if len(experts_str)>30 else ""}</div>
          <div style='text-align:center; font-weight:700;'>{consensus} 位</div>
          <div style='text-align:center;'>{star_str}</div>
        </div>"""
        st.markdown(row_html, unsafe_allow_html=True)

    st.markdown("")

    # 展開詳細論點
    with st.expander("📝 查看各標的詳細論點", expanded=False):
        for p in picks:
            st.markdown(f"**{p['ticker']}** — 共 {p['consensus']} 位專家推薦")
            for j, (expert, thesis, date) in enumerate(
                zip(p["experts"], p["theses"], p["dates"])
            ):
                days_old = (datetime.now() - datetime.strptime(date, "%Y-%m-%d")).days
                freshness = "🟢" if days_old <= 7 else ("🟡" if days_old <= 14 else "🔴")
                st.markdown(
                    f"&nbsp;&nbsp;{freshness} **{expert}** ({date})：{thesis}",
                    unsafe_allow_html=False,
                )
            st.markdown("---")


def _render_ai_cards(enhanced: List[Dict]) -> None:
    """渲染 Gemini AI 輸出的結構化卡片"""
    for p in enhanced:
        conf      = p.get("ai_confidence", 3)
        conf_star = "⭐" * conf
        summary   = p.get("ai_summary", "")
        reason    = p.get("ai_reason", "")
        consensus = p["consensus"]
        experts   = "、".join(set(p["experts"][:3]))

        bg  = "#0D2E1A" if conf >= 4 else ("#2D2010" if conf >= 3 else "#2D1B1B")
        bdr = "#238636" if conf >= 4 else ("#D29922" if conf >= 3 else "#DA3633")
        title_col = "#3FB950" if conf >= 4 else ("#FFD700" if conf >= 3 else "#FF7B72")

        card_html = f"""
        <div style='background:{bg}; border:1px solid {bdr}; border-radius:10px;
                    padding:14px 18px; margin-bottom:12px;'>
          <div style='display:flex; justify-content:space-between; align-items:center;'>
            <b style='font-size:18px; color:{title_col};'>{p['ticker']}</b>
            <div style='font-size:16px;'>{conf_star} <span style='font-size:12px;color:#8B949E;'>信心 {conf}/5</span></div>
          </div>
          <div style='font-size:13px; font-weight:600; color:#E6EDF3; margin:8px 0 4px;'>📌 {summary}</div>
          <div style='font-size:12px; color:#8B949E; margin-bottom:8px;'>{reason}</div>
          <div style='display:flex; gap:16px; font-size:12px; color:#8B949E;'>
            <span>👥 一致性：<b style='color:#E6EDF3;'>{consensus} 位</b>白名單專家推薦</span>
            <span>🧑‍💼 包含：{experts}</span>
          </div>
        </div>"""
        st.markdown(card_html, unsafe_allow_html=True)

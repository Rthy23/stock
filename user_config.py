import json
import os

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_config.json")

SECTION_META = [
    {"key": "Comparison",   "label": "📅 歷史表現對比分析"},
    {"key": "FactorSystem", "label": "📊 7-Factor 多因子分析"},
    {"key": "AIReport",     "label": "🤖 Gemini AI 深度分析"},
]

_DEFAULT_ORDER = [s["key"] for s in SECTION_META]


def load_order() -> list:
    """
    Load the 個股診斷 section display order from user_config.json.
    Returns the default order if the file is missing or corrupt.
    Automatically creates the file on first call.
    """
    if not os.path.exists(_CONFIG_FILE):
        save_order(_DEFAULT_ORDER)
        return _DEFAULT_ORDER.copy()
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        order = cfg.get("diag_order", _DEFAULT_ORDER)
        known = {s["key"] for s in SECTION_META}
        order = [k for k in order if k in known]
        for s in SECTION_META:
            if s["key"] not in order:
                order.append(s["key"])
        return order
    except Exception:
        return _DEFAULT_ORDER.copy()


def save_order(order: list) -> None:
    """
    Persist the section display order to user_config.json.
    Merges with any existing keys so other config data is preserved.
    """
    try:
        existing = {}
        if os.path.exists(_CONFIG_FILE):
            try:
                with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing["diag_order"] = order
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[user_config] save_order error: {e}")


def get_section_labels() -> dict:
    """Return {key: label} mapping for all sections."""
    return {s["key"]: s["label"] for s in SECTION_META}

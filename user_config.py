"""
user_config.py — Centralised user preference management.

Schema (user_config.json):
{
  "module_order":      ["Comparison", "FactorSystem", "AIReport"],
  "watchlist":         ["AAPL", "TSLA", "NVDA"],
  "analyst_whitelist": ["@example_handle"]
}
"""
import json
import os

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_config.json")

SECTION_META = [
    {"key": "Comparison",   "label": "📅 歷史表現對比分析"},
    {"key": "FactorSystem", "label": "📊 7-Factor 多因子分析"},
    {"key": "AIReport",     "label": "🤖 Gemini AI 深度分析"},
]

_DEFAULT_CONFIG: dict = {
    "module_order":      [s["key"] for s in SECTION_META],
    "watchlist":         [],
    "analyst_whitelist": [],
}


# ── Low-level read / write ────────────────────────────────────────────────────

def load_config() -> dict:
    """
    Load the full config dict, migrating legacy keys and filling defaults.
    Never raises — always returns a usable dict.
    """
    cfg = {}
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

    # ── Migrate legacy key name ─────────────────────────────────────────────
    if "diag_order" in cfg and "module_order" not in cfg:
        cfg["module_order"] = cfg.pop("diag_order")

    # ── Fill defaults for any missing keys ──────────────────────────────────
    for k, v in _DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = v

    # ── Validate module_order: only known keys, all present ─────────────────
    known = {s["key"] for s in SECTION_META}
    order = [x for x in cfg["module_order"] if x in known]
    for s in SECTION_META:
        if s["key"] not in order:
            order.append(s["key"])
    cfg["module_order"] = order

    return cfg


def save_config(cfg: dict) -> None:
    """Atomically overwrite the config file."""
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[user_config] save_config error: {e}")


# ── Module order ──────────────────────────────────────────────────────────────

def load_order() -> list:
    return load_config()["module_order"]


def save_order(order: list) -> None:
    cfg = load_config()
    cfg["module_order"] = order
    save_config(cfg)


def get_section_labels() -> dict:
    return {s["key"]: s["label"] for s in SECTION_META}


# ── Watchlist ─────────────────────────────────────────────────────────────────

def load_watchlist_cfg() -> list:
    """Return the persisted watchlist from user_config.json."""
    return load_config().get("watchlist", [])


def save_watchlist_cfg(watchlist: list) -> None:
    """Persist a new watchlist to user_config.json."""
    cfg = load_config()
    cfg["watchlist"] = [t.upper().strip() for t in watchlist if t.strip()]
    save_config(cfg)


# ── KOL / Analyst whitelist ───────────────────────────────────────────────────

def load_kol_whitelist() -> list:
    """Return list of user-added analyst handles (e.g. ['@handle1', '@handle2'])."""
    return load_config().get("analyst_whitelist", [])


def save_kol_whitelist(handles: list) -> None:
    cfg = load_config()
    cfg["analyst_whitelist"] = handles
    save_config(cfg)


def add_kol(handle: str) -> tuple:
    """
    Validate handle format and add to the whitelist if not already present.
    Returns (success: bool, message: str).
    """
    handle = handle.strip()
    if not handle:
        return False, "請輸入帳號名稱。"
    if not handle.startswith("@"):
        handle = "@" + handle
    if len(handle) < 2:
        return False, "帳號格式錯誤，請包含 @ 符號。"

    current = load_kol_whitelist()
    if handle.lower() in [h.lower() for h in current]:
        return False, f"{handle} 已在白名單中。"

    current.append(handle)
    save_kol_whitelist(current)
    return True, f"✅ {handle} 已加入白名單。"


def remove_kol(handle: str) -> None:
    """Remove a handle from the KOL whitelist (case-insensitive)."""
    current = load_kol_whitelist()
    updated = [h for h in current if h.lower() != handle.lower()]
    save_kol_whitelist(updated)

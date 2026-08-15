"""Small, dependency-light navigation helpers shared by Streamlit modules."""

from __future__ import annotations

import streamlit as st


def navigate_to_ticker(ticker: str) -> None:
    """Select a ticker and open the shared stock-diagnosis flow."""
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        return
    st.session_state["selected_ticker"] = normalized
    st.session_state["diag_ticker"] = normalized
    st.session_state["diag_stock_info"] = None
    st.session_state["diag_hist"] = None
    st.session_state["auto_fetch"] = True
    st.session_state["nav_page"] = "🔬 個股診斷 (Micro)"
    st.rerun()
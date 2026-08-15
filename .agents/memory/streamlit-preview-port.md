---
name: Streamlit preview port
description: Replit webview port and health-check behavior for this Streamlit project.
---

The Replit webview workflow expects the Streamlit server on port 5000 and bound to
0.0.0.0; keep the workflow command, Streamlit config, and port mapping aligned.

**Why:** A healthy Streamlit process on another port can still produce a blank
preview or a failed workflow check even when localhost responds.

**How to apply:** When changing the entry point or server settings, verify the
workflow, `.streamlit/config.toml`, and `.replit` all use port 5000, then check
the workflow logs and an app preview.
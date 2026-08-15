---
name: Streamlit pages routing
description: How page modules behave when a project already has a custom Streamlit router.
---

When a Streamlit project contains a `pages/` directory, Streamlit automatically
adds those Python files to its multipage navigation. A project that also uses a
custom sidebar router must make page modules usable both when imported by the
main app and when executed directly by Streamlit's automatic page runner.

**Why:** Imported-only page modules appeared as blank pages when selected from
the automatically generated navigation.

**How to apply:** Keep rendering in an importable function, call it only under
`if __name__ == "__main__":` for direct page execution, and keep shared
navigation state in `st.session_state`.
---
name: Streamlit pages routing
description: How page modules behave when a project already has a custom Streamlit router.
---

When a Streamlit project contains a `pages/` directory, Streamlit automatically
adds those Python files to its multipage navigation. A project that also uses a
custom sidebar router must make page modules usable both when imported by the
main app and when executed directly by Streamlit's automatic page runner.
Shared navigation helpers should live in a small dependency-light module rather
than being imported from a large UI component module.

**Why:** Imported-only page modules appeared as blank pages when selected from
the automatically generated navigation, and cloud snapshots can expose an older
UI module before a newly-added helper is available.

**How to apply:** Keep rendering in an importable function, call it only under
`if __name__ == "__main__":` for direct page execution, put shared navigation
helpers in a minimal module, and keep state in `st.session_state`.
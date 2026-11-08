---
name: Prospective validation regimes
description: Keep future evidence comparable when changing scientific scoring definitions.
---

Changes to raw-input definitions, directional preferences, percentile calibration,
fixed weights or score eligibility require a new method identity. Keep prior
snapshots intact but do not pool their validation evidence with the new regime.

**Why:** Mixing scoring regimes produces an IC for no single deployed scoring
method and can make a new formula appear to have prospective evidence it never
earned. The user explicitly requires forward-only validation, not historical
reconstructed scores.

**How to apply:** Start the new regime with genuinely observed scores and new
cohorts; do not recompute past scores under a new formula. Statistical summaries
must remain clear about which regime produced their inputs.

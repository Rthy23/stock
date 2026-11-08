---
name: Live build verification
description: Distinguish published behavior, workspace tests, and managed-database metadata.
---

Treat live-site observations and workspace verification as separate evidence.
A successful publish status or a GitHub sync does not prove that the public app
is serving the code currently checked out here.

**Why:** A production investigation found older public navigation than the
workspace, despite deployment metadata reporting a successful build. Workspace
fixes and real-source tests alone would have overstated what had changed live.

**How to apply:** Obtain the published URL through deployment metadata and compare
actual public page content with the workspace before claiming a live fix. Report
workspace verification separately and make the republish requirement explicit.

Likewise, “no managed production database” does not establish that DATABASE_URL
is absent or unusable: an existing secret/custom connection may still exist.

**Why:** The database service reported no managed production DB while the secret
existence check reported DATABASE_URL present.

**How to apply:** Check existence without exposing values, distinguish registered
managed DB metadata from the application's real connection outcome, and never
claim a successful or failed production connection without runtime evidence.

---
name: Secret confirmation
description: A confirmed secret does not necessarily mean its value was replaced.
---

When a secure form reports that a secret was "added or confirmed," do not assume the user replaced an invalid existing value. Check only existence and application-level validity without displaying the value. If validity still fails, ask whether they re-entered the value or merely confirmed the old one; request an explicit overwrite through the secure form if needed.

**Why:** Confirmation of an existing invalid SEC identification value repeatedly left the application validation failing; an explicit overwrite made the official-data panel work.

**How to apply:** For future secret-configuration troubleshooting, distinguish presence from validity. Never copy secret values into chat, code, logs, or memory.
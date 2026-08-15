#!/bin/bash
# Post-merge setup script — runs automatically after every task merge.
set -e

echo "=== Post-merge setup ==="

# Install / sync Python dependencies if requirements.txt was changed.
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt
fi

# Quick syntax check on the core modules that tasks most commonly touch.
python -m py_compile \
    app.py \
    kol_whitelist.py \
    kol_config.py \
    picks_store.py 2>/dev/null || python -m py_compile app.py kol_whitelist.py kol_config.py

echo "=== Post-merge setup complete ==="

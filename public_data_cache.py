"""Last successful PUBLIC-source responses; no credentials or request headers stored.

The local cache survives reruns/restarts on the same filesystem, not republishes.
ARK's dated PostgreSQL snapshots remain its cross-publish persistence mechanism.
"""
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import tempfile
import time

import requests

CACHE_DIR = Path(".cache/public_sources")


def request_get(url, **kwargs):
    """Retry transient upstream failures only; do not hammer access-denied endpoints."""
    for attempt in range(3):
        try:
            response = requests.get(url, **kwargs)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                delay = response.headers.get("Retry-After", "")
                time.sleep(min(float(delay), 8) if delay.isdigit() else .5 * 2**attempt)
                continue
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(.5 * 2**attempt)


def retry_call(fetch):
    for attempt in range(3):
        try:
            return fetch()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(.5 * 2**attempt)


def _encode(value):
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    raise TypeError(f"Unsupported public-cache type: {type(value).__name__}")


def _decode(value):
    return Decimal(value["__decimal__"]) if set(value) == {"__decimal__"} else value


def read_success(key):
    try:
        path = CACHE_DIR / (sha256(key.encode()).hexdigest() + ".json")
        data = json.loads(path.read_text(), object_hook=_decode)
        if data.get("key") == key and data.get("data") and data.get("fetched_at"):
            return data
    except (OSError, ValueError, TypeError):
        pass
    return None


def write_success(key, data):
    envelope = {"key": key, "data": data,
                "fetched_at": datetime.now(timezone.utc).isoformat()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (sha256(key.encode()).hexdigest() + ".json")
    fd, temporary = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(envelope, file, default=_encode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return envelope


def last_success(key, fetch, ttl=900, force=False):
    """Never replace a good response with an error or an empty result."""
    previous = read_success(key)
    if previous and not force:
        age = (datetime.now(timezone.utc) -
               datetime.fromisoformat(previous["fetched_at"])).total_seconds()
        if age < ttl:
            return {**previous, "cache_hit": True, "stale": False, "error": None}
    try:
        data = fetch()
        if not data:
            raise ValueError("來源回傳空白資料")
        try:
            envelope = write_success(key, data)
            cache_error = None
        except OSError:
            envelope = {"key": key, "data": data,
                        "fetched_at": datetime.now(timezone.utc).isoformat()}
            cache_error = "本機快取無法寫入；仍顯示本次來源資料"
        return {**envelope, "cache_hit": False, "stale": False, "error": cache_error}
    except Exception as exc:
        logging.getLogger(__name__).warning("public_source_failed key=%s error_type=%s",
                                           key, type(exc).__name__)
        if previous:
            return {**previous, "cache_hit": True, "stale": True,
                    "error": f"{type(exc).__name__}：{exc}"}
        raise

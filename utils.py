"""Shared utilities."""

from __future__ import annotations

import os
import re

_SAFE_CHARS_RE = re.compile(r"[^0-9A-Za-z\u3400-\u9fff _.-]")


def sanitize_meeting_name(name: str, fallback: str) -> str:
    """Sanitize user-provided meeting name for safe file paths."""
    name = (name or "").strip()
    if not name:
        return fallback
    name = name.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    name = _SAFE_CHARS_RE.sub("", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._- ")
    if not name:
        return fallback
    return name[:60]

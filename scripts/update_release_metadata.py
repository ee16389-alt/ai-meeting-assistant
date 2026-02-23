#!/usr/bin/env python3
"""Update RELEASE.md metadata/artifacts from current build outputs."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "RELEASE.md"
DESKTOP = ROOT / "desktop"
DIST = DESKTOP / "dist"
PKG = DESKTOP / "package.json"

META_START = "<!-- RELEASE_METADATA_START -->"
META_END = "<!-- RELEASE_METADATA_END -->"
ART_START = "<!-- RELEASE_ARTIFACTS_START -->"
ART_END = "<!-- RELEASE_ARTIFACTS_END -->"


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024.0:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{size:.1f}P"


def _read_version() -> str:
    data = json.loads(PKG.read_text(encoding="utf-8"))
    return str(data.get("version", "0.0.0")).strip()


def _find_artifacts(version: str) -> list[str]:
    items = []
    dmg_name = f"AI Meeting Assistant-{version}-arm64.dmg"
    exe_name = f"AI Meeting Assistant Setup {version}.exe"

    dmg_path = DIST / dmg_name
    if dmg_path.exists():
        items.append(f"- DMG: `desktop/dist/{dmg_name}` ({_human_size(dmg_path.stat().st_size)})")
    else:
        items.append(f"- DMG: `desktop/dist/{dmg_name}` (missing)")

    exe_path = DIST / exe_name
    if exe_path.exists():
        items.append(f"- EXE: `desktop/dist/{exe_name}` ({_human_size(exe_path.stat().st_size)})")
    else:
        items.append(f"- EXE: `desktop/dist/{exe_name}` (missing)")

    return items


def _replace_block(text: str, start: str, end: str, lines: list[str]) -> str:
    if start not in text or end not in text:
        raise SystemExit(f"missing markers: {start} / {end}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    block = "\n".join(lines)
    return f"{before}{start}\n{block}\n{end}{after}"


def main() -> int:
    if not RELEASE.exists():
        raise SystemExit(f"missing {RELEASE}")
    if not PKG.exists():
        raise SystemExit(f"missing {PKG}")

    version = _read_version()
    today = date.today().isoformat()

    metadata_lines = [
        f"- Version: {version}",
        f"- Build date: {today}",
        "- Target: macOS arm64 DMG + Windows x64 EXE",
    ]
    artifacts_lines = _find_artifacts(version)

    content = RELEASE.read_text(encoding="utf-8")
    content = _replace_block(content, META_START, META_END, metadata_lines)
    content = _replace_block(content, ART_START, ART_END, artifacts_lines)
    RELEASE.write_text(content, encoding="utf-8")

    print("updated RELEASE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

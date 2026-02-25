#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def parse_model(model: str) -> tuple[str, str]:
    if ":" not in model:
        raise ValueError("model must be like 'name:tag', e.g. qwen2.5:1.5b")
    name, tag = model.split(":", 1)
    name = name.strip()
    tag = tag.strip()
    if not name or not tag:
        raise ValueError("invalid model name/tag")
    return name, tag


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy one Ollama model (manifest + blobs) into desktop/ollama-models for installer bundling."
    )
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Model tag, e.g. qwen2.5:1.5b")
    parser.add_argument(
        "--source",
        default=str(Path.home() / ".ollama" / "models"),
        help="Source Ollama models directory (default: ~/.ollama/models)",
    )
    parser.add_argument(
        "--dest",
        default="desktop/ollama-models",
        help="Destination directory inside project (default: desktop/ollama-models)",
    )
    args = parser.parse_args()

    name, tag = parse_model(args.model)
    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).resolve()

    manifest_src = source / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest_src.exists():
        raise SystemExit(f"manifest not found: {manifest_src}")

    manifest_dst = dest / "manifests" / "registry.ollama.ai" / "library" / name / tag
    copy_file(manifest_src, manifest_dst)

    with manifest_src.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    digests: set[str] = set()
    for field in ("config",):
        item = manifest.get(field)
        if isinstance(item, dict) and item.get("digest"):
            digests.add(item["digest"])
    for layer in manifest.get("layers", []):
        if isinstance(layer, dict) and layer.get("digest"):
            digests.add(layer["digest"])

    if not digests:
        raise SystemExit("no blob digest found in manifest")

    copied = 0
    total_size = 0
    for digest in sorted(digests):
        if not digest.startswith("sha256:"):
            continue
        blob_name = "sha256-" + digest.split(":", 1)[1]
        blob_src = source / "blobs" / blob_name
        if not blob_src.exists():
            raise SystemExit(f"blob not found: {blob_src}")
        blob_dst = dest / "blobs" / blob_name
        copy_file(blob_src, blob_dst)
        copied += 1
        total_size += blob_src.stat().st_size

    print(f"Model bundled: {args.model}")
    print(f"Manifest: {manifest_dst}")
    print(f"Blobs copied: {copied}")
    print(f"Total blob size: {total_size / (1024 * 1024):.2f} MB")
    print(f"Destination root: {dest}")


if __name__ == "__main__":
    main()

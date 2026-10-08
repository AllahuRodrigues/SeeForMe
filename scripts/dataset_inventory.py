"""Inventory locally acquired videos without downloading or assigning labels.

python scripts/dataset_inventory.py data/external --out data/external_inventory.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}


def inventory(root: Path):
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT):
        relative = path.relative_to(root).as_posix()
        yield {
            "source_dataset": relative.split("/")[0] if "/" in relative else "unassigned",
            "source_video_id": path.stem,
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "clip_id": hashlib.sha256(relative.encode("utf8")).hexdigest()[:16],
            "annotation_status": "unreviewed",
            "split": "unassigned",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"missing directory: {args.root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf8") as stream:
        for record in inventory(args.root):
            stream.write(json.dumps(record) + "\n")
            count += 1
    print(f"Indexed {count} videos; no files downloaded or labeled")


if __name__ == "__main__":
    main()

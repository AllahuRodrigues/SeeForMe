"""Inventory of local datasets. Catalogues what is on disk; it never downloads or labels anything.

    python scripts/dataset_inventory.py                      # summary of every known data root -> data/inventory.jsonl
    python scripts/dataset_inventory.py data/external --out data/external_inventory.jsonl   # one directory of videos

The second form writes one record per video (source dataset = first directory level, stable clip id, annotation
status "unreviewed", split "unassigned") and is what the human-review pipeline consumes.
"""
import argparse, hashlib, json, time
from pathlib import Path

try:
    from _common import ROOT
except ImportError:                      # imported as scripts.dataset_inventory (tests)
    ROOT = Path(__file__).resolve().parents[1]

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
ROOTS = {"egoblind_videos": ROOT / "data" / "egoblind" / "videos", "egoblind_cache": ROOT / "data" / "egoblind" / "cache",
         "egoblind_scenarios": ROOT / "data" / "egoblind" / "scenarios", "campus": ROOT / "data" / "campus",
         "campus_v2": ROOT / "data" / "campus_v2", "guidedog": ROOT / "data" / "guidedog", "dada": ROOT / "data" / "dada",
         "blv_road_nav": ROOT / "data" / "blv_road_nav", "mendeley_navassist": ROOT / "data" / "mendeley_navassist",
         "vizwiz_cls": ROOT / "data" / "vizwiz_cls", "timeli": ROOT / "data" / "timeli", "orbit": ROOT / "data" / "orbit",
         "drive_folder": ROOT / "data" / "drive_1unHMor", "ego4d": ROOT / "data" / "ego4d", "aria": ROOT / "data" / "aria",
         "jrdb": ROOT / "data" / "jrdb", "real_recordings": ROOT / "data" / "real", "external_code": ROOT / "external"}
EXT = {".mp4": "video", ".mov": "video", ".avi": "video", ".mkv": "video", ".webm": "video", ".jpg": "image", ".jpeg": "image",
       ".png": "image", ".npz": "cache", ".json": "json", ".parquet": "parquet", ".csv": "csv", ".xlsx": "xlsx", ".vrs": "aria_vrs"}


def inventory(root: Path):
    """One record per video file under root (sorted, stable clip ids, nothing reviewed or assigned yet)."""
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


def summarise(out: Path = ROOT / "data" / "inventory.jsonl"):
    """Size and file-type counts for every known data root; media rows go to out, totals to inventory_summary.json."""
    rows, summary = [], {}
    for name, root in ROOTS.items():
        if not root.exists():
            summary[name] = {"present": False}; continue
        counts, size = {}, 0
        for p in root.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                kind = EXT.get(p.suffix.lower(), "other"); counts[kind] = counts.get(kind, 0) + 1; size += p.stat().st_size
                if kind in ("video", "parquet", "aria_vrs"):
                    rows.append({"dataset": name, "path": str(p.relative_to(ROOT)), "kind": kind, "bytes": p.stat().st_size})
        summary[name] = {"present": True, "files": counts, "gb": round(size / 1e9, 2)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary["_generated"] = time.strftime("%Y-%m-%d %H:%M")
    (out.parent / "inventory_summary.json").write_text(json.dumps(summary, indent=1))
    for name, s in summary.items():
        if not name.startswith("_"):
            print(f"{name:20s} " + ("absent" if not s["present"] else f"{s['gb']:7.2f} GB  {s['files']}"))
    print(f"-> {out} ({len(rows)} media files), {out.parent / 'inventory_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", type=Path, help="directory of videos to index (omit for the summary of all known roots)")
    parser.add_argument("--out", type=Path, help="output jsonl (required with root)")
    args = parser.parse_args()
    if args.root is None:
        summarise(args.out or ROOT / "data" / "inventory.jsonl"); return
    if not args.root.is_dir():
        parser.error(f"missing directory: {args.root}")
    if args.out is None:
        parser.error("--out is required when a root directory is given")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf8") as stream:
        for record in inventory(args.root):
            stream.write(json.dumps(record) + "\n"); count += 1
    print(f"Indexed {count} videos; no files downloaded or labeled")


if __name__ == "__main__":
    main()

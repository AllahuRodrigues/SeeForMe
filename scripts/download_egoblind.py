"""Download the EgoBlind videos (Xiao et al., NeurIPS 2025 D&B track) into data/egoblind/.

    python scripts/download_egoblind.py --splits test:3,train:1      # two smallest splits first
    python scripts/download_egoblind.py --all                        # everything (~67 GB of zips)

Each split is a zip of mp4 files on the authors' Google Drive. Zips are downloaded one at a time
into data/egoblind/zips/, the mp4s are extracted (flattened) into data/egoblind/videos/<train|test>/,
a marker file is written, and the zip is deleted unless --keep-zips is given. The script refuses to
start a split when free disk would drop under --min-free-gb. Re-running skips finished splits.
The dataset is research-only; cite the paper and keep data/egoblind/meta/video_source.* with any
redistribution (see external/EgoBlind/README.md).
"""
import argparse, json, shutil, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "egoblind"

# Google Drive file ids, taken from the two folders linked in external/EgoBlind/README.md (2026-10-08).
TRAIN = ["1lAs3zwb6oCLoSJp_PXj5xpmkzjHSAK08", "1rapxFy46tMWOk0ao5F43wzsnvsNdofy3", "1wa8TeBl-NAVwIX0nBWOMubi6-zVi-A3c",
         "1TQ2k6b3verB7wO5DCyWn4mo3jEsJVXoU", "1MMprKf2HTM5nr_a2cf8eJWiJPwXcBN9g", "1zK9GGlMX2k8K2lPvUsmJTYxIRgk92ctf",
         "1g_AEGE-_H0GAmyyD-zNyOgPig2h3v_MN", "1QM-tL6t8uorzGS1VScNhiD0_kGDJplqN", "1D_6L8dGIe2LG6FYNEkpOXcqZnq-QuBch",
         "15_CzfqnLgKmPYIzPBQngwB9XF2m5cnSr"]
TEST = ["1OFLheaxq_OgBfgAioqIHaIJLN42o7VRT", "1fxISJQqhGjgpCF9Adh1n_pnimUu3CNE9", "199Qy_uSkVpJtyctoSRhNYmU10VUUnbzU",
        "1SGf1aiHHir7m_Xk8_a3crlAINNEaE_Zu", "1ptv_GhC-Pgo_hcC5nxxj3YCNqmaCVD88", "1SpTtG3V0Q4EfJQ4v6czyE7j7ijo3Rxol",
        "1qU95zRO2LqczgB-UReJGiilOJiOB67-q", "1VYhj0FXlV0Tni5rygdTycWZnJmcnZZz2", "19C4MuNtW9_MOWD3x55d8hDLQIQcyljPG",
        "1LxZdUMYPjgY_0lWHO5WbyLqsga-lCrA_"]
# approximate zip sizes in GB (from the Drive confirm pages), used only for the free-disk check
SIZE_GB = {"train": [2.9, 1.8, 2.2, 2.0, 3.4, 2.9, 2.8, 6.6, 2.8, 7.5], "test": [2.3, 2.8, 2.5, 1.7, 2.5, 2.8, 4.1, 3.6, 3.5, 6.5]}
META = {"video_source.csv": "1Wob42tJ95DL1VvKad7BPc5QIlHtiuttI"}   # list of original video sources (cite on redistribution)
IDS = {"train": TRAIN, "test": TEST}


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(DATA / "download.log", "a") as f:
        f.write(line + "\n")


def extract(zip_path: Path, out_dir: Path) -> int:
    """Flatten every .mp4 member into out_dir (by basename only, so a hostile archive cannot escape)."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name.lower().endswith(".mp4") or name.startswith("."):
                continue
            target = out_dir / name
            if target.exists() and target.stat().st_size == info.file_size:
                n += 1
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            n += 1
    return n


def fetch(split: str, idx: int, keep: bool, min_free: float) -> bool:
    import gdown
    out_dir = DATA / "videos" / split
    marker = out_dir / f".split_{idx}.done"
    if marker.exists():
        log(f"{split}/split_{idx}: already extracted ({marker.read_text().strip()})")
        return True
    need = SIZE_GB[split][idx] * (1.0 if keep else 2.0) + min_free
    if free_gb(DATA) < need:
        log(f"{split}/split_{idx}: SKIP, free disk {free_gb(DATA):.1f} GB < {need:.1f} GB needed")
        return False
    zip_path = DATA / "zips" / f"{split}_split_{idx}.zip"
    t0 = time.time()
    log(f"{split}/split_{idx}: downloading (~{SIZE_GB[split][idx]} GB)")
    try:
        got = gdown.download(id=IDS[split][idx], output=str(zip_path), quiet=True, resume=True)
    except Exception as e:   # gdown raises FileURLRetrievalError when Drive's per-file download quota is exhausted
        msg = str(e).replace("\n", " ").replace("\t", " ")
        short = "Drive quota exhausted (too many downloads of this file recently)" if "Too many users" in msg else msg[:200]
        log(f"{split}/split_{idx}: DOWNLOAD FAILED: {short}")
        return False
    if not got or not zip_path.exists():
        log(f"{split}/split_{idx}: DOWNLOAD FAILED (no file written)")
        return False
    log(f"{split}/split_{idx}: downloaded {zip_path.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s, extracting")
    try:
        n = extract(zip_path, out_dir)
    except zipfile.BadZipFile as e:
        log(f"{split}/split_{idx}: BAD ZIP ({e}); deleting partial file so the next run re-downloads")
        zip_path.unlink(missing_ok=True)
        return False
    marker.write_text(f"{n} videos, {time.strftime('%Y-%m-%d %H:%M')}")
    if not keep:
        zip_path.unlink()
    log(f"{split}/split_{idx}: done, {n} videos -> {out_dir} (free {free_gb(DATA):.1f} GB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="", help="comma list like test:3,train:1 (order is respected)")
    ap.add_argument("--all", action="store_true", help="all 20 splits, smallest first")
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--retry-every", type=float, default=0.0, help="minutes between passes over the splits that are still missing (0 = single pass)")
    ap.add_argument("--max-hours", type=float, default=12.0, help="give up retrying after this long")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    for sub in ("zips", "videos/train", "videos/test", "meta"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    import gdown
    for name, fid in META.items():
        target = DATA / "meta" / name
        if not target.exists():
            gdown.download(id=fid, output=str(target), quiet=True)
            log(f"meta: {name} ({target.stat().st_size} bytes)")
    for csv in ("train.csv",):
        src = ROOT / "external" / "EgoBlind" / "finetune" / csv
        if src.exists() and not (DATA / "meta" / csv).exists():
            shutil.copy(src, DATA / "meta" / csv)
    for csv in ("test.csv", "test_half_release.csv", "test_half_hold.csv"):
        src = ROOT / "external" / "EgoBlind" / csv
        if src.exists() and not (DATA / "meta" / csv).exists():
            shutil.copy(src, DATA / "meta" / csv)
    if args.meta_only:
        return 0
    jobs = []
    if args.splits:
        for item in args.splits.split(","):
            split, idx = item.strip().split(":")
            jobs.append((split, int(idx)))
    if args.all:
        rest = [(s, i) for s in ("train", "test") for i in range(10) if (s, i) not in jobs]
        jobs += sorted(rest, key=lambda j: SIZE_GB[j[0]][j[1]])
    if not jobs:
        ap.error("give --splits or --all")
    ok = 0
    t_start = time.time()
    pending = list(jobs)
    while True:
        done_now = []
        for split, idx in pending:
            if fetch(split, idx, args.keep_zips, args.min_free_gb):
                ok += 1; done_now.append((split, idx))
        pending = [j for j in pending if j not in done_now]
        if not pending or args.retry_every <= 0 or (time.time() - t_start) / 3600 > args.max_hours:
            break
        log(f"{len(pending)} splits still missing; next attempt in {args.retry_every:.0f} min")
        time.sleep(args.retry_every * 60)
    summary = {s: sorted(int(p.name.split("_")[1].split(".")[0]) for p in (DATA / "videos" / s).glob(".split_*.done")) for s in ("train", "test")}
    n_videos = {s: len(list((DATA / "videos" / s).glob("*.mp4"))) for s in ("train", "test")}
    log(f"finished {ok}/{len(jobs)} jobs; extracted splits {json.dumps(summary)}; videos {json.dumps(n_videos)}; free {free_gb(DATA):.1f} GB")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())

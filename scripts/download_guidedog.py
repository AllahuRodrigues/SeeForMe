"""Download the GuideDog dataset (kjunh/GuideDog, ACL 2026; CC BY-NC 4.0, gated with auto-approval) into data/guidedog/.

The gate must be accepted once with your Hugging Face account:
  1. open https://huggingface.co/datasets/kjunh/GuideDog while logged in and click "Agree and access repository"
  2. run  .venv/bin/hf auth login   (or export HF_TOKEN=hf_...; `huggingface-cli` no longer works with huggingface_hub 1.x)
  3. python scripts/download_guidedog.py            # ~5.1 GB of parquet: gold (2,106), silver (19,978), depth (383), object (435)
     python scripts/download_guidedog.py --configs gold,depth,object    # skip the 4.5 GB silver split
"""
import argparse, sys
from pathlib import Path
from _common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="gold,depth,object,silver")
ap.add_argument("--out", default=str(ROOT / "data" / "guidedog"))
args = ap.parse_args()
try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
except ImportError:
    sys.exit("pip install huggingface_hub")
want = set(args.configs.split(","))
patterns = ["README.md"] + [f"data/{c}-*.parquet" if c in ("gold", "silver") else f"{c}/*.parquet" for c in want]
try:
    path = snapshot_download("kjunh/GuideDog", repo_type="dataset", local_dir=args.out, allow_patterns=patterns)
except (GatedRepoError, HfHubHTTPError) as e:
    sys.exit(f"GuideDog is gated: accept the conditions at https://huggingface.co/datasets/kjunh/GuideDog with your account, "
             f"then `.venv/bin/hf auth login` (or set HF_TOKEN). ({type(e).__name__}: {str(e)[:120]})")
files = sorted(Path(path).rglob("*.parquet"))
print(f"{len(files)} parquet files in {path}:")
for f in files:
    print(f"  {f.relative_to(path)}  {f.stat().st_size / 1e6:.0f} MB")

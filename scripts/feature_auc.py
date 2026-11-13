"""Feature gate: does a new feature help separate should-speak candidates from the rest, across videos?

    python scripts/feature_auc.py --tag looming          # fit on `train`, score on `finetune` (development only)

Every legal candidate at every decision on the dev stream is labelled as in the full-information learner:
1 inside its event's window before the event is announced, 0 near a labelled question otherwise (rows with no
label are dropped). A standardised logistic regression is fitted on `train` and scored on `finetune` with the base
29 P-CACS features and with base + each candidate feature group. The paired bootstrap resamples `finetune` videos.
A group passes the gate if it adds at least +0.02 AUC with a 95% interval above 0. Writes results/features/<tag>.{json,md}.
"""
import argparse, math, random
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, source_recordings
from seeforme.online.policy import OnlinePolicy
from seeforme.online.stream import run_stream

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="looming")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seg", action="store_true", help="join per-second segmentation shares from data/egoblind_seg/")
args = ap.parse_args()


def tau_feats(tau):
    """Bounded encodings of a time to contact (inf = no approach)."""
    return [1.0 if math.isfinite(tau) else 0.0, 1.0 / (1.0 + tau) if math.isfinite(tau) else 0.0, float(math.isfinite(tau) and tau < 3.0)]


GROUPS = {
    "looming_track": ["has_loom", "inv_loom", "loom_lt3"],
    "looming_decision": ["dec_has_loom", "dec_max_inv_loom", "dec_n_loom_lt3"],
}

rows = []


class Dump(OnlinePolicy):
    def _learn_full_info(self, obs, X, t):
        sc, ann = self._scenario, self._feedback.announced
        ev = {e.obj_id: e for e in sc.speak_events()}
        taus = [o.track.looming_ttc() for o in obs]
        tf = [tau_feats(x) for x in taus]
        dec = [float(any(f[0] for f in tf)), max(f[1] for f in tf), float(sum(f[2] for f in tf))]
        for o, tau_f in zip(obs, tf):
            sid = o.track.source_id
            e = ev.get(sid) if sid is not None else None
            if e is not None and e.in_window(t) and sid not in ann:
                y = 1
            elif sc.counts_fp(t):
                y = 0
            else:
                continue
            rows.append({"split": sc.split, "video": sc.video_id, "t": t, "y": y,
                         "base": np.asarray(o.event.as_vector(), float), "extra": np.array(tau_f + dec, float)})


cfg = load_campus_cfg()
src = source_recordings()
scen = []
for s in ("train", "finetune"):
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    scen += part
if args.limit:
    scen = scen[:args.limit]
pol = Dump(mode="full_info", lr=0.0, seed=0)
for _ in run_stream(pol, scen, cfg, p_tap=0.5, seed=0):
    pass

# causal scene-segmentation features (scripts/cache_segmentation.py): the last sample at or before t, and the
# maximum over the previous 3 s, per group. Only past frames are used.
SEG_DIR = ROOT / "data" / "egoblind_seg"
seg_names = []
if args.seg and SEG_DIR.exists():
    seg = {}
    for r in rows:
        v = r["video"]
        if v not in seg:
            f = SEG_DIR / f"{v}.npz"
            seg[v] = np.load(f) if f.exists() else None
        z = seg[v]
        if z is None or len(z["t"]) == 0:
            r["seg"] = None
            continue
        grp = [str(g) for g in z["groups"]]
        i = int(np.searchsorted(z["t"], r["t"] + 1e-6)) - 1
        if i < 0:
            r["seg"] = np.zeros(2 * len(grp)); continue
        j0 = int(np.searchsorted(z["t"], r["t"] - 3.0))
        r["seg"] = np.concatenate([z["share"][i], z["share"][j0:i + 1].max(0)])
        if not seg_names:
            seg_names = [f"seg_{g}" for g in grp] + [f"seg_{g}_max3s" for g in grp]
    n_missing = sum(r.get("seg") is None for r in rows)
    rows = [r for r in rows if r.get("seg") is not None]
    for r in rows:
        r["extra"] = np.concatenate([r["extra"], r["seg"]])
    GROUPS["segmentation"] = seg_names
    GROUPS["segmentation_architecture"] = [n for n in seg_names if any(k in n for k in ("stairs", "door", "railing", "walkable"))]
    print(f"segmentation features joined ({n_missing} rows without a cache dropped from every model)", flush=True)

tr = [r for r in rows if r["split"] == "train"]
te = [r for r in rows if r["split"] == "finetune"]
extra_names = GROUPS["looming_track"] + GROUPS["looming_decision"] + seg_names
print(f"rows train {len(tr)} (pos {sum(r['y'] for r in tr)}), finetune {len(te)} (pos {sum(r['y'] for r in te)})", flush=True)


def matrix(rs, cols):
    X = np.array([r["base"] for r in rs])
    if cols:
        E = np.array([r["extra"] for r in rs])
        X = np.hstack([X, E[:, [extra_names.index(c) for c in cols]]])
    return X


def fit_score(cols):
    Xt, Xe = matrix(tr, cols), matrix(te, cols)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    m = LogisticRegression(max_iter=5000, C=1.0).fit((Xt - mu) / sd, [r["y"] for r in tr])
    return m.predict_proba((Xe - mu) / sd)[:, 1], m.predict_proba((Xt - mu) / sd)[:, 1]


ye = np.array([r["y"] for r in te]); yt = np.array([r["y"] for r in tr])
by_vid = defaultdict(list)
for i, r in enumerate(te):
    by_vid[r["video"]].append(i)
vids = sorted(by_vid)


def boot_auc_diff(sa, sb, n, seed=0):
    rng = random.Random(seed); out = []
    for _ in range(n):
        idx = [i for v in (rng.choice(vids) for _ in vids) for i in by_vid[v]]
        y = ye[idx]
        if y.min() == y.max():
            continue
        out.append(roc_auc_score(y, sa[idx]) - roc_auc_score(y, sb[idx]))
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


base_e, base_t = fit_score([])
res = {"n_train": len(tr), "n_finetune": len(te), "pos_train": int(yt.sum()), "pos_finetune": int(ye.sum()),
       "base": {"auc_train": roc_auc_score(yt, base_t), "auc_finetune": roc_auc_score(ye, base_e)}, "groups": {}}
single = {}
E = np.array([r["extra"] for r in te])
for j, nme in enumerate(extra_names):
    if E[:, j].std() > 0:
        single[nme] = roc_auc_score(ye, E[:, j])
res["single_feature_auc_finetune"] = single
for gname, cols in list(GROUPS.items()) + [("looming_all", GROUPS["looming_track"] + GROUPS["looming_decision"])]:
    se, st = fit_score(cols)
    lo, hi = boot_auc_diff(se, base_e, args.boot)
    d = roc_auc_score(ye, se) - res["base"]["auc_finetune"]
    res["groups"][gname] = {"auc_train": roc_auc_score(yt, st), "auc_finetune": roc_auc_score(ye, se), "delta": d,
                            "ci": [lo, hi], "passes_gate": bool(d >= 0.02 and lo > 0)}

out = ROOT / "results" / "features"
out.mkdir(parents=True, exist_ok=True)
dump_json(res, out / f"{args.tag}.json")
lines = [f"Feature gate ({args.tag}): logistic fit on train ({len(tr)} rows, {int(yt.sum())} pos), scored on finetune "
         f"({len(te)} rows, {int(ye.sum())} pos)\n", "| features | AUC train | AUC finetune | delta vs base [95% CI] | gate |", "|---|---|---|---|---|",
         f"| base 29 | {res['base']['auc_train']:.3f} | {res['base']['auc_finetune']:.3f} | | |"]
for g, v in res["groups"].items():
    lines.append(f"| base + {g} | {v['auc_train']:.3f} | {v['auc_finetune']:.3f} | {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | "
                 f"{'pass' if v['passes_gate'] else 'fail'} |")
lines += ["", "Single-feature AUC on finetune: " + ", ".join(f"{k} {v:.3f}" for k, v in single.items())]
(out / f"{args.tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

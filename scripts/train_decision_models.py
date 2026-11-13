"""Controlled "speak or stay silent" model comparison on the development splits (train fits, finetune compares).

    OMP_NUM_THREADS=2 python scripts/train_decision_models.py            # dataset, offline fits, replay at equal speech

1. Dataset: replay `train` + `finetune` with a full-information Dump policy (lr 0), as in feature_auc.py. Every
   confirmed candidate at every decision keeps its raw 29-dim P-CACS vector, video, split, t, track id and label
   (1 inside its event window before announcement, 0 near a labelled moment otherwise, -1 unlabelled). Unlabelled
   rows are kept only as history for the causal K-step windows and are never fitted or scored.
2. Offline: logistic (last step), class-weighted logistic + Platt, MLP (last step), GRU and TCN over K = 2, 4, 8, 16
   steps. All fitted on `train`. Hyperparameters, early stopping, K and calibration are chosen on `finetune`, which is
   therefore a development comparison set, not a held-out test.
3. Replay: FrozenScorerPolicy (no learning) with the online BudgetController at 2 cues/min, streamed over
   `train` + `finetune` exactly like online greedy (the `train` part only lets the controller settle; it is in-sample
   for the models and is not reported). Metrics on the 104 `finetune` videos, seeds 0, 1, 2.

`select`, `holdout` and `test` are never loaded. Writes results/decision_models/ and models/decision_models/.
"""
import argparse, csv, json, math, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, source_recordings
from seeforme.campus.metrics import evaluate_run
from seeforme.egoblind.metrics import egoblind_extras
from seeforme.online.decision_models import (DecisionModel, FrozenScorerPolicy, Standardizer, causal_windows, make_net,
                                             step_of)
from seeforme.online.policy import OnlinePolicy
from seeforme.online.stream import run_stream

ALLOWED = ("train", "finetune")
OUT = ROOT / "results" / "decision_models"
MOD = ROOT / "models" / "decision_models"
KS = (2, 4, 8, 16)
KMAX = max(KS)

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="", help="npz cache of the candidate table (default: results/decision_models/.cache.npz)")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--skip-replay", action="store_true")
args = ap.parse_args()
torch.set_num_threads(2)
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

cfg = load_campus_cfg()
src = source_recordings()
scen = {}
for s in ALLOWED:
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    scen[s] = part[:args.limit] if args.limit else part
stream = scen["train"] + scen["finetune"]
assert {sc.split for sc in stream} <= set(ALLOWED)

# ------------------------------------------------------------------------------------------------ 1. dataset
cache = Path(args.cache) if args.cache else OUT / ".cache.npz"
if cache.exists() and not args.limit:
    z = np.load(cache, allow_pickle=False)
    X, y, vid, trk, stp, tt, split = (z[k] for k in ("X", "y", "vid", "trk", "stp", "t", "split"))
    print(f"loaded candidate table {cache} ({len(y)} rows)", flush=True)
else:
    rec = defaultdict(list)

    class Dump(OnlinePolicy):
        def _learn_full_info(self, obs, Xn, t):
            sc, ann = self._scenario, self._feedback.announced
            ev = {e.obj_id: e for e in sc.speak_events()}
            fp = sc.counts_fp(t)
            for o in obs:
                sid = o.track.source_id
                e = ev.get(sid) if sid is not None else None
                lab = 1 if (e is not None and e.in_window(t) and sid not in ann) else (0 if fp else -1)
                rec["X"].append(np.asarray(o.event.as_vector(), np.float32)); rec["y"].append(lab)
                rec["vid"].append(sc.video_id); rec["trk"].append(o.track.track_id)
                rec["stp"].append(step_of(t, self.decision_interval)); rec["t"].append(t); rec["split"].append(sc.split)

    t0 = time.time()
    pol = Dump(mode="full_info", lr=0.0, seed=0)
    for _ in run_stream(pol, stream, cfg, p_tap=0.5, seed=0):
        pass
    X = np.stack(rec["X"]); y = np.array(rec["y"], np.int8); vid = np.array(rec["vid"], np.int64)
    trk = np.array(rec["trk"], np.int64); stp = np.array(rec["stp"], np.int64); tt = np.array(rec["t"], np.float64)
    split = np.array(rec["split"])
    if not args.limit:
        np.savez_compressed(cache, X=X, y=y, vid=vid, trk=trk, stp=stp, t=tt, split=split)
    print(f"dumped {len(y)} candidate rows in {time.time() - t0:.0f}s", flush=True)

assert set(np.unique(split)) <= set(ALLOWED)
W16, M16 = causal_windows(vid.tolist(), trk.tolist(), stp.tolist(), X, KMAX)
lab = y >= 0
is_tr, is_ft = (split == "train") & lab, (split == "finetune") & lab
ytr, yft = y[is_tr].astype(int), y[is_ft].astype(int)
std = Standardizer.fit(X[is_tr])
ft_vid = vid[is_ft]

summary = {"splits_used": list(ALLOWED), "n_videos": {s: len(scen[s]) for s in ALLOWED},
           "minutes": {s: sum(sc.duration for sc in scen[s]) / 60 for s in ALLOWED},
           "rows_all": int(len(y)), "rows_unlabelled_history_only": int((y < 0).sum()),
           "rows": {s: int(((split == s) & lab).sum()) for s in ALLOWED},
           "positives": {s: int(((split == s) & (y == 1)).sum()) for s in ALLOWED},
           "videos_with_labelled_rows": {s: int(len(np.unique(vid[(split == s) & lab]))) for s in ALLOWED},
           "decision_interval_s": 0.5, "K_steps": list(KS), "K_seconds": [k * 0.5 for k in KS],
           "history_coverage_mean_slots": {f"K{k}": float(M16[lab][:, -k:].sum(1).mean()) for k in KS},
           "label_rule": "1 inside the event window before announcement; 0 where counts_fp(t); unlabelled dropped (kept as history)",
           "standardisation": "mean/sd of labelled train rows, clipped to +-5, padded slots 0"}
dump_json(summary, OUT / "dataset_summary.json")
print(json.dumps(summary, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ 2. offline
def ece(yv, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - yv[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def metrics(p_ft, p_tr):
    return {"auc_finetune": roc_auc_score(yft, p_ft), "ap_finetune": average_precision_score(yft, p_ft),
            "brier_finetune": brier_score_loss(yft, p_ft), "ece_finetune": ece(yft, p_ft),
            "auc_train": roc_auc_score(ytr, p_tr), "base_rate_finetune": float(yft.mean())}


def windows(mask, K):
    return W16[mask][:, -K:], M16[mask][:, -K:]


def fit_logistic(weighted):
    best = None
    Ztr = std(X[is_tr][:, None], np.ones((is_tr.sum(), 1), np.float32))[:, 0]
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        lr = LogisticRegression(max_iter=5000, C=C, class_weight="balanced" if weighted else None).fit(Ztr, ytr)
        m = DecisionModel("logistic", 1, std, coef=lr.coef_[0], intercept=float(lr.intercept_[0]), meta={"C": C, "class_weight": "balanced" if weighted else None})
        a = roc_auc_score(yft, m.predict_proba(*windows(is_ft, 1)))
        if best is None or a > best[0]:
            best = (a, m)
    return best[1]


def platt_fit(z, yv):
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(z.reshape(-1, 1), yv)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_net(family, K, wd, seed=0, dropout=0.2):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_net(family, dropout=dropout)
    Wt, Mt = windows(is_tr, K)
    Zt = torch.from_numpy(std(Wt, Mt)); Mt = torch.from_numpy(Mt); Yt = torch.from_numpy(ytr.astype(np.float32))
    Wf, Mf = windows(is_ft, K)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad, curve = -1.0, None, 0, 0, []
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Yt), generator=g)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(net(Zt[b], Mt[b]), Yt[b])
            loss.backward(); opt.step()
        m = DecisionModel(family, K, std, net=net, net_kw={"dropout": dropout})   # sets eval mode
        a = roc_auc_score(yft, m.predict_proba(Wf, Mf))
        curve.append(round(a, 4))
        if a > best:
            best, best_ep, bad = a, ep + 1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    return DecisionModel(family, K, std, net=net, net_kw={"dropout": dropout},
                         meta={"weight_decay": wd, "dropout": dropout, "best_epoch": best_ep, "epochs_run": len(curve),
                               "finetune_auc_curve": curve, "seed": seed})


# per-decision inference benchmark: real finetune decisions, all their confirmed candidates in one batch
dec_groups = defaultdict(list)
for i in np.where(split == "finetune")[0]:
    dec_groups[(int(vid[i]), int(stp[i]))].append(i)
bench = random.Random(0).sample(sorted(dec_groups), min(500, len(dec_groups)))


def time_per_decision(m):
    batches = [np.array(dec_groups[k]) for k in bench]
    for b in batches[:20]:
        m.predict_proba(W16[b][:, -m.K:], M16[b][:, -m.K:])
    t0 = time.perf_counter()
    for b in batches:
        m.predict_proba(W16[b][:, -m.K:], M16[b][:, -m.K:])
    return 1000 * (time.perf_counter() - t0) / len(batches)


def evaluate(name, m, extra=None):
    p_ft = m.predict_proba(*windows(is_ft, m.K)); p_tr = m.predict_proba(*windows(is_tr, m.K))
    r = {"family": m.family, "K": m.K, "K_seconds": m.K * 0.5, **metrics(p_ft, p_tr), "n_params": m.n_params(),
         "ms_per_decision": time_per_decision(m), "hyper": m.meta, **(extra or {})}
    print(f"{name:22s} AUC ft {r['auc_finetune']:.3f} tr {r['auc_train']:.3f} AP {r['ap_finetune']:.3f} "
          f"Brier {r['brier_finetune']:.4f} ECE {r['ece_finetune']:.4f} params {r['n_params']} {r['ms_per_decision']:.3f} ms", flush=True)
    return r


offline, models = {}, {}
t0 = time.time()
models["logistic"] = fit_logistic(False)
offline["logistic"] = evaluate("logistic", models["logistic"])

cw = fit_logistic(True)
offline["logistic_cw_raw"] = evaluate("logistic_cw (raw)", cw)
# Platt on finetune. Reported calibration metrics are cross-fitted over two halves of the finetune videos (each half
# scored with a calibrator fitted on the other half); the saved model uses a calibrator fitted on all of finetune.
z_ft = cw.logit(*windows(is_ft, 1))
vids_ft = sorted(set(ft_vid.tolist()))
half = set(random.Random(0).sample(vids_ft, len(vids_ft) // 2))
in_a = np.array([v in half for v in ft_vid])
p_cf = np.zeros(len(yft))
for fit_m, app_m in ((in_a, ~in_a), (~in_a, in_a)):
    a_, b_ = platt_fit(z_ft[fit_m], yft[fit_m])
    p_cf[app_m] = 1 / (1 + np.exp(-(a_ * z_ft[app_m] + b_)))
cwp = DecisionModel("logistic", 1, std, coef=cw.coef, intercept=cw.intercept, platt=platt_fit(z_ft, yft), meta={**cw.meta, "platt": "fitted on finetune"})
models["logistic_cw_platt"] = cwp
r = evaluate("logistic_cw_platt", cwp)
r.update({"brier_finetune_crossfit": brier_score_loss(yft, p_cf), "ece_finetune_crossfit": ece(yft, p_cf),
          "note": "Platt fitted on finetune: brier/ece_finetune are in-sample for the calibrator; *_crossfit use 2-fold video cross-fitting"})
offline["logistic_cw_platt"] = r

best_wd = {}
for fam, Ks in (("mlp", (1,)), ("gru", KS), ("tcn", KS)):
    for K in Ks:
        cands = [fit_net(fam, K, wd) for wd in (1e-4, 1e-3)]
        aucs = [roc_auc_score(yft, c.predict_proba(*windows(is_ft, K))) for c in cands]
        m = cands[int(np.argmax(aucs))]
        nm = fam if fam == "mlp" else f"{fam}_k{K}"
        models[nm] = m
        offline[nm] = evaluate(nm, m, {"finetune_auc_per_weight_decay": dict(zip(("1e-4", "1e-3"), [round(a, 4) for a in aucs]))})
print(f"offline fits {time.time() - t0:.0f}s", flush=True)

best_k = {fam: max((k for k in offline if k.startswith(fam + "_k")), key=lambda k: offline[k]["auc_finetune"]) for fam in ("gru", "tcn")}
for nm, m in models.items():
    m.save(MOD / f"{nm}.pt")
sizes = {f.name: f.stat().st_size for f in MOD.glob("*.pt")}
dump_json({"models": offline, "best_K_by_finetune_auc": best_k, "model_file_bytes": sizes}, OUT / "offline.json")

# paired video bootstrap of finetune AUC against logistic
by_v = defaultdict(list)
for i, v in enumerate(ft_vid):
    by_v[int(v)].append(i)
rng = random.Random(0)
boots = [[i for v in (rng.choice(vids_ft) for _ in vids_ft) for i in by_v[v]] for _ in range(1000)]
p_log = models["logistic"].predict_proba(*windows(is_ft, 1))
for nm, m in models.items():
    if nm == "logistic":
        continue
    p = m.predict_proba(*windows(is_ft, m.K))
    d = sorted(roc_auc_score(yft[b], p[b]) - roc_auc_score(yft[b], p_log[b]) for b in boots if 0 < yft[b].sum() < len(b))
    offline[nm]["auc_delta_vs_logistic_ci95"] = [d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]]
dump_json({"models": offline, "best_K_by_finetune_auc": best_k, "model_file_bytes": sizes}, OUT / "offline.json")

L = ["# Offline comparison: fitted on `train`, selected and scored on `finetune`", "",
     f"Rows: train {summary['rows']['train']} ({summary['positives']['train']} positive), finetune {summary['rows']['finetune']} "
     f"({summary['positives']['finetune']} positive, base rate {yft.mean():.3f}). Finetune chose every hyperparameter "
     "(logistic C, network weight decay, early-stopping epoch, K, Platt calibration), so these finetune numbers are a "
     "development comparison, not a held-out estimate. One training seed per network.", "",
     "| model | K (s) | AUC finetune | AUC delta vs logistic [95% CI] | AP | Brier | ECE | AUC train | params | ms / decision |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for nm, r in offline.items():
    ci = r.get("auc_delta_vs_logistic_ci95")
    dl = f"{r['auc_finetune'] - offline['logistic']['auc_finetune']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
    br = f"{r['brier_finetune']:.4f}" + (f" ({r['brier_finetune_crossfit']:.4f} cf)" if "brier_finetune_crossfit" in r else "")
    ec = f"{r['ece_finetune']:.4f}" + (f" ({r['ece_finetune_crossfit']:.4f} cf)" if "ece_finetune_crossfit" in r else "")
    L.append(f"| {nm} | {r['K']} ({r['K_seconds']:g}) | {r['auc_finetune']:.3f} | {dl} | {r['ap_finetune']:.3f} | {br} | {ec} | "
             f"{r['auc_train']:.3f} | {r['n_params']} | {r['ms_per_decision']:.3f} |")
L += ["", f"Best K by finetune AUC: GRU {best_k['gru']}, TCN {best_k['tcn']}. ECE: 15 equal-width bins. 'cf' = Platt "
      "cross-fitted over two halves of the finetune videos. ms / decision: all confirmed candidates of one real "
      "finetune decision scored in one batch, mean over 500 decisions, 2 CPU threads.",
      "logistic_cw_raw is the class-weighted fit before calibration (its probabilities are inflated by the weighting)."]
(OUT / "offline.md").write_text("\n".join(L) + "\n")
print("\n".join(L), flush=True)
if args.skip_replay:
    raise SystemExit

# ------------------------------------------------------------------------------------------------ 3. replay
REPLAY = ["logistic", "logistic_cw_platt", "mlp", best_k["gru"], best_k["tcn"]]
seeds = [int(s) for s in args.seeds.split(",")]
dur = {sc.video_id: sc.duration for sc in stream}
rows = []


def row_for(name, seed, sc, log):
    r = evaluate_run(log, sc).as_dict()
    r.update(egoblind_extras(log, sc))
    r.update({"policy": name, "seed": seed, "video_id": sc.video_id, "split": sc.split})
    return r


replay_meta = {}
for seed in seeds:
    for nm in REPLAY + ["online_greedy"]:
        t0 = time.time()
        pol = (OnlinePolicy(mode="greedy", target_per_min=2.0, tau0=0.6, lr=0.5, seed=seed) if nm == "online_greedy"
               else FrozenScorerPolicy(models[nm], target_per_min=2.0, tau0=0.6, seed=seed, name=nm))
        for sc, log in run_stream(pol, stream, cfg, p_tap=0.5, seed=seed):
            if sc.split == "finetune":
                rows.append(row_for(nm, seed, sc, log))
        replay_meta[f"{nm}_s{seed}"] = {**({} if nm == "online_greedy" else pol.to_dict()), "tau_end": pol.tau, "seconds": time.time() - t0}
        print(f"replay {nm} seed {seed}: {time.time() - t0:.0f}s tau_end {pol.tau:.3f}", flush=True)

with open(OUT / "replay_per_video.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore"); w.writeheader(); w.writerows(rows)

g = lambda r, k: float(r[k]) if r.get(k) is not None else 0.0
tab = defaultdict(dict)                                      # policy -> (seed, video) -> row
for r in rows:
    tab[r["policy"]][(r["seed"], r["video_id"])] = r
ft_ids = sorted({r["video_id"] for r in rows})


def pooled(pol, keys):
    t = defaultdict(float)
    for k in keys:
        r = tab[pol][k]
        ne, nc, nq, ns = g(r, "n_events"), g(r, "n_coverable"), g(r, "n_quiet"), g(r, "n_spoken")
        t["ev"] += ne; t["ev_hit"] += round(g(r, "important_cue_recall") * ne)
        t["cov"] += nc; t["cov_hit"] += round(g(r, "coverable_recall") * nc)
        t["q"] += nq; t["q_bad"] += round(g(r, "quiet_violation_rate") * nq)
        t["sp"] += ns; t["min"] += dur[k[1]] / 60; t["speech_s"] += g(r, "audio_occupancy") * dur[k[1]]
    dv = lambda a, b: a / b if b else float("nan")
    return {"recall_all": dv(t["ev_hit"], t["ev"]), "recall_coverable": dv(t["cov_hit"], t["cov"]),
            "quiet_violation": dv(t["q_bad"], t["q"]), "cues_per_min": dv(t["sp"], t["min"]),
            "speech_s_per_min": dv(t["speech_s"], t["min"]), "hits_all": t["ev_hit"], "n_events": t["ev"],
            "hits_coverable": t["cov_hit"], "n_coverable": t["cov"], "n_quiet": t["q"]}


keys_all = [(s, v) for s in seeds for v in ft_ids]
rng = random.Random(0)
samples = [[rng.choice(ft_ids) for _ in ft_ids] for _ in range(args.boot)]
res = {"_meta": {"videos": len(ft_ids), "seeds": seeds, "target_cues_per_min": 2.0, "boot": args.boot,
                 "unit": "finetune video, resampled once and used for all seeds", "runs": replay_meta}}
for pol in REPLAY + ["online_greedy"]:
    res[pol] = {"pooled": pooled(pol, keys_all), "per_seed": {s: pooled(pol, [(s, v) for v in ft_ids]) for s in seeds}}
    if pol != "logistic":
        for k in ("recall_all", "recall_coverable", "quiet_violation"):
            d = sorted(pooled(pol, [(s, v) for s in seeds for v in smp])[k] - pooled("logistic", [(s, v) for s in seeds for v in smp])[k]
                       for smp in samples)
            res[pol][f"diff_vs_logistic_{k}"] = {"value": res[pol]["pooled"][k] - pooled("logistic", keys_all)[k],
                                                 "ci95": [d[int(0.025 * (len(d) - 1))], d[int(0.975 * (len(d) - 1))]]}
dump_json(res, OUT / "replay.json")

L = [f"# Replay at equal speech rate on `finetune` ({len(ft_ids)} videos, seeds {', '.join(map(str, seeds))} pooled)", "",
     "Frozen scorers speak the top legal candidate when its score is at least the BudgetController's tau (target "
     "2 cues/min), exactly like online greedy. Every policy streams `train` then `finetune` with the same seeds; only "
     "the `finetune` videos are reported (the `train` part lets the controller settle and is in-sample for the frozen "
     "models). online_greedy learns online from simulated taps and replayed questions, as in docs/online.md. Seeds only "
     "change the simulated feedback, so frozen scorers give the same cues on every seed. Model choice (K, weight decay, "
     "epochs, Platt) was made on these same finetune videos.", "",
     "| policy | coverable recall | all-event recall | all-event recall vs logistic [95% CI] | quiet violations | cues/min | speech s/min |",
     "|---|---|---|---|---|---|---|"]
for pol in REPLAY + ["online_greedy"]:
    p = res[pol]["pooled"]
    d = res[pol].get("diff_vs_logistic_recall_all")
    ds = f"{d['value']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else "reference"
    L.append(f"| {pol} | {p['hits_coverable']:.0f}/{p['n_coverable']:.0f} = {p['recall_coverable']:.3f} | "
             f"{p['hits_all']:.0f}/{p['n_events']:.0f} = {p['recall_all']:.3f} | {ds} | {p['quiet_violation']:.3f} | "
             f"{p['cues_per_min']:.2f} | {p['speech_s_per_min']:.1f} |")
L += ["", "Paired bootstrap: 2,000 resamples of finetune videos, each resample used for all three seeds of both policies.",
      "Per-seed all-event recall: " + "; ".join(f"{pol} " + " / ".join(f"{res[pol]['per_seed'][s]['recall_all']:.3f}" for s in seeds)
                                               for pol in REPLAY + ["online_greedy"])]
(OUT / "replay.md").write_text("\n".join(L) + "\n")
print("\n".join(L), flush=True)

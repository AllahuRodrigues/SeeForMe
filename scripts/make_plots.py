"""Produce the figures for the report from results/*.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import RESULTS

FIG = RESULTS / "figures"; FIG.mkdir(exist_ok=True)
# brand-neutral, colour-blind-safe palette
PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]


def bar(agg, metrics, fname, title):
    names = list(agg.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(4.2, 0.55 * len(names) + 1.5) * len(metrics), 4.2))
    for ax, (m, lab) in zip(axes, metrics):
        vals = [agg[n][m] for n in names]
        ax.bar(range(len(names)), vals, color=[PAL[i % len(PAL)] for i in range(len(names))])
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(FIG / fname, dpi=160); plt.close(fig)


summ = json.loads((RESULTS / "summary_test.json").read_text())
bar(summ, [("important_cue_recall", "Important-cue recall ↑"), ("cue_precision", "Cue precision ↑"),
           ("audio_load_per_min", "Audio load (cues/min) ↓"), ("duplicate_rate", "Duplicate rate ↓"),
           ("mean_critical_delay", "Critical-event delay (s) ↓")], "main_results.png", "Baselines vs CACS on held-out scenarios")

# recall vs audio-load trade-off
fig, ax = plt.subplots(figsize=(5.5, 4))
for i, (n, a) in enumerate(summ.items()):
    ax.scatter(a["audio_load_per_min"], a["important_cue_recall"], s=70, color=PAL[i % len(PAL)], label=n)
ax.set_xlabel("audio load (cues / min)"); ax.set_ylabel("important-cue recall"); ax.set_title("Information preserved vs. speech emitted")
ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "tradeoff.png", dpi=160)

abl_path = RESULTS / "ablation_test.json"
if abl_path.exists():
    abl = json.loads(abl_path.read_text())
    bar(abl, [("important_cue_recall", "Important-cue recall ↑"), ("cue_precision", "Cue precision ↑"),
              ("audio_load_per_min", "Audio load ↓"), ("duplicate_rate", "Duplicate rate ↓")], "ablation.png", "CACS ablation")

kind_path = RESULTS / "summary_by_kind_test.json"
if kind_path.exists():
    bk = json.loads(kind_path.read_text())
    kinds = list(bk.keys()); pols = list(next(iter(bk.values())).keys())
    fig, ax = plt.subplots(figsize=(8, 4)); wdt = 0.8 / len(pols)
    for j, p in enumerate(pols):
        ax.bar([i + j * wdt for i in range(len(kinds))], [bk[k][p]["important_cue_recall"] for k in kinds], wdt, label=p, color=PAL[j % len(PAL)])
    ax.set_xticks([i + 0.4 - wdt / 2 for i in range(len(kinds))]); ax.set_xticklabels(kinds, fontsize=8)
    ax.set_ylabel("important-cue recall"); ax.set_title("Recall by scenario type"); ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "recall_by_kind.png", dpi=160)

pr = RESULTS / "priority_model_report.json"
if pr.exists():
    rep = json.loads(pr.read_text())
    w = rep["models"]["logreg"]["learned_weights"]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(list(w.keys()), list(w.values()), color=[PAL[2] if v >= 0 else PAL[3] for v in w.values()])
    ax.axvline(0, color="k", lw=0.8); ax.set_title("Learned logistic-regression weights per CACS feature")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "learned_weights.png", dpi=160)
print("figures written to", FIG)

deep_path = RESULTS / "deep_context_report.json"
if deep_path.exists():
    deep = json.loads(deep_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for i, (name, r) in enumerate(list(deep["dagger"].items()) + [("gru_imitation", deep["seq_il"])]):
        ax.plot([h["iter"] for h in r["history"]], [h["f1"] for h in r["history"]], marker="o", color=PAL[i % len(PAL)], label=name)
    ax.set_xlabel("DAgger iteration (0 = oracle data only)"); ax.set_ylabel("validation closed-loop F1"); ax.set_title("On-policy data aggregation")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    h = deep["seq_rl"]["history"]
    ax.plot([x["epoch"] for x in h], [x["f1"] for x in h], marker="o", color=PAL[0], label="val F1")
    ax2 = ax.twinx()
    ax2.plot([x["epoch"] for x in h if x["mean_return"] is not None], [x["mean_return"] for x in h if x["mean_return"] is not None], color=PAL[1], alpha=0.7, label="mean episode return")
    ax.set_xlabel("REINFORCE epoch (0 = imitation init)"); ax.set_ylabel("validation closed-loop F1"); ax2.set_ylabel("mean episode return")
    ax.set_title("Reward fine-tuning of the GRU policy"); ax.spines[["top"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "deep_context_training.png", dpi=160); plt.close(fig)
    print("deep-context training figure written")

# compact poster version: two baselines, tuned linear CACS, best on-policy feature model, RL sequence model
poster = {k: summ[k] for k in ["confidence_threshold", "confidence_cooldown", "cacs", "cacs_dagger:gbm", "cacs_seq_rl"] if k in summ}
bar(poster, [("important_cue_recall", "Important-cue recall ↑"), ("cue_precision", "Cue precision ↑"),
             ("audio_load_per_min", "Audio load (cues/min) ↓"), ("mean_critical_delay", "Critical-event delay (s) ↓")],
    "poster_results.png", "What gets said: baselines vs context-aware cue selection")
print("poster figure written")

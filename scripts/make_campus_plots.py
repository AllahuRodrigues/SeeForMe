"""Figures for the CampusBench results (results/campus/figures/)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from campus_common import RESULTS_C

FIG = RESULTS_C / "figures"; FIG.mkdir(parents=True, exist_ok=True)
summary = json.loads((RESULTS_C / "summary_test.json").read_text())
by_kind = json.loads((RESULTS_C / "summary_by_kind_test.json").read_text())
MAIN = [p for p in ["confidence_threshold", "confidence_cooldown", "cacs", "pcacs", "pcacs_learned:il", "pcacs_learned:rl", "oracle"] if p in summary]
NICE = {"confidence_threshold": "conf. threshold", "confidence_cooldown": "conf. + cooldown", "cacs": "CACS (linear)",
        "pcacs": "P-CACS (linear)", "pcacs_learned:il": "P-CACS GRU (imitation)", "pcacs_learned:rl": "P-CACS GRU (+RL)", "oracle": "oracle"}
COL = {"confidence_threshold": "#9aa5b1", "confidence_cooldown": "#6c7a89", "cacs": "#f0b13b", "pcacs": "#2e9e6b",
       "pcacs_learned:il": "#3b82f6", "pcacs_learned:rl": "#7c3aed", "oracle": "#111827"}

# 1. main comparison: four panels
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (key, title) in zip(axes, [("action_weighted_recall", "action-weighted recall ↑"), ("late_critical_rate", "late critical rate ↓"),
                                   ("audio_occupancy", "audio occupancy ↓"), ("regret", "communication regret ↓")]):
    vals = [summary[p][key] for p in MAIN]
    ax.bar([NICE[p] for p in MAIN], vals, color=[COL[p] for p in MAIN])
    ax.set_title(title); ax.tick_params(axis="x", rotation=35, labelsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("CampusBench (held-out test split): what gets said, how late, how much airtime")
fig.tight_layout(); fig.savefig(FIG / "campus_main.png", dpi=160); plt.close(fig)

# 2. recall vs occupancy trade-off
fig, ax = plt.subplots(figsize=(6, 4.5))
for p, a in summary.items():
    ax.scatter(a["audio_occupancy"], a["action_weighted_recall"], s=70, color=COL.get(p, "#999"), zorder=3)
    ax.annotate(NICE.get(p, p), (a["audio_occupancy"], a["action_weighted_recall"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("audio occupancy (fraction of walking time with speech)"); ax.set_ylabel("action-weighted recall")
ax.grid(alpha=.3); ax.set_title("Timely warnings per unit of airtime")
fig.tight_layout(); fig.savefig(FIG / "campus_tradeoff.png", dpi=160); plt.close(fig)

# 3. per scene type
kinds = list(by_kind)
show = [p for p in ["cacs", "pcacs", "pcacs_learned:rl", "oracle"] if p in summary]
x = np.arange(len(kinds)); wdt = 0.8 / len(show)
fig, ax = plt.subplots(figsize=(12, 4))
for i, p in enumerate(show):
    ax.bar(x + i * wdt, [by_kind[k].get(p, {}).get("action_weighted_recall", 0) for k in kinds], wdt, label=NICE[p], color=COL[p])
ax.set_xticks(x + 0.4 - wdt / 2); ax.set_xticklabels(kinds, rotation=30, ha="right"); ax.set_ylabel("action-weighted recall")
ax.legend(fontsize=8); ax.set_title("Recall by CampusBench scene type"); fig.tight_layout(); fig.savefig(FIG / "campus_by_kind.png", dpi=160); plt.close(fig)

# 4. ablations
abl = [p for p in summary if p.startswith("pcacs-")] + ["pcacs"]
if len(abl) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title) in zip(axes, [("action_weighted_recall", "action-weighted recall"), ("late_critical_rate", "late critical rate"), ("regret", "regret")]):
        vals = [summary[p][key] for p in abl]
        ax.barh([p.replace("pcacs-", "− ").replace("pcacs", "P-CACS full") for p in abl], vals, color=["#2e9e6b" if p == "pcacs" else "#9fd3b8" for p in abl])
        ax.set_title(title)
    fig.suptitle("P-CACS ablations (test split)"); fig.tight_layout(); fig.savefig(FIG / "campus_ablation.png", dpi=160); plt.close(fig)

# 5. scheduler training curves
tr = RESULTS_C / "scheduler_training.json"
if tr.exists():
    rep = json.loads(tr.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    il = rep["il"]["history"]
    axes[0].plot([h["iter"] for h in il], [h["score"] for h in il], "o-", color=COL["pcacs_learned:il"]); axes[0].set_xlabel("DAgger iteration"); axes[0].set_ylabel("val score"); axes[0].set_title("Imitation (DAgger)")
    rl = rep["rl"]["history"]
    axes[1].plot([h["epoch"] for h in rl], [h["score"] for h in rl], "o-", color=COL["pcacs_learned:rl"]); axes[1].set_xlabel("REINFORCE epoch"); axes[1].set_title("Reward fine-tuning")
    if "pcacs" in summary:
        for ax in axes:
            ax.axhline(-summary["pcacs"]["regret"] + 0, ls="--", color=COL["pcacs"], lw=1)
    fig.tight_layout(); fig.savefig(FIG / "campus_scheduler_training.png", dpi=160); plt.close(fig)
print("figures ->", FIG)

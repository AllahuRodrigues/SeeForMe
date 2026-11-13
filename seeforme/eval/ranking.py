"""Ranking diagnostics for a should-speak score.

A pooled AUC mixes two abilities: telling busy recordings from quiet ones, and ordering the moments inside one
recording. Only the second helps a scheduler decide when to speak, so `within_group_auc` measures it alone.
`null_summary` turns a set of label-permutation AUCs into a reference range and a one-sided p-value.
"""
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata


def _auc(y, s):
    """Mann-Whitney AUC with ties counted as one half. Needs both classes."""
    y = np.asarray(y, bool)
    r = rankdata(np.asarray(s, float))
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_group_auc(y, score, groups):
    """Row-weighted mean of the AUC computed inside each group; groups with a single class are skipped and counted."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    y, score = np.asarray(y, int), np.asarray(score, float)
    per, rows, skipped = {}, {}, 0
    for g, idx in by.items():
        yy = y[idx]
        if 0 < yy.sum() < len(idx):
            per[g], rows[g] = _auc(yy, score[idx]), len(idx)
        else:
            skipped += 1
    n = sum(rows.values())
    auc = sum(per[g] * rows[g] for g in per) / n if n else float("nan")
    return {"auc": auc, "groups": len(per), "rows": n, "skipped": skipped, "per_group": per, "group_rows": rows}


def null_summary(observed, null):
    """Mean, central 95% range and one-sided p of `observed` against permutation values (the observed one included)."""
    if len(null) == 0:
        raise ValueError("null_summary needs at least one permutation value")
    xs = np.sort(np.asarray(null, float))
    p = (1 + int((xs >= observed).sum())) / (len(xs) + 1)
    return {"mean": float(xs.mean()), "lo": float(np.quantile(xs, 0.025)), "hi": float(np.quantile(xs, 0.975)), "p": p, "n": len(xs)}

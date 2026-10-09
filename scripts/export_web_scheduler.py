"""Export the learned RL scheduler (models/scheduler/scheduler_rl.pt) for the web app.

Writes
    web/public/models/scheduler_rl.onnx   the GRU scheduler with a fixed, browser-friendly interface
    web/public/config/scheduler.json      feature names in order, K, mode, silence bias, version hash, parity check

ONNX interface (opset 17, N is dynamic):
    inputs   windows       float32 [N, K, 29]  the last K EventFeatures vectors of each candidate, RIGHT-aligned
                                               (row K-1 is the current step, missing older steps are zero rows)
             mask          float32 [N, K]      1 where the row holds a real step, 0 for the padding in front
    outputs  action_logits float32 [N, 4]      silence / speak / interrupt / haptic
             slot_logits   float32 [N, 6]      object / distance / direction / motion / action / detail
             silence_bias  float32 [1]         the learned global "say nothing" logit of the RL policy

The GRU is unrolled over the K steps with PyTorch's gate equations; a masked step keeps the hidden state, so a
right-aligned window with zeros in front gives exactly the hidden state of `pack_padded_sequence` over the real
steps (what `SchedulerNet.encode` computes in Python). The script checks the exported graph against
`SchedulerModel.forward` (the Python code path, left-aligned windows + lengths) on random inputs and fails if
the max abs difference is >= 1e-4.

    python scripts/export_web_scheduler.py [--model models/scheduler/scheduler_rl.pt]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from seeforme.cues.compose import SLOTS  # noqa: E402
from seeforme.cues.pcacs import ACTIONS, EventFeatures  # noqa: E402
from seeforme.ml.scheduler_model import SchedulerModel  # noqa: E402

OPSET = 17


class WebScheduler(nn.Module):
    """SchedulerNet (arch 'gru') with the GRU unrolled over a right-aligned, masked window."""

    def __init__(self, model: SchedulerModel):
        super().__init__()
        net = model.net
        if net.arch != "gru":
            raise SystemExit(f"only the 'gru' scheduler is exported for the web (got {net.arch})")
        g = net.gru
        if g.num_layers != 1 or g.bidirectional or not g.batch_first:
            raise SystemExit("expected a single-layer, unidirectional, batch_first GRU")
        self.K, self.H = model.K, g.hidden_size
        self.w_ih = nn.Parameter(g.weight_ih_l0.detach().clone())   # (3H, n_in), gate order r, z, n
        self.w_hh = nn.Parameter(g.weight_hh_l0.detach().clone())   # (3H, H)
        self.b_ih = nn.Parameter(g.bias_ih_l0.detach().clone())
        self.b_hh = nn.Parameter(g.bias_hh_l0.detach().clone())
        self.trunk = copy_module(net.trunk)
        self.action = copy_module(net.action)
        self.slots = copy_module(net.slots)
        self.silence_bias = nn.Parameter(net.silence_bias.detach().clone())
        self.register_buffer("h0_proj", torch.zeros(self.w_ih.shape[1], self.H))

    def forward(self, windows: torch.Tensor, mask: torch.Tensor):
        H = self.H
        h = windows[:, 0, :] @ self.h0_proj                     # zeros [N, H] with a dynamic N
        gi_all = windows @ self.w_ih.t() + self.b_ih             # [N, K, 3H]
        for k in range(self.K):
            gi = gi_all[:, k, :]
            gh = h @ self.w_hh.t() + self.b_hh
            r = torch.sigmoid(gi[:, :H] + gh[:, :H])
            z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h_new = (1 - z) * n + z * h
            m = mask[:, k:k + 1]
            h = m * h_new + (1 - m) * h
        zt = self.trunk(h)
        return self.action(zt), self.slots(zt), self.silence_bias * 1.0


def copy_module(m: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(m)


def random_windows(n: int, K: int, d: int, rng: np.random.Generator):
    """Random candidate histories of length 1..K, as lists (the Python path) and right-aligned (the web path)."""
    lists, X, M = [], np.zeros((n, K, d), np.float32), np.zeros((n, K), np.float32)
    for i in range(n):
        L = int(rng.integers(1, K + 1))
        w = (rng.normal(0, 1.0, (L, d)) * rng.choice([0.3, 1.0, 3.0])).astype(np.float32)
        lists.append(w.tolist())
        X[i, K - L:] = w
        M[i, K - L:] = 1.0
    return lists, X, M


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "scheduler" / "scheduler_rl.pt"))
    ap.add_argument("--report", default=str(ROOT / "models" / "scheduler_report.json"))
    ap.add_argument("--onnx", default=str(ROOT / "web" / "public" / "models" / "scheduler_rl.onnx"))
    ap.add_argument("--meta", default=str(ROOT / "web" / "public" / "config" / "scheduler.json"))
    ap.add_argument("--n-random", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)

    src = Path(args.model)
    model = SchedulerModel.load(str(src))
    K, d = model.K, model.net.n_in
    names = EventFeatures.names()
    assert d == len(names) == 29, (d, len(names))
    web = WebScheduler(model).eval()

    # 1) the unrolled module equals the packed GRU (torch vs torch)
    rng = np.random.default_rng(args.seed)
    lists, X, M = random_windows(args.n_random, K, d, rng)
    la_ref, _, ls_ref = model.forward(lists)                    # Python path: _pad (left-aligned) + pack_padded_sequence
    with torch.no_grad():
        la_w, ls_w, sb_w = web(torch.from_numpy(X), torch.from_numpy(M))
    d_torch = max(float((la_ref - la_w).abs().max()), float((ls_ref - ls_w).abs().max()))
    print(f"unrolled GRU vs packed GRU (torch): max abs diff {d_torch:.3e} on {args.n_random} random windows")

    # 2) export
    out = Path(args.onnx); out.parent.mkdir(parents=True, exist_ok=True)
    ex_X, ex_M = torch.from_numpy(X[:3]), torch.from_numpy(M[:3])
    torch.onnx.export(web, (ex_X, ex_M), str(out), input_names=["windows", "mask"],
                      output_names=["action_logits", "slot_logits", "silence_bias"],
                      dynamic_axes={"windows": {0: "N"}, "mask": {0: "N"}, "action_logits": {0: "N"}, "slot_logits": {0: "N"}},
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)
    import onnx
    m = onnx.load(str(out))
    # constant folding turns the silence-bias output into an initializer with a generated name: give it its name back
    want = ["action_logits", "slot_logits", "silence_bias"]
    for o, name in zip(m.graph.output, want):
        if o.name != name:
            old = o.name
            for init in m.graph.initializer:
                if init.name == old:
                    init.name = name
            for node in m.graph.node:
                node.input[:] = [name if x == old else x for x in node.input]
                node.output[:] = [name if x == old else x for x in node.output]
            o.name = name
    assert [o.name for o in m.graph.output] == want and [i.name for i in m.graph.input] == ["windows", "mask"]
    onnx.checker.check_model(m)
    m.producer_name = "seeforme/scripts/export_web_scheduler.py"
    m.doc_string = f"SeeForMe learned scheduler (RL), from {src.relative_to(ROOT) if src.is_relative_to(ROOT) else src}"
    onnx.save(m, str(out))

    # 3) ONNX Runtime vs PyTorch (Python path) on the same random inputs
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == want, [o.name for o in sess.get_outputs()]
    diffs = []
    for i in range(0, args.n_random, 512):
        la_o, ls_o, sb_o = sess.run(None, {"windows": X[i:i + 512], "mask": M[i:i + 512]})
        diffs.append(float(np.abs(la_o - la_ref[i:i + 512].numpy()).max()))
        diffs.append(float(np.abs(ls_o - ls_ref[i:i + 512].numpy()).max()))
    d_onnx = max(diffs)
    d_bias = float(abs(sb_o[0] - float(model.net.silence_bias.detach()[0])))
    # decisions too: argmax of [log-odds..., silence bias] per single candidate, no masks
    lo = la_ref[:, 1:] - la_ref[:, :1]
    full_ref = torch.cat([lo, model.net.silence_bias.detach().expand(len(lo), 1)], 1).argmax(1).numpy()
    la_all = np.concatenate([sess.run(None, {"windows": X[i:i + 512], "mask": M[i:i + 512]})[0] for i in range(0, args.n_random, 512)])
    lo_o = la_all[:, 1:] - la_all[:, :1]
    full_o = np.concatenate([lo_o, np.full((len(lo_o), 1), sb_o[0], np.float32)], 1).argmax(1)
    agree = float((full_o == full_ref).mean())
    print(f"ONNX Runtime {ort.__version__} vs PyTorch {torch.__version__}: max abs diff {d_onnx:.3e} (logits), "
          f"silence bias diff {d_bias:.1e}, per-candidate argmax agreement {agree:.4f} on {args.n_random} random windows")
    if not d_onnx < 1e-4 or not d_bias < 1e-6:
        raise SystemExit(f"FAIL: ONNX export differs from PyTorch by {d_onnx:.3e} (limit 1e-4)")

    # 4) metadata the web app checks before it uses the model
    rep = json.loads(Path(args.report).read_text()) if Path(args.report).exists() else {}
    src_sha, onnx_sha = sha256(src), sha256(out)
    meta = {
        "version": f"rl-{src_sha[:12]}",
        "model_url": f"/models/{out.name}?v={onnx_sha[:12]}",
        "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "source_sha256": src_sha,
        "onnx_sha256": onnx_sha,
        "arch": model.net.arch, "hidden": model.hidden, "K": K, "opset": OPSET,
        "mode": "rl",
        "decision": "argmax over [speak/interrupt/haptic log-odds vs silence per candidate (illegal = -1e9)] + [silence_bias]",
        "threshold": None,
        "il_threshold": (rep.get("il") or {}).get("threshold"),
        "silence_bias": float(model.net.silence_bias.detach()[0]),
        "features": names,
        "actions": list(ACTIONS),
        "slots": list(SLOTS),
        "inputs": {"windows": ["N", K, d], "mask": ["N", K]},
        "outputs": {"action_logits": ["N", len(ACTIONS)], "slot_logits": ["N", len(SLOTS)], "silence_bias": [1]},
        "window_alignment": "right (row K-1 = current step; mask 0 marks zero padding in front)",
        "check": {"n_random": args.n_random, "onnx_vs_torch_max_abs_diff": d_onnx, "unrolled_vs_packed_max_abs_diff": d_torch,
                  "argmax_agreement": agree, "onnxruntime": ort.__version__, "torch": torch.__version__},
        "trained_on": "CampusBench (generated by this repository)",
        "exported": _dt.date.today().isoformat(),
    }
    mp = Path(args.meta); mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"-> {out} ({out.stat().st_size / 1024:.1f} KB, sha256 {onnx_sha[:12]})")
    print(f"-> {mp} (version {meta['version']}, K={K}, silence_bias={meta['silence_bias']:.4f})")


if __name__ == "__main__":
    main()

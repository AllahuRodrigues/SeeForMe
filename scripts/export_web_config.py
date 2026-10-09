"""Export the tuned P-CACS config to web/public/config/pcacs.json (camelCase keys the web app reads at startup)."""
import json, yaml
from _common import CONFIGS, ROOT

raw = yaml.safe_load((CONFIGS / "pcacs_tuned.yaml").read_text())
w = raw.pop("weights")
camel = lambda k: "".join(p.capitalize() if i else p for i, p in enumerate(k.split("_")))
out = {camel(k): v for k, v in raw.items() if not k.startswith("use_")}
out["weights"] = {camel(k): v for k, v in w.items()}
# names that differ between the python and TS configs
ren = {"hapticAfterSpeechS": "hapticAfterSpeech", "hapticRepeatS": "hapticRepeat", "budgetCapacityS": "budgetCapacity",
       "reannounceMoveM": "reannounceMove", "reannounceTauS": "reannounceTau", "recentBlockS": "recentBlock"}
out = {ren.get(k, k): v for k, v in out.items()}
d = ROOT / "web" / "public" / "config"; d.mkdir(parents=True, exist_ok=True)
(d / "pcacs.json").write_text(json.dumps(out, indent=1))
print("->", d / "pcacs.json", sorted(out))

# benchmark numbers for the web app's results panel
import json as _json
res = ROOT / "results" / "campus" / "summary_test.json"
if res.exists():
    S = _json.loads(res.read_text())
    keep = ["confidence_threshold", "confidence_cooldown", "cacs", "pcacs", "pcacs_learned:il", "pcacs_learned:rl", "oracle"]
    names = {"confidence_threshold": "Confidence threshold", "confidence_cooldown": "Confidence + cooldown", "cacs": "CACS (v1)",
             "pcacs": "P-CACS linear", "pcacs_learned:il": "P-CACS GRU imitation", "pcacs_learned:rl": "P-CACS GRU + RL", "oracle": "Reference"}
    bench = [{"id": p, "name": names[p], **{k: round(S[p][k], 3) for k in ("action_weighted_recall", "late_critical_rate", "cue_precision",
             "audio_occupancy", "cues_per_min", "silence_precision", "min_message_ratio", "regret")}} for p in keep if p in S]
    dd = ROOT / "web" / "public" / "data"; dd.mkdir(parents=True, exist_ok=True)
    (dd / "benchmark.json").write_text(_json.dumps(bench, indent=1))
    print("->", dd / "benchmark.json")

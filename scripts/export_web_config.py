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

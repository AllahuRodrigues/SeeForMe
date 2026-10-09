"""Use GuideDog's human-verified annotations to measure SeeForMe's *perception* on street-view egocentric images:

  depth config  (383 items, which of two objects is closer): does Depth Anything V2 order the two boxes correctly,
                and how far are its metres from GuideDog's estimated depths?  -> the depth-reliability calibration
                the research plan asks for (perception residuals on real held-out images), split by distance band
                (closer object < 3 m / 3-8 m / > 8 m) and by how far apart the pair is (farther/closer ratio)
  gold split    (2,106 human-verified guidance paragraphs, no S1/S2/S3 markers: a scene sentence, hazards by clock
                direction and distance, then advice "To navigate safely, ..."): each clock mention is one clause;
                does YOLO11 detect *something* in that direction, and is the named hazard a COCO class at all?
                Clauses that say the direction is clear are counted separately, not as hazards.
                -> perception coverage of what sighted annotators warn blind pedestrians about

    python scripts/guidedog_eval.py                     # -> results/guidedog/summary.json
    python scripts/guidedog_eval.py --limit 200 --out-tag quick   # quick pass -> summary_quick.json
    python scripts/guidedog_eval.py --smoke             # synthetic rows + parser checks, no dataset needed (CI)
No training happens here; GuideDog images are static and carry no timing, so they cannot label the scheduler.
"""
import argparse, io, json, re, statistics as st
from collections import Counter
from pathlib import Path
import numpy as np
from _common import RESULTS, ROOT, dump_json, load_cfg
from seeforme.egoblind.data import match_concepts

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "guidedog"))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--depth-size", type=int, default=252)
ap.add_argument("--out-tag", default="", help="write summary_<tag>.json instead of summary.json")
args = ap.parse_args()
OUT = RESULTS / "guidedog"; OUT.mkdir(parents=True, exist_ok=True)

CLOCK = {"10": (0.0, 0.2), "11": (0.2, 0.4), "12": (0.4, 0.6), "1": (0.6, 0.8), "2": (0.8, 1.0)}   # clock position -> image x band
_HOUR_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_HOUR = r"(?:1[0-2]|[1-9]|" + "|".join(_HOUR_WORDS) + r")"
_OCLOCK = r"o\s*['’]?\s*clock"                         # o'clock, o’clock, o clock
# one clock mention, possibly naming several hours: "2 o'clock", "10 and 11 o'clock", "11, 12, 1, or 2 o’clock", "10 to 2 o'clock"
CLOCK_MENTION_RE = re.compile(rf"\b(?:{_HOUR}(?:\s*{_OCLOCK})?\s*(?:,\s*(?:and|or|to)?|and|or|to|through|&|-|–)\s*)*{_HOUR}\s*{_OCLOCK}\b", re.I)
CLOCK_WORD_RE = re.compile(rf"\b{_HOUR}\s*{_OCLOCK}\b|\b{_OCLOCK}\b", re.I)   # "2 o'clock" must not match the COCO class "clock"
_HOUR_TOK_RE = re.compile(rf"\b{_HOUR}\b|\b(?:to|through)\b|[-–]", re.I)
_SENT_SPLIT_RE = re.compile(r"[.!?\n]+(?!\d)")           # "2.5 m" stays one sentence
# where one clock clause ends and the next begins: the last comma / contrast word between two mentions, else the last "and"
_STRONG_SEP_RE = re.compile(r",|\b(?:but|while|whilst|whereas|however|though|although|yet)\b", re.I)
_WEAK_SEP_RE = re.compile(r"\b(?:and|with|plus)\b", re.I)
# the closing advice ("To navigate safely, ...", "It's best to ...") only repeats hazards already given by direction
_ADVICE_RE = re.compile(r"^\W*(?:t?o (?:\w+ ){0,2}?)?(?:safely |carefully |cautiously )?(?:navigate|navigating|move|proceed|continue|avoid|stay|sail|walk|steer|travel|keep|turn|head|veer|go|reach|use)\b"
                        r"|^\W*(?:it(?:'s|’s| is| would be) (?:best|safest|safer|advisable|wise|recommended)|the (?:best|safest) (?:way|option)"
                        r"|for (?:your )?safety|for safe(?:ly)? \w+|(?:for|in) navigating|i (?:recommend|suggest)|you should|consider"
                        r"|be (?:cautious|careful|mindful|aware)|ensure|carefully|cautiously)\b", re.I)
# a clause that says its direction is clear is a clear-direction mention, not a hazard
_CLEAR_RE = re.compile(r"(?<!no )(?<!not a )(?<!end of the )\bclear (?:path|way|route|walkway|space|area|passage)\b"
                       r"|\b(?:is|are|looks?|appears?|seems?|remains?) (?:(?!not\b)\w+ )?(?:clear|unobstructed)\b"
                       r"|\b(?:area|space|path|way|walkway|sidewalk|street|road|it) (?:is|looks|appears|seems|remains) (?:\w+ )?(?:open|empty|free)\b"
                       r"|\bnothing\b|\bno (?:\w+ ){0,2}(?:obstacles?|hazards?|obstructions?|dangers?|risks?|concerns?|issues?|threats?)\b"
                       r"|\bwithout (?:any )?(?:\w+ )?(?:obstacles?|hazards?|obstructions?)\b|\bunobstructed\b", re.I)
_CONTRAST_RE = re.compile(r"\b(?:but|however|though|although|except|yet)\b", re.I)
# "similar to 10 o'clock", "the one at 10 o'clock": a back-reference to a direction already described, not a new mention
_BACKREF_RE = re.compile(r"(?:similar to|same as|as at|than at|like (?:at |the one at )?|mirrors?|mirroring|(?:those|the ones?|that) at)\s*$", re.I)
_FILLER = {"at", "your", "you", "directly", "ahead", "and", "onward", "onwards", "to", "from", "around", "the", "in", "on", "s", "there"}


def _hours(mention):
    """Hours named by one clock mention; ranges go the short way round ("10 to 2" -> 10, 11, 12, 1, 2)."""
    out, rng = [], False
    for tok in _HOUR_TOK_RE.findall(mention):
        t = tok.lower()
        if t in ("to", "through", "-", "–"):
            rng = bool(out); continue
        h = _HOUR_WORDS.get(t) or int(t)
        if rng:
            step = 1 if (h - out[-1]) % 12 <= 6 else -1
            while out[-1] != h:
                out.append((out[-1] + step - 1) % 12 + 1)
        else:
            out.append(h)
        rng = False
    return list(dict.fromkeys(str(h) for h in out))


def strip_clock(text):
    return CLOCK_WORD_RE.sub(" ", CLOCK_MENTION_RE.sub(" ", text))


def says_clear(clause):
    """The clause says its direction is clear ("the path is clear", "no immediate hazards"), and no later "but ..." names
    something there anyway ("no immediate hazards, but there are pedestrians directly ahead")."""
    s = strip_clock(clause); m = _CLEAR_RE.search(s)
    if not m:
        return False
    c = _CONTRAST_RE.search(s, m.end())
    if c is None:
        return True
    coco, other = match_concepts(s[c.end():])
    return not (coco or [o for o in other if o != "road"])     # "road" is the generic street/path/sidewalk word


def gold_clauses(text):
    """[(hour, clause, says_clear)] for every in-view clock direction (10-2 o'clock) a gold paragraph mentions.

    Each clock mention gets its own clause (text before and after the clock phrase, up to the neighbouring mention), and
    a clause naming several hours ("at 10 and 11 o'clock") counts once per hour. The closing advice sentences are
    dropped (they repeat the hazards), and a direction from the opening scene sentence is kept only when the
    hazard sentences do not mention that hour again. Explicit S1/S2/S3 markers, when present, select S2 instead."""
    m = re.search(r"\bS2\b:?(.*?)(?:\bS3\b|$)", text, re.S)
    if m:
        sents, scene_first = [s for s in _SENT_SPLIT_RE.split(m.group(1)) if s.strip()], False
    else:
        sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        while len(sents) > 1 and _ADVICE_RE.search(sents[-1]):
            sents.pop()
        scene_first = len(sents) > 1
    out = []
    for i, sent in enumerate(sents):
        for part in sent.split(";"):
            ms = [mm for mm in CLOCK_MENTION_RE.finditer(part) if not _BACKREF_RE.search(part[max(0, mm.start() - 30): mm.start()])]
            cuts = [0]
            for a, b in zip(ms, ms[1:]):
                gap = part[a.end(): b.start()]
                seps = list(_STRONG_SEP_RE.finditer(gap)) or list(_WEAK_SEP_RE.finditer(gap))
                cuts.append(a.end() + seps[-1].start() if seps else b.start())
            cuts.append(len(part))
            clauses = [part[cuts[j]: cuts[j + 1]].strip(" ,") for j in range(len(ms))]
            # "At 12 o'clock, and at 1 and 2 o'clock, there are no hazards": a clause with nothing but its clock phrase
            # shares the content of the next clause
            for j in range(len(clauses) - 2, -1, -1):
                if not [w for w in re.findall(r"[a-z]+", strip_clock(clauses[j]).lower()) if w not in _FILLER]:
                    clauses[j] = clauses[j] + " " + clauses[j + 1]
            for mm, clause in zip(ms, clauses):
                clear = says_clear(clause)
                out += [(i, h, clause, clear) for h in _hours(mm.group(0)) if h in CLOCK]
    later = {h for i, h, _, _ in out if i > 0}
    return [(h, c, clear) for i, h, c, clear in out if not (scene_first and i == 0 and h in later)]


def _check_parser():
    """Fixed cases modelled on GuideDog phrasing (paraphrased; no dataset text); run with --smoke."""
    got = gold_clauses("You're on a street. At 2 o'clock, similar to 10 o'clock, walls narrow the walkway. At 10 o'clock, a bench.")
    assert sorted(h for h, _, _ in got) == ["10", "2"], got                      # the back-reference is not a second 10
    got = gold_clauses("You're on a street. At 12 o'clock, and at 1 and 2 o'clock, there are no specific hazards. Go ahead.")
    assert all(c for _, _, c in got) and sorted(h for h, _, _ in got) == ["1", "12", "2"], got
    assert not says_clear("a delivery van is parked nearby, at the end of the clear path")
    got = gold_clauses("You're on a street. At 1 o'clock, there's a clear path, and at 2 o'clock, about six steps away, are parked cars. "
                       "To navigate safely, keep left of the cars at 2 o'clock.")
    assert [(h, c) for h, _, c in got] == [("1", True), ("2", False)], got
    assert match_concepts(strip_clock(got[1][1]))[0] == ["car"], got
    got = gold_clauses("You're on a street. At 1 o'clock, the path is clear but foot traffic may become a concern. "
                       "At 11 o'clock, there are no obstacles nearby, but people are walking directly ahead. I recommend moving left.")
    assert [(h, c) for h, _, c in got] == [("1", True), ("11", False)], got
    got = gold_clauses("You are on a street with a taxi at 10 o’clock and buildings at 1 and 2 o clock. "
                       "At 11 o'clock, a person stands 3 steps away; there's nothing at 12 o'clock. It's best to keep right.")
    assert [(h, c) for h, _, c in got] == [("10", False), ("1", False), ("2", False), ("11", False), ("12", True)], got
    assert match_concepts(strip_clock(got[0][1]))[0] == ["car"] and "clock" not in match_concepts(strip_clock(got[3][1]))[0], got
    got = gold_clauses("You're between buildings at your 10 and 2 o'clock. At 10 to 12 o'clock, a bench. At 2 o'clock, a wall.")
    assert [h for h, _, _ in got] == ["10", "11", "12", "2"], got


def decode_image(cell):
    from PIL import Image
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))[:, :, ::-1].copy()   # BGR for the pipeline


def _png(img):
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(img[:, :, ::-1]).save(buf, format="PNG"); return buf.getvalue()


def smoke_rows():
    rng = np.random.default_rng(0)
    img = (rng.random((360, 640, 3)) * 255).astype(np.uint8)
    depth_rows = [{"image": {"bytes": _png(img)}, "closer_ratio_bbox": [0.4, 0.5, 0.6, 0.9], "farther_ratio_bbox": [0.1, 0.3, 0.2, 0.45],
                   "closer_depth": 2.0, "farther_depth": 8.0, "closer": "person", "farther": "car"}]
    gold_rows = [{"image": {"bytes": _png(img)}, "scene_location": "sidewalk",
                  "gold_label": "S1: You are on a sidewalk. S2: 12 o'clock: a person about 2 m ahead. 2 o'clock: a parked bicycle 3 m. S3: keep left."},
                 {"image": {"bytes": _png(img)}, "scene_location": "street",      # the real GuideDog format: one paragraph, no markers
                  "gold_label": "You're on a busy street with a taxi at 10 o'clock. At 1 o'clock, there's a clear path, and at 2 o'clock, "
                                "about 6 steps away, a motorcycle is parked. To navigate safely, keep left of the motorcycle at 2 o'clock."}]
    return depth_rows, gold_rows


def box_depth(dmap, box):
    h, w = dmap.shape
    x1, y1, x2, y2 = box
    if max(x1, y1, x2, y2) > 1.5:                      # pixel box
        x1, x2, y1, y2 = x1 / w, x2 / w, y1 / h, y2 / h
    cx1, cx2 = x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1); cy1, cy2 = y1 + 0.2 * (y2 - y1), y2 - 0.2 * (y2 - y1)
    patch = dmap[int(cy1 * h): max(int(cy1 * h) + 1, int(cy2 * h)), int(cx1 * w): max(int(cx1 * w) + 1, int(cx2 * w))]
    patch = patch[np.isfinite(patch) & (patch > 0.05)]
    return float(np.percentile(patch, 30)) if patch.size >= 4 else None


DEPTH_BANDS = (3.0, 8.0, ("lt3m", "3to8m", "gt8m"))       # GuideDog depth of the object: < 3 m, 3-8 m, > 8 m
RATIO_BANDS = (1.5, 3.0, ("lt1.5", "1.5to3", "gt3"))       # farther / closer GuideDog depth: < 1.5, 1.5-3, > 3


def _ref(x):
    x = None if x is None else float(x)
    return x if x is not None and np.isfinite(x) and x > 0 else None


def band(x, bands):
    lo, hi, names = bands
    return "unknown" if x is None else names[0] if x < lo else names[1] if x <= hi else names[2]


cfg = load_cfg()
if args.smoke:
    _check_parser()
    depth_rows, gold_rows = smoke_rows()
    det = depth = None
else:
    data = Path(args.data)
    dp = sorted(data.glob("depth/*.parquet")); gp = sorted(data.glob("data/gold-*.parquet"))
    if not dp and not gp:
        raise SystemExit(f"no GuideDog parquet files under {data}; run scripts/download_guidedog.py")
    import pandas as pd
    depth_rows = pd.concat([pd.read_parquet(p) for p in dp]).to_dict("records") if dp else []
    gold_rows = pd.concat([pd.read_parquet(p) for p in gp]).to_dict("records") if gp else []
    if args.limit:
        depth_rows, gold_rows = depth_rows[: args.limit], gold_rows[: args.limit]
    from seeforme.perception import YoloDetector
    from seeforme.perception.depth import DepthAnythingEstimator, OUTDOOR
    det = YoloDetector(cfg.perception); depth = DepthAnythingEstimator(OUTDOOR, input_size=args.depth_size)

# ---- depth ordering --------------------------------------------------------------
order_ok, abs_err, rel_err, n_depth = 0, [], [], 0
band_n, band_ok = Counter(), Counter()                     # (split, band) -> pairs / correctly ordered pairs
err_by_band = {}                                           # GuideDog depth band of each object -> ([abs err], [rel err])
for r in depth_rows:
    img = decode_image(r["image"])
    dmap = depth(img) if depth is not None else np.full(img.shape[:2], 5.0)
    zc, zf = box_depth(dmap, r["closer_ratio_bbox"]), box_depth(dmap, r["farther_ratio_bbox"])
    if zc is None or zf is None:
        continue
    n_depth += 1; order_ok += zc < zf
    rc, rf = _ref(r.get("closer_depth")), _ref(r.get("farther_depth"))
    for key in (("closer_depth", band(rc, DEPTH_BANDS)), ("depth_ratio", band(rf / rc if rc and rf else None, RATIO_BANDS))):
        band_n[key] += 1; band_ok[key] += zc < zf
    for z, ref in ((zc, rc), (zf, rf)):
        if ref is not None:
            abs_err.append(abs(z - ref)); rel_err.append(abs(z - ref) / ref)
            e = err_by_band.setdefault(band(ref, DEPTH_BANDS), ([], [])); e[0].append(abs(z - ref)); e[1].append(abs(z - ref) / ref)


def _by_band(split, bands):
    names = list(bands[2]) + (["unknown"] if band_n[(split, "unknown")] else [])
    return {b: {"n": band_n[(split, b)], "order_accuracy": band_ok[(split, b)] / band_n[(split, b)] if band_n[(split, b)] else None} for b in names}


depth_res = {"n": n_depth, "order_accuracy": order_ok / n_depth if n_depth else None,
             "abs_err_m_median": st.median(abs_err) if abs_err else None, "rel_err_median": st.median(rel_err) if rel_err else None,
             "order_accuracy_by_closer_depth": _by_band("closer_depth", DEPTH_BANDS),
             "order_accuracy_by_depth_ratio": _by_band("depth_ratio", RATIO_BANDS),
             "err_by_ref_depth": {b: {"n": len(err_by_band[b][0]), "abs_err_m_median": st.median(err_by_band[b][0]),
                                      "rel_err_median": st.median(err_by_band[b][1])} for b in DEPTH_BANDS[2] if b in err_by_band}}

# ---- gold hazards vs detections -----------------------------------------------------
n_haz = covered = seen = n_clear = n_img_haz = 0; concept_counter = Counter(); uncovered = Counter(); by_hour = Counter()
for r in gold_rows:
    clauses = gold_clauses(r.get("gold_label") or "")
    n_clear += sum(clear for _, _, clear in clauses)
    hazards = [(hour, clause) for hour, clause, clear in clauses if not clear]
    if not hazards:
        continue
    n_img_haz += 1
    img = decode_image(r["image"])
    dets = det.detect(img) if det is not None else []
    for hour, desc in hazards:
        n_haz += 1; by_hour[hour] += 1
        coco, other = match_concepts(strip_clock(desc))
        for c in coco:
            concept_counter[c] += 1
        for c in other:
            concept_counter[f"non-COCO:{c}"] += 1
        if coco:
            covered += 1
            lo, hi = CLOCK[hour]
            if any(d.label in coco and lo - 0.1 <= 0.5 * (d.bbox[0] + d.bbox[2]) <= hi + 0.1 for d in dets):
                seen += 1
        else:
            uncovered[(other or ["unnamed"])[0]] += 1
gold_res = {"n_images": len(gold_rows), "n_hazard_mentions": n_haz, "coco_coverable": covered, "coverable_rate": covered / n_haz if n_haz else None,
            "detected_in_direction": seen, "detected_rate_of_coverable": seen / covered if covered else None,
            "top_concepts": dict(concept_counter.most_common(20)), "top_uncoverable": dict(uncovered.most_common(15)),
            "n_images_with_hazards": n_img_haz, "n_clear_direction_mentions": n_clear,
            "hazard_mentions_by_hour": {h: by_hour[h] for h in CLOCK}}
summary = {"depth_mcqa": depth_res, "gold_hazards": gold_res, "smoke": args.smoke}
tag = re.sub(r"[^\w.-]", "_", args.out_tag)
dump_json(summary, OUT / ("summary" + ("_smoke" if args.smoke else "") + (f"_{tag}" if tag else "") + ".json"))
print(json.dumps(summary, indent=1)[:2500])

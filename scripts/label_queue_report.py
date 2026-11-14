"""Aggregate one rater's rubric v4 labels against the hidden design of a label queue.

    python scripts/label_queue_report.py --name priority160 --reviewer m1

Reads the answer key (`derived:` columns of results/egoblind/review_events_<name>.csv) and the rater's latest verdicts,
and writes totals only: usefulness by stratum, the rater's speak-or-not decision against the weak EgoBlind label, and
whether the proposed sentence was true by where the sentence came from. No per-moment answer is written, so a second
rater who has not finished can read the output without being unblinded on any item.

The queue is stratified, so these shares describe the strata, not EgoBlind as a whole. A model rater is reported as a
model. Writes results/egoblind/label_review_<name>_<reviewer>.{json,md}.
"""
import argparse, csv, json, sys

from _common import dump_json, ROOT
from seeforme.egoblind.label_queue import rater_report
from seeforme.egoblind.review import blind_row, latest_verdicts, rater_kind, two_axis

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="priority160")
ap.add_argument("--reviewer", required=True)
args = ap.parse_args()

rows = list(csv.DictReader(open(ROOT / "results" / "egoblind" / f"review_events_{args.name}.csv")))
store = ROOT / "data" / "review" / args.name / "verdicts.jsonl"
recs = [json.loads(l) for l in open(store)] if store.exists() else []
kind = rater_kind(recs, args.reviewer)
if kind is None:
    sys.exit(f"no verdicts by {args.reviewer!r} in {store}")
v = latest_verdicts(recs, reviewer=args.reviewer, human_only=False)
items, notes_sub = [], 0
for r in rows:
    k = blind_row(r)["key"]
    if k not in v:
        continue
    use, claim = two_axis(v[k])
    notes_sub += v[k].get("note", "").startswith("[sub]")
    items.append({"stratum": r["derived:stratum"], "weak_should_speak": {"1": True, "0": False}.get(r["derived:should_speak"].strip()),
                  "claim_source": r["derived:claim_source"], "usefulness": use, "claim": claim})
rep = rater_report(items)
rep.update({"queue": args.name, "reviewer": args.reviewer, "rater_kind": kind, "queue_size": len(rows), "notes_flagging_subtitles": int(notes_sub)})
out = ROOT / "results" / "egoblind"
dump_json(rep, out / f"label_review_{args.name}_{args.reviewer}.json")

pct = lambda a, b: f"{a}/{b} = {a / b:.0%}" if b else "0/0"
w = rep["against_weak_label"]
L = [f"Rater `{args.reviewer}` ({kind}) on queue `{args.name}`: {rep['n']} of {len(rows)} moments labelled. "
     f"Stratified sample: shares describe the strata, not EgoBlind as a whole.\n",
     "Usefulness: " + ", ".join(f"{k} {n}" for k, n in sorted(rep["usefulness"].items(), key=lambda kv: -kv[1])) + ".\n",
     "| stratum | moments | speak unprompted (of decided) | can't tell |", "|---|---|---|---|"]
for s, c in sorted(rep["by_stratum"].items(), key=lambda kv: -kv[1]["n"]):
    L.append(f"| {s} | {c['n']} | {pct(c['proactive'], c['decided'])} | {c['uncertain']} |")
L += ["", "| weak EgoBlind label | decided moments | rater says speak unprompted |", "|---|---|---|",
      f"| should speak (the user asked about it) | {w['weak_speak']['decided']} | {pct(w['weak_speak']['rater_proactive'], w['weak_speak']['decided'])} |",
      f"| quiet (the user asked and was told it is clear) | {w['weak_quiet']['decided']} | {pct(w['weak_quiet']['rater_proactive'], w['weak_quiet']['decided'])} |",
      "", f"Agreement with the weak label on speak-or-not: {w['agreement']:.2f} over decided moments ({w['excluded_uncertain']} can't-tell excluded, "
          f"not counted as silent; {w['no_weak_label']} moments nobody asked about have no weak label and are left out).",
      "", "| sentence came from | true | partly true | false | can't check |", "|---|---|---|---|---|"]
for s, c in sorted(rep["claim_by_source"].items()):
    L.append(f"| {s} | {c.get('correct', 0)} | {c.get('partially_correct', 0)} | {c.get('wrong', 0)} | {c.get('not_assessable', 0)} |")
L += ["", f"Notes flagging burned-in subtitles on the frames: {notes_sub} of {rep['n']}."]
if kind == "model":
    L += ["", "These are a model's labels. They are not human labels and are not used as training targets on their own."]
(out / f"label_review_{args.name}_{args.reviewer}.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

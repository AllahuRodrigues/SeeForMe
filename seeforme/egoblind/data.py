"""EgoBlind questions as weak supervision for cue scheduling.

EgoBlind is a VideoQA dataset: 1,392 first-person videos of blind users and 5.3K questions the users
asked at a known moment (`start-time/s`), each with up to four reference answers and a question type.
It has no object tracks and no communication windows, so it cannot label the scheduler directly. What
it does record is *when a blind user needed information, and about what*: a safety question at t_q
answered "No, a car is coming" says a car was action-relevant just before t_q, and a system that
only speaks after the user has had to ask is late. This module turns every proactive question
(safety warnings, navigation) into a weak event in the CampusBench annotation format:

    labels             COCO classes named in the answers (or in the question when the answer confirms it)
    should_speak       True for a confirmed hazard / landmark; False ("quiet window") for "no obstacle" answers
    earliest_useful_s  t_q - 6  (refined to the first detection of a matching label when a perception cache exists)
    ideal_s            t_q - 2
    latest_safe_s      t_q + 0.5   (the user had to ask: anything later is late)
    severity           1.0 safety warnings, 0.7 navigation
    coverable          whether the current perception stack can see the named object at all

Information-reading, tool-use, communication and resource questions are user-initiated queries,
not proactive cues; they are kept as `queries` (for FIND / ASK experiments) and produce no event.
Everything here is a heuristic label (`label_source = "egoblind_question"`); human-verified
annotations in the same format (docs/campus_annotation.md) take precedence when they exist.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..campus.world import EVENT_SPECS, LABEL_EVENT, EventAnnotation
from ..cues.compose import compose

# ---------------------------------------------------------------------------
# question types (train and test CSVs use different names for the same classes)
# ---------------------------------------------------------------------------

TYPE_MAP = {
    "information reading": "information",
    "safety warnings": "safety",
    "local navigation and positioning": "navigation",
    "navigation": "navigation",
    "resource acquisition and others": "resource",
    "other resources": "resource",
    "tool use": "tool_use",
    "communication and interaction": "communication",
    "social communication": "communication",
}
QUESTION_TYPES = ("safety", "navigation", "information", "resource", "tool_use", "communication")
PROACTIVE_SEVERITY = {"safety": 1.0, "navigation": 0.7}

# communication windows relative to t_q (seconds)
EARLIEST_BEFORE_S = 6.0
IDEAL_BEFORE_S = 2.0
LATEST_AFTER_S = 0.5
QUIET_BEFORE_S = 4.0
LOOKBACK_S = 10.0      # a cached detection this long before t_q may open the window
SUDDEN_S = 2.0         # a question this early in the clip cannot be anticipated
LABEL_SOURCE = "egoblind_question"


def normalize_type(raw: Optional[str]) -> str:
    key = (raw or "").strip().lower()
    return TYPE_MAP.get(key, key.replace(" ", "_") or "unknown")


@dataclass
class Question:
    question_id: str
    video_id: int
    split: str
    t_q: float
    qtype: str
    raw_type: str
    question: str
    answers: List[str] = field(default_factory=list)

    def text(self) -> str:
        return " ".join([self.question] + self.answers)


def load_questions(csv_path, split: str) -> List[Question]:
    out: List[Question] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answers = [(row.get(f"answer{i}") or "").strip() for i in range(4)]
            out.append(Question(question_id=row["question_id"], video_id=int(row["video_name"]), split=split,
                                t_q=float(row["start-time/s"]), qtype=normalize_type(row.get("type")),
                                raw_type=(row.get("type") or "").strip(), question=(row.get("question") or "").strip(),
                                answers=[a for a in answers if a]))
    return out


def group_by_video(questions: Sequence[Question]) -> Dict[int, List[Question]]:
    by: Dict[int, List[Question]] = {}
    for q in questions:
        by.setdefault(q.video_id, []).append(q)
    for v in by.values():
        v.sort(key=lambda q: q.t_q)
    return by


# ---------------------------------------------------------------------------
# object vocabulary: COCO classes the detector can see, and concepts it cannot
# ---------------------------------------------------------------------------

COCO_KEYWORDS: Dict[str, List[str]] = {
    "person": [r"\bpersons?\b", r"\bpeople\b", r"\bpedestrians?\b", r"\bpassers?-?by\b", r"\b(?:man|woman|men|women|boy|girl|guy|lady|gentleman)\b",
               r"\bsome(?:one|body)\b", r"\bstaff\b", r"\bworkers?\b", r"\bdrivers?\b", r"\bpassengers?\b", r"\bcrowds?\b", r"\bcrowded\b",
               r"\bchild(?:ren)?\b", r"\bkids?\b", r"\bconductor\b", r"\bcashiers?\b", r"\bofficers?\b", r"\bguards?\b", r"\bvendors?\b",
               r"\bshop ?owner\b", r"\bcompanion\b", r"\bcustomers?\b", r"\bcyclists?\b", r"\briders?\b", r"\bvolunteers?\b", r"\bwaiters?\b"],
    "bicycle": [r"\bbicycles?\b", r"\bbikes?\b", r"\bcyclists?\b", r"\btricycles?\b"],
    "motorcycle": [r"\bmotor ?(?:bike|cycle)s?\b", r"\bscooters?\b", r"\bmopeds?\b", r"\be-?bikes?\b", r"\btricycles?\b",
                   r"\belectric (?:bike|bicycle|scooter|vehicle|car|motorcycle)s?\b"],
    "car": [r"\bcars?\b", r"\bvehicles?\b", r"\btaxis?\b", r"\bvans?\b", r"\bsedans?\b", r"\bsuvs?\b", r"\bautomobiles?\b"],
    "bus": [r"\bbus(?:es)?\b", r"\bcoach(?:es)?\b"],
    "truck": [r"\btrucks?\b", r"\blorr(?:y|ies)\b", r"\bpickups?\b", r"\bvehicles?\b"],
    "train": [r"\btrains?\b", r"\b(?:subway|metro) (?:train|car)s?\b", r"\btrams?\b"],
    "traffic light": [r"\btraffic (?:light|signal)s?\b", r"\b(?:signal|crossing|pedestrian|red|green) lights?\b"],
    "stop sign": [r"\bstop signs?\b"],
    "bench": [r"\bbench(?:es)?\b"],
    "chair": [r"\bchairs?\b", r"\bseats?\b", r"\bstools?\b", r"\barmchairs?\b"],
    "couch": [r"\bsofas?\b", r"\bcouch(?:es)?\b"],
    "dining table": [r"\btables?\b", r"\bdesks?\b", r"\bcounters?\b"],
    "potted plant": [r"\bplants?\b", r"\bflower ?pots?\b", r"\bpotted\b"],
    "suitcase": [r"\bsuitcases?\b", r"\bluggage\b", r"\btrolley cases?\b"],
    "backpack": [r"\bbackpacks?\b", r"\bschool ?bags?\b", r"\bbags?\b"],
    "handbag": [r"\bhandbags?\b", r"\bpurses?\b", r"\bbags?\b"],
    "umbrella": [r"\bumbrellas?\b"],
    "dog": [r"\bdogs?\b", r"\bpupp(?:y|ies)\b"],
    "cat": [r"\bcats?\b", r"\bkittens?\b"],
    "horse": [r"\bhorses?\b", r"\bpon(?:y|ies)\b"],
    "bird": [r"\bbirds?\b", r"\bpigeons?\b"],
    "elephant": [r"\belephants?\b"],
    "zebra": [r"\bzebras?\b"],
    "bottle": [r"\bbottles?\b"],
    "cup": [r"\bcups?\b", r"\bmugs?\b"],
    "wine glass": [r"\bwine ?glass(?:es)?\b"],
    "bowl": [r"\bbowls?\b"],
    "fork": [r"\bforks?\b"],
    "knife": [r"\bkni(?:fe|ves)\b"],
    "spoon": [r"\bspoons?\b"],
    "cell phone": [r"\b(?:cell|mobile) ?phones?\b", r"\bphones?\b", r"\bsmartphones?\b"],
    "laptop": [r"\blaptops?\b"],
    "tv": [r"\btvs?\b", r"\btelevisions?\b", r"\bmonitors?\b"],
    "keyboard": [r"\bkeyboards?\b"],
    "remote": [r"\bremotes?\b"],
    "book": [r"\bbooks?\b"],
    "clock": [r"\bclocks?\b"],
    "vase": [r"\bvases?\b"],
    "scissors": [r"\bscissors\b"],
    "bed": [r"\bbeds?\b"],
    "toilet": [r"\btoilets?\b"],
    "sink": [r"\bsinks?\b"],
    "refrigerator": [r"\b(?:refrigerator|fridge)s?\b"],
    "microwave": [r"\bmicrowaves?\b"],
    "oven": [r"\bovens?\b"],
    "toaster": [r"\btoasters?\b"],
    "fire hydrant": [r"\b(?:fire )?hydrants?\b"],
    "parking meter": [r"\bparking meters?\b"],
    "skateboard": [r"\bskateboards?\b"],
    "sports ball": [r"\bballs?\b"],
    "banana": [r"\bbananas?\b"], "apple": [r"\bapples?\b"], "orange": [r"\boranges?\b"], "sandwich": [r"\bsandwich(?:es)?\b"],
    "pizza": [r"\bpizzas?\b"], "cake": [r"\bcakes?\b"], "donut": [r"\bdo(?:ugh)?nuts?\b"],
}

# action-relevant things the COCO detector cannot name (reported as perception coverage gaps)
NON_COCO_KEYWORDS: Dict[str, List[str]] = {
    "stairs": [r"\bstair(?:s|case|way)?\b", r"\bsteps? (?:up|down)\b", r"\bescalators?\b"],
    "door": [r"\bdoors?\b", r"\bgates?\b", r"\bentrances?\b", r"\bdoorway\b"],
    "curb": [r"\bcurbs?\b", r"\bkerbs?\b", r"\bedge of the (?:sidewalk|road|platform|pavement)\b", r"\bdrop-?offs?\b"],
    "pole": [r"\bpoles?\b", r"\bposts?\b", r"\bpillars?\b", r"\bcolumns?\b", r"\bbollards?\b", r"\blamp ?posts?\b"],
    "wall": [r"\bwalls?\b"],
    "fence": [r"\bfences?\b", r"\brailings?\b", r"\bhandrails?\b", r"\bguardrails?\b", r"\bbarriers?\b"],
    "tree": [r"\btrees?\b", r"\bbranch(?:es)?\b", r"\bbush(?:es)?\b", r"\bhedges?\b"],
    "box": [r"\bbox(?:es)?\b", r"\bcartons?\b", r"\bcrates?\b"],
    "cart": [r"\bcarts?\b", r"\btrolle(?:y|ys|ies)\b"],
    "trash can": [r"\btrash ?(?:can|bin)s?\b", r"\bgarbage (?:can|bin)s?\b", r"\b(?:dust|rubbish) ?bins?\b", r"\bbins?\b"],
    "cone": [r"\b(?:traffic )?cones?\b"],
    "obstacle": [r"\bobstacles?\b", r"\bobstructions?\b", r"\bblock(?:ed|ing|age)\b", r"\bin (?:your|the) way\b"],
    "ground hazard": [r"\bpuddles?\b", r"\bholes?\b", r"\bpotholes?\b", r"\buneven\b", r"\bslippery\b", r"\bwet floor\b", r"\bditch(?:es)?\b", r"\bgaps?\b"],
    "crossing": [r"\bcrosswalks?\b", r"\bpedestrian crossings?\b", r"\bintersections?\b", r"\bcross ?roads?\b"],
    "road": [r"\broads?\b", r"\bstreets?\b", r"\bsidewalks?\b", r"\bpavements?\b", r"\bpaths?\b", r"\blanes?\b", r"\btactile paving\b", r"\bblind path\b"],
    "elevator": [r"\belevators?\b", r"\blifts?\b"],
    "shelf": [r"\bshel(?:f|ves)\b", r"\bracks?\b"],
    "sign": [r"\bsigns?\b", r"\bsignage\b"],
    "machine": [r"\bmachines?\b", r"\bkiosks?\b", r"\bturnstiles?\b", r"\bticket gates?\b"],
    "water": [r"\briver\b", r"\blake\b", r"\bpond\b", r"\bpool\b", r"\bfountain\b"],
    "animal": [r"\banimals?\b", r"\bgoats?\b", r"\bducks?\b", r"\bchickens?\b", r"\bdeer\b", r"\bmonkeys?\b"],
}

_COCO_RE = {k: [re.compile(p) for p in v] for k, v in COCO_KEYWORDS.items()}
_NON_COCO_RE = {k: [re.compile(p) for p in v] for k, v in NON_COCO_KEYWORDS.items()}
# phrases whose words would otherwise match the wrong class
_REWRITES = [(re.compile(r"\b(?:1[0-2]|[1-9]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[\s-]*o['’]?\s*clock\b|\bo['’]clock\b"), " direction "),   # "at 2 o'clock" is not the COCO class clock
             (re.compile(r"\bhot ?dogs?\b"), "hotdog"), (re.compile(r"\bzebra (?:crossing|line|stripe)s?\b"), "crosswalk"),
             (re.compile(r"\bsteps? (?:ahead|away|forward|in front|behind|to the)\b"), "paces"), (re.compile(r"\bbus stop\b"), "busstop"),
             (re.compile(r"\bsubway station\b"), "station"), (re.compile(r"\btrain station\b"), "station"),
             (re.compile(r"\bcar(?:riage)? door\b"), "vehicledoor"), (re.compile(r"\bparking lot\b"), "parkinglot")]


def _first_match(text: str, pats) -> Optional[int]:
    best = None
    for p in pats:
        m = p.search(text)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


# EgoBlind label or concept -> names an open-vocabulary detector may use for the same thing (perception.NAV_VOCAB)
LABEL_ALIASES: Dict[str, List[str]] = {
    "dog": ["animal"], "cat": ["animal"], "horse": ["animal"], "cow": ["animal"], "sheep": ["animal"], "bird": ["animal"],
    "elephant": ["animal"], "bear": ["animal"], "motorcycle": ["scooter"], "trash can": ["trash can"], "ground hazard": ["ground hazard"],
}


def detector_labels(labels: Sequence[str], concepts: Sequence[str], vocab: Sequence[str]) -> List[str]:
    """Labels and concepts of an event as an open-vocabulary detector with this vocabulary would name them."""
    vocab = set(vocab)
    out: List[str] = []
    for name in list(labels) + [c for c in concepts if c not in GENERIC_CONCEPTS]:
        for cand in [name] + LABEL_ALIASES.get(name, []):
            if cand in vocab and cand not in out:
                out.append(cand)
    return out


def match_concepts(text: str) -> Tuple[List[str], List[str]]:
    """(COCO labels, non-COCO concepts) named in `text`, each ordered by first mention."""
    t = text.lower()
    for pat, rep in _REWRITES:
        t = pat.sub(rep, t)
    coco = sorted(((pos, lab) for lab, pats in _COCO_RE.items() if (pos := _first_match(t, pats)) is not None))
    other = sorted(((pos, lab) for lab, pats in _NON_COCO_RE.items() if (pos := _first_match(t, pats)) is not None))
    return [lab for _, lab in coco], [lab for _, lab in other]


# ---------------------------------------------------------------------------
# polarity: did the answer confirm something action-relevant, or confirm that the way is clear?
# ---------------------------------------------------------------------------

_DONTKNOW = re.compile(r"\b(?:don'?t|do not|can'?t|cannot) (?:know|tell|see)\b|\bnot sure\b|\bunclear\b|\bunknown\b")
_YES = re.compile(r"^\W*(?:yes|yeah|yep|yup|correct|right|sure)\b")
_NO = re.compile(r"^\W*(?:no|nope|none|not really|nothing)\b")
# explicit statements that nothing action-relevant is there ...
_CLEAR = re.compile(r"\bno (?:visible |obvious |potential |immediate |apparent |other |nearby |significant )?(?:obstacles?|hazards?|dangers?|vehicles?|cars?|traffic|people|one|person|risks?|obstructions?)\b"
                    r"|\bnothing\b|\b(?:path|road|way|route|sidewalk|area|front|floor|ground) (?:ahead |in front )?(?:is|looks|seems|appears) (?:mostly |quite |pretty |fairly )?(?:clear|safe|empty|free)\b"
                    r"|\bit'?s (?:clear|safe)\b|\bit is (?:clear|safe)\b|\bis clear\b|\bclear of\b|\bsafe to\b|\bno danger\b|\bare no\b|\bis no\b")
# "not safe to step forward" / "unsafe" must not read as clear through "safe to" ("not clear" is left out: "it's not
# clear whether ..." expresses uncertainty)
_UNSAFE = re.compile(r"\bnot (?:\w+ )?safe\b|\bunsafe\b|\bisn'?t safe\b")
# ... and advice that only implies it ("you can go"), accepted for safety / presence questions but not for "where is X?"
_CLEAR_ADVICE = re.compile(r"\byou can (?:go|walk|cross|proceed|continue|pass)\b|\bgo (?:ahead|straight)\b")
# question kinds decide what "yes" means. safety_q ("is it safe?", "can I cross?") and clear_q ("is the path clear?"):
# yes = nothing in the way. hazard_q ("is it dangerous?", "any risk?") and presence_q ("is there anything ahead?"):
# yes = something is there. The strong cue words (safe / clear / danger) win over the generic presence words, and among
# strong cues the first one in the question decides ("is there any danger, is the road clear?" is a hazard question).
_SAFETY_Q = re.compile(r"\bsafe(?:ly)?\b|\bok(?:ay)? (?:to|for me to)\b|\bcan i (?:cross|walk|go|pass|proceed|continue|step)\b|\bshould i (?:cross|go|proceed|continue)\b")
_HAZARD_Q = re.compile(r"\bdanger(?:s|ous)?\b|\brisks?\b|\brisky\b|\bhazard(?:s|ous)?\b")
_CLEAR_Q = re.compile(r"\bclear\b|\bunobstructed\b|\bfree (?:of|from)\b|\b(?:path|way|road|sidewalk|route|street|walkway) (?:\w+ )?(?:is |looks )?free\b")
_CLEAR_OF = re.compile(r"\b(?:clear|free) (?:of|from)\b(.*)")
_PRESENCE_Q = re.compile(r"\b(?:is|are) there\b|\bany\b|\banything\b|\banyone\b|\bobstacles?\b|\bhazards?\b|\bblock|\bbump|\bhit\b|\bin (?:my|the) (?:way|path)\b|\bclear\b|\btraffic\b|\bvehicles?\b|\bcoming\b|\bnearby\b|\bahead\b")


def question_kind(question: str) -> str:
    """safety_q | clear_q (yes = clear) or hazard_q | presence_q (yes = something there) or other."""
    ql = question.lower()
    strong = [(m.start(), kind) for kind, rx in (("safety_q", _SAFETY_Q), ("clear_q", _CLEAR_Q), ("hazard_q", _HAZARD_Q))
              for m in [rx.search(ql)] if m]
    if strong:
        return min(strong)[1]
    if _PRESENCE_Q.search(ql):
        return "presence_q"
    return "other"


YES_MEANS_CLEAR = {"safety_q", "clear_q"}


def answer_polarity(answer: str) -> str:
    """'yes' | 'no' | 'clear' | 'clear_advice' | 'unknown' | 'none' for one reference answer. An explicit
    "the path is clear" / "no obstacles" wins over the leading yes / no ("Yes, the path is clear")."""
    al = answer.lower().strip()
    if not al:
        return "none"
    if _DONTKNOW.search(al):
        return "unknown"
    if _UNSAFE.search(al):
        return "no"
    if _CLEAR.search(al):
        return "clear"
    if _YES.match(al):
        return "yes"
    if _NO.match(al):
        return "no"
    if _CLEAR_ADVICE.search(al):
        return "clear_advice"
    return "none"


# words a negated phrase can carry without naming a present object ("no obstacles", "clear of people")
_NEGATED = re.compile(r"\b(?:no|not any|without|free of|clear of|aren'?t any|isn'?t any|are no|is no|no other|no more)\s+(?:\w+\s+){0,2}?"
                      r"(?:obstacles?|hazards?|dangers?|vehicles?|cars?|traffic|people|persons?|pedestrians?|one|bikes?|bicycles?|objects?|things?|risks?|obstructions?|"
                      r"stairs|steps|doors?|poles?|animals?|dogs?|cats?|trees?|boxes|chairs?|tables?|buses|trucks?|motorcycles?|scooters?)\b")
_CONTRAST = re.compile(r"\b(?:but|however|although|though|except|apart from|other than)\b")
# concepts that name the question's topic rather than a present object
GENERIC_CONCEPTS = {"road", "obstacle", "crossing"}


def strip_negated(text: str) -> str:
    return _NEGATED.sub(" ", text.lower())


def contrast_clauses(text: str) -> str:
    """The part of each sentence after 'but' / 'however' / ... ('mostly clear, but there are people' -> 'there are people')."""
    out = []
    for sent in re.split(r"[.;!?]", text.lower()):
        m = _CONTRAST.search(sent)
        if m:
            out.append(sent[m.end():])
    return " ".join(out)


@dataclass
class WeakLabel:
    status: str                 # present | clear | skip
    labels: List[str]           # COCO labels, primary first
    concepts: List[str]         # non-COCO concepts named
    reason: str = ""


def weak_label(q: Question) -> WeakLabel:
    """Turn one proactive question + answers into present / clear / skip."""
    if q.qtype not in PROACTIVE_SEVERITY:
        return WeakLabel("skip", [], [], "query_type")
    answers = [a for a in q.answers if a]
    kind = question_kind(q.question)
    pols = [answer_polarity(a) for a in answers]
    pol = next((p for p in pols if p != "unknown"), pols[0] if pols else "none")
    full = strip_negated(" ".join(answers))
    # where may a *present* object be named? anywhere when the answer affirms that something is there: "yes" to a
    # presence / hazard question, "no" to a safety / clear question ("No, a car is coming"), or no polarity at all;
    # after a clear answer ("clear", "no obstacle", "yes" to "is it safe?") only in a contrast clause ("clear, but people ahead")
    yes_clear = kind in YES_MEANS_CLEAR
    affirmative = pol == "none" or (pol == "yes" and not yes_clear) or (pol == "no" and yes_clear)
    obj_text = full if affirmative else contrast_clauses(full)
    coco_a, other_a = match_concepts(obj_text)
    other_a = [c for c in other_a if c not in GENERIC_CONCEPTS]
    coco_q, other_q = match_concepts(q.question)
    other_q = [c for c in other_q if c not in GENERIC_CONCEPTS]
    if coco_a or other_a:
        return WeakLabel("present", coco_a, other_a, "answer_names_object")
    if pol == "unknown" or (pol == "none" and kind == "other"):
        return WeakLabel("skip", [], [], "no_information")
    if pol in ("no", "clear", "clear_advice", "yes") and not affirmative:
        # a clear answer that still names a *new* object outside a contrast clause ("No. There is a person six paces ahead")
        # is ambiguous; objects the question itself named ("... between you and the train entrance") are references
        # (only detector classes count: a non-COCO word in a clear answer is usually a reference, "to the train entrance")
        c2, o2 = match_concepts(full)
        c2 = [c for c in c2 if c not in coco_q]
        if c2:
            return WeakLabel("skip", c2, [c for c in o2 if c not in GENERIC_CONCEPTS], "ambiguous_clear")
    if kind in ("presence_q", "hazard_q"):
        if pol == "yes":
            return WeakLabel("present", coco_q, other_q, "answer_confirms_question") if (coco_q or other_q) else WeakLabel("present", [], [], "present_unnamed")
        if pol in ("no", "clear", "clear_advice"):
            return WeakLabel("clear", [], [], "user_confirmed_clear")
        return WeakLabel("skip", [], [], "no_information")
    if kind == "clear_q":
        if pol in ("yes", "clear", "clear_advice"):
            return WeakLabel("clear", [], [], "user_confirmed_clear")
        if pol == "no":
            # "is the counter clear of sharp objects? No." names the hazard after "clear of"; "is the path clear to the
            # door? No." names a destination, not the obstacle
            m = _CLEAR_OF.search(q.question.lower())
            c3, o3 = match_concepts(m.group(1)) if m else ([], [])
            o3 = [c for c in o3 if c not in GENERIC_CONCEPTS]
            return WeakLabel("present", c3, o3, "not_clear_per_answer") if (c3 or o3) else WeakLabel("present", [], [], "present_unnamed")
        return WeakLabel("skip", [], [], "no_information")
    if kind == "safety_q":
        if pol in ("yes", "clear", "clear_advice"):
            return WeakLabel("clear", [], [], "user_confirmed_clear")
        if pol == "no":
            return WeakLabel("present", coco_q, other_q, "unsafe_per_answer") if (coco_q or other_q) else WeakLabel("present", [], [], "present_unnamed")
        return WeakLabel("skip", [], [], "no_information")
    if pol == "clear":
        return WeakLabel("clear", [], [], "user_confirmed_clear")
    return WeakLabel("skip", [], [], "no_information")


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def coverability(labels: Sequence[str], concepts: Sequence[str]) -> Tuple[bool, str, str]:
    """(coverable, reason, primary label). Coverable = detectable by the COCO detector *and* in the P-CACS taxonomy."""
    for lab in labels:
        if LABEL_EVENT.get(lab, "decor") != "decor":
            return True, "", lab
    if labels:
        return False, "taxonomy_decor", labels[0]
    if concepts:
        return False, "not_detectable", concepts[0]
    return False, "no_object_named", "unknown"


def _window(t_q: float) -> Tuple[float, float, float]:
    earliest = max(0.0, t_q - EARLIEST_BEFORE_S)
    latest = t_q + LATEST_AFTER_S
    ideal = min(max(t_q - IDEAL_BEFORE_S, earliest), latest)
    return round(earliest, 2), round(ideal, 2), round(latest, 2)


def events_for_video(questions: Sequence[Question]) -> Tuple[List[EventAnnotation], List[Dict], Dict]:
    """Weak events (should-speak and quiet windows) plus the remaining questions as queries."""
    events: List[EventAnnotation] = []
    queries: List[Dict] = []
    stats = {"n_questions": len(questions), "present": 0, "clear": 0, "skip": 0, "coverable": 0,
             "uncoverable": {"taxonomy_decor": 0, "not_detectable": 0, "no_object_named": 0}, "by_type": {}}
    for q in sorted(questions, key=lambda q: q.t_q):
        stats["by_type"][q.qtype] = stats["by_type"].get(q.qtype, 0) + 1
        wl = weak_label(q)
        base_detail = {"question_id": q.question_id, "qtype": q.qtype, "question": q.question, "answers": list(q.answers),
                       "t_q": q.t_q, "label_source": LABEL_SOURCE, "labels": list(wl.labels), "concepts": list(wl.concepts),
                       "weak_reason": wl.reason}
        if wl.status == "skip":
            stats["skip"] += 1
            queries.append({**base_detail, "status": "skip"})
            continue
        obj_id = len(events) + 1
        if wl.status == "clear":
            stats["clear"] += 1
            e0 = round(max(0.0, q.t_q - QUIET_BEFORE_S), 2)
            events.append(EventAnnotation(obj_id, "none", "object", "quiet", None, False, "user_confirmed_clear", 0.0, "none", False,
                                          e0, None, round(q.t_q + LATEST_AFTER_S, 2), None, 0.0, None, None, False, False, (), "",
                                          {**base_detail, "coverable": False, "coverable_reason": "quiet", "detected": None, "first_seen_s": None}))
            continue
        stats["present"] += 1
        coverable, why, primary = coverability(wl.labels, wl.concepts)
        if coverable:
            stats["coverable"] += 1
        else:
            stats["uncoverable"][why] += 1
        etype = LABEL_EVENT.get(primary) or (primary if (primary in EVENT_SPECS or primary not in COCO_KEYWORDS) else "decor")
        spec = EVENT_SPECS.get(etype)
        severity = PROACTIVE_SEVERITY[q.qtype]
        slots = ["object"] + (list(spec.slots) if spec is not None and spec.severity > 0 else ["distance"])
        moving = etype in ("vehicle", "bike", "person_moving")
        if moving and "motion" not in slots:
            slots.append("motion")
        earliest, ideal, latest = _window(q.t_q)
        cue = compose(slots, primary, "object", None) if primary != "unknown" else "Something ahead."
        base = {"label": primary, "event_type": etype, "labels": list(wl.labels), "coverable": coverable, "coverable_reason": why,
                "window": [earliest, ideal, latest], "slots": list(slots), "cue": cue, "sudden": q.t_q < SUDDEN_S}
        events.append(EventAnnotation(obj_id, primary, "object", etype, None, True, "", severity,
                                      "high" if severity >= 0.9 else "medium", severity >= 0.75, earliest, ideal, latest, None,
                                      1.0, None, None, moving, q.t_q < SUDDEN_S, tuple(slots), cue,
                                      {**base_detail, "coverable": coverable, "coverable_reason": why, "detected": None, "first_seen_s": None,
                                       "question_kind": question_kind(q.question), "base": base}))
    return events, queries, stats


_OUTDOOR = re.compile(r"\b(?:street|road|cross|crossing|traffic|cars?|bus|sidewalk|outside|outdoor|park|station|intersection|bike|bicycle|crosswalk|curb|vehicle|walk(?:ing)?|pedestrian|tree|weather|rain|snow)\b")
_INDOOR = re.compile(r"\b(?:kitchen|stove|room|table|phone|cook|indoor|elevator|store|shop|counter|shelf|shelves|fridge|refrigerator|oven|microwave|desk|bed|bathroom|toilet|sink|cup|bowl|plate|screen|button|aisle|cashier|restaurant|hallway|office)\b")


def guess_outdoor(questions: Sequence[Question]) -> bool:
    """Pick the outdoor or indoor metric-depth model for a video from what its questions talk about."""
    text = " ".join(q.text() for q in questions).lower()
    return len(_OUTDOOR.findall(text)) > len(_INDOOR.findall(text))

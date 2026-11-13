"""VIABench proactive reminders as human labels of when an assistant should speak unprompted.

Each reminder is an interval [start_time, end_time] in a blind user's own video, a task category and the sentence an
assistant should say, marked by human annotators over the whole video. That is direct supervision for the question the
EgoBlind weak labels only approximate. The benchmark scores recall only, so anything built on it here also reports
speech outside every reminder.

Label rule for one candidate at one decision (`decision_label`): 1 if a reminder that names the candidate's object is
active (with a small lead and lag), 0 if no reminder of any kind is within `guard` seconds, None otherwise. None is
masked: a nearby reminder about something else, or the seconds just around a reminder, do not say that this candidate
was not worth mentioning.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..campus.world import EventAnnotation

SPLITS = ("via_train", "via_dev", "via_test")            # 70 / 20 / 10 by source video; via_test is a lockbox

# description keyword -> label. COCO names where the detector has the class, structure names otherwise.
_KEYWORDS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("pedestrian", "pedestrians", "person", "people", "passerby", "passersby", "child", "children", "crowd", "man", "woman", "someone"), "person"),
    (("bicycle", "bicycles", "bike", "bikes", "cyclist", "shared bike", "shared bikes"), "bicycle"),
    (("motorcycle", "motorcycles", "motorbike", "scooter", "scooters", "moped", "e-bike", "e-bikes", "electric bike", "electric bikes",
      "electric vehicle", "electric vehicles", "tricycle"), "motorcycle"),
    (("car", "cars", "vehicle", "vehicles", "taxi", "van", "suv", "sedan"), "car"),
    (("truck", "trucks", "lorry"), "truck"),
    (("bus", "buses"), "bus"),
    (("dog", "dogs"), "dog"),
    (("chair", "chairs", "stool", "stools"), "chair"),
    (("bench", "benches"), "bench"),
    (("table", "tables"), "dining table"),
    (("traffic light", "traffic lights", "signal light"), "traffic light"),
    (("fire hydrant",), "fire hydrant"),
    (("suitcase", "luggage"), "suitcase"),
    (("umbrella", "umbrellas"), "umbrella"),
    (("potted plant", "flower pot", "flowerpot", "planter", "planters"), "potted plant"),
    (("stairs", "stair", "staircase", "steps", "step", "escalator", "escalators"), "stairs"),
    (("door", "doors", "entrance", "exit", "gate", "doorway", "elevator", "lift"), "door"),
    (("pole", "poles", "pillar", "pillars", "bollard", "bollards", "post", "posts", "column", "lamp post", "utility pole"), "pole"),
    (("railing", "railings", "handrail", "handrails", "fence", "fences", "guardrail", "barrier", "barriers"), "railing"),
    (("curb", "kerb"), "curb"),
    (("crosswalk", "zebra crossing", "crossing"), "crosswalk"),
    (("tactile paving", "blind path", "tactile path"), "tactile paving"),
)
_TASK_LABEL = {"Stairs Up/Down": "stairs", "Staircase Up/Down": "stairs", "Escalator": "stairs", "Box Elevator": "door",
               "Exit/Entry Recognition": "door", "Crosswalk": "crosswalk", "Blind Path Recognition": "tactile paving",
               "Pedestrian Traffic Light": "traffic light"}
_PATTERNS = [(re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True)) + r")\b"), lab) for keys, lab in _KEYWORDS]


@dataclass(frozen=True)
class Reminder:
    qid: int
    start: float
    end: float
    task: str
    description: str
    labels: Tuple[str, ...] = ()


def detector_labels(description: str, task: str = "") -> List[str]:
    """Object labels a reminder is about, in order of first mention; the task category is the fallback."""
    text = (description or "").lower()
    found = []
    for pat, lab in _PATTERNS:
        m = pat.search(text)
        if m:
            found.append((m.start(), lab))
    labels = []
    for _, lab in sorted(found):
        if lab not in labels:
            labels.append(lab)
    if not labels and task in _TASK_LABEL:
        labels.append(_TASK_LABEL[task])
    return labels


def parse_reminders(raw: Dict) -> Dict[str, Dict]:
    """proactive_reminder.json -> {video path: {uid, caption, reminders}} with reminders in time order."""
    out = {}
    for key, v in raw.items():
        rem = [Reminder(int(a["qid"]), float(a["start_time"]), float(a["end_time"]), a.get("task", ""), a.get("description", ""),
                        tuple(detector_labels(a.get("description", ""), a.get("task", ""))))
               for a in v.get("annotations", []) if float(a["end_time"]) >= float(a["start_time"])]
        out[key] = {"uid": v.get("video_uid", key), "caption": v.get("caption", ""), "reminders": sorted(rem, key=lambda r: (r.start, r.qid))}
    return out


def clip_reminders(reminders: Sequence[Reminder], max_s: float) -> List[Reminder]:
    """Reminders that end inside the processed part of the video."""
    return [r for r in reminders if r.end <= max_s + 1e-6]


def split_of(key: str, seed: int = 0) -> str:
    """Stable 70 / 20 / 10 split by source video (each VIABench video is one source)."""
    u = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return SPLITS[0] if u < 0.7 else SPLITS[1] if u < 0.9 else SPLITS[2]


def events_from_reminders(reminders: Iterable[Reminder]) -> List[EventAnnotation]:
    ev = []
    for r in reminders:
        labels = list(r.labels)
        ev.append(EventAnnotation(
            obj_id=r.qid, label=labels[0] if labels else "unknown", kind="object", event_type="hazard_static", text=None,
            should_speak=True, reason="", severity=1.0, urgency="high", interrupt_current_speech=False,
            earliest_useful_s=r.start, ideal_s=r.start, latest_safe_s=r.end, t_contact=None, path_intersection=1.0,
            distance_at_ideal_m=None, direction_at_ideal_deg=None, moving=False, sudden=False, minimum_information=("object",),
            ideal_cue=r.description,
            detail={"t_q": r.end, "labels": labels, "task": r.task, "qtype": r.task, "description": r.description,
                    "question": r.description, "label_source": "viabench_reminder"}))
    return ev


def decision_label(t: float, label: str, reminders: Sequence[Reminder], lead: float = 1.0, lag: float = 0.5, guard: float = 3.0) -> Optional[int]:
    """1, 0 or None (masked) for a candidate with detector label `label` at decision time `t`."""
    if any(label in r.labels and r.start - lead <= t <= r.end + lag for r in reminders):
        return 1
    if any(r.start - guard <= t <= r.end + guard for r in reminders):
        return None
    return 0

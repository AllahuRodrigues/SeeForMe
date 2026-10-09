"""Predictive Context-Aware Cue Scheduling (P-CACS).

CACS asked "which detected object is important?". P-CACS asks, for every tracked event,
"will this change what the user should do in the next few seconds, what is the shortest
useful message, and when should it be said?", and treats speech as a scarce channel.

For candidate i at time t:

    U_i = w_h H_i + w_tau H_i f(TTC_i) + w_a A_i (0.3 + 0.7 P_i) + w_g G_i + w_n N_i (0.5 + 0.5 H_i)
          - w_u Unc_i - lambda_s SpeechCost_i - lambda_q QueueCost_i

    H   hazard = class severity x P(path intersection)         (signs: importance x legibility)
    f   urgency from time-to-contact, 1 at the reaction time, 0 at the horizon
    A   actionability of the event class
    G   relevance to the user's stated goal (FIND mode)
    N   novelty (never announced, moved closer, long since announced)
    Unc uncertainty from detector confidence and depth confidence
    SpeechCost  seconds the message would occupy the channel
    QueueCost   SpeechCost x most urgent *other* hazard it would block

Actions: SILENCE, SPEAK (channel free, U >= theta_s, budget allows), INTERRUPT (channel
busy, U >= theta_i and clearly above the message being spoken), HAPTIC (imminent hazard
while speech is unavailable or already given). A token bucket limits how much of the
time non-urgent speech may occupy; urgent hazards bypass it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from ..campus.world import EVENT_SPECS, LABEL_EVENT, default_event_type
from ..geometry.camera import CameraModel
from ..types import Cue, Observation, Track, bbox_center
from .compose import compose, speech_seconds
from .policy import CuePolicy
from .text_importance import KeywordTextImportance

ACTIONS = ("silence", "speak", "interrupt", "haptic")

ACTIONABILITY = {
    "stairs": 1.0, "curb": 1.0, "vehicle": 1.0, "bike": 1.0, "hazard_static": 0.8, "barrier": 0.85,
    "person_moving": 0.6, "person_static": 0.5, "door": 0.6, "landmark": 0.5, "sign_nav": 0.7, "goal": 0.9,
    "unknown_obstacle": 0.7, "decor": 0.0, "sign_decor": 0.0,
}
MIN_SLOTS = {
    "stairs": ("distance", "action"), "curb": ("distance", "action"), "vehicle": ("motion", "direction"),
    "bike": ("motion", "direction"), "hazard_static": ("distance",), "barrier": ("distance",),
    "person_moving": ("motion",), "person_static": ("distance",), "door": ("distance", "detail"),
    "landmark": ("distance", "direction"), "sign_nav": ("direction",), "goal": ("distance", "direction"),
    "unknown_obstacle": ("distance",),
}


@dataclass
class PCACSWeights:
    """Defaults are the operating point found by scripts/tune_pcacs.py on the CampusBench train split
    (random search on the communication score); w_goal is set so that a FIND request is always honoured."""
    w_hazard: float = 1.43
    w_ttc: float = 0.42
    w_action: float = 1.06
    w_goal: float = 2.5
    w_novelty: float = 0.40
    w_uncert: float = 0.90
    w_text: float = 1.52
    lam_speech: float = 0.48
    lam_queue: float = 1.13
    w_aware: float = 0.0     # awareness term: a severe object close in front, even off the collision course (0 = off, the CampusBench operating point)


@dataclass
class PCACSConfig:
    weights: PCACSWeights = field(default_factory=PCACSWeights)
    threshold_speak: float = 0.92
    threshold_interrupt: float = 2.14
    interrupt_margin: float = 0.54
    critical_severity: float = 0.9       # a hazard this severe, inside its reaction horizon, always cuts a non-critical message
    critical_urgency: float = 0.6
    haptic_ttc: float = 1.5
    haptic_min_path: float = 0.6
    haptic_min_severity: float = 0.5
    haptic_after_speech_s: float = 3.0   # a haptic only reinforces a cue given at least this long ago
    haptic_repeat_s: float = 1.0
    budget_capacity_s: float = 6.0
    budget_refill: float = 0.6           # seconds of speech earned per second (~target occupancy)
    urgent_bypass: float = 0.57          # H * f(TTC) above this ignores the budget
    min_confidence: float = 0.35
    sure_confidence: float = 0.6
    ttc_react: float = 1.5
    ttc_horizon: float = 9.7
    reannounce_move_m: float = 3.0
    reannounce_tau_s: float = 10.0
    recent_block_s: float = 2.8
    decision_interval: float = 0.5
    verbosity: str = "balanced"          # quiet | balanced | detailed
    units: str = "m"
    # ablation / deployment toggles
    use_ttc: bool = True
    use_budget: bool = True
    use_interrupt: bool = True
    use_queue_cost: bool = True
    use_haptic: bool = True
    use_depth: bool = True
    use_slots: bool = True               # False -> always speak every slot (verbose messages)


@dataclass
class EventFeatures:
    """Everything the scheduler (and the learned scheduler) sees about one candidate."""
    severity: float = 0.0
    path_prob: float = 0.0
    hazard: float = 0.0
    urgency: float = 0.0
    actionability: float = 0.0
    goal: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 0.0
    speech_cost: float = 0.0
    queue_cost: float = 0.0
    confidence: float = 0.0
    distance: float = 0.0
    depth_conf: float = 0.0
    angle: float = 0.0
    rel_speed: float = 0.0
    lateral_speed: float = 0.0
    ttc: float = 0.0
    growth: float = 0.0
    since_announced: float = 1.0
    announce_count: float = 0.0
    text_importance: float = 0.0
    is_text: float = 0.0
    moving: float = 0.0
    channel_busy: float = 0.0
    channel_remaining: float = 0.0
    budget: float = 1.0
    density: float = 0.0
    own_speed: float = 0.0        # object's own approach speed (relative speed minus the user's walking speed)
    ego_speed: float = 0.0

    @staticmethod
    def names() -> List[str]:
        return list(EventFeatures().__dict__.keys())

    def as_vector(self) -> List[float]:
        return [float(v) for v in self.__dict__.values()]


@dataclass
class ChannelState:
    busy: bool = False
    remaining_s: float = 0.0
    current_utility: float = 0.0
    current_severity: float = 0.0
    current_track: Optional[int] = None


def event_type_for(track: Track) -> str:
    if track.kind == "text":
        return "sign_nav"
    return default_event_type(track.label, track.kind)


class PCACSPolicy(CuePolicy):
    name = "pcacs"
    supports_actions = True

    def __init__(self, cfg: Optional[PCACSConfig] = None, text_model=None, camera: Optional[CameraModel] = None,
                 goal_tags: Optional[Sequence[str]] = None, name: Optional[str] = None):
        self.cfg = cfg or PCACSConfig()
        self.decision_interval = self.cfg.decision_interval
        self.text_model = text_model or KeywordTextImportance()
        self.cam = camera or CameraModel()
        self.goal_tags: List[str] = [g.lower() for g in (goal_tags or [])]
        self.budget = self.cfg.budget_capacity_s
        self._last_t: Optional[float] = None
        self._last_obs: List[Observation] = []
        self.last_features: Dict[int, EventFeatures] = {}
        self.last_action: str = "silence"
        self.announced_dist: Dict[int, float] = {}
        self.ego_speed = 1.0
        if name:
            self.name = name

    # ---- public controls (FIND / verbosity from the UI) -----------------------
    def set_goal(self, tags: Optional[Sequence[str]]) -> None:
        self.goal_tags = [g.lower() for g in (tags or [])]

    def reset(self) -> None:
        self.budget = self.cfg.budget_capacity_s
        self._last_t = None
        self._last_obs = []
        self.last_features = {}
        self.announced_dist.clear()
        self.ego_speed = 1.0

    def last_observations(self) -> List[Observation]:
        return self._last_obs

    # ---- features -----------------------------------------------------------
    def _urgency(self, trk: Track) -> float:
        c = self.cfg
        if c.use_ttc and math.isfinite(trk.ttc_s):
            ttc = trk.ttc_s
        else:
            z = trk.distance_m if trk.distance_m is not None else 6.0
            ttc = max(0.0, (z - 0.6) / 1.0)      # assume walking speed
        return max(0.0, min(1.0, (c.ttc_horizon - ttc) / (c.ttc_horizon - c.ttc_react)))

    def _goal(self, trk: Track) -> float:
        if not self.goal_tags:
            return 0.0
        words = {trk.label.lower()}
        if trk.text:
            words.add(trk.text.lower()); words |= set(trk.text.lower().split())
        if trk.label in ("chair", "bench"):
            words.add("seat")
        return 1.0 if any(g in words for g in self.goal_tags) else 0.0

    def _novelty(self, trk: Track, t: float) -> float:
        if trk.last_announced is None:
            return 1.0
        n = 1.0 - math.exp(-(t - trk.last_announced) / self.cfg.reannounce_tau_s)
        d0 = self.announced_dist.get(trk.track_id)
        if d0 is not None and trk.distance_m is not None and d0 - trk.distance_m >= self.cfg.reannounce_move_m:
            n = max(n, min(1.0, (d0 - trk.distance_m) / (2 * self.cfg.reannounce_move_m)))
        if trk.displacement_since_announced() > 0.25:
            n = max(n, 0.6)
        return n / (1 + 0.4 * trk.announce_count)

    def _slots(self, etype: str, trk: Track, f: EventFeatures) -> Tuple[str, ...]:
        c = self.cfg
        base = ["object"]
        if not c.use_slots:
            return ("object", "distance", "direction", "motion", "action", "detail")
        base += list(MIN_SLOTS.get(etype, ("distance",)))
        moving = f.moving > 0
        if moving and "motion" not in base and etype not in ("sign_nav", "goal", "landmark"):
            base.append("motion")
        if abs(f.angle) > (25 if c.verbosity == "quiet" else 12) and "direction" not in base:
            base.append("direction")
        if c.verbosity == "quiet":
            base = [s for s in base if s != "detail"]
        elif c.verbosity == "detailed":
            for s in ("direction", "detail"):
                if s not in base:
                    base.append(s)
        return tuple(base)

    def _update_ego(self, tracks: List[Track]) -> None:
        """Walking speed of the user from how fast confidently-ranged objects come closer (median)."""
        v = sorted(-tr.rel_speed_mps for tr in tracks
                   if tr.kind == "object" and tr.depth_conf >= 0.5 and tr.distance_m is not None and tr.distance_m < 12
                   and abs(tr.lateral_speed_mps) < 0.5 and len(tr.dist_history) >= 4)
        if v:
            med = v[len(v) // 2]
            self.ego_speed = 0.7 * self.ego_speed + 0.3 * max(0.0, min(2.0, med))

    def _features(self, tracks: List[Track], t: float, ch: ChannelState) -> List[Observation]:
        c, w = self.cfg, self.cfg.weights
        obs: List[Observation] = []
        raw = []
        self._update_ego(tracks)
        for trk in tracks:
            etype = event_type_for(trk)
            spec = EVENT_SPECS[etype]
            f = EventFeatures()
            f.confidence = trk.confidence
            f.distance = trk.distance_m if trk.distance_m is not None else 6.0
            f.depth_conf = trk.depth_conf if c.use_depth else 0.25
            cx, _ = bbox_center(trk.bbox)
            f.angle = self.cam.angle_deg_from_u(cx)
            f.rel_speed, f.lateral_speed = trk.rel_speed_mps, trk.lateral_speed_mps
            f.ttc = min(10.0, trk.ttc_s) if math.isfinite(trk.ttc_s) else 10.0
            f.growth = max(-1.0, min(1.0, trk.area_growth_rate()))
            f.since_announced = 1.0 if trk.last_announced is None else min(1.0, (t - trk.last_announced) / 10.0)
            f.announce_count = min(1.0, trk.announce_count / 3.0)
            f.ego_speed = self.ego_speed
            f.own_speed = -(trk.rel_speed_mps + self.ego_speed) if trk.depth_conf >= 0.4 else 0.0   # >0 = coming at the user
            lat_thr = 0.4 + 0.05 * f.distance     # lateral speed is noisier at range
            f.moving = 1.0 if (abs(trk.lateral_speed_mps) > lat_thr or f.own_speed > 0.5) else 0.0
            f.channel_busy, f.channel_remaining = (1.0 if ch.busy else 0.0), min(1.0, ch.remaining_s / 3.0)
            f.budget = self.budget / c.budget_capacity_s
            f.density = min(1.0, len(tracks) / 12.0)
            if trk.kind == "text":
                f.is_text = 1.0
                f.text_importance = self.text_model.score(trk.text or "")
                legible = min(1.0, (trk.bbox[3] - trk.bbox[1]) / 0.05)
                facing = 1.0 if abs(f.angle) < 30 else 0.3
                f.severity = 0.45 * f.text_importance
                f.path_prob = legible * facing
                f.hazard = f.severity * f.path_prob
                f.actionability = ACTIONABILITY["sign_nav"] * f.text_importance
            else:
                f.severity = spec.severity
                f.path_prob = trk.path_prob if c.use_depth else min(1.0, 0.6 * (1 - abs(cx - 0.5) * 2) + 0.4 * (trk.bbox[3] > 0.7))
                f.hazard = f.severity * f.path_prob
                f.actionability = ACTIONABILITY.get(etype, 0.0)
            f.urgency = self._urgency(trk) if f.hazard > 0 else 0.0
            # a remembered (currently unseen) object is scored like any other candidate: its decayed confidence and depth
            # confidence already raise `uncertainty`, so a memory speaks later and hedged, and never blocks others (below)
            f.goal = self._goal(trk)
            f.novelty = self._novelty(trk, t)
            f.uncertainty = 1.0 - f.confidence * (0.5 + 0.5 * f.depth_conf)
            slots = self._slots("goal" if f.goal > 0 else etype, trk, f)
            hedged = trk.confidence < c.sure_confidence
            msg = compose(slots, trk.label, trk.kind, trk.text, distance_m=f.distance, depth_conf=f.depth_conf,
                          angle_deg=f.angle, rel_speed=-f.own_speed, lateral_speed=(f.lateral_speed if f.moving else 0.0),
                          lateral_m=trk.lateral_m, hedged=hedged, units=c.units)
            dur = speech_seconds(msg)
            f.speech_cost = dur / 3.0
            raw.append((trk, f, slots, msg, dur, hedged))
        # queue cost needs the other candidates' urgency (remembered objects do not block the channel for others)
        hz = [(fx.hazard * fx.urgency) if not trk.remembered else 0.0 for (trk, fx, *_r) in raw]
        for i, (trk, f, slots, msg, dur, hedged) in enumerate(raw):
            others = [h for j, h in enumerate(hz) if j != i]
            f.queue_cost = f.speech_cost * (max(others) if others else 0.0) if c.use_queue_cost else 0.0
            near = 1.0 if f.distance <= 9.0 else 0.0
            # awareness: proximity in the frontal cone, independent of path probability and TTC (what blind users ask
            # about: "is it safe?" when a person or a car is close ahead, whether or not it is on a collision course)
            aware = (max(0.0, 1.0 - f.distance / 6.0) * (1.0 if abs(f.angle) < 25 else 0.4) * f.severity * (1 - f.is_text)
                     if w.w_aware > 0 else 0.0)
            value = (w.w_hazard * f.hazard * f.urgency                     # how bad x how soon
                     + w.w_ttc * f.hazard * max(0.0, f.urgency - 0.5) * 2   # extra push inside the reaction horizon
                     + w.w_action * f.actionability * f.path_prob * (0.4 + 0.6 * f.urgency) * (1 - f.is_text)
                     + w.w_text * f.text_importance * f.path_prob * near * f.is_text
                     + w.w_goal * f.goal * near
                     + w.w_aware * aware)
            u = (value * (0.35 + 0.65 * f.novelty)
                 - w.w_uncert * f.uncertainty - w.lam_speech * f.speech_cost - w.lam_queue * f.queue_cost)
            o = Observation(track=trk, t=t, features=None, score=float(u), extras=f.as_vector())  # type: ignore[arg-type]
            o.event = f; o.slots = slots; o.message = msg; o.duration = dur; o.hedged = hedged   # type: ignore[attr-defined]
            obs.append(o)
            self.last_features[trk.track_id] = f
        return obs

    # ---- decision -------------------------------------------------------------
    def _refill(self, t: float) -> None:
        if self._last_t is not None:
            self.budget = min(self.cfg.budget_capacity_s, self.budget + self.cfg.budget_refill * (t - self._last_t))
        self._last_t = t

    def _make_cue(self, o: Observation, t: float, action: str, ch: ChannelState) -> Cue:
        trk, f = o.track, o.event   # type: ignore[attr-defined]
        if action in ("speak", "interrupt"):
            trk.last_announced = t
            trk.last_announced_bbox = trk.bbox
            trk.announce_count += 1
            self.announced_dist[trk.track_id] = trk.distance_m if trk.distance_m is not None else 0.0
            if self.cfg.use_budget:
                self.budget = max(-self.cfg.budget_capacity_s, self.budget - o.duration)  # type: ignore[attr-defined]
        else:
            trk.last_haptic = t
        return Cue(t=t, track_id=trk.track_id, label=trk.label, kind=trk.kind,
                   message=o.message if action != "haptic" else "", score=o.score, policy=self.name,  # type: ignore[attr-defined]
                   text=trk.text, source_id=trk.source_id, hedged=o.hedged, action=action,  # type: ignore[attr-defined]
                   duration_s=o.duration if action != "haptic" else 0.0, slots=o.slots,  # type: ignore[attr-defined]
                   distance_m=trk.distance_m, ttc_s=(trk.ttc_s if math.isfinite(trk.ttc_s) else None),
                   severity=f.severity, utility=o.score,
                   interrupted_track=ch.current_track if action == "interrupt" else None)

    def decide(self, tracks: List[Track], t: float, ch: Optional[ChannelState] = None, scene_ctx=None) -> List[Cue]:
        c = self.cfg
        ch = ch or ChannelState()
        self._refill(t)
        obs = self._features(tracks, t, ch)
        obs.sort(key=lambda o: -o.score)
        self._last_obs = obs
        self.last_action = "silence"
        if not obs:
            return []
        for o in obs:
            trk, f = o.track, o.event  # type: ignore[attr-defined]
            if trk.confidence < c.min_confidence:
                continue
            imminent = (f.severity >= c.haptic_min_severity and f.path_prob >= c.haptic_min_path
                        and f.ttc <= c.haptic_ttc and f.depth_conf >= 0.4)
            recently = trk.last_announced is not None and t - trk.last_announced < c.recent_block_s
            haptic_ok = (c.use_haptic and imminent and (trk.last_haptic is None or t - trk.last_haptic >= c.haptic_repeat_s)
                         and (trk.last_announced is None or t - trk.last_announced >= c.haptic_after_speech_s))
            if recently:
                if haptic_ok:
                    self.last_action = "haptic"
                    return [self._make_cue(o, t, "haptic", ch)]
                continue
            if o.score < c.threshold_speak:
                break
            urgent = f.hazard * f.urgency >= c.urgent_bypass
            if not ch.busy:
                if c.use_budget and not urgent and self.budget < o.duration:  # type: ignore[attr-defined]
                    continue        # defer non-urgent speech until the budget recovers
                self.last_action = "speak"
                return [self._make_cue(o, t, "speak", ch)]
            # channel busy
            critical = (f.severity >= c.critical_severity and f.urgency >= c.critical_urgency and f.path_prob >= 0.5
                        and ch.current_severity < 0.7)
            if c.use_interrupt and ch.current_track != trk.track_id and f.severity > ch.current_severity and (
                    critical or (o.score >= c.threshold_interrupt and o.score >= ch.current_utility + c.interrupt_margin)):
                self.last_action = "interrupt"
                return [self._make_cue(o, t, "interrupt", ch)]
            if haptic_ok:
                self.last_action = "haptic"
                return [self._make_cue(o, t, "haptic", ch)]
        return []

    # CuePolicy compatibility: a free channel is assumed
    def select(self, tracks: List[Track], t: float, scene_ctx=None) -> List[Cue]:
        return [c for c in self.decide(tracks, t, ChannelState(), scene_ctx) if c.action != "haptic"]

    def to_dict(self) -> Dict:
        return asdict(self.cfg)

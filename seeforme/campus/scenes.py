"""CampusBench scenario generators.

Every generator places objects in metres along the path of a user who walks forward
at roughly 1.1 m/s. `ahead(t, d)` is the world z of a point d metres ahead of the user
at time t, so "a chair that is 6 m ahead when t = 8 s" reads naturally.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .world import CampusScenario, WorldObject
from ..cues.sign_vocab import translate
from ..geometry.camera import CameraModel

DECOR_OBJECTS = ["poster", "tv", "clock", "laptop", "book", "cup", "bottle", "vase", "cell phone", "painting"]
DECOR_SIGNS = ["SALE 50% OFF", "WELCOME", "COFFEE $2", "FOLLOW US ONLINE", "SPRING CONCERT", "HAPPY HOUR",
               "JOIN THE CLUB", "NEW MENU", "BUY ONE GET ONE", "ART SHOW FRIDAY", "www.example.com"]
NAV_SIGNS = ["EXIT", "RESTROOM", "ELEVATOR", "STAIRS", "ROOM 204", "ROOM 117B", "FIRE EXIT", "CAUTION WET FLOOR",
             "DO NOT ENTER", "GATE 12", "LIBRARY", "DINING HALL"]
OBSTACLES = ["chair", "trash can", "potted plant", "suitcase", "bench", "box", "cart", "backpack"]

CAMPUS_KINDS = ["courtyard", "hallway", "stairs", "doors", "crossing", "finding", "crowd",
                "temporary_hazard", "poor_capture", "negative", "changing",
                # v2 (research/scenario_catalog.json): channel contention, lost observation / memory,
                # multilingual airport wayfinding, and an out-of-distribution accident aftermath (evaluation only)
                "contention", "occlusion", "wayfinding_airport", "ood_aftermath"]
CAMPUS_KINDS_V1 = CAMPUS_KINDS[:11]
OOD_KINDS = {"ood_aftermath"}


class _Builder:
    def __init__(self, kind: str, seed: int, duration: float, fps: float):
        self.rng = random.Random(seed)
        self.kind, self.seed, self.duration, self.fps = kind, seed, duration, fps
        self.objects: List[WorldObject] = []
        self.user_kf: List[Tuple[float, float]] = [(0.0, 0.0), (duration, 1.1 * duration)]
        self.goal: Optional[str] = None
        self.goal_tags: List[str] = []
        self.camera = CameraModel()
        self.extra = {}

    # --- user ----------------------------------------------------------------
    def user_with_stops(self, stops: List[Tuple[float, float]], speed: float = 1.1):
        """stops: list of (t_stop, length_s)."""
        kf = [(0.0, 0.0)]
        t, z = 0.0, 0.0
        for ts, ln in sorted(stops):
            z += speed * (ts - t); kf.append((ts, z))
            t = ts + ln; kf.append((t, z))
        z += speed * (self.duration - t); kf.append((self.duration, z))
        self.user_kf = kf

    def user_z(self, t: float) -> float:
        kf = self.user_kf
        if t <= kf[0][0]:
            return kf[0][1]
        for (t0, z0), (t1, z1) in zip(kf, kf[1:]):
            if t0 <= t <= t1:
                return z0 + (z1 - z0) * (t - t0) / max(1e-6, t1 - t0)
        return kf[-1][1]

    def ahead(self, t: float, d: float) -> float:
        return self.user_z(t) + d

    # --- object helpers --------------------------------------------------------
    def add(self, o: WorldObject) -> WorldObject:
        o.obj_id = len(self.objects) + 1
        self.objects.append(o)
        return o

    def static(self, label: str, x: float, z_w: float, **kw) -> WorldObject:
        return self.add(WorldObject(obj_id=0, label=label, keyframes=[(0.0, x, z_w)], **kw))

    def sign(self, text: str, x: float, z_w: float, nav: bool, base_height: float = 1.6, **kw) -> WorldObject:
        kw.setdefault("detectability", 0.75); kw.setdefault("base_conf", 0.7)
        return self.add(WorldObject(obj_id=0, label="text", kind="text", text=text, event_type="sign_nav" if nav else "sign_decor",
                                    keyframes=[(0.0, x, z_w)], base_height=base_height, height=0.3, width=0.9, **kw))

    def crossing(self, label: str, z_w: float, t0: float, x_from: float, x_to: float, speed: float, **kw) -> WorldObject:
        t1 = t0 + abs(x_to - x_from) / speed
        return self.add(WorldObject(obj_id=0, label=label, keyframes=[(t0, x_from, z_w), (t1, x_to, z_w)],
                                    t_appear=t0, t_disappear=t1, **kw))

    def toward(self, label: str, x: float, t0: float, d0: float, speed: float, length: float = 12.0, **kw) -> WorldObject:
        """Walks toward the user starting d0 m ahead at t0."""
        z0 = self.ahead(t0, d0)
        return self.add(WorldObject(obj_id=0, label=label, keyframes=[(t0, x, z0), (t0 + length, x, z0 - speed * length)],
                                    t_appear=t0, t_disappear=t0 + length, **kw))

    def away(self, label: str, x: float, t0: float, d0: float, speed: float, length: float = 12.0, **kw) -> WorldObject:
        z0 = self.ahead(t0, d0)
        return self.add(WorldObject(obj_id=0, label=label, keyframes=[(t0, x, z0), (t0 + length, x, z0 + speed * length)],
                                    t_appear=t0, t_disappear=t0 + length, **kw))

    def decor(self, n: int, z_lo: float = 2.0, z_hi: float = 30.0):
        for _ in range(n):
            side = self.rng.choice([-1, 1])
            x = side * self.rng.uniform(1.2, 2.6)
            z = self.rng.uniform(z_lo, z_hi)
            lab = self.rng.choice(DECOR_OBJECTS)
            self.static(lab, x, z, base_height=self.rng.uniform(0.6, 1.5), height=0.4, width=0.5, base_conf=0.9)

    def decor_signs(self, n: int):
        for _ in range(n):
            self.sign(self.rng.choice(DECOR_SIGNS), self.rng.choice([-1, 1]) * self.rng.uniform(0.8, 2.0),
                      self.rng.uniform(3, 30), nav=False, base_conf=0.85)

    def build(self) -> CampusScenario:
        meta = {k: self.extra.pop(k) for k in ("occlusion", "language") if k in self.extra}
        desc = f"{self.kind} scenario, seed {self.seed}" + (f" {meta}" if meta else "")
        sc = CampusScenario(name=f"{self.kind}_{self.seed}", kind=self.kind, duration=self.duration, fps=self.fps,
                            objects=self.objects, user_keyframes=self.user_kf, goal=self.goal, goal_tags=self.goal_tags,
                            camera=self.camera, seed=self.seed, description=desc, **self.extra)
        return sc


# ---------------------------------------------------------------------------

def make_campus_scenario(kind: str, seed: int, duration: float = 30.0, fps: float = 10.0) -> CampusScenario:
    b = _Builder(kind, seed, duration, fps)
    rng = b.rng
    D = duration

    if kind == "courtyard":
        b.user_with_stops([(rng.uniform(12, 18), 2.0)])
        for z in (4, 11, 19, 27):
            b.static("bench", rng.choice([-1, 1]) * rng.uniform(1.6, 2.6), z + rng.uniform(-1, 1))
        b.decor(2); b.decor_signs(1)
        # pedestrians crossing at various ranges/times (only the close ones matter)
        for _ in range(3):
            t0 = rng.uniform(1, D - 6); d = rng.uniform(2.5, 12)
            side = rng.choice([-1, 1])
            b.crossing("person", b.ahead(t0, d), t0, side * 4, -side * 4, rng.uniform(1.0, 1.6))
        # one person walking straight at the user in the corridor, another off to the side
        b.toward("person", rng.uniform(-0.2, 0.2), rng.uniform(2, 8), 14, 1.2)
        b.toward("person", rng.choice([-1, 1]) * rng.uniform(1.3, 2.0), rng.uniform(4, 12), 12, 1.2)
        b.away("person", 0.3, 0.0, 6.0, 1.3)
        # cyclist cutting across close
        tb = rng.uniform(8, D - 6); side = rng.choice([-1, 1])
        b.crossing("bicycle", b.ahead(tb, rng.uniform(2.5, 4.5)), tb, side * 6, -side * 6, rng.uniform(3.5, 5.0))
        # bench partly in the path, and a temporary barrier near the end
        b.static("bench", rng.uniform(-0.2, 0.5), b.ahead(rng.uniform(5, 9), 7))
        b.static("barrier", 0.0, b.ahead(D - 8, 6), width=1.6)
        # person walking ahead who stops abruptly
        ts = rng.uniform(10, 16); v = 1.1
        z0 = 3.0
        b.add(WorldObject(obj_id=0, label="person", keyframes=[(0.0, 0.1, z0), (ts, 0.1, z0 + v * ts), (D, 0.1, z0 + v * ts)]))

    elif kind == "hallway":
        b.decor(rng.randint(3, 5)); b.decor_signs(2)
        for i, lab in enumerate(rng.sample(["backpack", "chair", "box", "trash can", "cart"], 3)):
            t = 4 + i * rng.uniform(7, 9)
            b.static(lab, rng.uniform(-0.35, 0.35), b.ahead(t, 7))
        # open door protruding into the corridor, and chairs along the wall (off path)
        b.static("door", 0.75, b.ahead(rng.uniform(14, 20), 7), width=0.9, detail={"open": True, "handle": "left"})
        for z in (6, 15, 24):
            b.static("chair", rng.choice([-1, 1]) * rng.uniform(1.3, 1.9), z)
        b.toward("person", rng.uniform(-0.15, 0.15), rng.uniform(3, 10), 15, 1.2)
        b.toward("person", rng.choice([-1, 1]) * 1.5, rng.uniform(6, 14), 15, 1.2)
        b.sign(rng.choice(NAV_SIGNS), rng.uniform(-0.6, 0.6), rng.uniform(8, 26), nav=True)
        if rng.random() < 0.4:
            b.goal, b.goal_tags = "find the exit", ["exit", "fire exit"]

    elif kind == "stairs":
        b.decor(2); b.decor_signs(1)
        tz = rng.uniform(6, 10)
        hand = rng.choice(["left", "right"])
        b.static("stairs", 0.0, b.ahead(tz, 8), width=1.6, height=1.4, detail={"direction": "down", "handrail": hand})
        b.sign("STAIRS", rng.uniform(-0.6, 0.6), b.ahead(tz, 7.5), nav=True)
        if rng.random() < 0.6:
            b.static("person", rng.uniform(-0.3, 0.3), b.ahead(tz, 7.3))   # blocking the stairway
        t2 = tz + rng.uniform(10, 13)
        b.static("stairs", 0.0, b.ahead(t2, 8), width=1.6, height=1.4, detail={"direction": "up", "handrail": rng.choice(["left", "right"])})
        b.static("potted plant", rng.choice([-1, 1]) * 1.4, b.ahead(t2, 4))

    elif kind == "doors":
        b.decor(2); b.decor_signs(1)
        for z in (5, 13, 21):
            b.static("door", rng.choice([-1, 1]) * 1.9, z, width=0.9, detail={"handle": rng.choice(["left", "right"])})
        te = rng.uniform(8, 14)
        b.static("door", 0.0, b.ahead(te, 8), width=1.0, detail={"handle": rng.choice(["left", "right"]),
                                                                   "push_pull": rng.choice(["push", "pull"])})
        b.sign(rng.choice(["PUSH", "PULL", "ENTRANCE"]), 0.3, b.ahead(te, 7.9), nav=True, base_height=1.4)
        t2 = te + rng.uniform(9, 12)
        b.static("door", 0.0, b.ahead(t2, 8), width=1.4, detail={"automatic": True})
        b.toward("person", 0.1, te + 2, 10, 1.1, length=8)

    elif kind == "crossing":
        b.user_with_stops([(rng.uniform(9, 12), 3.0)])
        b.extra["clutter_fp_rate"] = 0.08
        b.static("traffic light", 1.6, b.ahead(0, 12), base_height=2.5, height=1.0, width=0.35, base_conf=0.8)
        tc = rng.uniform(6, 9)
        b.static("curb", 0.0, b.ahead(tc, 5), width=3.0, height=0.15, detail={"direction": "down"})
        b.sign("STOP", 1.5, b.ahead(tc, 6), nav=True, base_height=2.0, base_conf=0.8)
        # cars crossing the street ahead: near ones matter, far ones do not
        for _ in range(4):
            t0 = rng.uniform(1, D - 4); d = rng.choice([3.0, 4.5, 12.0, 16.0]) + rng.uniform(-0.5, 0.5)
            side = rng.choice([-1, 1])
            b.crossing("car", b.ahead(t0, d), t0, side * 14, -side * 14, rng.uniform(6, 10), width=1.8, height=1.5)
        b.static("car", rng.choice([-1, 1]) * 2.6, b.ahead(0, 9), width=1.8, height=1.5)   # parked
        tb = rng.uniform(4, D - 6); side = rng.choice([-1, 1])
        b.crossing("bicycle", b.ahead(tb, 3.0), tb, side * 8, -side * 8, rng.uniform(4, 6))
        ts = rng.uniform(4, D - 6)
        b.crossing("scooter", b.ahead(ts, 2.5), ts, -7, 7, 5.0, width=0.5, height=1.1)
        for _ in range(3):
            b.static("person", rng.choice([-1, 1]) * rng.uniform(1.5, 3), rng.uniform(4, 28))
        b.static("pole", rng.uniform(-0.3, 0.3), b.ahead(rng.uniform(18, 24), 6), width=0.2, height=2.5)
        b.decor_signs(1)

    elif kind == "finding":
        goal = rng.choice(["room 204", "elevator", "seat", "exit", "restroom"])
        b.goal = f"find {'a' if goal == 'seat' else 'the'} {goal}"
        b.goal_tags = {"room 204": ["204", "room 204"], "elevator": ["elevator"], "seat": ["seat", "chair", "bench"],
                       "exit": ["exit", "fire exit"], "restroom": ["restroom", "toilet"]}[goal]
        b.decor(3); b.decor_signs(2)
        rooms = ["ROOM 201", "ROOM 202", "ROOM 203", "ROOM 204", "ROOM 205"]
        for i, r in enumerate(rooms):
            b.sign(r, rng.choice([-1, 1]) * 1.4, 4 + i * 5.5, nav=True, base_height=1.7)
        b.static("elevator", -1.9, b.ahead(14, 8), width=1.0, height=2.1, goal_tags=["elevator"])
        b.sign("ELEVATOR", -1.6, b.ahead(14, 8.2), nav=True, base_height=2.1)
        b.static("chair", 1.7, b.ahead(20, 6), goal_tags=["seat", "chair"])
        b.sign("EXIT", 0.2, b.ahead(26, 4), nav=True, base_height=2.2)
        b.sign("RESTROOM", 1.3, b.ahead(9, 7), nav=True)
        b.static(rng.choice(OBSTACLES), rng.uniform(-0.3, 0.3), b.ahead(rng.uniform(10, 20), 7))
        b.toward("person", rng.choice([-1, 1]) * 1.4, rng.uniform(4, 14), 14, 1.2)

    elif kind == "crowd":
        b.user_with_stops([(rng.uniform(6, 9), 1.5), (rng.uniform(16, 20), 1.5)])
        b.extra["clutter_fp_rate"] = 0.1
        for _ in range(rng.randint(7, 10)):
            b.static("person", rng.choice([-1, 1]) * rng.uniform(1.0, 3.0), rng.uniform(3, 32), base_conf=rng.uniform(0.6, 0.95))
        for i in range(4):     # a queue along the wall
            b.static("person", 1.1, b.ahead(12, 4) + i * 0.8)
        for _ in range(3):
            t0 = rng.uniform(1, D - 5); side = rng.choice([-1, 1])
            b.crossing("person", b.ahead(t0, rng.uniform(2.5, 10)), t0, side * 3, -side * 3, rng.uniform(1.0, 1.5))
        b.static("person", rng.uniform(-0.2, 0.2), b.ahead(rng.uniform(10, 16), 7))      # directly ahead
        b.toward("person", rng.uniform(-0.2, 0.2), rng.uniform(18, 22), 12, 1.3, length=8)
        b.decor(2); b.decor_signs(1)
        b.sign("DINING HALL", 0.4, b.ahead(4, 8), nav=True, base_height=2.2)

    elif kind == "temporary_hazard":
        b.decor(3); b.decor_signs(2)
        hz = rng.sample([("cone", 0.4, 0.7), ("barrier", 1.6, 1.0), ("wet floor sign", 0.3, 0.65), ("box", 0.5, 0.5),
                         ("cart", 0.7, 1.0), ("suitcase", 0.45, 0.7)], 4)
        for i, (lab, w, h) in enumerate(hz):
            t = 4 + i * rng.uniform(5.5, 7)
            b.static(lab, rng.uniform(-0.3, 0.3) if lab != "barrier" else 0.0, b.ahead(t, 7), width=w, height=h)
        b.static("cone", 1.5, b.ahead(14, 5), width=0.4, height=0.7)   # off path
        b.sign("CAUTION WET FLOOR", -0.5, b.ahead(9, 6), nav=True, base_height=0.4)
        b.toward("person", rng.choice([-1, 1]) * 1.5, rng.uniform(3, 10), 15, 1.2)

    elif kind == "poor_capture":
        base = make_campus_scenario("hallway", seed, duration, fps)
        b.objects = base.objects; b.user_kf = base.user_keyframes; b.goal, b.goal_tags = base.goal, base.goal_tags
        for o in b.objects:
            o.detectability *= 0.85
        b.camera = CameraModel(pitch_deg=rng.choice([5.0, 25.0, 15.0]))
        b.extra.update(pitch_jitter_deg=rng.uniform(5, 9), yaw_jitter_deg=rng.uniform(4, 7),
                       motion_blur=rng.uniform(0.3, 0.5), lighting=rng.choice(["normal", "low"]),
                       depth_noise_rel=0.12,
                       blackout_windows=[(t, t + rng.uniform(0.8, 1.6)) for t in (rng.uniform(5, 10), rng.uniform(16, 22))])

    elif kind == "negative":
        b.decor(rng.randint(6, 9)); b.decor_signs(rng.randint(2, 3))
        for _ in range(rng.randint(3, 5)):
            b.static("person", rng.choice([-1, 1]) * rng.uniform(1.4, 3.0), rng.uniform(3, 30), base_conf=0.95)
        b.away("person", rng.choice([-1, 1]) * 0.8, 0.0, 5.0, 1.4)
        for z in (7, 17, 26):
            b.static(rng.choice(["chair", "bench", "potted plant"]), rng.choice([-1, 1]) * rng.uniform(1.4, 2.2), z)
        b.static("door", -1.9, 12, width=0.9, detail={"handle": "right"})   # side door, not in the path
        b.static("fire hydrant", 1.8, 20, width=0.3, height=0.8)

    elif kind == "contention":
        # research/scenario_catalog.json crowd_contention: the channel is the scarce resource
        b.extra["clutter_fp_rate"] = 0.08
        b.decor(2); b.decor_signs(2)
        distractors = []
        # 1) a long sign reading (user-requested, low severity) is playing when stairs enter their window
        t1 = rng.uniform(6, 9)
        b.static("stairs", 0.0, b.ahead(t1 + 2.5, 5.5), width=1.6, height=1.4,
                 detail={"direction": "down", "handrail": rng.choice(["left", "right"])})
        distractors.append((round(t1, 2), round(rng.uniform(6.0, 8.0), 2), 0.3))
        # 2) two hazards with deadlines within a second of each other: a chair in the path and a person walking in
        t2 = rng.uniform(14, 18)
        b.static(rng.choice(["chair", "box", "cart"]), rng.uniform(-0.3, 0.3), b.ahead(t2, 3.5))
        b.toward("person", rng.uniform(-0.2, 0.2), t2 - 2.0, 9.0, 1.2, length=6)
        # 3) a critical message is already playing when a cyclist shoots across close in front: haptic territory
        t3 = rng.uniform(21, 25); side = rng.choice([-1, 1])
        distractors.append((round(t3 - 1.0, 2), 4.0, 0.9))
        b.crossing("bicycle", b.ahead(t3, 2.5), t3, side * 4, -side * 4, 4.5)
        b.sign(rng.choice(NAV_SIGNS), rng.uniform(-0.6, 0.6), rng.uniform(10, 26), nav=True)
        b.extra["distractor_speech"] = distractors

    elif kind == "occlusion":
        # lost_observation: a hazard is seen far ahead, then hidden (occluder or camera pan) for d seconds. Short
        # occlusions can still be handled after reappearance; long ones only by remembering. In 30% of seeds the
        # object is gone when the view clears, so a memory that is trusted blindly produces a false alarm.
        b.decor(2); b.decor_signs(1)
        d_occ = rng.choice([2.0, 4.0, 6.0, 10.0])
        t_see = rng.uniform(3, 5)
        lab = rng.choice(["chair", "box", "cart", "stairs"])
        kw = dict(width=1.6, height=1.4, detail={"direction": "down", "handrail": "left"}) if lab == "stairs" else {}
        o = b.static(lab, rng.uniform(-0.25, 0.25), b.ahead(t_see, 9.0), **kw)
        t_occ = t_see + 1.5
        moved = rng.random() < 0.3
        mechanism = rng.choice(["occluder", "pan"])
        if moved:
            o.t_disappear = t_occ                                   # removed while hidden ...
            b.static(lab, 1.7, b.ahead(t_see, 9.0), t_appear=t_occ + d_occ, **kw)   # ... and now off to the side
        if mechanism == "occluder" and not moved:
            o.occluded_windows = [(round(t_occ, 2), round(t_occ + d_occ, 2))]
        elif mechanism == "occluder":
            b.extra["blackout_windows"] = [(round(t_occ, 2), round(t_occ + d_occ, 2))]   # view blocked by a passing group
        else:
            b.extra["yaw_keyframes"] = [(round(t_occ, 2), 0.0), (round(t_occ + 0.4, 2), 55.0),
                                        (round(t_occ + d_occ - 0.4, 2), 55.0), (round(t_occ + d_occ, 2), 0.0)]
        b.extra["occlusion"] = {"seconds": d_occ, "mechanism": mechanism, "moved": moved, "t_occ": round(t_occ, 2)}
        b.extra["memory_windows"] = True      # the hazard was seen: it stays due while hidden (what memory is for)
        b.toward("person", rng.choice([-1, 1]) * 1.6, rng.uniform(16, 20), 12, 1.2)     # later, unrelated
        b.static(rng.choice(OBSTACLES), rng.uniform(-0.3, 0.3), b.ahead(rng.uniform(20, 24), 6))

    elif kind == "wayfinding_airport":
        # airport_terminal / transit_station / multilingual_and_symbols: dense signage, one goal, luggage and queues
        lang = rng.choice(["en", "en", "pt", "es", "fr"])
        goal_kind = rng.choice(["gate", "baggage", "platform", "restroom"])
        gate = f"A{rng.randint(10, 29)}"
        goals = {"gate": (f"find gate {gate}", [gate.lower(), f"gate {gate.lower()}"]),
                 "baggage": ("find baggage claim", ["baggage", "baggage claim", "bagagem", "equipaje", "bagages"]),
                 "platform": ("find platform 2", ["platform 2", "plataforma 2", "andén 2", "quai 2"]),
                 "restroom": ("find the restroom", ["restroom", "toilet", "banheiro", "baños", "toilettes"])}
        b.goal, b.goal_tags = goals[goal_kind]
        intents = rng.sample(["departures", "arrivals", "security check", "information", "exit", "elevator", "escalator", "gates", "passport control"], 5)
        for i, intent in enumerate(intents):
            b.sign(translate(intent, lang), rng.choice([-1, 1]) * rng.uniform(0.8, 1.6), 4 + i * 5.0, nav=True, base_height=2.0)
        for intent in rng.sample(["sale", "duty free", "free wifi", "coffee", "welcome"], 3):
            b.sign(translate(intent, lang), rng.choice([-1, 1]) * rng.uniform(0.8, 2.0), rng.uniform(3, 28), nav=False, base_conf=0.85)
        goal_text = {"gate": f"{translate('gate', lang)} {gate}", "baggage": translate("baggage claim", lang),
                     "platform": f"{translate('platform', lang)} 2", "restroom": translate("restroom", lang)}[goal_kind]
        b.sign(goal_text, rng.uniform(-0.5, 0.5), b.ahead(rng.uniform(16, 22), 6), nav=True, base_height=2.0)
        b.static("suitcase", rng.uniform(-0.3, 0.3), b.ahead(rng.uniform(5, 9), 7))
        for i in range(4):                                             # a queue along the wall
            b.static("person", 1.1, b.ahead(12, 4) + i * 0.8)
        b.toward("person", rng.uniform(-0.2, 0.2), rng.uniform(8, 14), 14, 1.2)
        b.static("cart", rng.uniform(-0.3, 0.3), b.ahead(rng.uniform(18, 24), 6), width=0.7, height=1.0)
        b.decor(2)
        b.extra["language"] = lang

    elif kind == "ood_aftermath":
        # accident_aftermath (OOD, evaluation only): smoke-like degraded capture, a stopped car across the path,
        # debris and glass the taxonomy has no class for, bystanders. Score: are unknown obstacles announced
        # (hedged) and is nothing said with false confidence when the view is gone?
        b.extra.update(lighting="low", motion_blur=0.4, clutter_fp_rate=0.12, depth_noise_rel=0.15,
                       blackout_windows=[(round(t, 2), round(t + rng.uniform(1.0, 2.5), 2)) for t in (rng.uniform(4, 8), rng.uniform(14, 20))])
        b.decor(2); b.decor_signs(1)
        tc = rng.uniform(5, 8)
        b.static("car", rng.uniform(-0.5, 0.5), b.ahead(tc, 8), width=1.8, height=1.5, base_conf=0.75)
        b.static("debris", rng.uniform(-0.3, 0.3), b.ahead(tc + 3, 6), width=0.8, height=0.4, base_conf=0.5, conf_noise=0.15, detectability=0.7)
        b.static("broken glass", rng.uniform(-0.4, 0.4), b.ahead(tc + 6, 5), width=1.0, height=0.08, base_conf=0.4, conf_noise=0.15, detectability=0.5)
        b.static("fallen sign", rng.choice([-1, 1]) * 0.4, b.ahead(tc + 9, 6), width=0.6, height=0.5, base_conf=0.55, detectability=0.7)
        for _ in range(rng.randint(4, 7)):
            b.static("person", rng.choice([-1, 1]) * rng.uniform(0.8, 2.5), b.ahead(tc, rng.uniform(3, 12)), base_conf=rng.uniform(0.5, 0.9))
        b.static("unknown object", 0.0, b.ahead(tc + 13, 6), width=1.2, height=0.9, base_conf=0.45, conf_noise=0.2)
        b.crossing("person", b.ahead(tc + 4, 3.0), tc + 4, -3, 3, 1.3)

    elif kind == "changing":
        b.decor(2); b.decor_signs(1)
        # chair that is dragged into the walking path
        tm = rng.uniform(6, 10)
        zc = b.ahead(tm, 8)
        b.add(WorldObject(obj_id=0, label="chair", keyframes=[(0.0, 1.6, zc), (tm, 1.6, zc), (tm + 1.5, 0.1, zc)]))
        # pedestrian walking beside the path who turns into it
        tp = rng.uniform(12, 16); zp = b.ahead(tp, 9)
        b.add(WorldObject(obj_id=0, label="person", t_appear=tp - 4,
                          keyframes=[(tp - 4, 1.8, zp + 4), (tp, 1.8, zp), (tp + 2.5, 0.0, zp - 2.0), (tp + 6, 0.0, zp - 5.5)]))
        # bike appearing suddenly from a side street close in front
        tb = rng.uniform(18, 23); side = rng.choice([-1, 1])
        b.crossing("bicycle", b.ahead(tb, 2.2), tb, side * 2.5, -side * 2.5, 4.5)
        # door that closes: an open door object replaced by a closed one
        td = rng.uniform(3, 6); zd = b.ahead(td, 10)
        b.add(WorldObject(obj_id=0, label="door", keyframes=[(0.0, 0.0, zd)], t_disappear=td + 3, width=1.0, detail={"open": True, "handle": "right"}))
        b.add(WorldObject(obj_id=0, label="door", keyframes=[(0.0, 0.0, zd)], t_appear=td + 3, width=1.0, detail={"open": False, "handle": "right"}))
        b.toward("person", -1.6, 0.5, 12, 1.1)
    else:
        raise ValueError(f"unknown campus scenario kind {kind}")

    return b.build()


def make_campus_benchmark(n_per_kind: int = 6, seed0: int = 500, duration: float = 30.0, fps: float = 10.0) -> List[CampusScenario]:
    out = []
    for k, kind in enumerate(CAMPUS_KINDS):
        for i in range(n_per_kind):
            out.append(make_campus_scenario(kind, seed0 + 1000 * k + i, duration, fps))
    return out

"""Render a CampusBench scenario as a video: simulated camera view with tracks, distance / TTC / path
probability, the scheduler's action, a top-down map in metres and the cue-window timeline. For the
poster and for showing *why* P-CACS speaks, buzzes, cuts in or stays silent.

    python scripts/demo_campus.py --kind stairs --policy pcacs
    python scripts/demo_campus.py --kind crossing --policy pcacs_learned:rl --seed 4403
"""
import argparse, math
import cv2, numpy as np
from _common import RESULTS, load_cfg
from campus_common import load_campus_cfg, make_campus_policy
from seeforme.campus import make_campus_scenario
from seeforme.campus.runner import run_on_campus

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="stairs"); ap.add_argument("--seed", type=int, default=4200)
ap.add_argument("--policy", default="pcacs"); ap.add_argument("--out", default=None)
args = ap.parse_args()
cfg = load_campus_cfg()
sc = make_campus_scenario(args.kind, args.seed)
pol = make_campus_policy(args.policy, cfg, sc)
W, H, MAPW = 640, 480, 260
out = args.out or str(RESULTS / "campus" / f"demo_{args.kind}_{args.policy.replace(':', '_')}.mp4")
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), sc.fps, (W + MAPW, H + 70))
events = {e.obj_id: e for e in sc.speak_events()}
last = [None, -1e9]
cue_times = []


def on_decision(t, tracks, obs, cue, ch, due):
    if cue is not None:
        last[0], last[1] = cue, t
        cue_times.append((t, cue))


def render(t, tracks, obs):
    cam = np.full((H, W, 3), 238, dtype=np.uint8)
    # faint ground truth with the cue window state
    for o in sc.objects:
        if not o.visible(t):
            continue
        b = o.bbox_at(t); e = events.get(o.obj_id)
        col = (200, 200, 200)
        if e is not None:
            col = (60, 60, 220) if e.in_window(t) else ((60, 160, 60) if t < e.earliest_useful_s else (120, 120, 120))
        x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
        cv2.rectangle(cam, (x1, y1), (x2, y2), col, 1)
        cv2.putText(cam, f"{o.text or o.label}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    score = {o.track.track_id: o for o in obs}
    for tr in tracks:
        x1, y1, x2, y2 = int(tr.bbox[0] * W), int(tr.bbox[1] * H), int(tr.bbox[2] * W), int(tr.bbox[3] * H)
        o = score.get(tr.track_id)
        hot = o is not None and getattr(o, "score", 0) >= getattr(pol.cfg, "threshold_speak", 1e9)
        col = (30, 30, 230) if hot else (40, 170, 90)
        cv2.rectangle(cam, (x1, y1), (x2, y2), col, 2)
        d = f"{tr.distance_m:.1f}m" if tr.distance_m is not None else ""
        ttc = f" ttc {tr.ttc_s:.1f}s" if math.isfinite(tr.ttc_s) else ""
        u = f" U={o.score:.2f}" if o is not None else ""
        cv2.putText(cam, f"{tr.text or tr.label} {d}{ttc} p{tr.path_prob:.2f}{u}", (x1, min(H - 4, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    # top-down map (x: lateral, z: forward), 1 px = 2.5 cm
    m = np.full((H, MAPW, 3), 250, dtype=np.uint8)
    ox, oz = MAPW // 2, H - 30
    sc_px = 25.0
    cv2.rectangle(m, (ox - int(0.45 * sc_px), 0), (ox + int(0.45 * sc_px), oz), (225, 240, 225), -1)   # walking corridor
    for zz in range(0, 17, 2):
        y = oz - int(zz * sc_px); cv2.line(m, (0, y), (MAPW, y), (230, 230, 230), 1); cv2.putText(m, f"{zz}m", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    cv2.circle(m, (ox, oz), 6, (0, 0, 0), -1)
    for o in sc.objects:
        if not (o.t_appear <= t < o.t_disappear):
            continue
        x, z = o.rel_pos(t)
        if z < -1 or z > 17 or abs(x) > 5:
            continue
        e = events.get(o.obj_id)
        col = (200, 200, 200) if e is None else ((60, 60, 220) if e.in_window(t) else (90, 90, 90))
        cv2.circle(m, (ox + int(x * sc_px), oz - int(z * sc_px)), 5 if e else 3, col, -1)
    frame = np.full((H + 70, W + MAPW, 3), 20, dtype=np.uint8)
    frame[:H, :W] = cam; frame[:H, W:] = m
    # footer: action + timeline
    cue, tc = last
    if cue is not None and t - tc < 2.5:
        txt = f"{cue.action.upper()}: {cue.message or '(buzz)'}"
        colr = (60, 60, 255) if cue.action != "haptic" else (40, 160, 255)
        cv2.putText(frame, txt, (8, H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colr, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "silent", (8, H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{sc.name}  policy={pol.name}  t={t:4.1f}s  goal={sc.goal or '-'}", (8, H + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    # timeline of windows and cues
    x0, x1_, y = 8, W + MAPW - 8, H + 62
    cv2.line(frame, (x0, y), (x1_, y), (90, 90, 90), 1)
    px = lambda tt: x0 + int((x1_ - x0) * tt / sc.duration)
    for e in events.values():
        cv2.line(frame, (px(e.earliest_useful_s), y), (px(e.latest_safe_s), y), (60, 60, 220), 3)
    for tt, c in cue_times:
        cv2.circle(frame, (px(tt), y), 3, (40, 160, 255) if c.action == "haptic" else (255, 255, 255), -1)
    cv2.circle(frame, (px(t), y), 4, (0, 255, 0), -1)
    return frame


frames_cache = []
def on_step(t, tracks, obs, cue, ch, due):
    on_decision(t, tracks, obs, cue, ch, due)

# run with a hook that renders every frame (decisions happen every 0.5 s; we render at every frame via a second pass)
from seeforme.campus.simulator import CampusDetectorSimulator
from seeforme.tracking.tracker import IoUTracker
from seeforme.geometry import CameraModel, TrackDynamics
from seeforme.eval.runner import SpeechChannel
from seeforme.cues.pcacs import ChannelState
tracker = IoUTracker(cfg.tracker, dynamics=TrackDynamics(CameraModel()))
channel = SpeechChannel(cfg.speech.min_gap, cfg.speech.seconds_per_word)
pol.reset(); nxt = 0.0
for t, dets in CampusDetectorSimulator(sc).frames():
    tracks = tracker.update(dets, t)
    if t + 1e-9 >= nxt:
        nxt = t + pol.decision_interval
        busy = not channel.free(t); cur = channel.current if busy else None
        ch = ChannelState(busy=busy, remaining_s=channel.remaining(t), current_utility=(cur.utility if cur else 0.0),
                          current_severity=(cur.severity if cur else 0.0), current_track=(cur.track_id if cur else None))
        cues = pol.decide(tracks, t, ch) if getattr(pol, "supports_actions", False) else (pol.select(tracks, t) if not busy else [])
        if cues:
            c = cues[0]
            if c.action != "haptic":
                channel.speak(c, interrupt=(c.action == "interrupt"))
            on_decision(t, tracks, pol.last_observations(), c, ch, set())
    writer.write(render(t, tracks, pol.last_observations() if hasattr(pol, "last_observations") else []))
writer.release()
print(f"{len(cue_times)} cues -> {out}")

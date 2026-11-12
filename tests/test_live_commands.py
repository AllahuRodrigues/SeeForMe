"""What voice commands do in the terminal demo: goal tags shared with the web app, the rule-based answers (never "the
path is clear"), the pause filter (routine cues stop, hazards never do) and the router's dispatch, with a fake
controller and speaker; then the same through the real LiveController, and a short headless run_live replay."""
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from seeforme.geometry import CameraModel
from seeforme.live.commands import (AID_ADVICE, HELP_VOICE, CommandRouter, answer, goal_tags, is_question, is_safety_cue,
                                    is_safety_track, names_goal, read_phrase, says_clear)
from seeforme.live.voice import Command
from seeforme.types import Cue, Track

CAM = CameraModel()
ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data" / "egoblind" / "videos" / "train" / "00700.mp4"


def track(tid, label="person", x=0.0, z=2.0, kind="object", text=None, path=0.0, ttc=math.inf, depth_conf=0.9):
    b = CAM.project_box(x, z, 1.7 if label == "person" else 0.9, 0.5)
    tr = Track(tid, label, kind, b, 0.9, 0.0, 1.0, text=text, confirmed=True)
    tr.distance_m, tr.depth_conf, tr.lateral_m, tr.path_prob, tr.ttc_s = z, depth_conf, x, path, ttc
    return tr


def cue(trk, msg=None, action="speak", severity=0.5, ttc=None, policy="fake"):
    return Cue(1.0, trk.track_id, trk.label, trk.kind, msg or f"{trk.label.capitalize()}.", 1.0, policy, text=trk.text,
               action=action, distance_m=trk.distance_m, ttc_s=ttc, severity=severity)


# ---- goal tags (same rules as web/src/main.ts goalTags) -----------------------------------------------------------
@pytest.mark.parametrize("raw,tags", [
    ("the exit", ["exit"]), ("room 204", ["room 204", "204"]), ("a seat", ["seat", "chair", "bench"]),
    ("gate 7", ["gate 7"]), ("people", ["people", "person"]), ("chairs", ["chairs", "chair"]),
    ("bike", ["bike", "bicycle"]), ("my phone", ["phone", "cell phone"]), ("it", []), ("", []),
    ("exit door", ["exit door", "exit", "door"]), ("cup", ["cup"])])
def test_goal_tags_match_the_web_app(raw, tags):
    assert goal_tags(raw) == tags


def test_goal_names_labels_aliases_and_whole_words_of_signs():
    assert names_goal("chair", None, ["seat"]) and names_goal("text", "ROOM 204 LAB", ["204"])
    assert not names_goal("text", "ROOM 1204", ["204"]) and not names_goal("person", None, ["chair"])


# ---- answers ------------------------------------------------------------------------------------------------------
SCENES = {
    "empty": [],
    "person in path": [track(1, path=0.9, z=1.5)],
    "chair left": [track(2, "chair", x=-1.5, z=2.0)],
    "busy": [track(1, path=0.9, z=1.5), track(2, "chair", x=-1.5), track(3, "car", x=2.5, z=6.0),
             track(4, "text", kind="text", text="EXIT", x=0.2, z=4.0)],
}
QUESTIONS = ["is the path clear", "is it safe to walk", "is the way clear ahead", "what's ahead", "what's on my left",
             "what's on my right", "is there a chair", "is there a person ahead", "any cars", "what do you see here",
             "is there a sign", "are there stairs", "can I walk forward"]


WALL_NOTE = "The depth map shows an obstacle ahead, about 1 meter."


def test_says_clear_finds_every_claim_and_allows_only_the_disclaimer():
    for claim in ("The path is clear.", "All clear ahead.", "It is safe to walk.", "The left side is clearer.",
                  "Nothing there.", "No obstacles ahead."):
        assert says_clear(claim), claim
    assert not says_clear("I did not detect anything in your path, but that does not mean it is clear.")
    assert not says_clear("I don't see anything I can name right now. That does not mean the way is clear.")


@pytest.mark.parametrize("scene", sorted(SCENES))
@pytest.mark.parametrize("q", QUESTIONS)
@pytest.mark.parametrize("note", ["", WALL_NOTE])
def test_answers_never_say_clear(scene, q, note):
    a = answer(q, SCENES[scene], CAM, cane_note=note)
    assert not says_clear(a), a


def test_path_answers():
    """Never "nothing in your path" alone (reviewer: it implied a clear path and ignored the cane): the web app's
    wording, and the cane's confirmed obstacle when there is one."""
    a = answer("is the path clear", [], CAM)
    assert a == ("I did not detect anything in your path, but that does not mean it is clear. I can miss obstacles, so "
                 "keep using your cane or guide dog.")
    assert answer("is the path clear", SCENES["person in path"], CAM).startswith("Careful, there's a person")
    for q in ("is the path clear", "is it safe to walk", "can I walk straight", "is the way blocked"):
        a = answer(q, [], CAM, cane_note=WALL_NOTE)
        assert a.startswith(WALL_NOTE) and "keep using your cane" in a and not says_clear(a), (q, a)
    a = answer("is the path clear", SCENES["person in path"], CAM, cane_note=WALL_NOTE)
    assert a.startswith("Careful, there's a person") and a.endswith(WALL_NOTE)
    assert answer("what's ahead", [], CAM, cane_note=WALL_NOTE).endswith(WALL_NOTE)
    assert WALL_NOTE not in answer("what's on my left", [], CAM, cane_note=WALL_NOTE)   # the note is about ahead


def test_path_answers_name_the_users_own_travel_aid():
    """Reviewer: the web app told a guide dog user (and one with neither) to "check with your cane". A path answer now
    names the aid the user chose (run_live --aid, the controller's aid; the web app's "I travel with"), in the same
    words as web/src/cane.ts aidAdvice, and never says clear."""
    from seeforme.live.cane import AIDS
    assert set(AID_ADVICE) == set(AIDS)
    for aid, end in (("cane", "keep using your cane."), ("guide-dog", "keep working with your guide dog."),
                     ("none", "go carefully."), (None, "keep using your cane or guide dog.")):
        for note in ("", WALL_NOTE):
            a = answer("is the path clear", [], CAM, cane_note=note, aid=aid)
            assert a.endswith(f"so {end}") and not says_clear(a), (aid, a)
            assert ("cane" in a) is (aid in ("cane", None)) and ("guide dog" in a) is (aid in ("guide-dog", None)), a
    r, ctl, _ = router()
    ctl.aid = "guide-dog"
    assert r.handle(Command("ask", "is the path clear"), []).endswith("so keep working with your guide dog.")
    cane_ts = (Path(__file__).resolve().parents[1] / "web" / "src" / "cane.ts").read_text()
    for aid, words in AID_ADVICE.items():          # the web app says the same
        assert f'"{aid}": "{words}"' in cane_ts or f'{aid}: "{words}"' in cane_ts, (aid, words)


def test_object_and_side_answers():
    assert answer("is there a chair", SCENES["chair left"], CAM) == "Yes, there's a chair about 2 meters away, on your left."
    assert answer("is there a chair on my right", SCENES["chair left"], CAM).startswith("I could not confirm a chair on your right")
    assert answer("is there a person", [], CAM).startswith("I could not confirm a person")
    assert answer("what's on my left", SCENES["busy"], CAM).startswith("On your left there's a chair")
    assert answer("is there a sign", SCENES["busy"], CAM) == "Yes, there's a sign that says EXIT, straight ahead."
    assert answer("any vehicles", SCENES["busy"], CAM).startswith("Yes, there's a car")
    near = track(9, "chair", z=1.0, depth_conf=0.2)            # a box-size guess: no number
    assert "very close" in answer("is there a chair", [near], CAM)


def test_unclear_text_is_not_answered_with_the_scene():
    assert is_question("is there a chair") and is_question("what's on my left") and is_question("chair")
    assert not is_question("newt") and not is_question("bars")


def test_read_phrase():
    sign = track(4, "text", kind="text", text="ROOM 204")
    assert read_phrase([sign]) == "The text says: ROOM 204"
    assert read_phrase([]).startswith("I don't see readable text")
    assert read_phrase([], ocr=False).startswith("Reading is off")


# ---- safety cues pass a pause -------------------------------------------------------------------------------------
def test_safety_cues():
    still = track(1, x=-1.5)                                   # a person standing on the left, out of the path
    assert not is_safety_cue(cue(still), still)
    assert is_safety_cue(cue(still, action="interrupt"), still)
    assert is_safety_cue(cue(still, action="haptic"), still)
    assert is_safety_cue(cue(still, severity=0.95), still)
    assert is_safety_cue(cue(still, ttc=3.0), still)
    assert is_safety_cue(cue(track(2, path=0.8)), track(2, path=0.8))          # in the walking path
    approaching = track(3, x=-1.5, ttc=4.0)
    assert is_safety_cue(cue(approaching), approaching)
    far = track(6, x=-1.5, z=9.0)
    far.rel_speed_mps = -0.6                                   # measured, closing, TTC 15 s: still approaching
    assert is_safety_cue(cue(far), far)
    far.depth_conf = 0.1                                       # a box guess "closes" whenever the user walks
    assert not is_safety_cue(cue(far), far)
    car = track(4, "car", x=-3.0, z=10.0)
    assert is_safety_cue(cue(car), car)
    chair = track(5, "chair", x=-1.5)
    assert not is_safety_cue(cue(chair), chair) and is_safety_cue(cue(chair), chair, goal=["seat"])
    cane = Cue(1.0, -7, "obstacle", "object", "Obstacle ahead, 1 meter.", 0.7, "cane", severity=0.7)
    assert is_safety_cue(cane, None)


def test_safety_tracks():
    """What lets the scheduler cut a cane cue or a reply (controller): at least everything is_safety_cue passes."""
    assert not is_safety_track(None) and not is_safety_track(track(1, x=-1.5))
    assert not is_safety_track(track(5, "chair", x=-1.5)) and is_safety_track(track(5, "chair", x=-1.5), ["chair"])
    assert is_safety_track(track(2, path=0.8)) and is_safety_track(track(3, x=-1.5, ttc=4.0))
    assert is_safety_track(track(4, "car", x=-3.0, z=10.0)) and is_safety_track(track(7, "stairs", x=-2.0, z=6.0))
    closing = track(6, x=-1.5, z=9.0)
    closing.rel_speed_mps = -0.6
    assert is_safety_track(closing)
    sign = track(8, "text", kind="text", text="ROOM 204", path=0.9)
    assert not is_safety_track(sign) and is_safety_track(sign, ["204"])
    for trk in (track(2, path=0.8), track(3, x=-1.5, ttc=4.0), track(4, "car", x=-3.0), closing):
        assert is_safety_cue(cue(trk), trk) and is_safety_track(trk)


# ---- the router, with fakes ---------------------------------------------------------------------------------------
class FakeAdapter:
    def __init__(self):
        self.mode, self.override, self.calls = "navigate", None, []

    def force_mode(self, mode):
        self.calls.append(mode)
        self.override = None if mode in (None, "auto") else mode
        if self.override:
            self.mode = self.override
        return self.override

    def step(self, activity, t, has_goal=False):
        self.calls.append(("step", activity, has_goal))
        self.mode = self.override or ("find" if has_goal else "scene")
        return self.mode


class FakeController:
    class _Policy:
        class _Cfg:
            verbosity = "balanced"
        cfg = _Cfg()

    class _Activity:
        activity = "stationary"

    def __init__(self):
        self.goal_tags = []
        self.adapter = FakeAdapter()
        self.activity = self._Activity()
        self.policy = self._Policy()
        self.camera = CAM
        self.units = "m"
        self.log = None

    def set_goal(self, tags):
        self.goal_tags = list(tags)

    def summary(self, tracks):
        return f"SUMMARY of {len(tracks)}"


class FakeSpeaker:
    def __init__(self, chime=True):
        self.enabled = True
        self._chime = chime
        self.chimes = 0

    def chime(self):
        self.chimes += 1
        return self._chime and self.enabled


class FakeLog:
    def __init__(self):
        self.rows = []

    def write(self, kind, t, **kw):
        self.rows.append((kind, t, kw))


class FakeScheduler:
    name, decision_interval, supports_actions = "fake", 0.5, True

    class _Cfg:
        verbosity = "balanced"
        units = "m"

    def __init__(self, make):
        self.cfg = self._Cfg()
        self.make = make           # tracks -> cues

    def decide(self, tracks, t, ch=None):
        return self.make(tracks)

    def reset(self):
        pass

    def last_observations(self):
        return []


def router(**kw):
    ctl, sp = FakeController(), FakeSpeaker()
    return CommandRouter(ctl, sp, **kw), ctl, sp


def test_find_sets_goal_tags_and_confirms():
    r, ctl, _ = router()
    assert r.handle(Command("find", "room 204")) == "Looking for room 204. I'll tell you when I see it."
    assert ctl.goal_tags == ["room 204", "204"]
    assert r.handle(Command("find", "seat")) == "Looking for a seat. I'll tell you when I see it."
    assert ctl.goal_tags == ["seat", "chair", "bench"]
    reply = r.handle(Command("find", "chair"), SCENES["chair left"])
    assert reply == "Looking for a chair. I can see it now: a chair about 2 meters away, on your left."
    assert r.handle(Command("find", "exit")).startswith("Looking for the exit.")


def test_find_without_a_target_keeps_the_goal():
    r, ctl, _ = router()
    ctl.goal_tags = ["door"]
    assert r.handle(Command("find", None)).startswith("Tell me what to look for")
    assert r.handle(Command("find", "it")).startswith("Tell me what to look for")
    assert ctl.goal_tags == ["door"]


def test_find_a_sign_without_ocr_says_so():
    r, _, _ = router(ocr=False)
    assert "Reading is off" in r.handle(Command("find", "room 204"))
    assert "Reading is off" not in r.handle(Command("find", "chair"))


def test_stop_find():
    r, ctl, _ = router()
    assert r.handle(Command("stop_find")) == "I wasn't looking for anything."
    ctl.goal_tags = ["door"]
    assert r.handle(Command("stop_find")) == "Okay, I stopped looking." and ctl.goal_tags == []


def test_pause_silences_routine_cues_but_not_hazards():
    still, car = track(1, x=-1.5), track(2, "car", x=2.5, z=8.0)
    offered = {"cues": []}
    r, ctl, _ = router()
    pol = r.wrap(FakeScheduler(lambda tracks: list(offered["cues"])))
    offered["cues"] = [cue(still)]
    assert pol.decide([still], 1.0) == [cue(still)]                 # not paused: as offered
    assert r.handle(Command("pause")) == "Paused. I'll still warn you about hazards." and r.paused
    assert pol.decide([still], 1.5) == []                           # paused: routine cue held
    assert [(t, c.message) for t, c in r.take_held()] == [(1.5, "Person.")] and r.take_held() == []
    for hazard, trk in ((cue(still, action="interrupt"), still), (cue(car), car), (cue(still, ttc=2.0), still)):
        offered["cues"] = [hazard]
        assert pol.decide([trk], 2.0) == [hazard]                   # hazards are said while paused
    assert r.handle(Command("pause")) == "Already paused."
    offered["cues"] = [cue(still)]
    assert r.handle(Command("start"), [still]) == "Resumed."
    assert not r.paused and pol.decide([still], 3.0) == [cue(still)]
    assert r.held_while_paused == 1 and r.passed_while_paused == 3
    assert r.handle(Command("start")) == "I'm already on."


def test_wrapped_policy_forwards_attributes():
    r, _, _ = router()
    inner = FakeScheduler(lambda tracks: [])
    pol = r.wrap(inner)
    assert pol.name == "fake" and pol.decision_interval == 0.5 and pol.supports_actions
    pol.cfg.verbosity = "quiet"
    assert inner.cfg.verbosity == "quiet"


def test_describe_returns_the_summary_and_repeat_says_it_again():
    r, _, _ = router()
    assert r.handle(Command("repeat")) == "I haven't said anything yet."
    assert r.handle(Command("describe"), [track(1)]) == "SUMMARY of 1"
    assert r.handle(Command("repeat")) == "I said: SUMMARY of 1"
    r.note_spoken(cue(track(1), "Person ahead, 2 meters."))
    r.note_spoken(cue(track(1), "Stop.", action="haptic"))        # not spoken
    assert r.handle(Command("repeat")) == "I said: Person ahead, 2 meters."


def test_ask_answers_about_the_view_and_never_says_clear():
    r, _, _ = router()
    for scene in SCENES.values():
        a = r.handle(Command("ask", "is the path clear"), scene)
        assert not says_clear(a), a
    assert r.handle(Command("ask", "is there a chair"), SCENES["chair left"]).startswith("Yes, there's a chair")
    assert r.handle(Command("ask", "newt")).startswith("Sorry, I didn't")


def test_typed_key_letters_toggle_pause_and_sound():
    """p and s typed in the terminal (voice.KEY_LINES) toggle, as the window keys do."""
    r, _, sp = router()
    assert r.handle(Command("pause", "toggle")).startswith("Paused") and r.paused
    assert r.handle(Command("pause", "toggle")) == "Resumed." and not r.paused
    assert r.handle(Command("mute", "toggle")) == "Sound off."
    r.after_reply()
    assert not sp.enabled
    assert r.handle(Command("mute", "toggle")) == "Sound on." and sp.enabled


def test_mute_waits_for_its_confirmation():
    r, _, sp = router()
    assert r.handle(Command("mute")) == "Sound off." and sp.enabled     # "Sound off." must still be heard
    r.after_reply()
    assert not sp.enabled
    assert r.handle(Command("unmute")) == "Sound on." and sp.enabled
    r.handle(Command("mute")); r.after_reply()
    assert r.handle(Command("start")).startswith("Resumed") and sp.enabled   # start also brings the sound back


def test_wake_chimes_or_says_yes():
    r, _, sp = router()
    assert r.handle(Command("wake")) == "" and sp.chimes == 1
    sp._chime = False
    assert r.handle(Command("wake")) == "Yes?"


def test_modes_and_verbosity():
    r, ctl, _ = router()
    assert r.handle(Command("mode", "scene")) == "Scene mode." and ctl.adapter.mode == "scene"
    assert r.handle(Command("mode", "navigate")) == "Navigation mode."
    assert r.handle(Command("mode", "read")) == "Reading mode."
    assert r.handle(Command("mode", "auto"), t=3.0) == "Automatic mode, now scene."
    assert ctl.adapter.calls[-2:] == [None, ("step", "stationary", False)]
    assert r.handle(Command("verbosity", "detailed")) == "More detail." and ctl.policy.cfg.verbosity == "detailed"
    assert r.handle(Command("verbosity", "quiet")) == "Less detail." and ctl.policy.cfg.verbosity == "quiet"


def test_help_flip_quit_read():
    r, _, _ = router()
    assert r.handle(Command("help")).startswith("Type a command")
    r.voice = True
    assert r.handle(Command("help")).startswith("Say SeeForMe, then a command")
    assert r.handle(Command("flip")) == "Switching cameras is only in the web app."
    r.flip = lambda: "Camera 1."
    assert r.handle(Command("flip")) == "Camera 1."
    assert r.handle(Command("read"), [track(4, "text", kind="text", text="EXIT")]) == "The text says: EXIT"
    assert not r.quit_requested and r.handle(Command("quit")) == "Goodbye." and r.quit_requested


def test_commands_are_logged():
    log = FakeLog()
    r = CommandRouter(FakeController(), FakeSpeaker(), log=log)
    r.handle(Command("find", "door", text="see for me find a door", source="vosk"), t=2.0)
    assert log.rows == [("command", 2.0, {"intent": "find", "arg": "door", "source": "vosk", "heard": "see for me find a door",
                                          "said": "Looking for a door. I'll tell you when I see it."})]
    assert r.report() == ["commands: find 1"]


# ---- through the real LiveController ------------------------------------------------------------------------------
def test_pause_through_the_live_controller():
    from seeforme.live.controller import LiveController
    still = track(1, x=-1.5, z=2.0)
    offered = {"cues": []}
    sched = FakeScheduler(lambda tracks: list(offered["cues"]))
    ctl = LiveController(sched, mode="navigate", gate=False)
    r = CommandRouter(ctl, FakeSpeaker())
    pol = r.wrap(ctl.policy)
    r.handle(Command("find", "chair"))
    assert ctl.goal_tags == ["chair"]
    r.handle(Command("pause"))
    offered["cues"] = [cue(still)]
    assert pol.decide([still], 1.0, None) == []
    chair = track(2, "chair", x=-1.5)                              # the FIND goal comes through a pause
    offered["cues"] = [cue(chair)]
    assert pol.decide([chair], 1.5, None) == [cue(chair)]
    assert r.handle(Command("describe"), [still]) == ctl.summary([still])


def test_path_question_reports_the_canes_wall():
    """Reviewer repro: a confirmed wall 1.2 m ahead, "is the path clear?" answered "I don't see anything in your path"."""
    from seeforme.live.controller import LiveController
    from tests.test_live_cane import FakeScheduler as CaneScheduler, render, wall
    ctl = LiveController(CaneScheduler(), setup="handheld")
    for i in range(30):
        ctl.on_frame(None, 0.1 * i, [], depth=render([wall(1.2)], seed=i))
    assert ctl.cane.summary()["obstacle"]["sector"] == "ahead"
    r = CommandRouter(ctl, FakeSpeaker())
    t = 2.95
    for q in ("is the path clear", "is it safe to walk", "can I walk straight", "is the way blocked"):
        a = r.handle(Command("ask", q), [], t)
        assert a.startswith("The depth map shows an obstacle ahead") and not says_clear(a), (q, a)
    assert r.handle(Command("describe"), [], t).startswith("The depth map shows an obstacle ahead")   # first
    # the reply said it: the cane holds that obstacle as said and does not cut the reply to say it again
    assert ctl.cane.stats["noted"] >= 1 and ctl.cane.propose(t, moving=True) is None
    assert ctl.cane_note(t + 5.0) == ""                  # no recent depth map: nothing to report
    ctl.desk = True
    assert r.handle(Command("ask", "is the path clear"), [], t).startswith("I did not detect anything")


# ---- run_live, headless, with scripted commands -------------------------------------------------------------------
# Each run pays the interpreter, torch and detector start-up plus one detector pass per frame (seconds each on a loaded
# machine), so they use the small detector, no depth, no OCR, the linear P-CACS rule and as few frames as the check
# needs: the commands are spread over the first 0.4 s (15 frames of the 30 fps clip).
FAST = ["--fast", "--headless", "--no-ocr", "--no-depth", "--no-log", "--policy", "pcacs", "--model", "yolo11n.pt"]


@pytest.mark.skipif(not CLIP.exists(), reason="local EgoBlind clip data/egoblind/videos/train/00700.mp4 not present")
def test_run_live_headless_with_scripted_commands():
    cmd = [sys.executable, str(ROOT / "scripts" / "run_live.py"), "--video", str(CLIP), *FAST, "--frames", "15",
           "--commands", "0.1:see for me find a chair;0.2:what's around me;0.3:pause;0.4:start"]
    env = dict(os.environ, OMP_NUM_THREADS="2")
    out = subprocess.run(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    text = out.stdout
    for expected in ("heard     find chair", "reply     Looking for a chair.", "heard     describe",
                     "heard     pause", "reply     Paused. I'll still warn you about hazards.", "heard     start",
                     "reply     Resumed", "commands: describe 1, find 1, pause 1, start 1", "aid cane"):
        assert expected in text, text[-3000:]
    assert "Voice commands: run make voice-setup" not in text      # headless: no microphone, no hint
    assert "scripted commands not reached" not in text


@pytest.mark.skipif(not CLIP.exists(), reason="local EgoBlind clip data/egoblind/videos/train/00700.mp4 not present")
def test_run_live_reads_commands_piped_into_stdin():
    """Reviewer repro: piped commands were ignored ("commands keys only") and a headless run never quit."""
    cmd = [sys.executable, str(ROOT / "scripts" / "run_live.py"), "--video", str(CLIP), *FAST, "--frames", "200"]
    env = dict(os.environ, OMP_NUM_THREADS="2")
    out = subprocess.run(cmd, cwd=ROOT, env=env, input="what's around me\nfind a chair\nq\n", capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    text = out.stdout
    assert "commands typed" in text and "commands keys only" not in text, text[-3000:]
    for expected in ("heard     describe", "heard     find chair", "heard     quit", "reply     Goodbye."):
        assert expected in text, text[-3000:]
    frames = int(text.split("frames: ")[1].split()[0])
    assert frames < 200                                            # quit ended the run
    assert "end of the piped commands" not in text                 # a video run: the commands' end is not the run's


@pytest.mark.parametrize("headless,video,frames,tty,sources,stops", [
    (True, False, 0, False, ["typed"], True),            # headless camera, commands piped in: nothing else ends it
    (True, True, 0, False, ["typed"], False),            # a replayed video ends by itself
    (True, False, 300, False, ["typed"], False),         # so does --frames
    (True, False, 0, True, ["typed"], False),            # a terminal's ctrl-D is not the end
    (True, False, 0, False, ["typed", "script"], False),  # a script may still quit
    (True, False, 0, False, ["vosk", "typed"], False),    # so may the microphone
    (False, False, 0, False, ["typed"], False)])          # a window has the q key
def test_end_of_piped_commands_stops_only_a_run_nothing_else_stops(headless, video, frames, tty, sources, stops):
    from seeforme.live.voice import input_end_stops_run
    assert input_end_stops_run(headless, video, frames, tty, sources) is stops

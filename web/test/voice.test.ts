// Voice commands: every shared vector in tests/data/voice_commands.json (the Python parser passes the same file), plus
// the listener's guards (the app never acts on its own voice; the wake phrase alone opens a follow-up window; a lower-ranked
// alternative never starts a command by itself, turns a question into one that changes something, or drops words to make
// one; a session that ends at once backs off), and what the camera commands decide in the browser (flipPlan, switchedReply;
// camcontrol.test.ts runs the commands themselves on a fake camera).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { HELP_TEXT, IDLE_TEXT, VoiceInput, facingOfSide, findTarget, flipPlan, isEcho, mayRefine, parseCommand, sideOfFacing,
  spokenNumbers, switchedReply, wordsToDigits, type Command, type VoiceHooks } from "../src/voice";

const here = dirname(fileURLToPath(import.meta.url));
// the bundle runs from test/.build: the repository root is three levels up
const vectors = JSON.parse(readFileSync(resolve(here, "../../../tests/data/voice_commands.json"), "utf8")) as {
  intents: string[]; wake: Array<{ text: string; intent: string | null; arg: string | null }>; direct: Array<{ text: string; intent: string | null; arg: string | null }>;
  unicode: Array<{ text: string; mode: "wake" | "direct"; intent: string | null; arg: string | null }>;
  refine: Array<{ best: string; alt: string; mode: "wake" | "direct"; take: boolean }>;
};

for (const mode of ["wake", "direct"] as const) {
  test(`voice vectors: every "${mode}" case`, () => {
    const bad: string[] = [];
    for (const v of vectors[mode]) {
      const got = parseCommand(v.text, mode);
      const intent = got?.intent ?? null, arg = got?.arg ?? null;
      if (intent !== v.intent || arg !== v.arg) bad.push(`${JSON.stringify(v.text)}: expected ${v.intent}/${v.arg}, got ${intent}/${arg}`);
    }
    assert.deepEqual(bad, []);
    assert.ok(vectors[mode].length > 5);
  });
}

// non-ASCII text: Unicode words, word boundaries and decimal digits, NFC first, as Python's re and voice.py normalize
// ("what does the café say" is a read; "josésee for me" has no wake phrase). tests/test_live_voice.py also cross-checks
// this parser against Python's on a seeded fuzz corpus.
test("voice vectors: every non-ASCII case", () => {
  const bad: string[] = [];
  for (const v of vectors.unicode) {
    const got = parseCommand(v.text, v.mode);
    const intent = got?.intent ?? null, arg = got?.arg ?? null;
    if (intent !== v.intent || arg !== v.arg) bad.push(`${JSON.stringify(v.text)} (${v.mode}): expected ${v.intent}/${v.arg}, got ${intent}/${arg}`);
  }
  assert.deepEqual(bad, []);
  assert.ok(vectors.unicode.length > 5);
});

test("every intent the parser returns is in the shared list", () => {
  for (const mode of ["wake", "direct"] as const) for (const v of [...vectors[mode], ...vectors.unicode]) {
    const got = parseCommand(v.text, mode);
    if (got) assert.ok(vectors.intents.includes(got.intent), got.intent);
  }
});

test("spoken numbers", () => {
  assert.deepEqual(spokenNumbers("room two oh four".split(" ")), ["room", "204"]);
  assert.deepEqual(spokenNumbers("gate twenty one".split(" ")), ["gate", "21"]);
  assert.deepEqual(spokenNumbers("room two hundred and five".split(" ")), ["room", "205"]);
  assert.deepEqual(spokenNumbers("oh look".split(" ")), ["oh", "look"]);   // "oh" is zero only inside a number
  // as voice.py: a leading "oh" keeps its zero, homophones are digits only in a run with an inner "oh"
  assert.equal(wordsToDigits("room oh four"), "room 04");
  assert.equal(wordsToDigits("room to oh for"), "room 204");
  assert.equal(wordsToDigits("table for two"), "table for 2");
  assert.equal(findTarget("the nearest bathroom"), "bathroom");
  assert.equal(findTarget("a chair for me please"), "chair");
  assert.equal(parseCommand("see for me find room twenty one b", "wake")?.arg, "room 21 b");
});

test("the app's own phrases, heard back, change nothing (help at most)", () => {
  // a screen reader reading #voice-state or the status region into the open microphone, or echo suppression failing
  const html = readFileSync(resolve(here, "../../index.html"), "utf8");
  const shown = [...html.matchAll(/<p id="voice-state"[^>]*>([^<]*)<\/p>/g)].map((m) => m[1].replace(/&quot;/g, '"'));
  assert.equal(shown.length, 1);
  for (const phrase of [IDLE_TEXT, `${IDLE_TEXT}.`, HELP_TEXT, ...shown, "SeeForMe ready. Say 'SeeForMe, help' for commands.",
    "Hi, this is SeeForMe. Stairs down, about 3 meters. Handrail left.", "Lost the SeeForMe server: text reading and metric depth are off.",
    "Without the SeeForMe server I can only find common objects such as a chair, a person or a bottle."]) {
    for (const mode of ["wake", "direct"] as const) {
      const got = parseCommand(phrase, mode);
      if (mode === "wake" || /see ?for ?me/i.test(phrase)) assert.ok(got == null || got.intent === "ask" || got.intent === "help", `${mode} ${JSON.stringify(phrase)}: ${JSON.stringify(got)}`);
    }
  }
  assert.equal(parseCommand(IDLE_TEXT, "wake")?.intent, "help");
});

test("the app's replies about the camera, heard back after Talk or a wake phrase, are not commands", () => {
  // web/src/main.ts runCommand: pause stops the camera here and says so; the flip replies name a side only when the
  // browser reports it. A reply such as "Front camera." would be a flip if the microphone caught it.
  const replies = ["Paused. The camera is off.", "The camera is already off.", "I couldn't switch the camera. The camera is off.",
    "Still starting the camera.", "The camera is still closing. Say start again in a moment.", "I couldn't start the camera. Press Start.",
    flipPlan(null, false, null).say, flipPlan("back", true, "back").say, flipPlan(null, true, null).say,
    flipPlan("front", true, null, null, "switch").say, flipPlan("front", false, null, null, "start").say, flipPlan("back", true, null, "back").say,
    switchedReply("front", "front"), switchedReply("front", "back"), switchedReply("back", null),
    switchedReply("front", "front", true), switchedReply("back", "front", true), switchedReply("back", null, true)];
  for (const r of replies) for (const mode of ["wake", "direct"] as const) {
    const got = parseCommand(r, mode);
    assert.ok(got == null || got.intent === "ask", `${mode} ${JSON.stringify(r)}: ${JSON.stringify(got)}`);
  }
});

test("front / back camera: already on that side does nothing; otherwise that side, said only as the browser reports it", () => {
  assert.equal(sideOfFacing("user"), "front");
  assert.equal(sideOfFacing("environment"), "back");
  assert.equal(sideOfFacing(undefined), null);   // most laptop and USB webcams do not say
  assert.equal(sideOfFacing("left"), null);
  assert.equal(facingOfSide("front"), "user");
  assert.equal(facingOfSide("back"), "environment");
  const plan = (text: string, running: boolean, current: "front" | "back" | null) => flipPlan(parseCommand(text, "direct")!.arg, running, current);
  assert.deepEqual(plan("front camera", true, "front"), { act: "none", say: "Already on the front camera." });
  assert.deepEqual(plan("switch to the back camera", true, "back"), { act: "none", say: "Already on the back camera." });
  assert.deepEqual(plan("selfie camera", true, "back"), { act: "switch", say: "" });
  assert.deepEqual(plan("rear camera", true, null), { act: "switch", say: "" });   // unknown side: ask the browser for it
  assert.deepEqual(plan("flip camera", true, "front"), { act: "next", say: "Switching camera." });
  assert.deepEqual(plan("other camera", true, null), { act: "next", say: "Switching camera." });
  assert.deepEqual(plan("flip camera", false, null), { act: "none", say: "The camera is off." });
  assert.equal(switchedReply("front", "front"), "Switched to the front camera.");
  assert.equal(switchedReply("front", "back"), "I couldn't find a front camera. This is the back camera.");
  assert.equal(switchedReply("back", null), "I asked for the back camera, but this camera does not say which way it faces.");
});

test("front / back camera with the camera off starts it on that side, and the reply names only the side the browser reports", () => {
  // before: "turn on the front camera" only stored the side, and the reply "When you start it, I'll use the front camera"
  // claimed a side no browser had reported (a laptop then opened its only camera and said nothing about it)
  const plan = (text: string) => flipPlan(parseCommand(text, "direct")!.arg, false, null);
  assert.deepEqual(plan("turn on the front camera"), { act: "start", say: "" });
  assert.deepEqual(plan("use the back camera"), { act: "start", say: "" });
  assert.equal(switchedReply("front", "front", true), "This is the front camera.");
  assert.equal(switchedReply("back", "front", true), "I couldn't find a back camera. This is the front camera.");
  assert.equal(switchedReply("back", null, true), "I asked for the back camera, but this camera does not say which way it faces.");
});

test("front / back camera: a request this camera already answered does not reopen it; one camera opens at a time", () => {
  // a laptop with one camera that does not report its side: every "back camera" stopped and reopened it, and the reset that
  // follows cleared the tracker, the repeat filter and the cane's memory
  assert.deepEqual(flipPlan("back", true, null, "back"),
    { act: "none", say: "I asked for the back camera, but this camera does not say which way it faces." });
  // a phone without a front camera: asked for the front, the browser gave the back one
  assert.deepEqual(flipPlan("front", true, "back", "front"), { act: "none", say: "I couldn't find a front camera. This is the back camera." });
  // another side, or no earlier request on this camera: switch as before
  assert.deepEqual(flipPlan("front", true, null, "back"), { act: "switch", say: "" });
  assert.deepEqual(flipPlan("back", true, null, null), { act: "switch", say: "" });
  // an open is still in progress: nothing more opens, and the reply says which open it waits for (a start was told
  // "Still switching the camera." although nothing was switching)
  for (const side of ["front", "back", null]) {
    assert.deepEqual(flipPlan(side, true, null, null, "switch"), { act: "none", say: "Still switching the camera." }, `${side}`);
    assert.deepEqual(flipPlan(side, false, null, null, "start"), { act: "none", say: "Still starting the camera." }, `${side}`);
  }
});

test("echo guard: the app's own words are not a command", () => {
  assert.equal(isEcho("person close ahead", ["Person, close, ahead."]), true);
  assert.equal(isEcho("see for me find a door", ["Looking for door."]), false);
});

function fakeHooks(speaking = false) {
  const got: Array<[Command, string]> = [], states: string[] = [];
  let acks = 0;
  const hooks: VoiceHooks = {
    onCommand: (c, how) => got.push([c, how]), onState: (_s, text) => states.push(text),
    appSpeaking: () => speaking, recentSpeech: () => ["Person, close, ahead."], ack: () => { acks++; },
  };
  return { hooks, got, states, acks: () => acks };
}

test("listener: results while the app speaks are ignored", () => {
  const f = fakeHooks(true);
  const v = new VoiceInput(f.hooks, null);
  v.heard(["see for me find a door"], "wake");
  assert.equal(f.got.length, 0);
});

test("listener: wake phrase alone opens a follow-up window for one command", () => {
  const f = fakeHooks(false);
  const v = new VoiceInput(f.hooks, null);
  v.handsFree = true;
  v.heard(["see for me"], "wake");
  assert.equal(f.acks(), 1);
  assert.equal(v.listeningForFollowUp, true);
  v.heard(["find the stairs"], "wake");             // no wake phrase needed inside the window
  assert.deepEqual(f.got.map(([c]) => c), [{ intent: "find", arg: "stairs" }]);
  assert.equal(v.listeningForFollowUp, false);
  v.heard(["find a chair"], "wake");                 // the window is used up
  assert.equal(f.got.length, 1);
  v.heard(["person close ahead"], "wake");           // the app's own cue: never a command
  assert.equal(f.got.length, 1);
});

test("listener: push-to-talk needs no wake phrase", () => {
  const f = fakeHooks(false);
  const v = new VoiceInput(f.hooks, null);
  v.heard(["pause"], "talk");
  assert.deepEqual(f.got.map(([c, how]) => [c.intent, how]), [["pause", "talk"]]);
});

test("listener: the best transcript decides; a lower alternative only refines a command addressed to the app", () => {
  const run = (alts: string[], how: "wake" | "talk" = "wake") => {
    const f = fakeHooks(false);
    const v = new VoiceInput(f.hooks, null);
    v.handsFree = true;
    v.heard(alts, how);
    return { cmds: f.got.map(([c]) => `${c.intent}:${c.arg}`), acks: f.acks(), follow: v.listeningForFollowUp };
  };
  // ordinary speech on top: a mis-heard wake phrase further down never runs a command ("pause" would stop the camera)
  assert.deepEqual(run(["nice weather", "c for me pause"]).cmds, []);
  // a bare wake phrase stays a wake phrase whatever the alternatives are
  assert.deepEqual(run(["see for me", "the cat"]), { cmds: [], acks: 1, follow: true });
  assert.deepEqual(run(["see for me", "see for me to"]), { cmds: [], acks: 1, follow: true });
  // ... but a known command among the alternatives of an addressed transcript is taken
  assert.deepEqual(run(["see for me fine a door", "see for me find a door"]).cmds, ["find:door"]);
  assert.deepEqual(run(["see for me", "see for me pause"]).cmds, ["pause:null"]);
  assert.deepEqual(run(["paws", "pause"], "talk").cmds, ["pause:null"]);
  // a command on top is never replaced
  assert.deepEqual(run(["see for me read", "see for me pause"]).cmds, ["read:null"]);
});

test("listener: a question addressed to the app stays a question whatever the alternatives say", () => {
  const run = (alts: string[], how: "wake" | "talk" = "wake") => {
    const f = fakeHooks(false);
    const v = new VoiceInput(f.hooks, null);
    v.handsFree = true;
    v.heard(alts, how);
    return f.got.map(([c]) => `${c.intent}:${c.arg}`);
  };
  // an alternative without "is" is a command ("the camera on" starts the camera, "the back camera" flips it): a question
  // about the camera must never start, stop or switch it
  assert.deepEqual(run(["see for me is the camera on", "see for me the camera on"]), ["ask:is the camera on"]);
  assert.deepEqual(run(["see for me is the camera off", "see for me the camera off"]), ["ask:is the camera off"]);
  assert.deepEqual(run(["see for me is that the back camera", "see for me the back camera"]), ["ask:is that the back camera"]);
  assert.deepEqual(run(["can you tell me is the camera on", "the camera on"], "talk"), ["ask:can you tell me is the camera on"]);
  // free text that is not a question is still refined by a command among the alternatives
  assert.deepEqual(run(["see for me fine a door", "see for me find a door"]), ["find:door"]);
});

test("listener: a misheard question is still corrected into an answer; free text is not shortened into a command", () => {
  // reviewer repros: the question guard of the last round also blocked these (HEAD took them), so a misheard "what's around
  // me" got a list of objects and "how do I get to the exit" set no goal; and "tell me goodbye", which the parser keeps as a
  // question, still quit through an alternative that only left "tell me" out
  const run = (alts: string[], how: "wake" | "talk" = "wake") => {
    const f = fakeHooks(false);
    const v = new VoiceInput(f.hooks, null);
    v.handsFree = true;
    v.heard(alts, how);
    return f.got.map(([c]) => `${c.intent}:${c.arg}`);
  };
  assert.deepEqual(run(["see for me what's a round me", "see for me what's around me"]), ["describe:null"]);
  assert.deepEqual(run(["see for me what does it stay", "see for me what does it say"]), ["read:null"]);
  assert.deepEqual(run(["see for me where am eye", "see for me where am i"]), ["describe:null"]);
  assert.deepEqual(run(["see for me how do i get too the exit", "see for me how do i get to the exit"]), ["find:exit"]);
  assert.deepEqual(run(["see for me tell me goodbye", "see for me goodbye"]), ["ask:tell me goodbye"]);
  assert.deepEqual(run(["see for me tell me stop", "see for me stop"]), ["ask:tell me stop"]);
  // the first alternative the rule allows is taken, not the first command
  assert.deepEqual(run(["see for me is the camera on", "see for me the camera on", "see for me what's around me"]), ["describe:null"]);
});

test("refine vectors: mayRefine, and the listener on [best, alt], pass every shared case", () => {
  for (const c of vectors.refine) {
    assert.equal(mayRefine(c.best, c.alt, c.mode), c.take, `${c.mode} ${JSON.stringify(c.best)} | ${JSON.stringify(c.alt)}`);
    const f = fakeHooks(false);
    const v = new VoiceInput(f.hooks, null);
    v.handsFree = true;
    v.heard([c.best, c.alt], c.mode === "wake" ? "wake" : "talk");
    const want = parseCommand(c.take ? c.alt : c.best, c.mode);
    assert.deepEqual(f.got.map(([cmd]) => cmd), want && want.intent !== "wake" ? [want] : [], `${c.mode} ${JSON.stringify(c.best)}`);
  }
});

test("a pause that names the camera has arg camera (the terminal demo says its camera stays on)", () => {
  for (const text of ["camera off", "stop the camera", "turn the camera off", "see for me close the camera"]) {
    assert.deepEqual(parseCommand(text, "direct"), { intent: "pause", arg: "camera" }, text);
  }
  for (const text of ["pause", "stop", "stop talking", "hold on a second"]) assert.deepEqual(parseCommand(text, "direct"), { intent: "pause", arg: null }, text);
});

test("listener: a hands-free session that ends at once backs off instead of restarting every 300 ms", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"] });
  let starts = 0;
  class QuickEnd {
    continuous = false; interimResults = false; lang = ""; maxAlternatives = 1;
    onresult: ((ev: unknown) => void) | null = null; onerror: ((ev: unknown) => void) | null = null;
    onend: (() => void) | null = null; onstart: (() => void) | null = null;
    start() { starts++; setTimeout(() => { this.onstart?.(); this.onend?.(); }, 5); }
    stop() {} abort() { setTimeout(() => this.onend?.(), 1); }
  }
  const f = fakeHooks(false);
  const v = new VoiceInput(f.hooks, QuickEnd as unknown as ConstructorParameters<typeof VoiceInput>[1]);
  v.enable();
  for (let ms = 0; ms < 10000; ms += 50) t.mock.timers.tick(50);
  v.disable();
  assert.ok(starts <= 6, `${starts} starts in 10 s`);   // 300, 600, 1200, 2400, 4800 ms apart (was ~33)
  assert.ok(starts >= 3);
  assert.ok(f.states.includes("Voice input keeps stopping, retrying…"));
});

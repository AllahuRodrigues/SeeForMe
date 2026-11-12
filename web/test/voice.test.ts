// Voice commands: every shared vector in tests/data/voice_commands.json (the Python parser passes the same file), plus
// the listener's guards (the app never acts on its own voice; the wake phrase alone opens a follow-up window; a lower-ranked
// alternative never starts a command by itself; a session that ends at once backs off).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { VoiceInput, isEcho, parseCommand, spokenNumbers, type Command, type VoiceHooks } from "../src/voice";

const here = dirname(fileURLToPath(import.meta.url));
// the bundle runs from test/.build: the repository root is three levels up
const vectors = JSON.parse(readFileSync(resolve(here, "../../../tests/data/voice_commands.json"), "utf8")) as {
  intents: string[]; wake: Array<{ text: string; intent: string | null; arg: string | null }>; direct: Array<{ text: string; intent: string | null; arg: string | null }>;
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

test("every intent the parser returns is in the shared list", () => {
  for (const mode of ["wake", "direct"] as const) for (const v of vectors[mode]) {
    const got = parseCommand(v.text, mode);
    if (got) assert.ok(vectors.intents.includes(got.intent), got.intent);
  }
});

test("spoken numbers", () => {
  assert.deepEqual(spokenNumbers("room two oh four".split(" ")), ["room", "204"]);
  assert.deepEqual(spokenNumbers("gate twenty one".split(" ")), ["gate", "21"]);
  assert.deepEqual(spokenNumbers("room two hundred and five".split(" ")), ["room", "205"]);
  assert.deepEqual(spokenNumbers("oh look".split(" ")), ["oh", "look"]);   // "oh" is zero only inside a number
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

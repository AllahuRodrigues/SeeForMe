// Voice commands: one parser for spoken and typed commands, and a listener on the Web Speech API.
//
// Hands-free ("wake" mode): the wake phrase "see for me" (also "seeforme", "hey see for me" and the usual mis-hearings
// "sea for me", "c for me", "see four me", "see 4 me") is required, and the command is the text after the LAST wake
// phrase, so "... the dog barked, see for me, find a chair" is "find a chair" and a conversation nearby is ignored. The
// wake phrase alone opens a 5 s window for one command without it (with a short earcon). Push-to-talk and typed text
// ("direct" mode): the wake phrase is optional. tests/data/voice_commands.json holds the vectors this parser and the
// Python one (seeforme/live/voice.py) must both pass.
//
// The app must never hear itself: results that arrive while the app is speaking (and 0.4 s after), or that repeat what
// it said a moment ago, are ignored. The listener restarts itself (Chrome ends continuous recognition after silence or a
// network hiccup) with a growing back-off on errors and on sessions that end at once, and stops while the page is hidden.
// Of a result's alternatives the best decides; a lower one may only turn a bare wake phrase or a free question that
// was addressed to the app into a known command, never start one by itself.
export type Intent = "wake" | "start" | "pause" | "quit" | "find" | "stop_find" | "read" | "describe" | "repeat" | "mute" | "unmute"
  | "help" | "mode" | "verbosity" | "flip" | "ask";
export interface Command { intent: Intent; arg: string | null }

/** Lower case, accents and punctuation removed (apostrophes inside words kept: "what's"), whitespace collapsed. */
export function normaliseSpeech(s: string): string {
  return s.toLowerCase().replace(/[‘’`]/g, "'").normalize("NFKD").replace(/\p{M}/gu, "")
    .replace(/[^a-z0-9'\s]/g, " ").replace(/(^|\s)'+|'+(?=\s|$)/g, " ").replace(/\s+/g, " ").trim();
}

const WAKE = /\b(?:hey\s+)?(?:(?:see|sea|c)\s+(?:for|four|4)\s+me|seeforme)\b/g;
/** The text after the last wake phrase, or null when there is none. */
export function afterWake(norm: string): string | null {
  let last: RegExpExecArray | null = null;
  for (const m of norm.matchAll(WAKE)) last = m;
  return last ? norm.slice((last.index ?? 0) + last[0].length).trim() : null;
}

const UNITS: Record<string, number> = { zero: 0, oh: 0, o: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 };
const TEENS: Record<string, number> = { ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19 };
const TENS: Record<string, number> = { twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };
const isNum = (w: string) => w in UNITS || w in TEENS || w in TENS || w === "hundred";
/** Spoken numbers as digits: "room two oh four" -> "room 204", "gate twenty one" -> "gate 21", "two hundred and five" -> "205".
 *  "oh" / "o" are zero only inside a number. */
export function spokenNumbers(words: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < words.length;) {
    const w = words[i];
    if (!isNum(w) || w === "oh" || w === "o" || w === "hundred") { out.push(w); i++; continue; }
    const groups: number[] = [];
    while (i < words.length) {
      const x = words[i];
      if (x in TENS) {
        let v = TENS[x]; i++;
        if (i < words.length && words[i] in UNITS && UNITS[words[i]] > 0) { v += UNITS[words[i]]; i++; }
        groups.push(v);
      } else if (x in TEENS) { groups.push(TEENS[x]); i++; }
      else if (x === "hundred" && groups.length && groups[groups.length - 1] < 10) {
        let v = groups.pop()! * 100; i++;
        if (words[i] === "and") i++;
        if (i < words.length && words[i] in TENS) { v += TENS[words[i]]; i++; if (i < words.length && words[i] in UNITS && UNITS[words[i]] > 0) { v += UNITS[words[i]]; i++; } }
        else if (i < words.length && words[i] in TEENS) { v += TEENS[words[i]]; i++; }
        else if (i < words.length && words[i] in UNITS && UNITS[words[i]] > 0) { v += UNITS[words[i]]; i++; }
        groups.push(v);
      } else if (x in UNITS) { groups.push(UNITS[x]); i++; }
      else break;
    }
    out.push(groups.map(String).join(""));
  }
  return out;
}

const DROP = new Set(["a", "an", "the", "nearest", "closest", "please"]);
/** A FIND target: spoken numbers as digits, articles and "nearest" / "closest" removed, a trailing "for me" dropped. */
export function findTarget(text: string): string {
  let ws = spokenNumbers(text.split(" ").filter(Boolean)).filter((w) => !DROP.has(w));
  if (ws.length >= 2 && ws[ws.length - 2] === "for" && ws[ws.length - 1] === "me") ws = ws.slice(0, -2);
  return ws.join(" ").trim();
}

const FILLER = /^(?:(?:please|ok|okay|um|uh|so|and|now|hey|can you|could you|would you|will you)\s+)+/;
const RULES: Array<[RegExp, Intent, ((m: RegExpMatchArray) => string | null)?]> = [
  [/^(?:stop (?:looking|searching|finding)|cancel(?: (?:the )?(?:find|search|goal|that))?|never ?mind|forget it|clear (?:the )?(?:find|goal|search))$/, "stop_find"],
  [/^(?:stop|pause|wait|hold on|stop announcing|stop the camera|camera off|turn (?:off )?the camera(?: off)?)$/, "pause"],
  [/^(?:start|resume|continue|go|begin|start again|start announcing|start (?:the )?camera|turn (?:on )?the camera(?: on)?|camera on)$/, "start"],
  [/^(?:quit|goodbye|good bye|bye|bye bye|exit (?:the )?app|close (?:the )?app|shut down)$/, "quit"],
  [/^(?:(?:switch|change|go|set it|set)(?: back)? to |use )?(?:the )?(navigation|navigate|navigating|walking|walk|scene|reading|read|automatic|auto)(?: mode)$|^(navigate|navigation)$/, "mode",
    (m) => { const w = m[1] ?? m[2]; return /^(?:nav|walk)/.test(w) ? "navigate" : w === "scene" ? "scene" : /^read/.test(w) ? "read" : "auto"; }],
  [/^(?:read(?: (?:it|this|that|out|aloud|the text|this text|that text|text|the sign|this sign|that sign|signs?|the label|this label))?(?: (?:for me|out loud|aloud))?|what does (?:it|this|that|the sign|this sign|that sign) say|what's written(?: here| there)?|what is written(?: here| there)?)$/, "read"],
  [/^(?:repeat(?: (?:that|it|again))?|say (?:that|it) again|again|what did you(?: just)? say|pardon(?: me)?|come again|one more time)$/, "repeat"],
  [/^(?:what's around(?: me)?|what is around(?: me)?|what do you see|what can you see|describe(?: (?:the scene|it|this|what you see|my surroundings|the room))?|look around|tell me what you see|what's near me|what is near me)$/, "describe"],
  [/^(?:help|help me|what can i say|what can you do|commands|list(?: the)? commands|what are the commands|how does this work)$/, "help"],
  [/^(?:mute|mute (?:the )?(?:sound|voice|audio)|be quiet|quiet|sound off|silence|shut up|stop talking|stop speaking|turn (?:off )?the sound(?: off)?|voice off)$/, "mute"],
  [/^(?:unmute|un mute|unmute (?:the )?(?:sound|voice|audio)|sound on|speak again|talk again|turn (?:on )?the sound(?: on)?|voice on)$/, "unmute"],
  [/^(?:more details?|more detailed|be more detailed|more information|more verbose|detailed mode)$/, "verbosity", () => "detailed"],
  [/^(?:less details?|less detailed|fewer details|less information|less verbose|be brief|brief mode)$/, "verbosity", () => "quiet"],
  [/^(?:normal details?|default details?|balanced(?: details?| mode)?|normal verbosity|standard details?)$/, "verbosity", () => "balanced"],
  [/^(?:flip|flip (?:the )?camera|switch (?:the )?camera|change (?:the )?camera|turn (?:the )?camera around|other camera|front camera|back camera|rear camera|selfie camera)$/, "flip"],
  [/^(?:find me|find|look for|looking for|search for|where is|where's|wheres|where are|take me to|guide me to|lead me to|bring me to|show me|locate|navigate to|get me to|i'm looking for|im looking for|i am looking for|i need to find|help me find|where can i find)\s+(.+)$/, "find",
    (m) => findTarget(m[1])],
  [/^(?:the )?(exit|emergency exit)$/, "find", (m) => m[1]],
];

/** The command in a recognised (or typed) utterance, or null to ignore it. mode "wake": hands-free, the wake phrase is
 *  required; "direct": push-to-talk / typed, it is optional. */
export function parseCommand(text: string, mode: "wake" | "direct" = "direct"): Command | null {
  const norm = normaliseSpeech(text ?? "");
  if (!norm) return null;
  const after = afterWake(norm);
  if (after == null && mode === "wake") return null;
  const rest = after ?? norm;
  if (!rest) return after != null ? { intent: "wake", arg: null } : null;
  const c = rest.replace(FILLER, "").replace(/\s+please$/, "").trim();
  for (const [re, intent, arg] of RULES) {
    const m = c.match(re);
    if (!m) continue;
    const a = arg ? arg(m) : null;
    if (intent === "find" && !a) break;   // "find the": nothing to look for, treat it as a question
    return { intent, arg: a };
  }
  return { intent: "ask", arg: rest };
}

/** Is the heard text most likely the app's own voice (most of its words are in something the app just said)? */
export function isEcho(heard: string, recent: string[]): boolean {
  const ws = normaliseSpeech(heard).split(" ").filter((w) => w.length > 1);
  if (ws.length < 2) return false;
  return recent.some((r) => { const said = new Set(normaliseSpeech(r).split(" ")); return ws.filter((w) => said.has(w)).length >= 0.8 * ws.length; });
}

export const HELP_TEXT = "Say see for me, then a command. For example: find a door. Start. Pause. Read. What's around me. Repeat. Mute. " +
  "More detail. Flip camera. Or ask a question, like is there a chair.";

// ---- the listener -------------------------------------------------------------------------------------------------------
type Rec = {
  continuous: boolean; interimResults: boolean; lang: string; maxAlternatives: number;
  onresult: ((ev: any) => void) | null; onerror: ((ev: any) => void) | null; onend: (() => void) | null; onstart: (() => void) | null;
  start(): void; stop(): void; abort(): void;
};
export type VoiceState = "off" | "starting" | "idle" | "follow" | "talk" | "heard" | "retry" | "hidden" | "blocked" | "unsupported";
export interface VoiceHooks {
  onCommand(cmd: Command, how: "wake" | "talk", heard: string): void;
  /** what the voice input is doing (#voice-state); `important` changes are mirrored to the status region */
  onState(state: VoiceState, text: string, important: boolean): void;
  /** the app is speaking now, or stopped less than 0.4 s ago */
  appSpeaking(): boolean;
  /** what the app said in the last few seconds (echo guard) */
  recentSpeech(): string[];
  /** a short acknowledgement after a bare wake phrase */
  ack(): void;
}

export const IDLE_TEXT = 'Say "SeeForMe, find a door"';
const FOLLOW_MS = 5000;
const QUICK_END_MS = 1000;   // a hands-free session shorter than this with nothing heard did not really run

export function speechRecognitionCtor(): (new () => Rec) | null {
  const w = (typeof window !== "undefined" ? window : globalThis) as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

const BLOCKED = "Microphone blocked: allow it in the browser's site settings to use voice.";

export class VoiceInput {
  readonly supported: boolean;
  handsFree = false;
  private rec: Rec | null = null;
  private ptt: Rec | null = null;
  private hidden = false;
  private followUntil = 0;
  private backoff = 300;
  private restartTimer = 0;
  private stateTimer = 0;
  private aborted = new WeakSet<Rec>();   // recognizers we stopped ourselves: their end is not a reason to restart

  constructor(private hooks: VoiceHooks, private SR: (new () => Rec) | null = speechRecognitionCtor(), public lang = "en-US") {
    this.supported = SR != null;
  }

  /** Start hands-free listening (call it from a user gesture the first time: it opens the microphone). */
  enable() {
    if (!this.supported) { this.state("unsupported", "Voice commands need a browser with speech recognition (Chrome, Edge or Safari). Type in Ask instead.", true); return; }
    this.handsFree = true; this.backoff = 300;
    this.state("starting", "Starting the microphone…", false);
    this.startRec();
  }
  disable(state: VoiceState = "off", text = "Voice off", important = true) {
    this.handsFree = false; this.followUntil = 0;
    clearTimeout(this.restartTimer);
    this.stopRec();
    if (!this.ptt) this.state(state, text, important);
  }
  /** Push-to-talk: one command, the wake phrase optional. A second press cancels. */
  talk() {
    if (!this.supported || !this.SR) { this.state("unsupported", "Voice input isn't available in this browser. Type your question in Ask.", true); return; }
    if (this.ptt) { try { this.ptt.abort(); } catch { /* ignore */ } return; }
    this.stopRec();   // one recognition at a time: hands-free resumes when this one ends
    const r = this.make(false);
    this.ptt = r;
    r.onresult = (ev) => this.result(ev, "talk");
    r.onerror = (ev) => {
      const e = String(ev?.error ?? "");
      if (e === "not-allowed" || e === "service-not-allowed") this.state("blocked", BLOCKED, true);
      else if (e !== "aborted") this.state("heard", e === "no-speech" ? "Didn't hear anything. Press Talk and speak." : "Sorry, I didn't catch that.", true);
    };
    r.onend = () => {
      if (this.ptt === r) this.ptt = null;
      if (this.handsFree && !this.hidden) this.scheduleRestart(150);
      else if (!this.stateTimer) this.settle();
    };
    try { r.start(); this.state("talk", "Listening…", false); }
    catch { this.ptt = null; this.state("heard", "The microphone is busy. Try again.", true); }
  }
  /** The page was hidden / shown: no listening in the background. */
  setHidden(hidden: boolean) {
    this.hidden = hidden;
    if (hidden) {
      clearTimeout(this.restartTimer);
      this.stopRec();
      if (this.ptt) { try { this.ptt.abort(); } catch { /* ignore */ } }
      if (this.handsFree) this.state("hidden", "Voice paused while the page is hidden", false);
    } else if (this.handsFree) this.startRec();
  }
  get listeningForFollowUp() { return Date.now() < this.followUntil; }

  private make(continuous: boolean): Rec {
    const r = new this.SR!();
    r.continuous = continuous; r.interimResults = false; r.lang = this.lang; r.maxAlternatives = 3;
    return r;
  }
  private startRec() {
    clearTimeout(this.restartTimer);
    if (!this.handsFree || this.hidden || this.rec || this.ptt) return;
    const r = this.make(true);
    this.rec = r;
    const started = Date.now();
    let results = false, failed = false;
    r.onstart = () => { if (this.rec === r) this.settle(); };
    r.onresult = (ev) => { results = true; this.backoff = 300; this.result(ev, "wake"); };
    r.onerror = (ev) => {
      const e = String(ev?.error ?? "");
      if (e === "not-allowed" || e === "service-not-allowed") this.disable("blocked", BLOCKED, true);
      else if (e === "no-speech" || e === "aborted") { /* normal: restarted on end */ }
      else { failed = true; this.backoff = Math.min(15000, this.backoff * 2); this.state("retry", e === "network" ? "Speech service not reachable, retrying…" : `Voice input error (${e}), retrying…`, false); }
    };
    r.onend = () => {
      if (this.rec === r) this.rec = null;
      if (this.aborted.has(r)) return;
      // a session that ends at once without a result or an error (another tab holds the microphone, a platform that
      // ends sessions early) must not restart every 300 ms forever: back off as for an error
      if (!results && !failed && Date.now() - started < QUICK_END_MS) {
        this.backoff = Math.min(15000, this.backoff * 2);
        if (this.backoff >= 2400) this.state("retry", "Voice input keeps stopping, retrying…", false);
      }
      if (this.handsFree && !this.hidden && !this.ptt) this.scheduleRestart(this.backoff);
    };
    try { r.start(); }
    catch { this.rec = null; this.backoff = Math.min(15000, this.backoff * 2); this.scheduleRestart(this.backoff); }
  }
  private stopRec() {
    const r = this.rec;
    if (!r) return;
    this.aborted.add(r);
    this.rec = null;
    try { r.abort(); } catch { /* ignore */ }
  }
  private scheduleRestart(ms: number) {
    clearTimeout(this.restartTimer);
    this.restartTimer = setTimeout(() => this.startRec(), ms) as unknown as number;
  }
  private settle() {
    if (this.ptt) return;
    if (!this.handsFree) { this.state("off", "Voice off", false); return; }
    if (this.listeningForFollowUp) this.state("follow", "Listening…", false);
    else this.state("idle", IDLE_TEXT, false);
  }
  private state(s: VoiceState, text: string, important: boolean) {
    clearTimeout(this.stateTimer); this.stateTimer = 0;
    this.hooks.onState(s, text, important);
    // transient messages go back to the steady state after a moment
    if (s === "heard" || s === "retry") this.stateTimer = setTimeout(() => { this.stateTimer = 0; this.settle(); }, 4000) as unknown as number;
  }

  private result(ev: any, how: "wake" | "talk") {
    const results = ev?.results;
    if (!results) return;
    for (let i = ev.resultIndex ?? 0; i < results.length; i++) {
      const res = results[i];
      if (!res?.isFinal) continue;
      const alts: string[] = [];
      for (let k = 0; k < (res.length ?? 0); k++) if (res[k]?.transcript) alts.push(String(res[k].transcript));
      this.heard(alts, how);
    }
  }
  /** One final result (its alternatives, best first). */
  heard(alts: string[], how: "wake" | "talk") {
    if (!alts.length) return;
    // never act on the app's own voice
    if (this.hooks.appSpeaking() || alts.some((a) => isEcho(a, this.hooks.recentSpeech()))) return;
    const follow = how === "talk" || this.listeningForFollowUp;
    const mode = follow ? "direct" : "wake";
    // The best transcript decides. A lower-ranked alternative may only refine it, never start a command on its own: when
    // the best one was addressed to the app (it has the wake phrase; or Talk / the follow-up window) but is only the bare
    // wake phrase or a free question, an alternative that is a known command replaces it ("see for me fine a door" ->
    // "see for me find a door"). So "nice weather" with an alternative "c for me pause" does nothing.
    let cmd = parseCommand(alts[0], mode), text = alts[0];
    if (cmd && (cmd.intent === "wake" || cmd.intent === "ask")) {
      for (const a of alts.slice(1)) {
        const c = parseCommand(a, mode);
        if (c && c.intent !== "wake" && c.intent !== "ask") { cmd = c; text = a; break; }
      }
    }
    if (!cmd) return;
    if (cmd.intent === "wake") {
      if (!this.handsFree) { this.state("heard", 'Press Talk and say a command, like "find a door".', true); return; }
      this.followUntil = Date.now() + FOLLOW_MS;
      this.hooks.ack();
      this.state("follow", "Listening…", false);
      this.stateTimer = setTimeout(() => { this.stateTimer = 0; if (!this.listeningForFollowUp) this.settle(); }, FOLLOW_MS + 50) as unknown as number;
      return;
    }
    this.followUntil = 0;
    this.state("heard", `Heard: "${normaliseSpeech(text)}"`, false);
    this.hooks.onCommand(cmd, how, text);
  }
}

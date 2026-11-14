// Voice commands: one parser for spoken and typed commands, and a listener on the Web Speech API.
//
// The parser is a port of seeforme/live/voice.py (parse_command), rule for rule and in the same order, so a phrase means
// the same thing in the terminal demo and in the browser; tests/data/voice_commands.json holds the vectors both pass.
// Hands-free ("wake" mode): the wake phrase "see for me" (also "seeforme", "hey / ok see for me" and the usual
// mis-hearings "sea for me", "c for me", "see four me", "see 4 me") is required, and the command is the text after the
// LAST wake phrase, so "... the dog barked, see for me, find a chair" is "find a chair" and a conversation nearby is
// ignored. The wake phrase alone opens a 5 s window for one command without it (with a short earcon). Push-to-talk and
// typed text ("direct" mode): the wake phrase is optional. Polite openings ("please", "can you") are dropped, and "tell
// me", "do you know", "find out" when a question word, "the way to" or "the way out" follows; then the rules run in
// voice.py's order: stop_find; a FIND verb ("find", "where can I find", "take me to" ...) or "where X is"; a weak FIND verb
// ("get me", "how do I get to", "the way to") only with a noun phrase after it ("get me a chair") or a known target
// ("how do I get to elevator"; "get me started" stays a question); "show me ..." (a describe, read or help phrase after it
// keeps its meaning, a question stays a question, a noun phrase is a FIND target: "show me the exit"); the mode and detail
// phrases, which must be the whole command, so "find the scene mode button" is a FIND and "switch to reading mode" a mode;
// a named camera side ("front camera": flip with arg "front" or "back"); the fixed phrases (a pause that names the camera,
// "camera off", has arg "camera"); a bare "exit" or "the way out (of here)". FIND targets lose a leading "me", "the way
// to", articles and "nearest" / "closest", "the way out (of here)" is the exit, and spoken numbers become digits ("room
// two oh four" -> "room 204", "room oh four" -> "room 04"). The app's own phrases never turn into a command that changes
// anything: the start-up and idle texts parse as "help" at most.
//
// The app must never hear itself: results that arrive while the app is speaking (and 0.4 s after), or that repeat what
// it said a moment ago, are ignored. The listener restarts itself (Chrome ends continuous recognition after silence or a
// network hiccup) with a growing back-off on errors and on sessions that end at once, and stops while the page is hidden.
// Of a result's alternatives the best decides; a lower one may only turn a bare wake phrase or a free text that was
// addressed to the app into a known command (mayRefine), never start one by itself: a question only into a command that
// answers or looks for something ("is the camera on" never becomes "the camera on", a start), and no free text into a
// command that changes something by leaving words out ("tell me goodbye" is not "goodbye", nor is "goodbye please").
import { COCO } from "./yolo";

export type Intent = "wake" | "start" | "pause" | "quit" | "find" | "stop_find" | "read" | "describe" | "repeat" | "mute" | "unmute"
  | "help" | "mode" | "verbosity" | "flip" | "ask";
export interface Command { intent: Intent; arg: string | null }

// Python's \w on str (letters, numbers, "_"). JavaScript's \w and \b are ASCII-only even with the u flag, so the parser
// spells its word class and word boundaries out: "what does the café say" is a read, and "josésee for me" has no wake
// phrase, in both parsers (tests/data/voice_commands.json "unicode", tests/test_live_voice.py cross-checks a fuzz corpus).
const W = "[\\p{L}\\p{N}_]";
const WORD = new RegExp(W, "u");
const NOT_AFTER_WORD = `(?<!${W})`, NOT_BEFORE_WORD = `(?!${W})`;   // \b before / after a word character
/** NFC, lowercase, punctuation to spaces, apostrophes kept only inside words ("what's"), single spaces (voice.py
 *  normalize). */
export function normaliseSpeech(s: string): string {
  const x = (s ?? "").normalize("NFC").toLowerCase().replace(/[’‘]/g, "'").replace(/[^\p{L}\p{N}_\s']/gu, " ").replace(/_/g, " ");
  const kept = x.replace(/'/g, (_m, i: number) => (WORD.test(x[i - 1] ?? "") && WORD.test(x[i + 1] ?? "") ? "'" : " "));
  return kept.split(/\s+/).filter(Boolean).join(" ");
}

// "see for me" and what small recognisers make of it; "hey" / "ok" in front belong to the wake phrase
const WAKE = new RegExp(`${NOT_AFTER_WORD}(?:(?:hey|ok|okay)\\s+)?(?:see|sea|c)\\s*(?:for|four|4)\\s*me${NOT_BEFORE_WORD}`, "gu");
/** The text after the last wake phrase, or null when there is none. */
export function afterWake(norm: string): string | null {
  let last: RegExpExecArray | null = null;
  for (const m of norm.matchAll(WAKE)) last = m;
  return last ? norm.slice((last.index ?? 0) + last[0].length).trim() : null;
}

// ---- spoken numbers ---------------------------------------------------------------------------------------------------
const UNITS = new Map(Object.entries({ zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 }));
const TEENS = new Map(Object.entries({ ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16,
  seventeen: 17, eighteen: 18, nineteen: 19 }));
const TENS = new Map(Object.entries({ twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 }));
const ZERO = new Set(["oh", "o"]);   // zero only inside a number: "two oh four"
// how a recogniser spells digits it hears as words ("room two oh four" -> "room to oh for"); read as digits only in a run
// with an inner "oh", where they cannot be the preposition ("a table for two" stays "table for 2")
const HOMOPHONES = new Map(Object.entries({ to: "two", too: "two", for: "four", fore: "four", won: "one", ate: "eight" }));
const isDigits = (w: string) => /^\p{Nd}+$/u.test(w);   // decimal digits of any script (voice.py: str.isdecimal)
const isNum = (w: string) => UNITS.has(w) || TEENS.has(w) || TENS.has(w) || w === "hundred" || isDigits(w);

/** Digits for a run of number words: digit by digit ("two oh four" -> 204), tens with a unit as one number ("twenty
 *  one" -> 21), teens ("twelve" -> 12), "X hundred [and] Y". */
function numberRun(ws: string[]): string {
  const out: string[] = [];
  const small = (j: number): [number | null, number] => {   // a value below 100 starting at j, and the next index
    const w = ws[j] ?? "";
    const u = (k: number) => UNITS.get(ws[k] ?? "") ?? 0;
    if (TENS.has(w)) return j + 1 < ws.length && u(j + 1) > 0 ? [TENS.get(w)! + u(j + 1), j + 2] : [TENS.get(w)!, j + 1];
    if (TEENS.has(w)) return [TEENS.get(w)!, j + 1];
    if (UNITS.has(w)) return [UNITS.get(w)!, j + 1];
    return [null, j];
  };
  let i = 0;
  while (i < ws.length) {
    const w = ws[i];
    if (w === "hundred" || (UNITS.has(w) && ws[i + 1] === "hundred")) {
      const base = 100 * (UNITS.get(w) ?? 1);
      i += w === "hundred" ? 1 : 2;
      if (ws[i] === "and") i++;
      const [rest, next] = small(i);
      out.push(String(base + (rest ?? 0))); i = next;
      continue;
    }
    if (ZERO.has(w)) { out.push("0"); i++; continue; }
    if (isDigits(w)) { out.push(w); i++; continue; }
    const [v, next] = small(i);
    if (v == null) { out.push(w); i++; continue; }   // not a number word (cannot happen for a run; never loop forever)
    out.push(String(v)); i = next;
  }
  return out.join("");
}

/** Spoken numbers as digits: "room two oh four" -> "room 204", "room to oh for" -> "room 204", "room oh four" ->
 *  "room 04", "gate twenty one" -> "gate 21", "two hundred and five" -> "205"; "oh look" stays. */
export function wordsToDigits(text: string): string {
  const ws = text.split(/\s+/).filter(Boolean), out: string[] = [];
  let i = 0;
  while (i < ws.length) {
    let j = i;
    while (j < ws.length && (isNum(ws[j]) || ZERO.has(ws[j]) || HOMOPHONES.has(ws[j]))) j++;
    const run = ws.slice(i, j);
    if (run.length >= 2 && run.slice(1).some((w) => ZERO.has(w)) && !ZERO.has(run[0])) {
      out.push(numberRun(run.map((w) => HOMOPHONES.get(w) ?? w))); i = j;
      continue;
    }
    if (ZERO.has(ws[i]) && i + 1 < ws.length && isNum(ws[i + 1])) {   // a leading "oh" before a number word
      j = i + 1;
      while (j < ws.length && (isNum(ws[j]) || ZERO.has(ws[j]))) j++;
      out.push(numberRun(ws.slice(i, j))); i = j;
      continue;
    }
    if (!isNum(ws[i])) { out.push(ws[i]); i++; continue; }
    j = i;
    while (j < ws.length && (isNum(ws[j]) || (ZERO.has(ws[j]) && j > i)
      || (ws[j] === "and" && j > i && ws[j - 1] === "hundred" && j + 1 < ws.length && isNum(ws[j + 1])))) j++;
    out.push(numberRun(ws.slice(i, j))); i = j;
  }
  return out.join(" ");
}
/** wordsToDigits on a word list. */
export const spokenNumbers = (words: string[]): string[] => wordsToDigits(words.join(" ")).split(" ").filter(Boolean);

const ARG_DROP = new Set(["a", "an", "the", "nearest", "closest", "please"]);
/** A FIND target: no articles, "nearest" / "closest" or "please", no trailing "for me", spoken numbers as digits. */
export function findTarget(text: string): string {
  const s = normaliseSpeech(text).split(" ").filter((w) => w && !ARG_DROP.has(w)).join(" ");
  return wordsToDigits(s.replace(/(?:^|\s+)for me$/, "").trim());
}

// ---- commands (voice.py: _STOP_FIND, _FIND, _FIND_WHERE, _FIND_WEAK, _SHOW, _MODE, _VERBOSITY, _CAMERA_SIDE, _EXACT, in order)
// polite openings, dropped before the rules; the indirect ones ("tell me", "do you know", "find out", and a "please" after
// them) only before a question word, "the way to" or "the way out": "can you tell me what this says" is "what this says"
// and "tell me please where the exit is" a FIND, but "tell me goodbye" and "tell me stop" stay as said (a question, not quit
// or pause)
const FILLER = new RegExp("^(?:(?:please|ok|okay|um|uh|so|and|now|hey|can you|could you|would you|will you) )*"
  + "(?:(?:tell me|do you know|find out) (?:please )?(?=(?:what|what's|whats|where|where's|wheres|how|which|who|who's|"
  + "whether|if|when|why|the way to|the way out)(?: |$)))?", "u");
const TRAILING = /(?:\s+(?:please|now|thanks|thank you))+$/u;
const FIND_PREFIX = "(?:help me |i want to |i need to |i'd like to |i would like to )";
// FIND verbs: whatever follows is the target ("find the exit", "where is room 204"); "find out", "where are we / you / they
// ..." are not ("where are we" is a describe, "where are we going" a question)
const FIND = new RegExp(`^(?:${FIND_PREFIX}|start |keep )?`
  + "(?:find(?! out(?: |$))|look for|looking for|search for|locate|where is|where's|wheres|"
  + "where are(?! (?:we|you|they)(?: |$))|take me to|guide me to|lead me to|bring me to|navigate to|"
  + "i'm looking for|i am looking for|where (?:can|could|do|would|should|will|might) (?:i|we) find|"
  + "where (?:i|we) (?:can|could) find)(?:\\s+(.*))?$", "u");
// FIND verbs that also begin ordinary sentences ("get me started", "how do I get to sleep", "the way to go"): a FIND only when
// a noun phrase follows (nounTarget). "get me to" before "get me" ("get me to the exit" is the exit); a bare "get to" only
// after a prefix ("help me get to the exit"; "get to work" is not a FIND)
const FIND_WEAK = new RegExp(`^(?:${FIND_PREFIX}?(?:get me to|get me|how (?:do|can|could|would|should) (?:i|we) `
  + `(?:get to|find|reach)|how to (?:get to|find|reach)|the way to)|${FIND_PREFIX}get to) (.+)$`, "u");
// "where the exit is" (after "show me", "tell me", "do you know"): a FIND too, but not "where it is" or "where you are"
const FIND_WHERE = /^where (?!(?:it|this|that|these|those|they|you|we|he|she|there|here) (?:is|are)$)(.+) (?:is|are)$/u;
// what makes the words after a weak verb a noun phrase (nounTarget)
const DETERMINER = /^(?:a|an|the|my|our|this|that|these|those|some|any|nearest|closest) /u;
// targets that need no determiner after a weak verb: recognisers drop a short "the" ("how do I get to elevator"). The
// detector's labels and the places and things blind travellers ask for, the same set as voice.py known_targets()
// (commands.COCO, commands.NAV_WORDS, GRAMMAR_TARGETS; tests/test_live_voice.py compares them). Only the whole target
// counts: "how do I get to know the person" stays a question.
const NAV_WORDS = ["stairs", "staircase", "steps", "door", "doors", "exit", "entrance", "elevator", "lift", "escalator",
  "restroom", "bathroom", "curb", "pole", "crosswalk", "crossing"];
const GRAMMAR_TARGETS = ["door", "exit", "entrance", "stairs", "steps", "staircase", "elevator", "lift", "escalator", "bathroom",
  "restroom", "toilet", "room", "gate", "counter", "cashier", "checkout", "desk", "seat", "table", "bench", "chair", "sign",
  "bus stop", "crosswalk", "crossing", "platform", "train", "bus", "taxi", "car", "person", "dog", "cup", "bottle", "phone",
  "keys", "bag", "backpack", "laptop", "window", "wall", "pole", "curb", "path", "hallway", "corridor", "lobby", "office",
  "kitchen", "store", "shop"];
export const KNOWN_TARGETS: ReadonlySet<string> = new Set([...COCO, ...NAV_WORDS, ...GRAMMAR_TARGETS]);
const CLAUSE_VERB = /(?:^| )(?:is|are|was|were|isn't|aren't|wasn't|weren't)(?: |$)/u;
// "show me ...": a describe or read phrase after it keeps its meaning, a question stays a question, anything else is a FIND
// target ("show me the exit"); see show()
const SHOW = /^show me(?: (.+))?$/u;
const SHOW_DESCRIBE = /^(?:around(?: (?:here|me|us))?|(?:my |the )?surroundings|everything|the (?:scene|room|area|view))$/u;
const SUBJ = `(?:it|this|that|(?:this|that|the) ${W}+(?: ${W}+)?)`;   // what a sign question is about: "it", "the sign", "this menu"
const SHOW_READ = new RegExp(`^(?:the |this |that |any )?(?:text|writing|words)(?: (?:here|there|on ${SUBJ}))?$`, "u");
const QUESTION_WORD = new RegExp("^(?:what|what's|whats|where|where's|wheres|how|how's|which|who|who's|why|when|whether|if|is|"
  + "are|was|were|can|could|do|does|did|will|would|should)(?: |$)", "u");
const STOP_FIND = new RegExp(`^(?:(?:stop|quit) (?:looking|searching|finding)${NOT_BEFORE_WORD}.*|cancel${NOT_BEFORE_WORD}.*|`
  + `never ?mind${NOT_BEFORE_WORD}.*|forget it${NOT_BEFORE_WORD}.*|clear (?:the )?(?:find|goal|search)|stop find)$`, "u");
// mode and detail phrases are the whole command (a FIND target may contain them: "find the scene mode button")
const MODE = new RegExp("^(?:(?:switch|change|go|turn|set|put|use|enter|back)(?: it| me| back)?(?: (?:to|into|on|in))? )?(?:the )?"
  + "(navigation|navigate|navigating|walking|walk|scene|read|reading|text|auto|automatic) mode(?: on)?$", "u");
const MODE_ARG: Record<string, string> = { navigation: "navigate", navigate: "navigate", navigating: "navigate", walking: "navigate",
  walk: "navigate", scene: "scene", read: "read", reading: "read", text: "read", auto: "auto", automatic: "auto" };
const VERB_PREFIX = "^(?:(?:give me|i want|i'd like|i would like|use|switch to|go to|set|turn on|with) )?";
const VERBOSITY: Array<[string, RegExp]> = ([
  ["detailed", "more (?:detail|details|detailed|information|info)|more verbose|be more detailed|detailed"],
  ["quiet", "less (?:detail|details|detailed|information|info|talking)|fewer details|talk less|be brief|briefer"],
  ["balanced", "(?:normal|default|medium|standard|usual) (?:detail|details|verbosity)|balanced"],
] as const).map(([level, pat]): [string, RegExp] => [level, new RegExp(`${VERB_PREFIX}(?:${pat})(?: (?:please|now|mode))?$`, "u")]);
// a named camera side: flip with arg "front" or "back" ("flip camera", "other camera": flip without one)
const SIDE = "(front|back|rear|selfie)(?: facing)?";
const CAMERA_SIDE = new RegExp(
  `^(?:(?:(?:use|switch to|change to|go to|flip to|swap to|turn to|turn on|open|select|try)(?: the)? |the )?${SIDE} (?:camera|cam)`
  + `|(?:flip|switch|change|swap|turn|toggle) (?:the )?(?:camera|cam) (?:to|around to) (?:the )?${SIDE}(?: (?:one|camera|cam|side))?`
  + `|(?:flip|switch|swap) to (?:the )?${SIDE}(?: one| side)?)$`, "u");
const SIDE_ARG: Record<string, string> = { front: "front", selfie: "front", back: "back", rear: "back" };
const EXACT: Array<[Intent, string | null, RegExp]> = ([   // first match wins
  ["pause", null, "(?:stop|pause|paws|wait|hold on|hang on)(?: (?:talking|announcing|announcements|it|that|for now|a (?:moment|second|sec|minute)))*"],
  // a pause that names the camera has arg "camera": the terminal demo keeps the camera on while paused and says so
  ["pause", "camera", "(?:stop|pause|turn off|switch off|shut off|close)(?: the)? (?:camera|cam)|(?:turn|switch|shut) (?:the )?(?:camera|cam) off|"
    + "(?:the )?(?:camera|cam) off"],
  ["start", null, "(?:start|resume|continue|unpause|restart|keep going|go on|carry on)(?: (?:the|camera|announcing|announcements|talking|again|it|up|listening))*|"
    + "(?:turn on|switch on|open)(?: the)? (?:camera|cam)|(?:turn|switch) (?:the )?(?:camera|cam) on|(?:the )?(?:camera|cam) on"],
  ["quit", null, "(?:quit|goodbye|good bye|bye|bye bye|shut down|shutdown|(?:exit|quit|close) (?:the )?(?:app|program|demo))"],
  ["read", null, `(?:read(?: .+)?|what (?:does|do) ${SUBJ} say|what ${SUBJ} says|(?:what's|what is|whats) written(?: (?:here|there|on ${SUBJ}))?)`],
  ["describe", null, "(?:(?:what's|what is|whats) (?:around(?: me| here| us)?|here|near me|nearby)|what (?:do|can) you see(?: now| here)?|"
    + "what you (?:can )?see|describe(?: .+)?|look around|where am i|where are we|where i am|where we are)"],
  ["repeat", null, "(?:repeat(?: (?:that|it|again))*|say (?:that|it) again|say again|again|come again|pardon(?: me)?|what did you say|"
    + "what was that|one more time|tell me (?:please )?(?:that |it )?again)"],
  ["unmute", null, "(?:unmute|un mute|sound on|turn (?:the )?sound on|turn on (?:the )?sound|voice on|unmute (?:sound|voice|audio))"],
  ["mute", null, "(?:mute|be quiet|quiet|silence|hush|shut up|sound off|turn (?:the )?sound off|turn off (?:the )?sound|voice off|mute (?:sound|voice|audio))"],
  ["help", null, "(?:help(?: me| please| (?:for|with) (?:the )?commands)?|what can i say|what can you do|(?:the )?commands|list (?:the )?commands|what are the commands)"],
  ["flip", null, "(?:flip|(?:flip|switch|change|swap|toggle) (?:the )?(?:camera|cameras|cam)|(?:next|other) camera|use the other camera)"],
] as const).map(([intent, arg, pat]): [Intent, string | null, RegExp] => [intent, arg, new RegExp(`^(?:${pat})$`, "u")]);
const BARE_EXIT = /^(?:exit|exits|(?:the |a )?way out(?: of here)?)$/u;
const SAME_GOAL = new Map([["way out", "exit"], ["way out of here", "exit"], ["out of here", "exit"]]);

/** A FIND target from the words after the verb: findTarget without a leading "me" ("find me a chair") or "the way to"
 *  ("show me the way to the exit"); "the way out" and "the way out of here" are the exit. null when nothing is left ("find
 *  me"). */
function verbTarget(raw: string): string | null {
  const t = findTarget(raw.replace(/^me(?: |$)/u, "").replace(/^(?:the |a )?way to /u, ""));
  return (SAME_GOAL.get(t) ?? t) || null;
}
/** The FIND target after a weak verb (FIND_WEAK) or "show me" when the words are a noun phrase: a determiner first ("the
 *  exit", "a chair", "my keys", "some water"), a name with a number ("room 204", "gate twenty one"), a known target
 *  without a determiner ("elevator", "bus stop": KNOWN_TARGETS) or "out of here" / "the way out (of here)". null for the
 *  rest of an ordinary sentence: "get me started", "how do I get to sleep", "the way to go", "the way to the exit is
 *  blocked" (a clause), "show me the way" (voice.py _noun_target). */
function nounTarget(raw: string): string | null {
  const s = raw.replace(/^me(?: |$)/u, "").replace(/^(?:the |a )?way to /u, "");
  if (CLAUSE_VERB.test(s)) return null;
  const t = findTarget(s);
  const same = SAME_GOAL.get(t);
  if (same) return same;
  if (!t || t === "way") return null;
  if (DETERMINER.test(s) || (t.includes(" ") && /\p{Nd}/u.test(t)) || KNOWN_TARGETS.has(t)) return t;
  return null;
}
const strip = (text: string) => text.replace(FILLER, "").replace(TRAILING, "").trim() || text;
/** Is this body a question (a question word first, after the polite and indirect openings: "can you tell me is the camera
 *  on")? A lower reading may turn a question only into a command that answers or looks for something (mayRefine): "is the
 *  camera on" must not become "the camera on", a start. */
function isQuestion(body: string): boolean {
  const s = strip(body);
  return QUESTION_WORD.test(s) || QUESTION_WORD.test(s.replace(/^(?:tell me|do you know|find out) (?:please )?/u, ""));
}

/** "show me <rest>": describe ("show me around", "show me what you see"), read ("show me the text", "show me what it says")
 *  or help when <rest> reads as one; a FIND when <rest> is one ("show me where the exit is"); a question stays a question
 *  ("show me what's on my left"); a noun phrase is a FIND target ("show me the exit", "show me a chair", "show me to my
 *  seat"); anything else stays as said ("show me something", "show me the way"). */
function show(rest: string, body: string): Command {
  if (!rest) return { intent: "ask", arg: body };
  if (SHOW_DESCRIBE.test(rest)) return { intent: "describe", arg: null };
  if (SHOW_READ.test(rest)) return { intent: "read", arg: null };
  const cmd = coreCommand(strip(rest), rest, false);
  if (cmd.intent === "find" || cmd.intent === "describe" || cmd.intent === "read" || cmd.intent === "help") return cmd;
  if (cmd.intent === "ask" && !QUESTION_WORD.test(rest)) {
    const t = nounTarget(rest.replace(/^to /u, ""));
    if (t) return { intent: "find", arg: t };
  }
  return { intent: "ask", arg: body };
}

/** Rules in order: stop_find; a FIND verb (it may name mode or detail words: "find the scene mode button") or "where X is";
 *  a weak FIND verb with a noun phrase ("get me a chair", "how do I get to the exit"); "show me ..." (show()); the
 *  whole-command mode and detail phrases; a named camera side; the fixed phrases; a bare "exit"; else a question (arg = the
 *  body as heard). */
function coreCommand(core: string, body: string, withShow = true): Command {
  if (STOP_FIND.test(core)) return { intent: "stop_find", arg: null };
  const f = core.match(FIND) ?? core.match(FIND_WHERE);
  if (f) return { intent: "find", arg: verbTarget(f[1] ?? "") };   // "find" alone: find with nothing named
  const w = core.match(FIND_WEAK);
  const wt = w ? nounTarget(w[1]) : null;
  if (wt) return { intent: "find", arg: wt };
  const s = withShow ? core.match(SHOW) : null;
  if (s) return show(s[1] ?? "", body);
  const m = core.match(MODE);
  if (m) return { intent: "mode", arg: MODE_ARG[m[1]] };
  for (const [level, rx] of VERBOSITY) if (rx.test(core)) return { intent: "verbosity", arg: level };
  const c = core.match(CAMERA_SIDE);
  if (c) return { intent: "flip", arg: SIDE_ARG[c.slice(1).find((g) => g)!] };
  for (const [intent, arg, rx] of EXACT) if (rx.test(core)) return { intent, arg };
  if (BARE_EXIT.test(core)) return { intent: "find", arg: "exit" };
  return { intent: "ask", arg: body };
}

/** The command in the words after the wake phrase (normalised); null only for empty text. */
function bodyCommand(body: string): Command | null {
  if (!body) return null;
  return coreCommand(strip(body), body);
}

/** The command in a recognised (or typed) utterance, or null to ignore it. mode "wake": hands-free, the wake phrase is
 *  required; "direct": push-to-talk / typed, it is optional. */
export function parseCommand(text: string, mode: "wake" | "direct" = "direct"): Command | null {
  const s = normaliseSpeech(text ?? "");
  if (!s) return null;
  const body = afterWake(s);
  if (body != null) return body ? bodyCommand(body) : { intent: "wake", arg: null };
  return mode === "wake" ? null : bodyCommand(s);
}

// ---- lower readings of the same speech (VoiceInput.heard; in the terminal, voice.py snap_text) ----------------------
// the commands that only answer or look for something (a FIND goal is dropped with "stop looking"): the only ones a lower
// reading may turn a question into
const ANSWERS: ReadonlySet<Intent> = new Set<Intent>(["find", "describe", "read", "help", "repeat"]);
/** Are the words of `short` all in `long`, in the same order, with some left out? */
function dropsWords(short: string[], long: string[]): boolean {
  if (short.length >= long.length) return false;
  let i = 0;
  for (const w of long) if (i < short.length && w === short[i]) i++;
  return i === short.length;
}
/** The words the rules see (coreCommand): those after the last wake phrase, else all of them (normalised), without the
 *  polite openings and trailing words that strip() drops ("so", "please", "now" ...). mayRefine compares these, so a
 *  filler that a lower reading adds ("goodbye please" for "tell me goodbye") does not hide the words it left out (voice.py
 *  _core_words). */
function coreWords(text: string): string[] {
  const n = normaliseSpeech(text);
  const body = (afterWake(n) ?? n).trim();
  return body ? strip(body).split(" ").filter(Boolean) : [];
}
/** May `alt`, a lower reading of the same speech, replace `best`, which reads as the bare wake phrase or free text ("ask")?
 *  The best reading decides; a lower one may only correct a word it misheard, and only into a command:
 *  - after a bare wake phrase, any command (the best reading heard nothing after it);
 *  - after a question ("what's a round me", "is the camera on"), only a command that answers or looks for something
 *    (ANSWERS: "what's around me"), never one that starts, stops, switches or silences anything ("the camera on");
 *  - after other free text, any command, except one that changes something and is the best reading with words left out
 *    ("tell me goodbye" is not "goodbye", "tell me stop" is not "stop"): the words it lacks made the best reading free text.
 *    Words left out are counted on the words the rules see (coreWords), so "goodbye please" or "so goodbye" for "tell me
 *    goodbye", and "and stop" for "don't stop", leave words out too.
 *  voice.py may_refine is the same rule; tests/data/voice_commands.json "refine" holds the cases both pass. */
export function mayRefine(best: string, alt: string, mode: "wake" | "direct" = "direct"): boolean {
  const b = parseCommand(best, mode), a = parseCommand(alt, mode);
  if (!b || !a || a.intent === "wake" || a.intent === "ask") return false;
  if (b.intent === "wake") return true;
  if (b.intent !== "ask") return false;
  if (ANSWERS.has(a.intent)) return true;
  return !isQuestion(b.arg ?? "") && !dropsWords(coreWords(alt), coreWords(best));
}

/** Is the heard text most likely the app's own voice (most of its words are in something the app just said)? */
export function isEcho(heard: string, recent: string[]): boolean {
  const ws = normaliseSpeech(heard).split(" ").filter((w) => w.length > 1);
  if (ws.length < 2) return false;
  return recent.some((r) => { const said = new Set(normaliseSpeech(r).split(" ")); return ws.filter((w) => said.has(w)).length >= 0.8 * ws.length; });
}

export const HELP_TEXT = "Say see for me, then a command. For example: find a door. Start. Pause. Read. What's around me. Repeat. Mute. " +
  "More detail. Flip camera. Or ask a question, like is there a chair.";

// ---- camera sides (what a "flip" command with arg "front" / "back" does in the browser; web/src/camcontrol.ts) ----------
export type CameraSide = "front" | "back";
/** The side a getUserMedia facingMode names ("user": front, "environment": back); null when the camera does not say
 *  (most laptop and USB webcams), so no reply claims a side nobody reported. */
export const sideOfFacing = (facingMode: string | null | undefined): CameraSide | null =>
  facingMode === "user" ? "front" : facingMode === "environment" ? "back" : null;
export const facingOfSide = (side: CameraSide): "user" | "environment" => (side === "front" ? "user" : "environment");
/** A flip command: `side` the side it names (null: the next camera), `running` whether the camera runs, `current` the side
 *  the running camera reports, `asked` the side already asked for on this very camera (null when none was, or the camera
 *  changed since), `pending` the open still in progress (camcontrol.ts): a start (the camera was off) or a switch, null
 *  when none. act "next": the Flip button; "switch": open the camera of that side, then say switchedReply(); "start": the
 *  camera is off, start it on that side, then say switchedReply(.., true); "none": only `say`. While an open is pending
 *  nothing more opens, and the reply says which open it is. Asking again for a side this camera already answered does not
 *  reopen it (on a laptop with one camera every "back camera" stopped and reopened it and cleared what the app had seen). */
export function flipPlan(side: string | null, running: boolean, current: CameraSide | null, asked: CameraSide | null = null,
  pending: "start" | "switch" | null = null): { act: "none" | "next" | "switch" | "start"; say: string } {
  const s: CameraSide | null = side === "front" || side === "back" ? side : null;
  if (pending) return { act: "none", say: pending === "start" ? "Still starting the camera." : "Still switching the camera." };
  if (!running) return s ? { act: "start", say: "" } : { act: "none", say: "The camera is off." };
  if (!s) return { act: "next", say: "Switching camera." };
  if (current === s) return { act: "none", say: `Already on the ${s} camera.` };
  if (asked === s) return { act: "none", say: switchedReply(s, current) };
  return { act: "switch", say: "" };
}
/** What to say once the camera of side `want` was asked for and `got` is the side the new camera reports (`started`: the
 *  camera was off and has just started). Never a side the browser did not report. */
export function switchedReply(want: CameraSide, got: CameraSide | null, started = false): string {
  if (got === want) return started ? `This is the ${want} camera.` : `Switched to the ${want} camera.`;
  if (got) return `I couldn't find a ${want} camera. This is the ${got} camera.`;
  return `I asked for the ${want} camera, but this camera does not say which way it faces.`;
}

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

// a screen reader may read #voice-state (Hands-free's description) into the open microphone: heard back, it is "help" at
// most, never a FIND or a mode
export const IDLE_TEXT = 'Say "SeeForMe, help" for commands';
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
    // wake phrase or free text, the first alternative that mayRefine() allows replaces it ("see for me fine a door" ->
    // "see for me find a door", "what's a round me" -> "what's around me"). So "nice weather" with an alternative "c for me
    // pause" does nothing, a question never becomes a command that changes anything ("is the camera on" is not "the camera
    // on", a start), and "tell me goodbye" is not "goodbye".
    let cmd = parseCommand(alts[0], mode), text = alts[0];
    if (cmd && (cmd.intent === "wake" || cmd.intent === "ask")) {
      const better = alts.slice(1).find((a) => mayRefine(alts[0], a, mode));
      if (better) { cmd = parseCommand(better, mode); text = better; }
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

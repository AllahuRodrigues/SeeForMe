// The recorded voice pack. Cues are slot-based sentences ("Car, about 3 meters, slightly left."), so a natural voice can
// say all of them from a finite set of recorded fragments joined at the commas, where a speaker pauses anyway. Each
// fragment is recorded twice: with a continuing end (before a comma, key "m:") and a falling end (before a full stop,
// key "f:"). Fixed app sentences are recorded whole (key "s:"). Text the pack does not contain (sign text, answers)
// is never patched together: planClips returns null and the caller uses another voice for the whole sentence.
//
// This module is pure. voice/build.mjs imports it to decide what to record; clipvoice.ts plays the result.
import { COCO } from "./yolo";
import { actionPhrase, directionPhrase, distancePhrase, motionPhrase, objectPhrase } from "./compose";

/** index.json next to the sprite: clip key -> [start seconds, length seconds] inside the sprite file. */
export interface VoiceIndex { voice: string; sprite: string; clips: Record<string, [number, number]> }
export interface PlannedClip { key: string; gapAfterS: number }

/** Pause after a fragment that ends in a comma, and between two sentences (the clips themselves are trimmed). */
export const COMMA_GAP_S = 0.07, SENTENCE_GAP_S = 0.22;

/** Sentences the app says as they are (main.ts, camcontrol.ts, voice.ts). Dynamic replies are not here on purpose. */
export const FIXED_PHRASES: string[] = [
  "Camera on. I'll tell you what matters.",
  "Paused. The camera is off.",
  "Already on.",
  "Still starting the camera.",
  "The camera is already off.",
  "I couldn't start the camera. Press Start.",
  "I couldn't switch the camera. The camera is off.",
  "Switched to the front camera.",
  "Switched to the back camera.",
  "This is the front camera.",
  "This is the back camera.",
  "Object detection is ready.",
  "Object detection is not working, obstacle cues are off. Reload the page to retry.",
  "Find cleared.",
  "Stopped looking.",
  "Tell me what to look for, for example a chair or the exit.",
  "Reading.",
  "I can't read any text here.",
  "The camera is off. Start it first.",
  "Sorry, reading failed. Check the connection to the SeeForMe server and try again.",
  "Goodbye. Hands-free voice is off.",
  "Sound off.",
  "Sound on.",
  "Read mode.",
  "Navigate mode.",
  "Navigate mode. Scene descriptions follow automatically when you stand still.",
  "Automatic: Navigate mode.",
  "More detail.",
  "Less detail.",
  "Normal detail.",
  "Okay.",
  "Voice input isn't available in this browser. Type your question in Ask.",
  "Hi, this is SeeForMe. Stairs down, about 3 meters. Handrail left.",
  "Say see for me, then a command. For example: find a door. Start. Pause. Read. What's around me. Repeat. Mute. More detail. Flip camera. Or ask a question, like is there a chair.",
];

/** Labels a live track can carry: the detector's classes, the cane layer's obstacle, and the composer's special cases. */
const LABELS = [...COCO, "obstacle", "unknown object", "curb"];

function fragments(): string[] {
  const out = new Set<string>();
  for (const label of LABELS) for (const hedged of [false, true]) out.add(objectPhrase(label, "object", null, hedged));
  out.add(objectPhrase("", "text", "", false));                       // "a sign" (a sign whose text could not be read)
  for (const units of ["m", "ft"] as const) for (const conf of [0.3, 0.9])
    for (let d = 0; d <= 40.5; d += 0.01) { const p = distancePhrase(d, conf, units); if (p) out.add(p); }
  for (let a = -90; a <= 90; a += 1) out.add(directionPhrase(a));
  for (const rel of [0, -2]) for (const ls of [0, 0.6, -0.6]) for (const lat of [-1, 1]) { const p = motionPhrase(rel, ls, lat); if (p) out.add(p); }
  for (const label of LABELS) { const p = actionPhrase(label); if (p) out.add(p); }
  // cane and guide-dog layer (cane.ts caneMessage)
  for (const what of ["something at head height", "low obstacle", "obstacle"]) for (const dir of ["ahead", "on your left", "on your right"]) out.add(`${what} ${dir}`);
  out.add("and on your left"); out.add("and on your right"); out.add("stop");
  return [...out].map((f) => f.toLowerCase());
}

let cached: string[] | null = null;
/** Every clip key of a complete pack. */
export function packKeys(): string[] {
  cached ??= [...fragments().flatMap((f) => [`m:${f}`, `f:${f}`]), ...FIXED_PHRASES.map((p) => `s:${p}`)];
  return cached;
}

/** The text to record for a clip: the fragment with the punctuation its position needs. */
export function clipText(key: string): string {
  const body = key.slice(2);
  if (key.startsWith("s:")) return body;
  return body[0].toUpperCase() + body.slice(1) + (key.startsWith("m:") ? "," : ".");
}

/** The clips that say `message`, or null when any part of it is not recorded. */
export function planClips(message: string, has: (key: string) => boolean): PlannedClip[] | null {
  const msg = message.trim();
  if (!msg) return null;
  if (has(`s:${msg}`)) return [{ key: `s:${msg}`, gapAfterS: 0 }];
  const sentences = msg.split(/(?<=\.)\s+/);
  const plan: PlannedClip[] = [];
  for (const [si, sentence] of sentences.entries()) {
    if (!sentence.endsWith(".")) return null;
    const parts = sentence.slice(0, -1).split(", ");
    for (const [pi, part] of parts.entries()) {
      const last = pi === parts.length - 1;
      const key = `${last ? "f" : "m"}:${part.toLowerCase()}`;
      if (!part || !has(key)) return null;
      plan.push({ key, gapAfterS: !last ? COMMA_GAP_S : si < sentences.length - 1 ? SENTENCE_GAP_S : 0 });
    }
  }
  return plan;
}

/** How long a plan takes to play. */
export function planSeconds(plan: readonly PlannedClip[], index: VoiceIndex): number {
  return plan.reduce((s, c) => s + index.clips[c.key][1] + c.gapAfterS, 0);
}

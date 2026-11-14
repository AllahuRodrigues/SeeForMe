// The device voice says what the recorded pack (voicepack.ts) does not hold: answers, sign text, and anything before
// the pack has loaded. A browser lists every voice the system has, and the first English one is usually a compact,
// robotic voice, so the choice is made here: the most natural English female voice that is installed. Downloaded
// "Premium" and "Enhanced" system voices and Edge's online "Natural" voices come first; novelty voices never.
export interface VoiceLike { name: string; lang: string }
/** How a voice sounds, by what its name declares: "natural" (premium or neural), "enhanced", or "standard". */
export type VoiceGrade = "natural" | "enhanced" | "standard";

const FEMALE = /\b(zoe|ava|allison|samantha|susan|joelle|noelle|nicky|aria|jenny|michelle|emma|ana|sonia|libby|karen|moira|tessa|serena|kate|martha|fiona|veena|zira|hazel|catherine|female)\b|^google us english$/i;
const MALE = /\b(aaron|evan|nathan|tom|alex|daniel|oliver|arthur|gordon|rishi|david|mark|guy|davis|tony|jason|brandon|christopher|eric|roger|steffan|andrew|brian|ryan|james|george|male)\b/i;
/** Joke voices, the 1980s voices macOS still ships, and the synthesiser-style set listed as "Name (English (US))". */
const NOVELTY = /^(albert|bad news|bahh|bells|boing|bubbles|cellos|good news|jester|organ|superstar|trinoids|whisper|wobble|zarvox|fred|junior|kathy|ralph|princess)$|^(eddy|flo|grandma|grandpa|reed|rocko|sandy|shelley) \(/i;
/** Between voices of one grade: the younger, conversational ones first. */
const ORDER = ["nicky", "karen", "susan", "samantha", "allison", "jenny", "ava", "aria", "zoe"];

export function voiceQuality(v: VoiceLike): VoiceGrade {
  if (/\(premium\)|natural|neural/i.test(v.name)) return "natural";
  if (/\(enhanced\)|^google (us|uk) english/i.test(v.name)) return "enhanced";
  return "standard";
}

function score(v: VoiceLike): number {
  if (!/^en([-_]|$)/i.test(v.lang) || NOVELTY.test(v.name)) return -Infinity;
  const grade = voiceQuality(v), name = v.name.toLowerCase();
  let s = grade === "natural" ? 300 : grade === "enhanced" ? 200 : 0;
  if (MALE.test(v.name)) s -= 1000;            // a female voice of any grade is preferred to a male one
  else if (FEMALE.test(v.name)) s += 100;
  if (/^en[-_]us/i.test(v.lang)) s += 50;      // the recorded pack is American English
  return s + 1 + ORDER.findIndex((n) => name.startsWith(n) || name.includes(` ${n} `));
}

/** The voice to use for device speech, or null when no usable English voice is listed (the browser's default speaks). */
export function bestVoice<T extends VoiceLike>(voices: readonly T[]): T | null {
  let best: T | null = null, top = -Infinity;
  for (const v of voices) { const s = score(v); if (s > top) { best = v; top = s; } }
  return best;
}

/** One line for Settings: which device voice speaks (everything, or only what the recorded pack cannot say), and how
 *  to get a better one. */
export function voiceNote(v: VoiceLike | null, everything = false): string {
  if (!v) return "";
  const basic = voiceQuality(v) === "standard";
  return `${everything ? "Everything is said by" : "Answers and sign text use"} ${v.name}${basic ? ", a basic voice. Adding a Premium voice such as Zoe or Ava in the system's voice settings may improve it." : "."}`;
}

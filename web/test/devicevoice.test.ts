// The device voice (web/src/devicevoice.ts) says what the recorded pack cannot: answers, sign text, and everything
// before the pack has loaded. Every platform lists dozens of voices and puts a compact, robotic one first, so the
// choice must find the most natural English female voice that is installed, and never a novelty or a male one while
// a female one exists.
import { test } from "node:test";
import assert from "node:assert/strict";
import { bestVoice, voiceNote, voiceQuality } from "../src/devicevoice";

const v = (name: string, lang = "en-US") => ({ name, lang });
const pick = (voices: Array<{ name: string; lang: string }>) => bestVoice(voices)?.name ?? null;

test("on an iPhone with no extra voices, the standard US female voice", () => {
  assert.equal(pick([v("Aaron"), v("Albert"), v("Daniel", "en-GB"), v("Fred"), v("Karen", "en-AU"), v("Nicky"), v("Samantha"),
    v("Flo (English (US))"), v("Shelley (English (US))"), v("Trinoids"), v("Amélie", "fr-CA")]), "Samantha");
});

test("a downloaded premium or enhanced voice wins over the standard one", () => {
  assert.equal(pick([v("Samantha"), v("Samantha (Enhanced)"), v("Evan (Enhanced)")]), "Samantha (Enhanced)");
  assert.equal(pick([v("Samantha"), v("Samantha (Enhanced)"), v("Ava (Premium)"), v("Evan (Premium)")]), "Ava (Premium)");
  assert.equal(pick([v("Ava (Premium)"), v("Zoe (Premium)"), v("Serena (Premium)", "en-GB")]), "Zoe (Premium)");
});

test("a female voice of lower grade is preferred to a male voice of higher grade", () => {
  assert.equal(pick([v("Evan (Premium)"), v("Samantha")]), "Samantha");
  assert.equal(pick([v("Microsoft Guy Online (Natural) - English (United States)"), v("Microsoft Zira - English (United States)")]),
    "Microsoft Zira - English (United States)");
});

test("in Edge, a natural online voice; in Chrome, the Google US voice", () => {
  assert.equal(pick([v("Microsoft David - English (United States)"), v("Microsoft Zira - English (United States)"),
    v("Microsoft Aria Online (Natural) - English (United States)"), v("Microsoft Sonia Online (Natural) - English (United Kingdom)", "en-GB")]),
    "Microsoft Aria Online (Natural) - English (United States)");
  assert.equal(pick([v("Samantha"), v("Google US English"), v("Google UK English Male", "en-GB"), v("Google Deutsch", "de-DE")]), "Google US English");
});

test("novelty and synthesiser-style voices are never chosen while anything else speaks English", () => {
  assert.equal(pick([v("Bad News"), v("Bells"), v("Zarvox"), v("Whisper"), v("Eddy (English (US))"), v("Kathy"), v("Tessa", "en-ZA")]), "Tessa");
  assert.equal(pick([v("Grandma (English (US))"), v("Moira", "en-IE")]), "Moira");
});

test("an unknown English voice is better than none, and another language is never used", () => {
  assert.equal(pick([v("English United States", "en_US"), v("español Estados Unidos", "es_US")]), "English United States");
  assert.equal(pick([v("Amélie", "fr-CA"), v("Anna", "de-DE")]), null);
  assert.equal(pick([]), null);
});

test("the grade of a voice is reported, so the app can say when a better one could be installed", () => {
  assert.equal(voiceQuality(v("Zoe (Premium)")), "natural");
  assert.equal(voiceQuality(v("Microsoft Aria Online (Natural) - English (United States)")), "natural");
  assert.equal(voiceQuality(v("Samantha (Enhanced)")), "enhanced");
  assert.equal(voiceQuality(v("Samantha")), "standard");
});

test("the settings note names the voice and says when a more natural one can be installed", () => {
  assert.equal(voiceNote(v("Zoe (Premium)")), "Answers and sign text use Zoe (Premium).");
  assert.equal(voiceNote(v("Samantha")), "Answers and sign text use Samantha, a basic voice. Adding a Premium voice such as Zoe or Ava in the system's voice settings may improve it.");
  assert.equal(voiceNote(v("Samantha (Enhanced)"), true), "Everything is said by Samantha (Enhanced).");
  assert.equal(voiceNote(null), "");
});

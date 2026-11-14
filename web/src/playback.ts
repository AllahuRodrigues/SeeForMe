// Measured speech accounting. The scheduler charges its speech budget with an estimated cue length at the moment it
// decides to speak (pcacs.ts makeCue, gate.ts charge). The voice may then play for less (cut off, cancelled, failed) or
// more (a slow voice) than that estimate. This ledger settles each charge once against the seconds that were actually
// played, so the budget follows what the user heard: unplayed seconds are refunded (never above the capacity), overruns
// become debt. It has no DOM dependency; Speech (speech.ts) feeds it the start and end events of the browser voice and of
// the audio element. The Python counterpart is SpeechLedger.reserve / settle (seeforme/scheduling/speech_ledger.py).

export type Outcome = "completed" | "interrupted" | "cancelled" | "failed" | "lost";

/** The scheduler whose budget (seconds) was charged: SchedulerBase and its subclasses fit. */
export interface BudgetHolder { budget: number; cfg: { budgetCapacity: number } }

export interface PlaybackRecord {
  id: number; text: string; backend: string;
  /** false for speech the scheduler never charged (answers to a question): measured, but the budget is left alone */
  charged: boolean;
  reservedS: number; issuedMs: number; startedMs: number | null; endedMs: number | null;
  playedS: number;
  /** false when no start was observed, so playedS is the estimate (completed) or zero (anything else) */
  measured: boolean;
  /** seconds added to the budget at settlement (negative: the cue ran long) */
  adjustS: number;
  outcome: Outcome | null;
}

export interface PlaybackSummary {
  cues: number; reservedS: number; playedS: number; refundedS: number; overrunS: number;
  outcomes: Record<Outcome, number>;
  /** share of cues that did not play to their end */
  cutShare: number;
}

/** A cue whose end event never arrives is closed this long after it should have ended (a browser can drop `onend`). */
const LOST_FACTOR = 3, LOST_SLACK_MS = 3000;

export class PlaybackLedger {
  records: PlaybackRecord[] = [];
  private open = new Map<number, PlaybackRecord>();
  private nextId = 1;

  constructor(private holder: () => BudgetHolder | null | undefined, private now: () => number = () => performance.now(), private keep = 500) {}

  /** A cue was handed to the voice. Its estimated seconds are already charged by the scheduler when `charged`. */
  issue(text: string, reservedS: number, backend: string, charged = true): number {
    const rec: PlaybackRecord = { id: this.nextId++, text, backend, charged, reservedS: Math.max(0, reservedS), issuedMs: this.now(),
      startedMs: null, endedMs: null, playedS: 0, measured: false, adjustS: 0, outcome: null };
    this.records.push(rec); this.open.set(rec.id, rec);
    if (this.records.length > this.keep) this.records.splice(0, this.records.length - this.keep);
    return rec.id;
  }

  /** The first audible moment of the cue. */
  started(id: number) {
    const rec = this.open.get(id);
    if (rec && rec.startedMs == null) rec.startedMs = this.now();
  }

  /** Playback ended. Settles the charge once and returns the seconds added to the budget; later calls return 0.
   *  `playedS` overrides the wall-clock measurement when the voice knows its own position (an audio element). */
  ended(id: number, outcome: Outcome, playedS?: number): number {
    const rec = this.open.get(id);
    if (!rec) return 0;
    this.open.delete(id);
    rec.endedMs = this.now(); rec.outcome = outcome;
    if (playedS != null && Number.isFinite(playedS)) { rec.playedS = Math.max(0, playedS); rec.measured = true; }
    else if (rec.startedMs != null) { rec.playedS = Math.max(0, (rec.endedMs - rec.startedMs) / 1000); rec.measured = true; }
    else rec.playedS = outcome === "completed" ? rec.reservedS : 0;
    if (!rec.charged) return 0;
    const h = this.holder();
    if (!h) return 0;
    const cap = h.cfg.budgetCapacity, before = h.budget;
    h.budget = Math.min(cap, Math.max(-cap, before + rec.reservedS - rec.playedS));
    rec.adjustS = h.budget - before;
    return rec.adjustS;
  }

  /** Stop, pause or session reset: every open cue ends now. Started cues were interrupted, the others never sounded. */
  abandonAll(): number {
    let total = 0;
    for (const rec of [...this.open.values()]) total += this.ended(rec.id, rec.startedMs != null ? "interrupted" : "cancelled");
    return total;
  }

  /** Is a cue still open (issued and not yet ended)? Closes cues whose end event never came. */
  active(): boolean {
    const now = this.now();
    for (const rec of [...this.open.values()]) {
      // a cue that has not started LOST_SLACK_MS after it was issued is not going to sound; a started one may run long
      const limit = rec.startedMs != null ? rec.reservedS * 1000 * LOST_FACTOR + LOST_SLACK_MS : LOST_SLACK_MS;
      if (now - (rec.startedMs ?? rec.issuedMs) <= limit) continue;
      // no end event: a started cue is taken to have played as estimated, an unstarted one never sounded
      this.ended(rec.id, "lost", rec.startedMs != null ? rec.reservedS : 0);
      rec.measured = false;
    }
    return this.open.size > 0;
  }

  summary(): PlaybackSummary {
    const done = this.records.filter((r) => r.outcome != null);
    const outcomes: Record<Outcome, number> = { completed: 0, interrupted: 0, cancelled: 0, failed: 0, lost: 0 };
    let reservedS = 0, playedS = 0, refundedS = 0, overrunS = 0;
    for (const r of done) {
      outcomes[r.outcome!]++; reservedS += r.reservedS; playedS += r.playedS;
      refundedS += Math.max(0, r.reservedS - r.playedS); overrunS += Math.max(0, r.playedS - r.reservedS);
    }
    return { cues: done.length, reservedS, playedS, refundedS, overrunS, outcomes,
      cutShare: done.length ? (done.length - outcomes.completed) / done.length : 0 };
  }

  clear() { this.records = []; this.open.clear(); }
}

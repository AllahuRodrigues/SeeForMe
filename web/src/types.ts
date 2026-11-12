export type BBox = [number, number, number, number]; // normalised x1,y1,x2,y2

export interface Detection {
  label: string;
  confidence: number;
  bbox: BBox;
  kind: "object" | "text";
  text?: string;
  distance_m?: number;
  depth_conf: number;
}

export interface Track {
  id: number;
  label: string;
  kind: "object" | "text";
  text?: string;
  bbox: BBox;
  confidence: number;
  firstSeen: number;
  lastSeen: number;
  hits: number;
  misses: number;
  confirmed: boolean;
  lastAnnounced: number | null;
  lastAnnouncedBBox: BBox | null;
  announceCount: number;
  lastHaptic: number | null;
  bboxHistory: Array<[number, BBox]>;
  // metric dynamics
  distance: number | null;
  depthConf: number;
  lateral: number;
  relSpeed: number;
  lateralSpeed: number;
  ttc: number;
  pathProb: number;
  distHistory: Array<[number, number, number]>; // t, z, x
  remembered?: boolean;  // kept alive by a memory ledger (Python only; the web tracker has none)
}

export type Action = "silence" | "speak" | "interrupt" | "haptic";

export interface Cue {
  t: number;
  trackId: number;
  label: string;
  message: string;
  action: Action;
  duration: number;
  utility: number;
  severity: number;
  distance: number | null;
  depthConf?: number;   // of `distance`: below 0.45 it is a bounding-box guess, never spoken as a number
  ttc: number | null;
  slots: string[];
  /** identity override for the announcement gate (cane obstacles: "obstacle:<sector>:<level>"); objects use their label */
  identity?: string;
  /** who proposed it: the scheduler (default), the gate saying a held cue itself, or the cane obstacle consumer */
  source?: "scheduler" | "gate" | "cane";
}

/** The "cane" summary of POST /api/depth (server/app.py): free distance per image sector within a 4 m horizon and the
 *  nearest class-agnostic obstacle. Directions are in the camera image (left = left side of the picture). */
export type CaneSector = "left" | "ahead" | "right";
export type CaneLevel = "ground" | "body" | "head";
export interface CaneObstacle { sector: CaneSector; distance_m: number; level: CaneLevel; confidence: number; bbox: BBox }
export interface CaneSummary {
  clear_m: Record<CaneSector, number | null>;   // null = cannot tell (never read as "clear")
  obstacle: CaneObstacle | null;
  suggest: null | "left" | "right" | "stop";
}

export interface ChannelState {
  busy: boolean;
  remaining: number;
  currentUtility: number;
  currentSeverity: number;
  currentTrack: number | null;
}

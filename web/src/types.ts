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
  ttc: number | null;
  slots: string[];
}

export interface ChannelState {
  busy: boolean;
  remaining: number;
  currentUtility: number;
  currentSeverity: number;
  currentTrack: number | null;
}

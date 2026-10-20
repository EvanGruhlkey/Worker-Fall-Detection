export type IconName =
  | "alert" | "cam" | "pulse" | "shield" | "fall" | "warn" | "ok" | "eye"
  | "pin" | "clock" | "gear" | "chevron" | "download" | "wifi";

export type Tone = "red" | "green" | "default";

export interface Stat {
  label: string;
  value: string;
  note: string;
  tone: Tone;
  icon: IconName;
}

export type EventStatus = "Unacknowledged" | "Needs review" | "Resolved" | "False alarm";

export interface FallEvent {
  id: number;
  time: string;
  name: string;
  icon: IconName;
  color: string;
  confidence: number | null; // percent, null when not applicable
  status: EventStatus;
  camera?: string | null;
  location?: string | null;
}

export interface CameraInfo {
  id: string;
  name: string;
  label: string;
  fps: number;
  online: boolean;
  stream_url?: string | null; // MJPEG feed with pose overlay, while online
}

export interface Location {
  name: string;
  coords: string;
}

/** GET /api/dashboard */
export interface Dashboard {
  location: Location;
  stats: Stat[];
  camera: CameraInfo;
  events: FallEvent[];
}

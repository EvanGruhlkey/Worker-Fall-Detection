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
}

export interface CameraInfo {
  id: string;
  name: string;
  label: string;
  fps: number;
  online: boolean;
}

export type IconName =
  | "alert" | "cam" | "pulse" | "shield" | "fall" | "warn" | "ok" | "eye"
  | "pin" | "clock" | "gear" | "chevron" | "download" | "upload" | "wifi" | "close";

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
  // Extra details from the backend, used by the review panel.
  detected_at?: string; // ISO 8601, UTC
  source?: "camera" | "video";
  job_id?: number;
  coords?: string | null;
  video_time?: number; // seconds into the video / camera session
  track_id?: number | null;
  reason?: string;
  reviewer?: string | null;
  notes?: string | null;
  reviewed_at?: string | null;
  snapshot_url?: string | null;
  clip_url?: string | null;
}

export type UploadStatus = "queued" | "running" | "completed" | "stopped" | "failed";

/** An uploaded video being (or already) checked for falls. */
export interface Upload {
  id: number;
  name: string;
  status: UploadStatus;
  progress: number | null; // 0 to 1
  fall_count: number;
  error: string | null;
  created_at: string;
  output_video_url: string | null;
}

export interface CameraInfo {
  id: string;
  name: string;
  label: string;
  fps: number;
  online: boolean;
  job_id?: number | null;
  source?: string | null; // local camera index; credentials in stream URLs stay server-side
  stream_url?: string | null; // MJPEG feed with pose overlay, while online
}

export interface CameraOption {
  source: string;
  label: string;
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
  uploads: Upload[];
}

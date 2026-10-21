// MOCK DATA ONLY. Replace with real data once the backend exists.
import type { CameraInfo, FallEvent, Stat } from "../types";

export const LOCATION = {
  name: "City, State, Country (Gen Location)",
  coords: "Cooridnates",
};

export const STATS: Stat[] = [
  { label: "Active incidents", value: "-", note: "Text Here", tone: "default", icon: "alert" }, //Switch the tone and dependent on whether one is active or not
  { label: "Cameras online", value: "-", note: "Text Here", tone: "default", icon: "cam" }, //Switch the tone dependent on whether camera is active or not
  { label: "Detection uptime", value: "-", note: "Text Here", tone: "default", icon: "pulse" }, //Switch the tone dependent on percentage level
  { label: "Camera health", value: "-", note: "Text Here", tone: "default", icon: "shield" }, //Switch the tone dependent on whether camera feed is blocked or fuzzy
];

export const CAMERA: CameraInfo = {
  id: "Camera ID",
  name: "Camera Name",
  label: "Label",
  fps: 0,
  online: false,
};

export const EVENTS: FallEvent[] = [ //Make sure to fill this in with database fabricated data
  { id: 1, time: "10:42:11 AM", name: "Fall detected", icon: "fall", color: "var(--red)", confidence: 97, status: "Unacknowledged" },
  { id: 2, time: "10:38:04 AM", name: "Worker prone > 30s", icon: "fall", color: "var(--red)", confidence: 91, status: "Needs review" },
  { id: 3, time: "10:25:33 AM", name: "Unsteady gait", icon: "warn", color: "var(--amber)", confidence: 78, status: "Needs review" },
  { id: 4, time: "10:07:52 AM", name: "Slip recovered", icon: "ok", color: "var(--green)", confidence: 88, status: "Resolved" },
  { id: 5, time: "09:54:19 AM", name: "Crouch misread as fall", icon: "warn", color: "var(--amber)", confidence: 61, status: "False alarm" },
  { id: 6, time: "09:31:06 AM", name: "Motion in restricted zone", icon: "eye", color: "var(--muted)", confidence: 84, status: "Resolved" },
  { id: 7, time: "09:12:45 AM", name: "Feed obstructed", icon: "warn", color: "var(--amber)", confidence: null, status: "Resolved" },
];

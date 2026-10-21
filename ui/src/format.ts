import type { EventStatus } from "./types";

/** CSS class of each status pill. */
export const PILL: Record<EventStatus, string> = {
  Unacknowledged: "unack",
  "Needs review": "need",
  Resolved: "resolved",
  "False alarm": "false",
};

/** 75.4 -> "1:15" */
export function formatVideoTime(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

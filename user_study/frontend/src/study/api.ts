import { z } from "zod";

const alertSchema = z.object({
  id: z.number(), camera: z.string(), time: z.number(), emitted_at: z.number(),
  acknowledged_at: z.number().nullable(), review_url: z.string().nullable().default(null),
  decision: z.object({ action: z.enum(["dispatch", "investigate", "false_alarm"]), time: z.number() }).nullable(),
});
export const sessionSchema = z.object({
  id: z.string(), code: z.string(), stage: z.enum(["instructions", "paused", "running", "intermission", "survey", "completed"]),
  mode: z.enum(["demo", "pilot", "study"]), duration: z.number(), condition: z.enum(["manual", "automated"]).nullable(),
  condition_number: z.number(), elapsed: z.number(), consented: z.boolean(),
  wall_url: z.string().nullable().default(null),
  feeds: z.array(z.object({ id: z.string(), name: z.string(), url: z.string() })),
  reports: z.array(z.object({ camera: z.string(), time: z.number(), reported_at: z.number(), notes: z.string().default("") })),
  alerts: z.array(alertSchema), task: z.object({ index: z.number(), left: z.number(), right: z.number(), operator: z.enum(["+", "-", "*", "/"]), score: z.number() }).nullable(),
});
export type StudySession = z.infer<typeof sessionSchema>;
export type StudyAlert = z.infer<typeof alertSchema>;
export const enrollmentSchema = z.object({ token: z.string(), session: sessionSchema });
export const sessionsSchema = z.array(z.object({ id: z.string(), code: z.string(), group: z.string(), stage: z.string(), mode: z.string() }));
export const summarySchema = z.object({ summary: z.array(z.object({
  group: z.string(), condition: z.string(), participants: z.number(), detected: z.number(), missed: z.number(),
  false_alarms: z.number(), precision: z.number().nullable(), recall: z.number().nullable(), mean_latency: z.number().nullable(),
})) });

export async function request(path: string, method = "GET", body?: unknown): Promise<unknown> {
  const response = await fetch(`/api/study${path}`, {
    method, credentials: "same-origin", cache: "no-store",
    headers: { ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) {
    const error = z.object({ detail: z.unknown() }).safeParse(await response.json().catch(() => null));
    throw new Error(error.success && typeof error.data.detail === "string" ? error.data.detail : `Request failed (${response.status}). Check the configuration and try again.`);
  }
  return response.status === 204 ? null : response.json();
}

export async function download(format: "json" | "csv") {
  const response = await fetch(`/api/study/export?format=${format}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Export failed (${response.status})`);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url; link.download = `user-study.${format}`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function clock(seconds: number) {
  return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

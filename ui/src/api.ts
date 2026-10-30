import { useCallback, useEffect, useRef, useState } from "react";
import type { Dashboard, EventScope, EventStatus, Stat, Upload } from "./types";

const POLL_MS = 2500;
const LOCATION_REFRESH_MS = 10 * 60 * 1000;
const LOCATION_RETRY_MS = 15 * 1000;

function placeholderStats(note: string): Stat[] {
  return [
    { label: "Active incidents", value: "-", note, tone: "default", icon: "alert" },
    { label: "Cameras online", value: "-", note, tone: "default", icon: "cam" },
    { label: "Detection uptime", value: "-", note, tone: "default", icon: "pulse" },
    { label: "Camera health", value: "-", note, tone: "default", icon: "shield" },
  ];
}

function placeholder(note: string): Dashboard {
  return {
    location: { name: "—", coords: "" },
    stats: placeholderStats(note),
    camera: { id: "—", name: "No camera", label: note, fps: 0, online: false },
    events: [],
    uploads: [],
  };
}

/**
 * Send this computer's location (browser geolocation) to the backend, which
 * names it and stamps it on new events. The browser asks for permission once;
 * if it's blocked, the backend keeps using FALL_SITE_LOCATION.
 */
export function useReportLocation(): void {
  useEffect(() => {
    if (!("geolocation" in navigator)) return;
    let retry: number | undefined;

    function report() {
      navigator.geolocation.getCurrentPosition(
        ({ coords }) => {
          fetch("/api/location", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              latitude: coords.latitude,
              longitude: coords.longitude,
              accuracy: coords.accuracy,
            }),
          })
            .then((response) => {
              if (!response.ok) throw new Error(`HTTP ${response.status}`);
            })
            // Backend not up yet: try again shortly.
            .catch(() => { retry = window.setTimeout(report, LOCATION_RETRY_MS); });
        },
        () => {}, // permission denied or position unavailable
        { timeout: 15000, maximumAge: 5 * 60 * 1000 },
      );
    }

    report();
    const interval = window.setInterval(report, LOCATION_REFRESH_MS);
    return () => {
      window.clearInterval(interval);
      window.clearTimeout(retry);
    };
  }, []);
}

/**
 * Dashboard data from the backend, refreshed every few seconds. `scope`
 * picks the events listed; changing it reloads at once.
 * Call `refresh()` after changing something to show it without waiting.
 */
export function useDashboard(scope: EventScope): { data: Dashboard; refresh: () => void } {
  const [data, setData] = useState<Dashboard>(() => placeholder("Connecting to backend…"));
  const loadRef = useRef<() => void>(() => {});

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    // Only the newest request may update the page, so a slow older response
    // can't overwrite a newer one.
    let latest = 0;

    async function load() {
      const request = ++latest;
      try {
        const response = await fetch(`/api/dashboard?scope=${scope}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body = (await response.json()) as Dashboard;
        if (!cancelled && request === latest) setData(body);
      } catch {
        // Keep the last known events on screen, but don't show stale live status.
        if (!cancelled && request === latest) {
          setData((prev) => ({ ...placeholder("Backend offline"), events: prev.events, uploads: prev.uploads }));
        }
      } finally {
        if (!cancelled && request === latest) {
          window.clearTimeout(timer);
          timer = window.setTimeout(load, POLL_MS);
        }
      }
    }

    loadRef.current = load;
    load();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [scope]);

  const refresh = useCallback(() => loadRef.current(), []);
  return { data, refresh };
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // not JSON
  }
  return `Request failed (HTTP ${response.status})`;
}

/** Upload a recorded video; the backend queues it for fall detection. */
export async function uploadVideo(file: File): Promise<Upload> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/videos", { method: "POST", body: form });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

/** Set an event's review status (and optional notes). */
export async function reviewEvent(id: number, status: EventStatus, notes?: string | null): Promise<void> {
  const response = await fetch(`/api/events/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, notes: notes || null }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
}

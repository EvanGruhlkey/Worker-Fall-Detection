import { useEffect, useState } from "react";
import type { Dashboard, Stat } from "./types";

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

/** Dashboard data from the backend, refreshed every few seconds. */
export function useDashboard(): Dashboard {
  const [data, setData] = useState<Dashboard>(() => placeholder("Connecting to backend…"));

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function load() {
      try {
        const response = await fetch("/api/dashboard");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body = (await response.json()) as Dashboard;
        if (!cancelled) setData(body);
      } catch {
        // Keep the last known events on screen, but don't show stale live status.
        if (!cancelled) setData((prev) => ({ ...placeholder("Backend offline"), events: prev.events }));
      } finally {
        if (!cancelled) timer = window.setTimeout(load, POLL_MS);
      }
    }

    load();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  return data;
}

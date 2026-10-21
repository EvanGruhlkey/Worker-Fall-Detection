import { useEffect, useRef, useState } from "react";
import { listCameraOptions, startCamera, stopCamera } from "../api";
import type { CameraInfo, CameraOption } from "../types";

const CUSTOM_SOURCE = "__custom__";

export default function CameraControls({
  camera,
  onChanged,
}: {
  camera: CameraInfo;
  onChanged: () => void;
}) {
  const [options, setOptions] = useState<CameraOption[]>([]);
  const [selected, setSelected] = useState(camera.source ?? "");
  const [customSource, setCustomSource] = useState("");
  const [scanning, setScanning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initialScanStarted = useRef(false);

  async function scan() {
    setScanning(true);
    setError(null);
    try {
      const found = await listCameraOptions();
      setOptions(found);
      setSelected((current) => {
        if (current === CUSTOM_SOURCE || found.some((option) => option.source === current)) {
          return current;
        }
        return camera.source ?? found[0]?.source ?? CUSTOM_SOURCE;
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not scan for cameras");
    } finally {
      setScanning(false);
    }
  }

  useEffect(() => {
    // React Strict Mode replays effects in development; do not open each
    // camera twice during that check.
    if (initialScanStarted.current) return;
    initialScanStarted.current = true;
    void scan();
  }, []);

  useEffect(() => {
    if (camera.online && camera.source) setSelected(camera.source);
  }, [camera.online, camera.source]);

  const source = selected === CUSTOM_SOURCE ? customSource.trim() : selected;
  const isCurrent = camera.online && camera.source === source;

  async function changeCamera() {
    if (!source || isCurrent) return;
    setBusy(true);
    setError(null);
    try {
      if (camera.online && camera.job_id != null) await stopCamera(camera.job_id);
      await startCamera(source);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start camera");
    } finally {
      setBusy(false);
      onChanged();
    }
  }

  async function stop() {
    if (camera.job_id == null) return;
    setBusy(true);
    setError(null);
    try {
      await stopCamera(camera.job_id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not stop camera");
    } finally {
      setBusy(false);
      onChanged();
    }
  }

  return (
    <div className="camera-controls">
      <label>
        <span>Camera source</span>
        <select value={selected} onChange={(event) => setSelected(event.target.value)} disabled={busy || scanning}>
          {scanning && !options.length && <option value="">Scanning…</option>}
          {!scanning && !options.length && <option value="">No local cameras found</option>}
          {options.map((option) => (
            <option key={option.source} value={option.source}>{option.label}</option>
          ))}
          <option value={CUSTOM_SOURCE}>Network stream URL…</option>
        </select>
      </label>
      {selected === CUSTOM_SOURCE && (
        <label className="stream-source">
          <span>Stream URL</span>
          <input
            value={customSource}
            onChange={(event) => setCustomSource(event.target.value)}
            placeholder="rtsp://camera/stream"
            disabled={busy}
          />
        </label>
      )}
      <div className="camera-actions">
        <button className="btn" type="button" onClick={() => void scan()} disabled={busy || scanning}>
          {scanning ? "Scanning…" : "Rescan"}
        </button>
        <button className="btn camera-primary" type="button" onClick={() => void changeCamera()} disabled={busy || !source || isCurrent}>
          {busy ? "Please wait…" : camera.online ? "Switch camera" : "Start camera"}
        </button>
        {camera.online && (
          <button className="btn danger" type="button" onClick={() => void stop()} disabled={busy}>
            Stop
          </button>
        )}
      </div>
      {error && <span className="camera-error">{error}</span>}
    </div>
  );
}

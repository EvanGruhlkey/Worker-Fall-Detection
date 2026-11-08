import { useRef, useState } from "react";
import { simulateVideo, uploadVideo } from "../api";
import type { Upload } from "../types";
import Icon from "./Icon";

const ACCEPT = ".mp4,.mov,.avi,.mkv,.m4v,.webm,video/*";

function statusText(upload: Upload): string {
  switch (upload.status) {
    case "queued":
      return "Waiting to start…";
    case "running":
      return upload.progress === null ? "Analyzing…" : `Analyzing… ${Math.round(upload.progress * 100)}%`;
    case "completed":
      if (upload.fall_count === 0) return "Done · no falls found";
      return `Done · ${upload.fall_count} fall${upload.fall_count === 1 ? "" : "s"} found (see event log)`;
    case "failed":
      return `Failed: ${upload.error ?? "unknown error"}`;
    case "stopped":
      return "Playback finished";
    default:
      return "Stopped";
  }
}

/** Upload a recorded video and follow the backend's YOLO + ByteTrack + fall analysis. */
export default function VideoUpload({ uploads, onUploaded }: { uploads: Upload[]; onUploaded: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      await uploadVideo(file);
      onUploaded();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  async function playLive(id: number) {
    setBusy(true);
    setError(null);
    try {
      await simulateVideo(id);
      onUploaded();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start playback");
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className="uploads">
      <div className="uploads-head">
        <div>
          <strong>Test a video as a live feed</strong>
          <small>Plays at normal speed in the feed above, with live tracking and fall alerts.</small>
        </div>
        <button className="btn" onClick={() => input.current?.click()} disabled={busy}>
          <Icon name="upload" />{busy ? "Starting…" : "Choose video"}
        </button>
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>
      {error && <p className="upload-error">{error}</p>}
      {uploads.length > 0 && (
        <ul className="upload-list">
          {uploads.map((u) => (
            <li key={u.id}>
              <div className="upload-row">
                <span className="upload-name" title={u.name}>{u.name}</span>
                <button className="btn" disabled={busy || u.status === "queued" || u.status === "running"}
                  onClick={() => playLive(u.id)}>Play live</button>
                {u.output_video_url && (
                  <a href={u.output_video_url} target="_blank" rel="noreferrer">Watch analyzed video</a>
                )}
              </div>
              <small className={u.status === "failed" ? "red" : u.fall_count > 0 ? "falls" : undefined}>
                {statusText(u)}
              </small>
              {(u.status === "running" || u.status === "queued") && (
                <div className="bar"><span style={{ width: `${Math.round((u.progress ?? 0) * 100)}%` }} /></div>
              )}
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

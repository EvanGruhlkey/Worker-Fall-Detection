import { useEffect, useRef, useState } from "react";
import { reviewEvent } from "../api";
import { PILL, formatVideoTime } from "../format";
import type { EventStatus, FallEvent } from "../types";
import Icon from "./Icon";

function formatDate(iso?: string | null): string {
  return iso ? new Date(iso).toLocaleString() : "—";
}

interface Props {
  event: FallEvent;
  onClose: () => void;
  onChanged: () => void;
}

/** Evidence for one event, and buttons to confirm it or mark it a false alarm. */
export default function EventReview({ event, onClose, onChanged }: Props) {
  const [notes, setNotes] = useState(event.notes ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const acknowledged = useRef(false);

  // Opening a new alert counts as acknowledging it: Unacknowledged -> Needs review.
  useEffect(() => {
    if (event.status !== "Unacknowledged" || acknowledged.current) return;
    acknowledged.current = true;
    reviewEvent(event.id, "Needs review", event.notes).then(onChanged).catch(() => {});
  }, [event.id, event.status, event.notes, onChanged]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function decide(status: EventStatus) {
    setSaving(true);
    setError(null);
    try {
      await reviewEvent(event.id, status, notes);
      onChanged();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save");
    } finally {
      setSaving(false);
    }
  }

  const fromVideo = event.source === "video";
  const decided = event.status === "Resolved" || event.status === "False alarm";

  return (
    <div className="backdrop" onClick={onClose}>
      <section className="review" role="dialog" aria-modal="true" aria-label="Review event" onClick={(e) => e.stopPropagation()}>
        <header className="review-head">
          <h2>
            <span style={{ color: event.color, display: "flex" }}><Icon name={event.icon} /></span>
            {event.name}
            <span className={`pill ${PILL[event.status]}`}>{event.status}</span>
          </h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="close" size={16} /></button>
        </header>

        <div className="review-media">
          {event.clip_url ? (
            <video key={event.clip_url} src={event.clip_url} controls autoPlay muted loop playsInline />
          ) : event.snapshot_url ? (
            <img src={event.snapshot_url} alt="Frame where the fall was detected" />
          ) : (
            <div className="placeholder"><span>No image saved</span></div>
          )}
        </div>
        {!event.clip_url && event.snapshot_url && (
          <small className="media-note">Showing the snapshot. The clip (5 s before to 5 s after) appears here once it has been recorded.</small>
        )}

        <dl className="details">
          <dt>Detected</dt><dd>{formatDate(event.detected_at)}</dd>
          <dt>Source</dt>
          <dd>
            {fromVideo ? "Uploaded video" : "Live camera"}: {event.camera ?? "—"}
            {event.video_time !== undefined && <small>{fromVideo ? "At" : "Session time"} {formatVideoTime(event.video_time)}</small>}
          </dd>
          <dt>Location</dt><dd>{event.location ?? "—"}{event.coords && <small>{event.coords}</small>}</dd>
          <dt>Confidence</dt><dd>{event.confidence === null ? "—" : `${event.confidence}%`}</dd>
          <dt>Person</dt><dd>{event.track_id != null ? `Track ID ${event.track_id}` : "—"}</dd>
          <dt>Why flagged</dt><dd>{event.reason ?? "—"}</dd>
          {event.reviewed_at && decided && (<><dt>Reviewed</dt><dd>{formatDate(event.reviewed_at)}</dd></>)}
        </dl>

        <label className="notes">
          <span>Notes (optional)</span>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            maxLength={2000}
            rows={2}
            placeholder="e.g. Worker slipped on oil, first aid given"
          />
        </label>

        {error && <p className="upload-error">{error}</p>}

        <footer className="review-actions">
          <button className="btn danger" disabled={saving} onClick={() => decide("Resolved")}>
            <Icon name="fall" />Confirm real fall
          </button>
          <button className="btn" disabled={saving} onClick={() => decide("False alarm")}>
            <Icon name="warn" />False alarm
          </button>
        </footer>
        <small className="media-note">"Confirm real fall" marks the event Resolved. You can change the decision later by opening the event again.</small>
      </section>
    </div>
  );
}

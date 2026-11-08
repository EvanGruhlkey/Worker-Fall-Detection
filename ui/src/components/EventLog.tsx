import { PILL, formatVideoTime } from "../format";
import type { EventScope, FallEvent } from "../types";
import Icon from "./Icon";

interface Props {
  events: FallEvent[];
  scope: EventScope;
  onScopeChange: (scope: EventScope) => void;
  onSelect: (event: FallEvent) => void;
}

const SCOPES: { value: EventScope; label: string }[] = [
  { value: "today", label: "Today" },
  { value: "all", label: "All time" },
];

export default function EventLog({ events, scope, onScopeChange, onSelect }: Props) {
  return (
    <div className="log">
      <div className="log-head">
        <div>
          <h2>Event log</h2>
          <small>
            {scope === "today" ? "Today" : "All time, newest first"} · click an event to review it
          </small>
        </div>
        <div className="actions">
          <div className="segmented" role="group" aria-label="Events to show">
            {SCOPES.map((s) => (
              <button
                key={s.value}
                className={s.value === scope ? "active" : undefined}
                aria-pressed={s.value === scope}
                onClick={() => onScopeChange(s.value)}
              >
                {s.label}
              </button>
            ))}
          </div>
          {/* The CSV holds the same events as the list, with every detail. */}
          <a className="btn" href={`/api/dashboard/events.csv?scope=${scope}`} download>
            <Icon name="download" />Export log
          </a>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Time</th><th>Event</th><th>Camera / Location</th><th>Confidence</th><th>Status</th></tr>
          </thead>
          <tbody>
            {events.length === 0 && (
              <tr><td colSpan={5}><small>{scope === "today" ? "No events today" : "No events yet"}</small></td></tr>
            )}
            {events.map((e) => (
              <tr
                key={e.id}
                className="clickable"
                tabIndex={0}
                onClick={() => onSelect(e)}
                onKeyDown={(k) => (k.key === "Enter" || k.key === " ") && (k.preventDefault(), onSelect(e))}
              >
                <td className="time">{e.time}</td>
                <td>
                  <span className="ev">
                    <span style={{ color: e.color, display: "flex" }}><Icon name={e.icon} size={12} /></span>
                    {e.name}
                  </span>
                </td>
                <td>
                  {e.camera ?? "—"}
                  <small>
                    {e.source === "video" && e.video_time !== undefined
                      ? `Uploaded video · at ${formatVideoTime(e.video_time)}`
                      : e.location}
                  </small>
                </td>
                <td>{e.confidence === null ? "—" : `${e.confidence}%`}</td>
                <td><span className={`pill ${PILL[e.status]}`}>{e.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

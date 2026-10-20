import { PILL, formatVideoTime } from "../format";
import type { FallEvent } from "../types";
import Icon from "./Icon";

interface Props {
  events: FallEvent[];
  onSelect: (event: FallEvent) => void;
}

export default function EventLog({ events, onSelect }: Props) {
  // TODO: wire up filtering and export once there is real data.
  return (
    <div className="log">
      <div className="log-head">
        <div>
          <h2>Event log</h2>
          <small>Today · All events · click an event to review it</small>
        </div>
        <div className="actions">
          <button className="btn"><Icon name="pulse" />All events<Icon name="chevron" size={12} /></button>
          <button className="btn"><Icon name="download" />Export log</button>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Time</th><th>Event</th><th>Camera / Location</th><th>Confidence</th><th>Status</th></tr>
          </thead>
          <tbody>
            {events.length === 0 && (
              <tr><td colSpan={5}><small>No events today</small></td></tr>
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

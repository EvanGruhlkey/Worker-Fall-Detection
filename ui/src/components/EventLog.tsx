import { LOCATION } from "../data/mock";
import type { EventStatus, FallEvent } from "../types";
import Icon from "./Icon";

const PILL: Record<EventStatus, string> = {
  Unacknowledged: "unack",
  "Needs review": "need",
  Resolved: "resolved",
  "False alarm": "false",
};

export default function EventLog({ events }: { events: FallEvent[] }) {
  // TODO: wire up filtering and export once there is real data.
  return (
    <div className="log">
      <div className="log-head">
        <div>
          <h2>Event log</h2>
          <small>Today · All events</small>
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
            {events.map((e) => (
              <tr key={e.id}>
                <td className="time">{e.time}</td>
                <td>
                  <span className="ev">
                    <span style={{ color: e.color, display: "flex" }}><Icon name={e.icon} size={12} /></span>
                    {e.name}
                  </span>
                </td>
                <td>Camera 01<small>{LOCATION.name}</small></td>
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

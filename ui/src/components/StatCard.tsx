import type { Stat } from "../types";
import Icon from "./Icon";

export default function StatCard({ label, value, note, tone, icon }: Stat) {
  const toneClass = tone === "default" ? "" : tone;
  return (
    <div className="card">
      <div className="row">
        <span>{label}</span>
        <span className={toneClass}><Icon name={icon} /></span>
      </div>
      <div className={`value ${toneClass}`}>{value}</div>
      <small>{note}</small>
    </div>
  );
}

import { useState } from "react";
import { enrollmentSchema, request, type StudySession } from "./api";

export default function ParticipantForm({ onSession }: { onSession: (session: StudySession) => void }) {
  const [group, setGroup] = useState("");
  const [age, setAge] = useState("");
  const [experience, setExperience] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  return <section className="study-panel study-intake">
    <h2>Before you begin</h2>
    <p>This study compares manual monitoring with automated fall alerts. First, watch nine camera feeds and mark possible falls. Then, solve arithmetic problems and review alerts from the system.</p>
    <p>Each part lasts two minutes, followed by a short survey. Participation is voluntary; you can stop at any time.</p>
    <form onSubmit={event => {
      event.preventDefault(); setBusy(true); setError("");
      void request("/sessions", "POST", { group, age_range: age, experience })
        .then(value => onSession(enrollmentSchema.parse(value).session))
        .catch(reason => setError(reason instanceof Error ? reason.message : "Could not start the study."))
        .finally(() => setBusy(false));
    }}>
      <div className="study-intake-fields">
        <label>Group<select required value={group} onChange={event => setGroup(event.target.value)}><option value="" disabled>Select group</option><option value="student">Student</option><option value="oxy">Oxy worker</option></select></label>
        <label>Age range<select required value={age} onChange={event => setAge(event.target.value)}><option value="" disabled>Select age range</option>{["18-24", "25-34", "35-44", "45-54", "55+"].map(value => <option key={value}>{value}</option>)}</select></label>
        <label>Monitoring experience<select required value={experience} onChange={event => setExperience(event.target.value)}><option value="" disabled>Select experience</option><option value="none">None</option><option value="some">Some</option><option value="regular">Regular</option></select></label>
      </div>
      {error && <p role="alert" className="study-error">{error}</p>}
      <button className="btn study-primary" disabled={busy}>{busy ? "Starting..." : "Continue"}</button>
    </form>
  </section>;
}

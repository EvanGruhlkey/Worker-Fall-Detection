import { useState } from "react";
import { enrollmentSchema, request, type StudySession } from "./api";

export default function ParticipantForm({ onSession }: { onSession: (session: StudySession) => void }) {
  const [age, setAge] = useState("");
  const [experience, setExperience] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  return <section className="study-panel study-intake">
    <h2>Before you begin</h2>
    <p>This study compares manual monitoring with automated fall alerts. In one part, you’ll watch nine prerecorded camera feeds and mark possible falls. In the other, you’ll do a separate activity and respond when the system flags a possible fall.</p>
    <p>This study is open to students. Each part lasts two minutes, followed by a short survey. Participation is voluntary; you can stop at any time.</p>
    <form onSubmit={event => {
      event.preventDefault(); setBusy(true); setError("");
      void request("/sessions", "POST", { group: "student", age_range: age, experience })
        .then(value => onSession(enrollmentSchema.parse(value).session))
        .catch(reason => setError(reason instanceof Error ? reason.message : "Could not start the study."))
        .finally(() => setBusy(false));
    }}>
      <div className="study-intake-fields">
        <label>Age range<select required value={age} onChange={event => setAge(event.target.value)}><option value="" disabled>Select age range</option>{["18-24", "25-34", "35-44", "45-54", "55+"].map(value => <option key={value}>{value}</option>)}</select></label>
        <label>Monitoring experience<select required value={experience} onChange={event => setExperience(event.target.value)}><option value="" disabled>Select experience</option><option value="none">Never done this</option><option value="some">Tried it or done it occasionally</option><option value="regular">Done it regularly</option></select></label>
      </div>
      {error && <p role="alert" className="study-error">{error}</p>}
      <button className="btn study-primary" disabled={busy}>{busy ? "Starting..." : "Continue"}</button>
    </form>
  </section>;
}

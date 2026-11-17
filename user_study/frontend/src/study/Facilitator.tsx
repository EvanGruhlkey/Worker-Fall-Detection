import { useEffect, useState } from "react";
import { z } from "zod";
import { download, request, sessionsSchema, summarySchema } from "./api";

const footageSchema = z.object({ duration: z.number(), sets: z.array(z.object({ feeds: z.array(z.unknown()) })) });

export default function Facilitator() {
  const [view, setView] = useState<"results" | "settings">("results");
  const [footage, setFootage] = useState<z.infer<typeof footageSchema> | null>(null);
  const [sessions, setSessions] = useState<z.infer<typeof sessionsSchema>>([]);
  const [summary, setSummary] = useState<z.infer<typeof summarySchema>["summary"]>([]);
  const [group, setGroup] = useState("combined");
  const [manifest, setManifest] = useState<unknown>(null);
  const [filename, setFilename] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function act(work: () => Promise<void>) {
    setBusy(true); setError(""); setMessage("");
    try { await work(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Request failed"); }
    finally { setBusy(false); }
  }
  async function refresh() {
    const config = await request("/config");
    setFootage(config ? footageSchema.parse(config) : null);
    setSessions(sessionsSchema.parse(await request("/sessions")));
    setSummary(summarySchema.parse(await request("/export")).summary);
  }
  useEffect(() => { void act(refresh); }, []);

  const participants = sessions.filter(session => session.mode === "study" && (group === "combined" || session.group === group));
  const completed = participants.filter(session => session.stage === "completed").length;
  const rows = summary.filter(row => row.group === group);

  return <div className="study-management">
    <nav className="management-tabs" aria-label="Study management">
      <button aria-pressed={view === "results"} onClick={() => setView("results")}>Results</button>
      <button aria-pressed={view === "settings"} onClick={() => setView("settings")}>Settings</button>
    </nav>
    {error && <p role="alert" className="study-error">{error}</p>}
    {message && <p role="status" className="study-notice">{message}</p>}
    {view === "results" ? <section className="study-panel management-card">
      <div className="management-title"><h2>Participant progress</h2><button className="study-text-button" disabled={busy} onClick={() => void act(refresh)}>Refresh</button></div>
      <label className="management-filter">Participants<select value={group} onChange={event => setGroup(event.target.value)}><option value="combined">Everyone</option><option value="student">Students</option><option value="oxy">Oxy workers</option></select></label>
      <div className="management-stats"><div><strong>{completed}</strong><span>Completed</span></div><div><strong>{participants.length - completed}</strong><span>Not finished</span></div></div>
      {completed > 0 ? <div className="table-wrap"><table><thead><tr><th>Condition</th><th>Falls detected</th><th>Missed</th><th>False alarms</th></tr></thead><tbody>{rows.map(row => <tr key={row.condition}><td>{row.condition === "manual" ? "Manual" : "Automated"}</td><td>{row.detected}</td><td>{row.missed}</td><td>{row.false_alarms}</td></tr>)}</tbody></table></div> : <p>Results appear after a participant finishes both conditions and the survey.</p>}
      <div className="study-actions"><button className="btn study-primary" disabled={busy || sessions.length === 0} onClick={() => void act(() => download("csv"))}>Download results (CSV)</button><button className="btn" disabled={busy || sessions.length === 0} onClick={() => void act(() => download("json"))}>Full data (JSON)</button></div>
      <p className="management-caption">Downloads include all groups. Pilot and demo sessions are excluded above.</p>
      <details className="management-records"><summary>View participant records</summary>
        {participants.length > 0 ? <div className="table-wrap"><table><thead><tr><th>ID</th><th>Group</th><th>Progress</th></tr></thead><tbody>{participants.map(session => <tr key={session.id}><td>{session.code}</td><td>{session.group === "oxy" ? "Oxy worker" : "Student"}</td><td>{session.stage === "instructions" ? "Not started" : session.stage === "running" ? "Monitoring" : session.stage === "intermission" ? "Between conditions" : session.stage === "survey" ? "Survey" : session.stage === "completed" ? "Completed" : "Paused"}</td></tr>)}</tbody></table></div> : <p>No participant records yet.</p>}
      </details>
    </section> : <section className="study-panel management-card">
      <h2>Study settings</h2>
      {footage ? <dl className="management-settings"><div><dt>Order</dt><dd>Manual, then automated</dd></div><div><dt>Time per condition</dt><dd>{footage.duration / 60} minutes</dd></div><div><dt>Camera feeds</dt><dd>{footage.sets[0]?.feeds.length ?? 0}</dd></div></dl> : <p>No footage configured.</p>}
      <form onSubmit={event => { event.preventDefault(); void act(async () => {
        await request("/config", "PUT", manifest); setFootage(footageSchema.parse(manifest));
        setManifest(null); setFilename(""); setMessage("Footage updated for new participants.");
      }); }}>
        <label>Replace study footage<input type="file" accept=".json,application/json" onChange={event => {
          const file = event.target.files?.[0];
          setManifest(null); setFilename("");
          if (file) void act(async () => { const value: unknown = JSON.parse(await file.text()); footageSchema.parse(value); setManifest(value); setFilename(file.name); });
        }} /></label>
        <p>{filename || "Choose a prepared footage manifest (.json). Existing participants keep their footage."}</p>
        <button className="btn study-primary" disabled={busy || manifest === null}>Save footage</button>
      </form>
    </section>}
  </div>;
}

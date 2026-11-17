import { useCallback, useEffect, useRef, useState } from "react";
import { request, sessionSchema, type StudySession } from "./api";
import Condition from "./Condition";
import Facilitator from "./Facilitator";
import ParticipantForm from "./ParticipantForm";
import Survey from "./Survey";
import "./study.css";
import "./oxy-theme.css";

function Instructions({ session, command, busy }: { session: StudySession; command: (path: string, body?: unknown) => Promise<boolean>; busy: boolean }) {
  const [checks, setChecks] = useState({ adult: false, vision: false, eligible: false, voluntary: false });
  const eligible = Object.values(checks).every(Boolean);
  return <div className="study-stack">
    <section className="study-panel"><h2>Before you begin</h2>
      <p>{session.mode === "study" ? "Two conditions of two minutes each, followed by a short survey. Allow about 6 minutes in total." : "A short trial with recorded footage. It checks the study workflow and is excluded from study results."}</p>
      <p>You will watch prerecorded footage that may show falls. This can cause discomfort or screen fatigue. You can pause or withdraw at any time without penalty. Participation is voluntary and has no effect on your employment.</p>
      <p>The project team records your group, age range, monitoring experience, reports, response times, task answers, and survey. We do not ask for your name or share individual results with Oxy.</p>
      <p>Before agreeing, ask the facilitator about access to your records and how long they are kept. Read the consent form supplied by the team.</p>
      <form onSubmit={e => { e.preventDefault(); void command("/consent", checks); }}>
        <label className="study-check"><input type="checkbox" required checked={checks.adult} onChange={e => setChecks(c => ({ ...c, adult: e.target.checked }))} />I am at least 18 years old.</label>
        <label className="study-check"><input type="checkbox" required checked={checks.vision} onChange={e => setChecks(c => ({ ...c, vision: e.target.checked }))} />I have normal or corrected vision and can watch a screen for two minutes.</label>
        <label className="study-check"><input type="checkbox" required checked={checks.eligible} onChange={e => setChecks(c => ({ ...c, eligible: e.target.checked }))} />I am not a project member and have not seen, built, or labeled the test footage.</label>
        <label className="study-check"><input type="checkbox" required checked={checks.voluntary} onChange={e => setChecks(c => ({ ...c, voluntary: e.target.checked }))} />My questions have been answered and I voluntarily agree to participate.</label>
        <button className="btn" disabled={busy || !eligible || session.consented}>{session.consented ? "Consent recorded" : "Record consent"}</button>
      </form>
    </section>
    {session.consented && <button className="btn study-primary" disabled={busy} onClick={() => void command("/start")}>Start monitoring</button>}

  </div>;
}

export default function Study() {
  const [session, setSession] = useState<StudySession | null>(null);
  const [setup, setSetup] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [withdrawOpen, setWithdrawOpen] = useState(false);
  const pending = useRef(false);
  const queuedPause = useRef<unknown>(null);
  const recovery = useRef(true);
  const [connected, setConnected] = useState(false);
  const version = useRef(0);
  const monitoring = useRef(true);
  const parkedSession = useRef<StudySession | null>(null);
  const latest = useRef(session);
  latest.current = session;

  const command = useCallback(async function execute(path: string, body?: unknown): Promise<boolean> {
    if (pending.current) {
      if (path === "/pause") queuedPause.current = body;
      return false;
    }
    pending.current = true; setBusy(true); setError("");
    const id = ++version.current;
    try {
      const next = sessionSchema.parse(await request(`/session${path}`, "POST", body));
      if (id === version.current) setSession(next);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Study request failed");
      return false;
    } finally {
      pending.current = false; setBusy(false);
      if (queuedPause.current !== null) {
        const pauseBody = queuedPause.current; queuedPause.current = null;
        void execute("/pause", pauseBody);
      }
    }
  }, []);

  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      if (!pending.current && monitoring.current) {
        const id = ++version.current;
        try {
          const response = await fetch("/api/study/session", { cache: "no-store" });
          if (response.status === 401) {
            monitoring.current = false;
            if (!disposed) setSession(null);
          } else {
            if (!response.ok) throw new Error("Study server unavailable. Playback is paused until the connection returns.");
            let next = sessionSchema.parse(await response.json());
            if (recovery.current && next.stage === "running") {
              next = sessionSchema.parse(await request("/session/pause", "POST", { reason: "buffering" }));
            }
            recovery.current = false;
            if (!disposed && id === version.current) { setSession(next); setConnected(true); setError(""); if (next.stage === "completed") monitoring.current = false; }
          }
        } catch (e) {
          if (!disposed) {
            recovery.current = true; setConnected(false);
            setError(e instanceof Error ? e.message : "Connection lost");
            setSession(s => s?.stage === "running" ? { ...s, stage: "paused" } : s);
          }
        } finally { if (!disposed) setLoading(false); }
      }
      if (!disposed) timer = setTimeout(() => void poll(), 500);
    }
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, []);

  useEffect(() => {
    function visibility() {
      if (document.hidden && latest.current?.stage === "running") void command("/pause", { reason: "hidden" });
    }
    function unload(e: BeforeUnloadEvent) {
      if (latest.current && !["completed", "instructions"].includes(latest.current.stage)) { e.preventDefault(); e.returnValue = ""; }
    }
    document.addEventListener("visibilitychange", visibility);
    window.addEventListener("beforeunload", unload);
    return () => { document.removeEventListener("visibilitychange", visibility); window.removeEventListener("beforeunload", unload); };
  }, [command]);

  async function withdraw() {
    pending.current = true; ++version.current; setBusy(true);
    try { await request("/session", "DELETE"); monitoring.current = false; setSession(null); setWithdrawOpen(false); setError(""); }
    catch (e) { setError(e instanceof Error ? e.message : "Withdrawal failed"); }
    finally { pending.current = false; setBusy(false); }
  }
  function showFacilitator() {
    parkedSession.current = latest.current;
    monitoring.current = false; ++version.current; setSession(null); setSetup(true);
  }
  function toggleSetup() {
    if (setup && parkedSession.current) {
      setSession(parkedSession.current); parkedSession.current = null;
      monitoring.current = true;
    }
    setSetup(value => !value);
  }
  return <main className="study-page">
    <header className={`study-heading ${session && ["running", "paused"].includes(session.stage) ? "study-heading-hidden" : ""}`}><img className="study-logo" src="/oxy-logo.png" alt="Oxy" width="40" height="42" /><h1>User study</h1></header>
    {session?.mode === "pilot" && <p className="study-notice">Recorded footage / {session.duration}-second pilot / excluded from study results</p>}
    {session?.mode === "demo" && <p className="study-demo">Synthetic test session. Excluded from study results.</p>}
    {error && <div className="study-error" role="alert">{error}</div>}
    {loading ? <p>Connecting to study server...</p> : !session ? setup ? <Facilitator /> : <ParticipantForm onSession={s => { ++version.current; monitoring.current = true; recovery.current = false; setConnected(true); setSetup(false); setSession(s); }} />
      : session.stage === "instructions" ? <Instructions session={session} command={command} busy={busy || !connected} />
        : session.stage === "running" || session.stage === "paused" ? <Condition key={`${session.id}-${session.condition_number}`} session={session} command={command} busy={busy || !connected} onWithdraw={() => setWithdrawOpen(true)} onFacilitator={showFacilitator} />
          : session.stage === "intermission" ? <section className="study-panel"><h2>First condition complete</h2><p>The next condition uses a different set of footage. Continue when you are ready.</p><button className="btn study-primary" disabled={busy} onClick={() => void command("/start")}>Prepare second condition</button></section>
            : session.stage === "survey" ? <Survey busy={busy} submit={async body => { await command("/survey", body); }} />
              : <section className="study-panel"><h2>Session complete</h2><p>Responses saved. Thank you for participating.</p><button className="btn" onClick={() => { monitoring.current = false; ++version.current; setSession(null); setSetup(false); }}>New participant</button></section>}
    {!session && <footer className="study-entry-footer"><button className="study-text-button" onClick={toggleSetup}>{setup ? "Back to study" : "Study setup"}</button></footer>}
    {session && !["running", "paused", "completed"].includes(session.stage) && <footer className="study-entry-footer"><button className="study-text-button" disabled={busy} onClick={() => setWithdrawOpen(true)}>Withdraw</button></footer>}
    {withdrawOpen && <div className="study-modal-backdrop"><section className="study-panel" role="alertdialog" aria-modal="true" aria-labelledby="withdraw-title"><h2 id="withdraw-title">Withdraw from this study?</h2><p>This removes this participant session and its responses from the study database and exports. Previously downloaded copies must be removed by the facilitator.</p><div className="study-actions"><button autoFocus className="btn" disabled={busy} onClick={() => setWithdrawOpen(false)}>Keep participating</button><button className="btn danger" disabled={busy} onClick={() => void withdraw()}>Withdraw and delete records</button></div></section></div>}
  </main>;
}

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { StudyAlert, StudySession } from "./api";
import { clock } from "./api";

type Props = {
  session: StudySession;
  busy: boolean;
  command: (path: string, body?: unknown) => Promise<boolean>;
  audio: AudioContext | null;
  cameraView: ReactNode;
};

function chime(audio: AudioContext) {
  for (const delay of [0, .22]) {
    const oscillator = audio.createOscillator();
    const gain = audio.createGain();
    const start = audio.currentTime + delay;
    oscillator.frequency.value = 880;
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(.15, start + .01);
    gain.gain.exponentialRampToValueAtTime(.001, start + .16);
    oscillator.connect(gain); gain.connect(audio.destination);
    oscillator.start(start); oscillator.stop(start + .18);
    oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
  }
}

function result(task: NonNullable<StudySession["task"]>) {
  switch (task.operator) {
    case "+": return task.left + task.right;
    case "-": return task.left - task.right;
    case "*": return task.left * task.right;
    case "/": return task.left / task.right;
  }
}

function Review({ alert, url }: { alert: StudyAlert; url: string }) {
  const video = useRef<HTMLVideoElement>(null);
  const beginning = Math.max(0, alert.time - .5);
  useEffect(() => { const player = video.current; return () => { player?.pause(); }; }, []);
  if (alert.review_url) return <video ref={video} src={alert.review_url} muted controls autoPlay preload="auto" />;
  return <video ref={video} src={url} muted autoPlay preload="auto" onLoadedMetadata={() => {
    if (video.current) video.current.currentTime = beginning;
  }} onTimeUpdate={() => { if (video.current && video.current.currentTime >= alert.time + .5) video.current.pause(); }} />;
}

export default function AutomatedCondition({ session: s, busy, command, audio, cameraView }: Props) {
  const [answer, setAnswer] = useState("");
  const [view, setView] = useState<"arithmetic" | "cameras">("arithmetic");
  const mathTab = useRef<HTMLButtonElement>(null);
  const cameraTab = useRef<HTMLButtonElement>(null);
  const [reviewing, setReviewing] = useState<number | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const reviewButton = useRef<HTMLButtonElement>(null);
  const sending = useRef(false);
  const sounded = useRef(new Set<number>());
  const alert = s.alerts.find(item => item.decision === null);
  const running = s.stage === "running";
  const disabled = busy || !running || !!alert;
  const task = s.task;

  useEffect(() => {
    const popup = dialog.current;
    if (!popup) return;
    if (alert && running && reviewing === alert.id) {
      if (!popup.open) popup.showModal();
      popup.querySelector<HTMLButtonElement>("button")?.focus();
    } else if (popup.open) popup.close();
  }, [alert?.id, alert?.acknowledged_at, running, reviewing]);
  useEffect(() => {
    if (running && alert) reviewButton.current?.focus();
    else if (running && view === "arithmetic") input.current?.focus();
  }, [running, alert?.id, task?.index, view]);
  useEffect(() => { setAnswer(""); }, [task?.index]);
  useEffect(() => {
    if (!running || !audio || audio.state !== "running") return;
    const fresh = s.alerts.filter(item => !sounded.current.has(item.id) && item.decision === null);
    if (fresh.length === 0) return;
    fresh.forEach(item => sounded.current.add(item.id));
    chime(audio);
  }, [s.alerts, running, audio]);

  async function submit(value: string) {
    if (disabled || sending.current || !task || value.trim() === "") return;
    sending.current = true;
    try { await command("/task", { index: task.index, answer: Number(value) }); }
    finally { sending.current = false; }
  }
  async function openReview() {
    if (!alert) return;
    if (alert.acknowledged_at === null && !await command(`/alerts/${alert.id}/acknowledge`)) return;
    setReviewing(alert.id);
  }

  return <section className="automated-layout">
    <div className="automated-tabs" role="tablist" aria-label="Study view" onKeyDown={event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? "arithmetic" : event.key === "End" ? "cameras" : view === "arithmetic" ? "cameras" : "arithmetic";
      setView(next);
      (next === "arithmetic" ? mathTab : cameraTab).current?.focus();
    }}>
      <button ref={mathTab} id="arithmetic-tab" role="tab" aria-selected={view === "arithmetic"} aria-controls="arithmetic-panel" tabIndex={view === "arithmetic" ? 0 : -1} onClick={() => setView("arithmetic")}>Arithmetic</button>
      <button ref={cameraTab} id="cameras-tab" role="tab" aria-selected={view === "cameras"} aria-controls="cameras-panel" tabIndex={view === "cameras" ? 0 : -1} onClick={() => setView("cameras")}>Cameras</button>
    </div>
    <section className="arithmetic-task" id="arithmetic-panel" role="tabpanel" aria-labelledby="arithmetic-tab" hidden={view !== "arithmetic"}>
      {task && <form onSubmit={event => { event.preventDefault(); void submit(answer); }}>
        <label><span>{task.left} {task.operator === "*" ? "×" : task.operator === "/" ? "÷" : task.operator} {task.right} = ?</span>
          <input ref={input} type="number" min={-10000} max={10000} aria-label="Arithmetic answer" placeholder="Enter answer..." autoComplete="off" value={answer} disabled={disabled} onChange={event => {
            const value = event.target.value;
            setAnswer(value);
            if (value !== "" && Number(value) === result(task)) void submit(value);
          }} />
        </label>
      </form>}
      <p className="arithmetic-score">Score: {task?.score ?? 0}</p>
      <div className="arithmetic-status"><span>{running ? alert ? "Check the alert below" : "Correct answers advance automatically" : "Paused"}</span><button className="study-text-button" disabled={!audio} onClick={() => { if (audio) void audio.resume().then(() => chime(audio)); }}>Test sound</button></div>
    </section>
    <section className="pipeline-monitor" id="cameras-panel" role="tabpanel" aria-labelledby="cameras-tab" hidden={view !== "cameras"}>{cameraView}<p className="monitoring-status">{running ? "Monitoring in background" : "Monitoring paused"}</p></section>
    {alert && <section className="auto-alert-panel" role="alert" aria-labelledby="incident-heading">
      <h2 id="incident-heading">Possible fall / {s.feeds.find(feed => feed.id === alert.camera)?.name}</h2>
      {s.alerts.filter(item => item.decision === null).length > 1 && <span>{s.alerts.filter(item => item.decision === null).length} waiting</span>}
      <button ref={reviewButton} className="btn auto-review-button" disabled={!running || busy} onClick={() => void openReview()}>Review incident</button>
    </section>}
    <dialog ref={dialog} className="auto-alert-dialog" aria-labelledby="auto-alert-title" onCancel={event => event.preventDefault()}>
      {alert && <>
        <div className="auto-alert-heading"><h2 id="auto-alert-title">Detected fall replay</h2><span>{s.feeds.find(feed => feed.id === alert.camera)?.name} / {clock(alert.time)}</span></div>
        {reviewing === alert.id && running && <Review key={alert.id} alert={alert} url={s.feeds.find(feed => feed.id === alert.camera)?.url ?? ""} />}
        <div className="study-actions">{[["dispatch", "Dispatch help"], ["investigate", "Investigate"], ["false_alarm", "False alarm"]].map(([action, label]) => <button className="btn" key={action} disabled={busy} onClick={() => void command(`/alerts/${alert.id}/decision`, { action })}>{label}</button>)}</div>
        <button className="study-text-button" onClick={() => setReviewing(null)}>Close review</button>
        <button className="study-text-button" disabled={busy} onClick={() => void command("/pause", { reason: "operator" })}>Pause study</button>
      </>}
    </dialog>
  </section>;
}

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { StudyAlert, StudySession } from "./api";
import { clock } from "./api";

type Props = {
  session: StudySession;
  busy: boolean;
  command: (path: string, body?: unknown) => Promise<boolean>;
  audio: AudioContext | null;
  cameraView: ReactNode;
  tourTarget?: string;
  view: "gameplay" | "cameras";
  setView: (view: "gameplay" | "cameras") => void;
};

const responses = [["dispatch", "Dispatch help"], ["false_alarm", "False alarm"]] as const;

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

function Review({ alert, url }: { alert: StudyAlert; url: string }) {
  const video = useRef<HTMLVideoElement>(null);
  const [usePreparedClip, setUsePreparedClip] = useState(Boolean(alert.review_url));
  const beginning = Math.max(0, alert.time - 1);
  const end = alert.time + 1;
  useEffect(() => { const player = video.current; return () => { player?.pause(); }; }, []);
  if (usePreparedClip && alert.review_url) return <video
    ref={video}
    src={alert.review_url}
    muted
    controls
    autoPlay
    preload="auto"
    onLoadedMetadata={event => {
      if (event.currentTarget.duration < end - beginning - 0.1) setUsePreparedClip(false);
    }}
    onError={() => setUsePreparedClip(false)}
  />;
  return <video
    ref={video}
    src={`${url}#t=${beginning},${end}`}
    muted
    autoPlay
    preload="auto"
    onTimeUpdate={event => {
      if (event.currentTarget.currentTime >= end) event.currentTarget.pause();
    }}
  />;
}

export default function AutomatedCondition({ session: s, busy, command, audio, cameraView, tourTarget, view, setView }: Props) {
  const gameplayTab = useRef<HTMLButtonElement>(null);
  const cameraTab = useRef<HTMLButtonElement>(null);
  const game = useRef<HTMLIFrameElement>(null);
  const [reviewing, setReviewing] = useState<number | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const reviewButton = useRef<HTMLButtonElement>(null);
  const sounded = useRef(new Set<number>());
  const alert = s.alerts.find(item => item.decision === null);
  const running = s.stage === "running";
  const playing = running && !alert;

  function syncGame() {
    game.current?.contentWindow?.postMessage({ type: "study:runner", enabled: playing }, window.location.origin);
  }

  useEffect(() => {
    game.current?.contentWindow?.postMessage({ type: "study:runner", enabled: playing }, window.location.origin);
    if (playing && game.current?.getClientRects().length) game.current.focus({ preventScroll: true });
  }, [playing]);

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
  }, [running, alert?.id]);
  useEffect(() => {
    if (!running || !audio || audio.state !== "running") return;
    const fresh = s.alerts.filter(item => !sounded.current.has(item.id) && item.decision === null);
    if (fresh.length === 0) return;
    fresh.forEach(item => sounded.current.add(item.id));
    chime(audio);
  }, [s.alerts, running, audio]);

  async function openReview() {
    if (!alert) return;
    if (alert.acknowledged_at === null && !await command(`/alerts/${alert.id}/acknowledge`)) return;
    setReviewing(alert.id);
  }

  return <section className="automated-layout">
    <div className="automated-tabs" role="tablist" aria-label="Study view" onKeyDown={event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? "gameplay" : event.key === "End" ? "cameras" : view === "gameplay" ? "cameras" : "gameplay";
      setView(next);
      (next === "gameplay" ? gameplayTab : cameraTab).current?.focus();
    }}>
      <button ref={gameplayTab} id="gameplay-tab" role="tab" aria-selected={view === "gameplay"} aria-controls="gameplay-panel" tabIndex={view === "gameplay" ? 0 : -1} onClick={() => setView("gameplay")}>Flappy Bird</button>
      <button ref={cameraTab} id="cameras-tab" role="tab" aria-selected={view === "cameras"} aria-controls="cameras-panel" tabIndex={view === "cameras" ? 0 : -1} onClick={() => setView("cameras")}>Cameras</button>
    </div>
    <section className="gameplay-task" id="gameplay-panel" data-tour="gameplay" role="tabpanel" aria-labelledby="gameplay-tab" hidden={view !== "gameplay"}>
      <iframe ref={game} title="Flappy Bird by Samuel Crevier" src="/samuel-flappy/index.html"
        onLoad={syncGame} tabIndex={playing ? 0 : -1}
        style={{ pointerEvents: playing ? undefined : "none" }} />
    </section>
    <section className="pipeline-monitor" id="cameras-panel" role="tabpanel" aria-labelledby="cameras-tab" hidden={view !== "cameras"}>{cameraView}<p className="monitoring-status">{running ? "Monitoring in background" : "Monitoring paused"}</p></section>
    {tourTarget === "alert" && <section className="auto-alert-panel round-tour-example" data-tour="alert">
      <h2>Example alert · Possible fall</h2>
      <button className="btn auto-review-button" type="button">Review incident</button>
    </section>}
    {tourTarget === "decision" && <section className="auto-alert-panel round-tour-example" data-tour="decision">
      <h2>Example replay response</h2>
      <div className="study-actions">{responses.map(([action, label]) => <button className="btn" type="button" key={action}>{label}</button>)}</div>
    </section>}
    {alert && <section className="auto-alert-panel" role="alert" aria-labelledby="incident-heading">
      <h2 id="incident-heading">Possible fall / {s.feeds.find(feed => feed.id === alert.camera)?.name}</h2>
      {s.alerts.filter(item => item.decision === null).length > 1 && <span>{s.alerts.filter(item => item.decision === null).length} waiting</span>}
      <button ref={reviewButton} className="btn auto-review-button" disabled={!running || busy} onClick={() => void openReview()}>Review incident</button>
    </section>}
    <dialog ref={dialog} className="auto-alert-dialog" aria-labelledby="auto-alert-title" onCancel={event => event.preventDefault()}>
      {alert && <>
        <div className="auto-alert-heading"><h2 id="auto-alert-title">Detected fall replay</h2><span>{s.feeds.find(feed => feed.id === alert.camera)?.name} / {clock(alert.time)}</span></div>
        {reviewing === alert.id && running && <Review key={alert.id} alert={alert} url={s.feeds.find(feed => feed.id === alert.camera)?.url ?? ""} />}
        <div className="study-actions">{responses.map(([action, label]) => <button className="btn" key={action} disabled={busy} onClick={() => void command(`/alerts/${alert.id}/decision`, { action })}>{label}</button>)}</div>
        <button className="study-text-button" onClick={() => setReviewing(null)}>Close review</button>
        <button className="study-text-button" disabled={busy} onClick={() => void command("/pause", { reason: "operator" })}>Pause study</button>
      </>}
    </dialog>
  </section>;
}

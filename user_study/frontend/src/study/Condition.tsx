import { useEffect, useRef, useState } from "react";
import { clock, type StudySession } from "./api";
import AutomatedCondition from "./AutomatedCondition";
import "./console.css";

type Props = {
  session: StudySession;
  busy: boolean;
  command: (path: string, body?: unknown) => Promise<boolean>;
  onWithdraw: () => void;
  onFacilitator: () => void;
};

export default function Condition({ session: s, busy, command, onWithdraw, onFacilitator }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const picture = useRef<HTMLCanvasElement>(null);
  const alertAudio = useRef<AudioContext | null>(null);
  const [ready, setReady] = useState(false);
  const [starting, setStarting] = useState(false);
  const [camera, setCamera] = useState("");
  const [error, setError] = useState("");
  const paused = s.stage === "paused";
  const manual = s.condition === "manual";
  const disabled = paused || busy;

  useEffect(() => { if (s.stage !== "running") video.current?.pause(); }, [s.stage]);
  useEffect(() => {
    const player = video.current;
    return () => {
      player?.pause();
      if (player && !player.isConnected) { player.removeAttribute("src"); player.load(); }
    };
  }, []);

  useEffect(() => {
    const player = video.current;
    const canvas = picture.current;
    const context = canvas?.getContext("2d");
    if (!player || !canvas || !context) return;
    let frame = 0;
    let lastTime = -1;
    const paint = () => {
      if (player.readyState >= 2 && player.currentTime !== lastTime) {
        context.drawImage(player, 0, 0, canvas.width, canvas.height);
        lastTime = player.currentTime;
      }
      frame = requestAnimationFrame(paint);
    };
    frame = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(frame);
  }, []);

  useEffect(() => () => { void alertAudio.current?.close(); }, []);

  function interrupt(reason: "buffering" | "media_error" | "drift") {
    if (s.stage === "running") { video.current?.pause(); void command("/pause", { reason }); }
  }
  async function start() {
    const player = video.current;
    if (!player) return;
    setStarting(true); setError("");
    try {
      if (!manual) {
        alertAudio.current ??= new AudioContext();
        await alertAudio.current.resume();
      }
      player.currentTime = s.elapsed;
      await player.play();
      if (document.hidden || !await command("/resume")) throw new Error("Return to this tab and try again.");
    } catch (cause) { player.pause(); setError(cause instanceof Error ? cause.message : "Playback failed."); }
    finally { setStarting(false); }
  }

  if (!s.wall_url || ![6, 9].includes(s.feeds.length)) return <section className="study-panel">
    <h2>Start a camera session</h2><p>This session uses the previous camera layout. Your records are saved.</p>
    <button className="btn" onClick={onFacilitator}>Study setup</button>
  </section>;

  const cameraView = (
      <div className={`console-wall ${s.feeds.length === 9 ? "console-wall-nine" : ""}`} role="group" aria-label={`${s.feeds.length} simultaneous camera feeds`}>
        <video ref={video} src={s.wall_url} muted playsInline preload="auto" onLoadedMetadata={event => { event.currentTarget.currentTime = s.elapsed; }} onCanPlay={() => setReady(true)}
          onWaiting={() => { setReady(false); interrupt("buffering"); }}
          onError={() => { setReady(false); setError("Camera video could not load."); interrupt("media_error"); }}
          onEnded={() => { if (s.elapsed < s.duration - 1) { setError("Video ended early."); interrupt("media_error"); } }}
          onTimeUpdate={e => { if (!starting && Math.abs(e.currentTarget.currentTime - s.elapsed) > 2) interrupt("drift"); }} />
        <canvas ref={picture} width={1920} height={s.feeds.length === 9 ? 1080 : 720} aria-label={`${s.feeds.length} live camera views`} />
        <div className="console-tiles">{s.feeds.map((feed, index) => <button key={feed.id} aria-label={`Select ${feed.name}`} aria-pressed={camera === feed.id} disabled={!manual} onClick={() => setCamera(feed.id)}>
          <span>CAM {String(index + 1).padStart(2, "0")}</span>
        </button>)}</div>
      </div>
  );

  return <section className={manual ? "study-console" : "study-console study-console-auto"}>
    <header className="console-header">
      <strong className="console-timer" aria-label="Time remaining">{clock(Math.max(0, s.duration - s.elapsed))}</strong>
      <h1 className="console-brand"><img src="/oxy-logo.png" alt="Oxy" width="28" height="29" />Worker safety study</h1>
      <button className="btn" disabled={busy || starting || (paused && !ready)} onClick={() => paused ? void start() : void command("/pause", { reason: "operator" })}>
        {starting ? "Starting..." : !ready ? "Loading..." : paused ? s.elapsed === 0 ? "Start" : "Resume" : "Pause"}
      </button>
    </header>
    {error && <p className="console-error" role="alert">{error} {paused && <button onClick={() => { setError(""); setReady(false); video.current?.load(); }}>Retry video</button>}</p>}
    <div className="console-body">
      {manual && cameraView}
      <aside className="console-sidebar">{manual ? <>
        <p>Select a camera, then mark a possible fall.</p>
        <button className="console-mark" disabled={disabled || !camera} onClick={() => void command("/reports", { camera, time: Math.floor(s.elapsed) })}>Mark possible fall</button>
        <section className="console-events"><h2>Events ({s.reports.length})</h2>
          {s.reports.length === 0 ? <p>No events marked.</p> : <table><thead><tr><th>Camera</th><th>Time</th></tr></thead><tbody>
            {s.reports.map((report, index) => <tr key={index}><td>{String(s.feeds.findIndex(feed => feed.id === report.camera) + 1).padStart(2, "0")}</td><td>{clock(report.reported_at)}</td></tr>)}
          </tbody></table>}
        </section>
      </> : <AutomatedCondition session={s} busy={busy} command={command} audio={alertAudio.current} cameraView={cameraView} />}
        <footer className="console-controls"><button disabled={!paused || busy} onClick={onFacilitator}>Study setup</button><button disabled={busy} onClick={onWithdraw}>Withdraw</button></footer>
      </aside>
    </div>
  </section>;
}

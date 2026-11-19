import { useEffect, useRef, useState } from "react";
import { clock, type StudySession } from "./api";
import AutomatedCondition from "./AutomatedCondition";
import RoundTour, { type TourStep } from "./RoundTour";
import "./console.css";

type Props = {
  session: StudySession;
  busy: boolean;
  command: (path: string, body?: unknown) => Promise<boolean>;
  onWithdraw: () => void;
  onFacilitator?: () => void;
};

const manualTour: TourStep[] = [
  { target: "wall", title: "Watch the cameras", text: "Look across the camera feeds for possible falls." },
  { target: "camera", title: "Select a camera", text: "If you see a possible fall, click the camera where it happened." },
  { target: "mark", title: "Mark the possible fall", text: "Click Mark possible fall to record what you saw, then keep watching." },
];

const automatedTour: TourStep[] = [
  { target: "gameplay", title: "Play Flappy Bird", text: "Tap the game or press Space to flap. Fly through the gaps between the pipes." },
  { target: "alert", title: "Review an alert", text: "When you hear the chime and see an alert like this one, click Review incident to see the replay." },
  { target: "decision", title: "Choose a response", text: "Choose Dispatch help if the person needs help or False alarm if you don't see a fall. Then return to the game." },
];

const startStep: TourStep = {
  target: "start", title: "Start when ready", text: "Select Start when you’re ready. The timer begins when the round starts.",
};

export default function Condition({ session: s, busy, command, onWithdraw, onFacilitator }: Props) {
  const consoleRef = useRef<HTMLElement>(null);
  const startButton = useRef<HTMLButtonElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const picture = useRef<HTMLCanvasElement>(null);
  const alertAudio = useRef<AudioContext | null>(null);
  const [ready, setReady] = useState(false);
  const [starting, setStarting] = useState(false);
  const [camera, setCamera] = useState("");
  const [automatedView, setAutomatedView] = useState<"gameplay" | "cameras">("gameplay");
  const [error, setError] = useState("");
  const [tourIndex, setTourIndex] = useState<number | null>(s.elapsed === 0 && s.stage === "paused" ? 0 : null);
  const paused = s.stage === "paused";
  const manual = s.condition === "manual";
  const showAutomatedCameras = !manual && automatedView === "cameras";
  const disabled = paused || busy;
  const tourSteps = [...(manual ? manualTour : automatedTour), startStep];
  const tourStep = paused && tourIndex !== null ? tourSteps[tourIndex] : undefined;

  function advanceTour() {
    if (tourIndex === null) return;
    if (tourIndex < tourSteps.length - 1) {
      setTourIndex(tourIndex + 1);
      return;
    }
    setTourIndex(null);
    requestAnimationFrame(() => startButton.current?.focus());
  }

  useEffect(() => { if (s.stage !== "running") video.current?.pause(); }, [s.stage]);
  useEffect(() => {
    const player = video.current;
    if (manual || !player) return;
    if (s.stage !== "running" || !showAutomatedCameras) {
      player.pause();
      return;
    }
    if (!ready || player.seeking) return;
    if (player.paused || Math.abs(player.currentTime - s.elapsed) > 2) {
      player.currentTime = s.elapsed;
      if (player.paused) void player.play().catch(() => {});
    }
  }, [manual, ready, s.elapsed, s.stage, showAutomatedCameras]);
  useEffect(() => {
    const player = video.current;
    return () => {
      player?.pause();
      if (player && !player.isConnected) { player.removeAttribute("src"); player.load(); }
    };
  }, [s.wall_url]);

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
  }, [s.wall_url]);

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
      } else {
        player.currentTime = s.elapsed;
        await player.play();
      }
      if (document.hidden || !await command("/resume")) throw new Error("Return to this tab and try again.");
    } catch (cause) { player.pause(); setError(cause instanceof Error ? cause.message : "Playback failed."); }
    finally { setStarting(false); }
  }

  if (!s.wall_url || ![6, 9].includes(s.feeds.length)) return <section className="study-panel">
    <h2>Start a camera session</h2><p>This session uses the previous camera layout. Your records are saved.</p>
    {onFacilitator && <button className="btn" onClick={onFacilitator}>Study setup</button>}
  </section>;

  const cameraView = (
      <div className={`console-wall ${s.feeds.length === 9 ? "console-wall-nine" : ""}`} data-tour="wall" role="group" aria-label={`${s.feeds.length} simultaneous camera feeds`}>
        <video ref={video} src={s.wall_url} muted playsInline preload="auto" onLoadedMetadata={event => { event.currentTarget.currentTime = s.elapsed; }} onCanPlay={() => setReady(true)}
          onWaiting={() => { setReady(false); if (manual) interrupt("buffering"); }}
          onError={() => { setReady(false); setError("Camera video could not load."); if (manual) interrupt("media_error"); }}
          onEnded={() => { if (manual && s.elapsed < s.duration - 1) { setError("Video ended early."); interrupt("media_error"); } }}
          onTimeUpdate={e => { if (manual && !starting && Math.abs(e.currentTarget.currentTime - s.elapsed) > 2) interrupt("drift"); }} />
        <canvas ref={picture} width={1920} height={s.feeds.length === 9 ? 1080 : 720} aria-label={`${s.feeds.length} live camera views`} />
        <div className="console-tiles">{s.feeds.map((feed, index) => <button key={feed.id} data-tour={index === 0 ? "camera" : undefined} aria-label={`Select ${feed.name}`} aria-pressed={camera === feed.id} disabled={!manual} onClick={() => setCamera(feed.id)}>
          <span>CAM {String(index + 1).padStart(2, "0")}</span>
        </button>)}</div>
      </div>
  );

  return <section ref={consoleRef} className={manual ? "study-console" : "study-console study-console-auto"}>
    <header className="console-header">
      <strong className="console-timer" aria-label="Time remaining">{clock(Math.max(0, s.duration - s.elapsed))}</strong>
      <h1 className="console-brand"><img src="/oxy-logo.png" alt="Oxy" width="28" height="29" />Worker safety study</h1>
      <button ref={startButton} data-tour="start" className="btn" disabled={busy || starting || (manual && paused && !ready)} onClick={() => paused ? void start() : void command("/pause", { reason: "operator" })}>
        {starting ? "Starting..." : manual && paused && !ready ? "Loading..." : paused ? s.elapsed === 0 ? "Start" : "Resume" : "Pause"}
      </button>
    </header>
    {error && <p className="console-error" role="alert">{error} {paused && <button onClick={() => { setError(""); setReady(false); video.current?.load(); }}>Retry video</button>}</p>}
    <div className="console-body">
      {manual && cameraView}
      <aside className="console-sidebar">{manual ? <>
        <p>Select a camera, then mark a possible fall.</p>
        <button data-tour="mark" className="console-mark" disabled={disabled || !camera} onClick={() => void command("/reports", { camera, time: Math.floor(s.elapsed) })}>Mark possible fall</button>
        <section className="console-events"><h2>Events ({s.reports.length})</h2>
          {s.reports.length === 0 ? <p>No events marked.</p> : <table><thead><tr><th>Camera</th><th>Time</th></tr></thead><tbody>
            {s.reports.map((report, index) => <tr key={index}><td>{String(s.feeds.findIndex(feed => feed.id === report.camera) + 1).padStart(2, "0")}</td><td>{clock(report.reported_at)}</td></tr>)}
          </tbody></table>}
        </section>
      </> : <AutomatedCondition session={s} busy={busy} command={command} audio={alertAudio.current} cameraView={cameraView} tourTarget={tourStep?.target} view={automatedView} setView={setAutomatedView} />}
        <footer className="console-controls">{onFacilitator && <button disabled={!paused || busy} onClick={onFacilitator}>Study setup</button>}<button disabled={busy} onClick={onWithdraw}>Withdraw</button></footer>
      </aside>
    </div>
    {tourStep && tourIndex !== null && <RoundTour step={tourStep} index={tourIndex} total={tourSteps.length} part={s.condition_number}
      root={consoleRef} onBack={() => setTourIndex(tourIndex - 1)} onNext={advanceTour} />}
  </section>;
}

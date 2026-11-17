"""Facilitator and scoped participant endpoints for replay-based user studies."""
import csv
import hashlib
import io
import json
import os
import secrets
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response as FastAPIResponse
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRoute

from .models import Consent, Decision, Enrollment, Manifest, Note, Pause, Report, ReportNote, Survey, TaskAnswer
from .service import SEQUENCES, current, log, public, require, results, task, task_result, tick, video_set
from .storage import Store

class PrivateRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def private_response(request):
            response = await handler(request)
            response.headers["Cache-Control"] = "private, no-store"
            response.headers["Vary"] = "Cookie, Authorization"
            return response

        return private_response


router = APIRouter(prefix="/api/study", tags=["user study"], route_class=PrivateRoute)


def store(request: Request):
    if not hasattr(request.app.state, "study_store"):
        directory = os.environ.get("FALL_STUDY_DIR", str(Path(__file__).resolve().parents[2] / "data" / "study"))
        request.app.state.study_store = Store(directory)
    return request.app.state.study_store



def participant(request: Request, authorization: str = Header(default=""), study_token: str = Cookie(default=""), data=Depends(store)):
    token = authorization.removeprefix("Bearer ") or study_token
    digest = hashlib.sha256(token.encode()).hexdigest()
    with data.transaction() as db:
        row = db.execute("SELECT body FROM sessions WHERE token=?", (digest,)).fetchone()
        if not row:
            raise HTTPException(401, "Invalid or withdrawn participant session")
        session = json.loads(row["body"])
        tick(session)
        yield session
        Store.save(db, session)


@router.get("/config")
def get_config(data=Depends(store)):
    with data.transaction() as db:
        row = db.execute("SELECT body FROM config WHERE id=1").fetchone()
        return json.loads(row["body"]) if row else None


@router.put("/config")
def put_config(manifest: Manifest, data=Depends(store)):
    import cv2

    paths = set()
    for video_set_ in manifest.sets:
        from .models import Feed
        files = list(video_set_.feeds)
        extra = [f.pipeline_file for f in video_set_.feeds if f.pipeline_file]
        extra.extend(f for f in (video_set_.wall_file, video_set_.pipeline_wall_file) if f)
        files.extend(Feed(id="pipeline", name="Pipeline", file=f, duration=manifest.duration) for f in extra)
        for feed in files:
            path = (data.media / feed.file).resolve()
            if not path.is_relative_to(data.media.resolve()) or not path.is_file():
                raise HTTPException(422, f"Study media missing: {feed.file}. Place raw footage inside the study media directory.")
            if path in paths:
                raise HTTPException(422, "Each camera and condition must use distinct media")
            paths.add(path)
            video = cv2.VideoCapture(str(path))
            try:
                fps = video.get(cv2.CAP_PROP_FPS)
                frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
                if not video.isOpened() or fps <= 0 or frames / fps + 0.1 < manifest.duration:
                    raise HTTPException(422, f"Media is unreadable or too short: {feed.file}")
            finally:
                video.release()
    for video_set_ in manifest.sets:
        for alert in video_set_.alerts:
            if alert.review_file:
                path = (data.media / alert.review_file).resolve()
                if not path.is_relative_to(data.media.resolve()) or not path.is_file():
                    raise HTTPException(422, "Incident replay media missing")
    body = manifest.model_dump()
    with data.transaction() as db:
        db.execute("INSERT OR REPLACE INTO config VALUES (1, ?)", (json.dumps(body),))
    return {"saved": True, "mode": manifest.mode}


@router.post("/sessions", status_code=201)
def enroll(enrollment: Enrollment, request: Request, response: FastAPIResponse, data=Depends(store)):
    with data.transaction() as db:
        row = db.execute("SELECT body FROM config WHERE id=1").fetchone()
        require(row is not None, "Configure study footage first")
        existing = [json.loads(r["body"]) for r in db.execute("SELECT body FROM sessions")]
        code = enrollment.code
        if code is None:
            used_codes = {s["code"] for s in existing}
            code = "P_" + secrets.token_hex(4)
            while code in used_codes:
                code = "P_" + secrets.token_hex(4)
        require(not any(s["code"] == code for s in existing), "Participant code already exists")
        assignments = (0, 2)  # Manual first; alternate which footage set is used.
        counts = {i: sum(s["group"] == enrollment.group and s["manifest"]["mode"] == json.loads(row["body"])["mode"] and s["sequence"] == i for s in existing) for i in assignments}
        sequence = min(assignments, key=lambda i: (counts[i], i))
        token = secrets.token_urlsafe(32)
        s = {**enrollment.model_dump(), "code": code, "id": uuid.uuid4().hex, "sequence": sequence,
             "manifest": json.loads(row["body"]), "stage": "instructions", "created_at": time.time(),
             "consent": None, "runs": [], "survey": None, "notes": [], "log": []}
        log(s, "enrolled")
        db.execute("INSERT INTO sessions VALUES (?, ?, ?)", (s["id"], hashlib.sha256(token.encode()).hexdigest(), json.dumps(s)))
    response.set_cookie("study_token", token, httponly=True, samesite="strict", secure=request.url.scheme == "https", path="/api/study/session")
    return {"token": token, "session": public(s)}


@router.get("/session")
def state(s=Depends(participant, scope="function")):
    return public(s)


@router.post("/session/consent")
def consent(body: Consent, s=Depends(participant, scope="function")):
    require(s["stage"] == "instructions", "Consent must precede the conditions")
    s["consent"] = {**body.model_dump(), "wall_time": time.time()}
    log(s, "consent")
    return public(s)


@router.post("/session/start")
def start(s=Depends(participant, scope="function")):
    require(s["consent"] is not None and s["stage"] in {"instructions", "intermission"}, "Complete consent or the previous condition first")
    index = len(s["runs"])
    condition, set_id = SEQUENCES[s["sequence"]][index]
    s["runs"].append({"index": index, "condition": condition, "set": set_id, "elapsed": 0,
                      "last_tick": time.monotonic(), "completed": False, "reports": [], "alerts": [], "task_answers": []})
    # Do not start time until the browser has loaded every feed.
    s["stage"] = "paused"
    log(s, "condition_prepared", run=index)
    return public(s)


@router.post("/session/resume")
def resume(s=Depends(participant, scope="function")):
    require(s["stage"] == "paused", "Condition is not paused")
    current(s)["last_tick"] = time.monotonic()
    s["stage"] = "running"
    log(s, "resumed", study_time=current(s)["elapsed"])
    return public(s)


@router.post("/session/pause")
def pause(body: Pause, s=Depends(participant, scope="function")):
    if s["stage"] == "running":
        s["stage"] = "paused"
        log(s, "paused", reason=body.reason, study_time=current(s)["elapsed"])
    return public(s)


@router.post("/session/reports", status_code=201)
def report(body: Report, s=Depends(participant, scope="function")):
    require(s["stage"] == "running" and current(s)["condition"] == "manual", "Manual reporting is unavailable")
    run = current(s)
    require(body.camera in {f["id"] for f in video_set(s, run)["feeds"]}, "Unknown camera")
    require(body.time <= run["elapsed"], "Incident time cannot be in the future")
    event = {**body.model_dump(), "reported_at": run["elapsed"], "wall_time": time.time()}
    run["reports"].append(event)
    log(s, "manual_report", **event)
    return public(s)


@router.post("/session/reports/{report_index}/notes")
def report_note(report_index: int, body: ReportNote, s=Depends(participant, scope="function")):
    require(bool(s["runs"]) and 0 <= report_index < len(current(s)["reports"]), "Report not found")
    current(s)["reports"][report_index]["notes"] = body.text
    log(s, "report_note", report=report_index, notes=body.text)
    return public(s)


@router.post("/session/alerts/{alert_id}/acknowledge")
def acknowledge(alert_id: int, s=Depends(participant, scope="function")):
    require(s["stage"] == "running" and current(s)["condition"] == "automated", "Automated review is unavailable")
    alert = next((a for a in current(s)["alerts"] if a["id"] == alert_id), None)
    require(alert is not None, "Alert has not been emitted")
    if alert["acknowledged_at"] is None:
        alert["acknowledged_at"] = current(s)["elapsed"]
        log(s, "acknowledged", alert=alert_id, study_time=current(s)["elapsed"])
    return public(s)


@router.post("/session/alerts/{alert_id}/decision")
def decide(alert_id: int, body: Decision, s=Depends(participant, scope="function")):
    require(s["stage"] == "running", "Condition is not running")
    alert = next((a for a in current(s)["alerts"] if a["id"] == alert_id), None)
    require(alert is not None and alert["acknowledged_at"] is not None, "Acknowledge the alert before choosing an action")
    require(alert["decision"] is None, "Decision already recorded")
    alert["decision"] = {"action": body.action, "time": current(s)["elapsed"]}
    log(s, "decision", alert=alert_id, action=body.action, study_time=current(s)["elapsed"])
    return public(s)


@router.post("/session/task")
def answer(body: TaskAnswer, s=Depends(participant, scope="function")):
    require(s["stage"] == "running" and current(s)["condition"] == "automated", "Task is unavailable")
    prompt = task(s)
    require(body.index == prompt["index"], "This question has already changed")
    current(s)["task_answers"].append({**prompt, "answer": body.answer, "correct": body.answer == task_result(prompt), "time": current(s)["elapsed"]})
    return public(s)


@router.post("/session/survey")
def survey(body: Survey, s=Depends(participant, scope="function")):
    require(s["stage"] == "survey", "Complete both conditions before the survey")
    s["survey"] = body.model_dump()
    s["stage"] = "completed"
    log(s, "survey_completed")
    return public(s)


@router.delete("/session", status_code=204)
def withdraw(response: FastAPIResponse, authorization: str = Header(default=""), study_token: str = Cookie(default=""), data=Depends(store)):
    digest = hashlib.sha256((authorization.removeprefix("Bearer ") or study_token).encode()).hexdigest()
    with data.transaction() as db:
        db.execute("DELETE FROM sessions WHERE token=?", (digest,))
    response.delete_cookie("study_token", path="/api/study/session")


@router.get("/session/media/{session_id}/{run_index}/{camera}")
def media(session_id: str, run_index: int, camera: str, request: Request, s=Depends(participant, scope="function"), data=Depends(store)):
    require(s["stage"] in {"running", "paused"}, "Media unavailable outside conditions")
    require(session_id == s["id"] and run_index == current(s)["index"], "Media belongs to a different session or condition")
    feed = next((f for f in video_set(s, current(s))["feeds"] if f["id"] == camera), None)
    if not feed:
        raise HTTPException(404, "Camera not found")
    filename = feed.get("pipeline_file") if current(s)["condition"] == "automated" else None
    path = (data.media / (filename or feed["file"])).resolve()
    if not path.is_relative_to(data.media.resolve()) or not path.is_file():
        raise HTTPException(404, "Study media unavailable")
    return FileResponse(path)


@router.get("/session/mosaic/{session_id}/{run_index}")
def mosaic(session_id: str, run_index: int, s=Depends(participant, scope="function"), data=Depends(store)):
    require(s["stage"] in {"running", "paused"} and s["id"] == session_id and current(s)["index"] == run_index, "Camera wall unavailable")
    config = video_set(s, current(s))
    filename = config.get("pipeline_wall_file") if current(s)["condition"] == "automated" else None
    filename = filename or config.get("wall_file")
    require(filename is not None, "Camera wall has not been prepared")
    path = (data.media / filename).resolve()
    if not path.is_relative_to(data.media.resolve()) or not path.is_file():
        raise HTTPException(404, "Camera wall media unavailable")
    return FileResponse(path)


@router.get("/sessions")
def list_sessions(data=Depends(store)):
    with data.transaction() as db:
        sessions = [json.loads(r["body"]) for r in db.execute("SELECT body FROM sessions")]
    return [{"id": s["id"], "code": s["code"], "group": s["group"], "stage": s["stage"], "mode": s["manifest"]["mode"]} for s in sessions]


@router.post("/sessions/{session_id}/notes")
def note(session_id: str, body: Note, data=Depends(store)):
    with data.transaction() as db:
        row = db.execute("SELECT body FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Session not found")
        s = json.loads(row["body"])
        s["notes"].append({"text": body.text, "wall_time": time.time()})
        Store.save(db, s)
    return {"saved": True}


@router.get("/export")
def export(format: str = "json", data=Depends(store)):
    with data.transaction() as db:
        sessions = [results(json.loads(r["body"])) for r in db.execute("SELECT body FROM sessions")]
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["participant", "group", "mode", "stage", "sequence", "condition", "video_set", "completed", "detected", "missed", "false_alarms", "precision", "recall", "mean_latency_seconds"])
        for s in sessions:
            for run in s["runs"]:
                m = run["metrics"] or {}
                writer.writerow([s["code"], s["group"], s["mode"], s["stage"], s["sequence"], run["condition"], run["set"], run["completed"], *[m.get(k) for k in ("detected", "missed", "false_alarms", "precision", "recall", "mean_latency")]])
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="user-study.csv"'})
    if format != "json":
        raise HTTPException(422, "Use json or csv")
    return {"evaluation": "prerecorded replay; video-time detection latency excludes live processing and delivery delay",
            "sessions": sessions, "summary": summarize(sessions)}


def summarize(sessions):
    summary = []
    for group in ("combined", "oxy", "student"):
        for condition in ("manual", "automated"):
            runs = [r for s in sessions if s["stage"] == "completed" and s["mode"] == "study" and (group == "combined" or s["group"] == group)
                    for r in s["runs"] if r["completed"] and r["condition"] == condition]
            detected = sum(r["metrics"]["detected"] for r in runs)
            missed = sum(r["metrics"]["missed"] for r in runs)
            false = sum(r["metrics"]["false_alarms"] for r in runs)
            latencies = [v for r in runs for v in r["metrics"]["latencies"]]
            summary.append({"group": group, "condition": condition, "participants": len(runs), "detected": detected, "missed": missed,
                            "false_alarms": false, "precision": detected / (detected + false) if detected + false else None,
                            "recall": detected / (detected + missed) if detected + missed else None,
                            "mean_latency": sum(latencies) / len(latencies) if latencies else None})
    return summary


@router.get("/session/review/{session_id}/{run_index}/{alert_id}")
def review_media(session_id: str, run_index: int, alert_id: int, s=Depends(participant, scope="function"), data=Depends(store)):
    require(s["stage"] in {"running", "paused"} and s["id"] == session_id and current(s)["index"] == run_index and current(s)["condition"] == "automated", "Incident review unavailable")
    alert = next((a for a in current(s)["alerts"] if a["id"] == alert_id), None)
    require(alert is not None and alert.get("review_file"), "Incident replay unavailable")
    path = (data.media / alert["review_file"]).resolve()
    if not path.is_relative_to(data.media.resolve()) or not path.is_file():
        raise HTTPException(404, "Incident replay unavailable")
    return FileResponse(path)

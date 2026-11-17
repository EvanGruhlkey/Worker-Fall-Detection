"""Study clock and state transitions. All mutations run inside a storage transaction."""
import time
from fastapi import HTTPException
from .scoring import score

SEQUENCES = [(("manual", "A"), ("automated", "B")),
             (("automated", "B"), ("manual", "A")),
             (("manual", "B"), ("automated", "A")),
             (("automated", "A"), ("manual", "B"))]


def require(condition, message):
    if not condition:
        raise HTTPException(409, message)


def current(s):
    return s["runs"][-1]


def video_set(s, run):
    return next(v for v in s["manifest"]["sets"] if v["id"] == run["set"])


def log(s, kind, **data):
    s["log"].append({"kind": kind, "wall_time": time.time(), **data})


def tick(s):
    if s["stage"] != "running":
        return
    run = current(s)
    now = time.monotonic()
    gap = max(0, now - run["last_tick"])
    if gap > 3 or now < run["last_tick"]:
        s["stage"] = "paused"
        log(s, "paused", reason="connection_gap", study_time=run["elapsed"], last_contact_monotonic=run["last_tick"])
        return
    run["elapsed"] = min(s["manifest"]["duration"], run["elapsed"] + gap)
    run["last_tick"] = now
    if run["condition"] == "automated":
        for i, event in enumerate(video_set(s, run)["alerts"]):
            if event["time"] <= run["elapsed"] and not any(a["id"] == i for a in run["alerts"]):
                run["alerts"].append({**event, "id": i, "emitted_at": run["elapsed"], "acknowledged_at": None, "decision": None})
                log(s, "alert", run=run["index"], alert=i, study_time=run["elapsed"], scheduled_time=event["time"])
    if run["elapsed"] >= s["manifest"]["duration"]:
        run["completed"] = True
        s["stage"] = "intermission" if len(s["runs"]) == 1 else "survey"
        log(s, "condition_completed", run=run["index"])


def task(s):
    answers = current(s)["task_answers"]
    n = len(answers)
    operator = ("+", "-", "*", "/")[n % 4]
    left, right = 11 + (n * 7) % 79, 13 + (n * 11) % 67
    if operator == "-":
        left, right = max(left, right), min(left, right)
    elif operator == "*":
        left, right = 2 + (n * 7) % 11, 2 + (n * 11) % 11
    elif operator == "/":
        right = 2 + (n * 5) % 11
        left = right * (2 + (n * 3) % 11)
    return {"index": n, "left": left, "right": right, "operator": operator,
            "score": sum(answer["correct"] for answer in answers)}


def task_result(prompt):
    left, right = prompt["left"], prompt["right"]
    match prompt["operator"]:
        case "+": return left + right
        case "-": return left - right
        case "*": return left * right
        case "/": return left // right
    raise ValueError("Unknown arithmetic operator")


def public(s):
    run = current(s) if s["runs"] else None
    feeds = []
    if run and s["stage"] in {"running", "paused"}:
        feeds = [{"id": f["id"], "name": f["name"], "url": f'/api/study/session/media/{s["id"]}/{run["index"]}/{f["id"]}'} for f in video_set(s, run)["feeds"]]
    return {"id": s["id"], "code": s["code"], "stage": s["stage"], "mode": s["manifest"]["mode"],
            "duration": s["manifest"]["duration"], "condition": run["condition"] if run else None,
            "condition_number": len(s["runs"]), "elapsed": run["elapsed"] if run else 0,
            "wall_url": (f'/api/study/session/mosaic/{s["id"]}/{run["index"]}' if run and feeds and video_set(s, run).get("wall_file") else None),
            "feeds": feeds, "reports": run["reports"] if run else [],
            "alerts": [{**{k: v for k, v in a.items() if k != "review_file"}, "review_url": (f'/api/study/session/review/{s["id"]}/{run["index"]}/{a["id"]}' if a.get("review_file") else None)} for a in run["alerts"]] if run else [],
            "task": task(s) if run and run["condition"] == "automated" else None,
            "consented": s["consent"] is not None}


def results(s):
    rows = []
    for run in s["runs"]:
        config = s["manifest"]
        detections = run["reports"] if run["condition"] == "manual" else video_set(s, run)["alerts"]
        metrics = score(video_set(s, run)["truth"], detections, config["early_tolerance"], config["late_tolerance"])
        acknowledgments = [a["acknowledged_at"] - a["emitted_at"] for a in run["alerts"] if a["acknowledged_at"] is not None]
        rows.append({"condition": run["condition"], "set": run["set"], "completed": run["completed"],
                     "elapsed": run["elapsed"], "metrics": metrics if run["completed"] else None,
                     "acknowledgment_latencies": acknowledgments, "raw": run})
    return {"id": s["id"], "code": s["code"], "group": s["group"], "stage": s["stage"],
            "mode": s["manifest"]["mode"], "age_range": s["age_range"], "experience": s["experience"],
            "sequence": s["sequence"], "manifest": s["manifest"], "consent": s["consent"],
            "created_at": s["created_at"], "survey": s["survey"], "notes": s["notes"], "log": s["log"], "runs": rows}

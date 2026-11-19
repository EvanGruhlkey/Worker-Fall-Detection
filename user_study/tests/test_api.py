import json
import pytest
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from user_study.backend.router import router
from user_study.backend.storage import Store


@pytest.fixture
def study(tmp_path):
    app = FastAPI()
    app.state.study_store = Store(tmp_path)
    app.include_router(router)
    client = TestClient(app)
    manifest = {"title": "Test", "mode": "demo", "duration": 10, "detector_version": "test", "alert_source": "fixture",
                "sets": [{"id": key, "feeds": [{"id": f"c{i}", "name": f"Camera {i}", "file": f"{key}{i}.mp4", "duration": 10} for i in range(6)],
                          "truth": [{"camera": "c0", "time": 3}], "alerts": [{"camera": "c0", "time": 4}]} for key in ("A", "B")]}
    for vs in manifest["sets"]:
        for feed in vs["feeds"]:
            writer = cv2.VideoWriter(str(tmp_path / "media" / feed["file"]), cv2.VideoWriter_fourcc(*"mp4v"), 2, (32, 32))
            for _ in range(20):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
    admin = {}
    assert client.put("/api/study/config", json=manifest, headers=admin).status_code == 200
    return client, app.state.study_store, admin, manifest


def enroll(client, admin, code="P01", group=None):
    payload = {"code": code, "age_range": "18-24", "experience": "none"}
    if group is not None:
        payload["group"] = group
    response = client.post("/api/study/sessions", headers=admin, json=payload)
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["token"]}, response.json()["session"]["id"]


def advance(store, session_id, seconds):
    with store.transaction() as db:
        s = json.loads(db.execute("SELECT body FROM sessions WHERE id=?", (session_id,)).fetchone()["body"])
        s["runs"][-1]["elapsed"] += seconds
        Store.save(db, s)


def consent_and_start(client, headers):
    assert client.post("/api/study/session/consent", headers=headers, json=dict(adult=True, vision=True, eligible=True, voluntary=True)).status_code == 200
    assert client.post("/api/study/session/start", headers=headers).json()["stage"] == "paused"
    assert client.post("/api/study/session/resume", headers=headers).status_code == 200


def start_automated(client, store, headers, sid):
    consent_and_start(client, headers)
    assert client.get("/api/study/session", headers=headers).json()["condition"] == "manual"
    advance(store, sid, 11)
    assert client.get("/api/study/session", headers=headers).json()["stage"] == "intermission"
    assert client.post("/api/study/session/start", headers=headers).json()["condition"] == "automated"
    assert client.post("/api/study/session/resume", headers=headers).status_code == 200


def test_auth_and_participant_separation(study):
    client, store, admin, manifest = study
    assert client.get("/api/study/config").status_code == 200
    headers, sid = enroll(client, admin)
    body = client.get("/api/study/session", headers=headers).json()
    assert "truth" not in json.dumps(body) and "manifest" not in body
    assert client.get("/api/study/export", headers=headers).status_code == 200
    assert client.get(f"/api/study/session/media/{sid}/0/c0", headers=headers).status_code == 409
    assert client.post("/api/study/session/start", headers=headers).status_code == 409
    assert client.get("/api/study/session", headers={"Authorization": "Bearer invalid"}).status_code == 401


def test_full_session_scoring_and_withdrawal(study):
    client, store, admin, _ = study
    headers, sid = enroll(client, admin)
    consent_and_start(client, headers)
    advance(store, sid, 4)
    assert client.post("/api/study/session/reports", headers=headers, json={"camera": "c0", "time": 3}).status_code == 201
    assert client.post("/api/study/session/reports", headers=headers, json={"camera": "c0", "time": 99}).status_code == 409
    assert client.get("/api/study/session", headers=headers).json()["alerts"] == []
    advance(store, sid, 11)
    assert client.get("/api/study/session", headers=headers).json()["stage"] == "intermission"
    assert client.post("/api/study/session/start", headers=headers).status_code == 200
    assert client.post("/api/study/session/resume", headers=headers).status_code == 200
    advance(store, sid, 5)
    state = client.get("/api/study/session", headers=headers).json()
    assert len(state["alerts"]) == 1
    assert client.post("/api/study/session/alerts/0/decision", headers=headers, json={"action": "investigate"}).status_code == 409
    assert client.post("/api/study/session/alerts/0/acknowledge", headers=headers).status_code == 200
    assert client.post("/api/study/session/alerts/0/decision", headers=headers, json={"action": "investigate"}).status_code == 200
    assert client.post("/api/study/session/task", headers=headers, json={"index": 0, "answer": 24}).status_code == 200
    advance(store, sid, 11)
    assert client.get("/api/study/session", headers=headers).json()["stage"] == "survey"
    survey = dict(manual_confidence=3, manual_effort=4, manual_ease=2, automated_confidence=4, automated_effort=2, automated_ease=4, trust=4, feedback="test")
    assert client.post("/api/study/session/survey", headers=headers, json=survey).json()["stage"] == "completed"
    export = client.get("/api/study/export", headers=admin).json()
    assert [r["metrics"]["detected"] for r in export["sessions"][0]["runs"]] == [1, 1]
    assert all(r["participants"] == 0 for r in export["summary"]), "Demo must not enter real results"
    assert "mean_latency_seconds" in client.get("/api/study/export?format=csv", headers=admin).text
    assert client.delete("/api/study/session", headers=headers).status_code == 204
    assert client.get("/api/study/session", headers=headers).status_code == 401
    assert client.get("/api/study/export", headers=admin).json()["sessions"] == []


def test_balancing_validation_and_pause(study):
    client, store, admin, manifest = study
    for i in range(4):
        _, sid = enroll(client, admin, code=f"P{i}")
    data = client.get("/api/study/export", headers=admin).json()
    assert sorted(s["sequence"] for s in data["sessions"]) == [0, 0, 2, 2]
    manifest["sets"][0]["feeds"][0]["file"] = "../secret.mp4"
    assert client.put("/api/study/config", headers=admin, json=manifest).status_code == 422
    headers, sid = enroll(client, admin, code="P5", group="faculty")
    with store.transaction() as db:
        enrolled = json.loads(db.execute("SELECT body FROM sessions WHERE id=?", (sid,)).fetchone()["body"])
        assert enrolled["group"] == "student"
    consent_and_start(client, headers)
    advance(store, sid, 2)
    before = client.post("/api/study/session/pause", headers=headers, json={"reason": "hidden"}).json()["elapsed"]
    with store.transaction() as db:
        paused = json.loads(db.execute("SELECT body FROM sessions WHERE id=?", (sid,)).fetchone()["body"])
        paused["runs"][-1]["last_tick"] -= 100
        Store.save(db, paused)
    assert client.get("/api/study/session", headers=headers).json()["elapsed"] == before
    assert client.post("/api/study/session/survey", headers=headers, json={}).status_code == 422


def test_media_scope_cookie_and_recovery(study):
    client, store, admin, manifest = study
    headers, sid = enroll(client, admin)
    consent_and_start(client, headers)
    state = client.get("/api/study/session").json()
    media_url = state["feeds"][0]["url"]
    response = client.get(media_url)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert client.get(media_url.replace(sid, "wrong")).status_code == 409
    assert client.get(media_url.replace("/0/", "/1/")).status_code == 409
    with store.transaction() as db:
        s = json.loads(db.execute("SELECT body FROM sessions WHERE id=?", (sid,)).fetchone()["body"])
        s["runs"][-1]["last_tick"] -= 10
        Store.save(db, s)
    state = client.get("/api/study/session", headers=headers).json()
    assert state["stage"] == "paused"
    assert state["elapsed"] < 1
    assert client.get("/api/study/export", headers=admin).json()["sessions"][0]["log"][-1]["reason"] == "connection_gap"


def test_invalid_real_duration_and_media_length(study):
    client, store, admin, manifest = study
    manifest["mode"] = "study"
    assert client.put("/api/study/config", headers=admin, json=manifest).status_code == 422
    manifest["duration"] = 120
    for vs in manifest["sets"]:
        for feed in vs["feeds"]:
            feed["duration"] = 120
    response = client.put("/api/study/config", headers=admin, json=manifest)
    assert response.status_code == 422
    assert "too short" in response.json()["detail"]


def test_recorded_pilot_keeps_real_alerts_and_stays_out_of_summary(study):
    client, store, admin, manifest = study
    manifest["mode"] = "pilot"
    manifest["detector_version"] = "recorded-checkpoint"
    manifest["alert_source"] = "actual pipeline output"
    manifest["sets"][0]["alerts"] = []
    assert client.put("/api/study/config", headers=admin, json=manifest).status_code == 200
    headers, sid = enroll(client, admin)
    consent_and_start(client, headers)
    advance(store, sid, 11)
    assert client.get("/api/study/session", headers=headers).json()["mode"] == "pilot"
    exported = client.get("/api/study/export", headers=admin).json()
    assert exported["sessions"][0]["manifest"]["sets"][0]["alerts"] == []
    assert exported["sessions"][0]["runs"][0]["metrics"]["missed"] == 1
    assert all(row["participants"] == 0 for row in exported["summary"])


def test_parallel_camera_ranges_and_polling(study):
    from concurrent.futures import ThreadPoolExecutor
    client, store, admin, manifest = study
    headers, sid = enroll(client, admin)
    consent_and_start(client, headers)
    urls = [f"/api/study/session/media/{sid}/0/c{i % 6}" for i in range(9)] + ["/api/study/session"] * 3
    def read(url):
        response = client.get(url, headers={**headers, "Range": "bytes=0-127"})
        assert response.status_code in (200, 206), response.text
        return response.status_code
    with ThreadPoolExecutor(max_workers=12) as executor:
        statuses = list(executor.map(read, urls))
    assert len(statuses) == 12
    assert client.post("/api/study/session/pause", headers=headers, json={"reason": "operator"}).status_code == 200


def test_event_notes_preserve_original_response_time(study):
    client, store, admin, manifest = study
    headers, sid = enroll(client, admin)
    consent_and_start(client, headers)
    advance(store, sid, 4)
    response = client.post("/api/study/session/reports", headers=headers, json={"camera": "c0", "time": 3})
    original = response.json()["reports"][0]
    assert client.post("/api/study/session/pause", headers=headers, json={"reason": "operator"}).status_code == 200
    updated = client.post("/api/study/session/reports/0/notes", headers=headers, json={"text": "Worker beside the ladder"})
    assert updated.status_code == 200
    event = updated.json()["reports"][0]
    assert event["notes"] == "Worker beside the ladder"
    assert event["reported_at"] == original["reported_at"]
    assert event["time"] == 3
    assert client.post("/api/study/session/reports/99/notes", headers=headers, json={"text": "invalid"}).status_code == 409
    assert client.post("/api/study/session/reports/0/notes", headers=headers, json={"text": "x" * 501}).status_code == 422
    raw = client.get("/api/study/export", headers=admin).json()["sessions"][0]["runs"][0]["raw"]["reports"][0]
    assert raw["notes"] == "Worker beside the ladder"


def test_synchronized_wall_media_is_scoped_and_private(study):
    import shutil
    client, store, admin, manifest = study
    for video_set in manifest["sets"]:
        name = video_set["id"] + "_wall.mp4"
        shutil.copyfile(store.media / video_set["feeds"][0]["file"], store.media / name)
        video_set["wall_file"] = name
    assert client.put("/api/study/config", headers=admin, json=manifest).status_code == 200
    headers, sid = enroll(client, admin)
    consent_and_start(client, headers)
    state = client.get("/api/study/session", headers=headers).json()
    response = client.get(state["wall_url"], headers={**headers, "Range": "bytes=0-127"})
    assert response.status_code == 206
    assert response.headers["cache-control"] == "private, no-store"
    assert client.get(state["wall_url"].replace(sid, "wrong"), headers=headers).status_code == 409
    assert client.get(state["wall_url"], headers={"Authorization": "Bearer invalid"}).status_code == 401
    manifest["sets"][0]["wall_file"] = "../private.mp4"
    assert client.put("/api/study/config", headers=admin, json=manifest).status_code == 422


def test_arithmetic_cycles_operations_and_rejects_stale_answers(study):
    client, store, admin, manifest = study
    enroll(client, admin, code="manual-first")
    headers, sid = enroll(client, admin, code="arithmetic-second")
    start_automated(client, store, headers, sid)
    expected = [(11, 13, "+", 24), (24, 18, "-", 6), (5, 2, "*", 10), (66, 6, "/", 11)]
    for index, (left, right, operator, answer) in enumerate(expected):
        prompt = client.get("/api/study/session", headers=headers).json()["task"]
        assert (prompt["left"], prompt["right"], prompt["operator"]) == (left, right, operator)
        response = client.post("/api/study/session/task", headers=headers, json={"index": index, "answer": answer})
        assert response.status_code == 200
        assert response.json()["task"]["score"] == index + 1
    assert client.post("/api/study/session/task", headers=headers, json={"index": 3, "answer": 11}).status_code == 409
    raw = client.get("/api/study/export", headers=admin).json()["sessions"][1]["runs"][1]["raw"]["task_answers"]
    assert len(raw) == 4
    assert all(answer["correct"] for answer in raw)

def test_pipeline_media_only_in_automated_condition(study):
    client, store, admin, manifest = study
    for vs in manifest['sets']:
        for feed in vs['feeds']:
            feed['pipeline_file'] = 'pipeline_' + feed['file']
            writer = cv2.VideoWriter(str(store.media / feed['pipeline_file']), cv2.VideoWriter_fourcc(*'mp4v'), 2, (32, 32))
            for _ in range(20):
                writer.write(np.full((32, 32, 3), 255, dtype=np.uint8))
            writer.release()
        import shutil
        vs['wall_file'] = vs['id'] + '_raw_wall.mp4'
        vs['pipeline_wall_file'] = vs['id'] + '_overlay_wall.mp4'
        shutil.copyfile(store.media / vs['feeds'][0]['file'], store.media / vs['wall_file'])
        shutil.copyfile(store.media / vs['feeds'][0]['pipeline_file'], store.media / vs['pipeline_wall_file'])
    assert client.put('/api/study/config', headers=admin, json=manifest).status_code == 200
    for code, condition in [('raw', 'manual'), ('overlay', 'automated')]:
        headers, sid = enroll(client, admin, code=code)
        if condition == 'automated':
            start_automated(client, store, headers, sid)
        else:
            consent_and_start(client, headers)
        state = client.get('/api/study/session', headers=headers).json()
        assert state['condition'] == condition
        vs = next(v for v in manifest['sets'] if v['id'] == 'A')
        expected = vs['feeds'][0]['pipeline_file' if condition == 'automated' else 'file']
        for url in [state['feeds'][0]['url'], state['wall_url']]:
            assert client.get(url, headers=headers).content == (store.media / expected).read_bytes()
    manifest['sets'][0]['feeds'][0]['pipeline_file'] = '../private.mp4'
    assert client.put('/api/study/config', headers=admin, json=manifest).status_code == 422

def test_incident_replay_is_scoped_to_emitted_automated_alert(study):
    import shutil
    client, store, admin, manifest = study
    for vs in manifest['sets']:
        name = vs['id'] + '_incident.mp4'
        shutil.copyfile(store.media / vs['feeds'][0]['file'], store.media / name)
        vs['alerts'][0]['review_file'] = name
    assert client.put('/api/study/config', headers=admin, json=manifest).status_code == 200
    enroll(client, admin, code='manual')
    headers, sid = enroll(client, admin, code='automated')
    start_automated(client, store, headers, sid)
    url = f'/api/study/session/review/{sid}/1/0'
    assert client.get(url, headers=headers).status_code == 409
    advance(store, sid, 5)
    state = client.get('/api/study/session', headers=headers).json()
    assert state['alerts'][0]['review_url'] == url
    assert 'review_file' not in state['alerts'][0]
    response = client.get(url, headers=headers)
    assert response.content == (store.media / 'A_incident.mp4').read_bytes()
    assert response.headers['cache-control'] == 'private, no-store'
    assert client.get(url.replace('/1/0', '/0/0'), headers=headers).status_code == 409
    assert client.get(url, headers={'Authorization': 'Bearer invalid'}).status_code == 401
    manifest['sets'][0]['alerts'][0]['review_file'] = '../secret.mp4'
    assert client.put('/api/study/config', headers=admin, json=manifest).status_code == 422


def test_every_assignment_runs_manual_then_automated(study):
    client, store, admin, manifest = study
    for i in range(4):
        headers, sid = enroll(client, admin, code=f"order-{i}")
        start_automated(client, store, headers, sid)
    exported = client.get("/api/study/export", headers=admin).json()["sessions"]
    assert [[r["condition"] for r in s["runs"]] for s in exported] == [["manual", "automated"]] * 4
    assert [[r["set"] for r in s["runs"]] for s in exported] == [["A", "B"], ["B", "A"], ["A", "B"], ["B", "A"]]


def test_participant_ids_are_assigned_automatically(study):
    client, store, admin, manifest = study
    body = {"group": "student", "age_range": "18-24", "experience": "none"}
    first = client.post("/api/study/sessions", json=body)
    second = client.post("/api/study/sessions", json=body)
    assert first.status_code == second.status_code == 201
    codes = [response.json()["session"]["code"] for response in (first, second)]
    assert all(code.startswith("P_") and len(code) == 10 for code in codes)
    assert codes[0] != codes[1]
    exported = client.get("/api/study/export").json()["sessions"]
    assert [s["code"] for s in exported] == codes

def test_nine_camera_manifest_and_participant_view(study):
    import shutil
    client, store, admin, manifest = study
    for vs in manifest["sets"]:
        source = store.media / vs["feeds"][0]["file"]
        for index in range(6, 9):
            name = f'{vs["id"]}{index}.mp4'
            shutil.copyfile(source, store.media / name)
            vs["feeds"].append({"id": f"c{index}", "name": f"Camera {index}", "file": name, "duration": 10})
    assert client.put("/api/study/config", json=manifest).status_code == 200
    headers, sid = enroll(client, admin)
    consent_and_start(client, headers)
    state = client.get("/api/study/session", headers=headers).json()
    assert len(state["feeds"]) == 9
    assert client.get(state["feeds"][8]["url"], headers=headers).status_code == 200
    manifest["sets"][0]["feeds"].pop()
    assert client.put("/api/study/config", json=manifest).status_code == 422

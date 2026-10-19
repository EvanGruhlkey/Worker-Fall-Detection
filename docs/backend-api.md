# Backend API

FastAPI backend for fall detection. The frontend is a separate app that talks to it over HTTP/JSON.

## Run it

```bash
uv sync
uv run uvicorn backend.main:app --app-dir src --reload
```

- API base URL: `http://127.0.0.1:8000`
- Interactive docs (try every endpoint in the browser): `http://127.0.0.1:8000/docs`
- CORS allows any origin during development, so a frontend dev server on any port can call it.

Run the tests with `uv run pytest`.

## How it works

1. A **job** is one processing run: an uploaded video or a live camera.
2. Each frame goes through YOLO Pose + ByteTrack (`src/yolo_pose.py`, unchanged), then a fall detector.
3. When a fall is detected, the backend creates a **fall event** with status `pending`, saves a snapshot, and records a short clip (5 s before to 5 s after).
4. A person reviews the event in the frontend and marks it `confirmed` or `false_alarm`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Server is up |
| POST | `/api/videos` | Upload a video (multipart form, field `file`); queues a job |
| POST | `/api/cameras` | Start a live camera: `{"source": "0", "name": "Bay 3"}`; `source` is a camera index or `rtsp://...` URL |
| POST | `/api/cameras/{job_id}/stop` | Stop a live camera |
| GET | `/api/cameras/{job_id}/stream` | Live annotated feed (MJPEG) |
| GET | `/api/jobs` | List jobs, newest first (`?kind=video\|camera`, `?status=...`, `?limit=`) |
| GET | `/api/jobs/{job_id}` | One job (poll this for upload progress) |
| GET | `/api/jobs/{job_id}/video` | Full annotated output video (uploads only) |
| GET | `/api/jobs/{job_id}/json` | Per-frame pose data (same format as `yolo_pose.py`) |
| GET | `/api/events` | Fall log, newest first (`?status=pending\|confirmed\|false_alarm`, `?job_id=`, `?limit=`, `?offset=`) |
| GET | `/api/events/{event_id}` | One event |
| PATCH | `/api/events/{event_id}` | Review: `{"status": "confirmed", "reviewer": "Sam", "notes": "..."}` |
| GET | `/api/events/{event_id}/snapshot` | JPEG of the frame where the fall was detected |
| GET | `/api/events/{event_id}/clip` | MP4 clip around the fall |
| GET | `/api/stats` | Counts for a summary bar |

## Response shapes

**Event** (`GET /api/events` returns `{"items": [Event, ...], "total": 12}`)

```json
{
  "id": 4,
  "job_id": 2,
  "job_name": "site_cam.mp4",
  "track_id": 3,
  "frame_index": 412,
  "video_time": 13.733,
  "detected_at": "2026-10-18T21:34:30Z",
  "confidence": 0.82,
  "reason": "hip dropped 46% of body height in 0.7s; torso 74 deg from vertical",
  "status": "pending",
  "reviewer": null,
  "notes": null,
  "reviewed_at": null,
  "snapshot_url": "/api/events/4/snapshot",
  "clip_url": "/api/events/4/clip"
}
```

- `track_id` is the worker's tracking ID, matching the "ID n" label drawn on the video.
- `video_time` is in seconds from the start of the video (or of the camera session).
- `clip_url` is `null` for about 5 seconds after detection while the post-fall footage is recorded. Re-fetch the event to get it.

**Job**

```json
{
  "id": 2,
  "kind": "video",
  "name": "site_cam.mp4",
  "status": "running",
  "frames_processed": 300,
  "total_frames": 1800,
  "fps": 30.0,
  "fall_count": 1,
  "error": null,
  "created_at": "...", "started_at": "...", "finished_at": null,
  "progress": 0.17,
  "output_video_url": null,
  "output_json_url": null,
  "stream_url": null
}
```

- `status` is one of `queued`, `running`, `completed`, `stopped`, `failed`. When it is `failed`, `error` says why.
- `stream_url` is set only while a camera is running.

**Stats**

```json
{"total_events": 12, "pending": 3, "confirmed": 7, "false_alarm": 2, "active_cameras": 1, "running_jobs": 1}
```

Errors return `{"detail": "message"}` with a 4xx status.

## Frontend tips

- URLs in responses are relative (`/api/...`). Prefix them with the API base URL.
- Use the media URLs directly in HTML. Clips and output videos are H.264, so they play in any browser and support seeking.
  ```html
  <video src="http://127.0.0.1:8000/api/events/4/clip" controls></video>
  <img src="http://127.0.0.1:8000/api/events/4/snapshot">
  <img src="http://127.0.0.1:8000/api/cameras/2/stream">  <!-- live feed -->
  ```
- There is no push notification yet. To show new falls, poll `GET /api/events?limit=1` every few seconds and alert when the newest `id` changes.
- Timestamps are UTC in ISO 8601 format.

## Code layout (`src/backend/`)

| File | Purpose |
|---|---|
| `main.py` | Creates the FastAPI app, CORS, startup/shutdown |
| `routers/jobs.py` | Video upload, camera start/stop/stream, job endpoints |
| `routers/events.py` | Fall log, review, snapshot/clip, stats |
| `schemas.py` | JSON request/response models |
| `db.py` | Database tables (`Job`, `FallEvent`), SQLite at `data/falls.db` |
| `pipeline.py` | Runs frames through the model and fall detector, records events and clips |
| `manager.py` | Background threads: video queue (one at a time) and live cameras |
| `fall_detector.py` | **Placeholder** rule-based fall detector (see below) |
| `media.py` | Writes browser-playable H.264 video using a bundled ffmpeg |
| `config.py` | Settings from environment variables |

## Fall detector (placeholder)

`fall_detector.py` uses a simple rule so the system works end to end. It flags a fall when a tracked person's hips drop by 30% or more of their height within 1.5 s, and they then stay horizontal for 0.5 s. It has not been tuned on real fall footage. When the team has a trained classifier, replace `FallDetector.update(people, timestamp)`. As long as it returns a list of `FallDetection(track_id, confidence, reason)`, nothing else needs to change.

## Settings

| Env variable | Default |
|---|---|
| `FALL_DATA_DIR` | `data/` (uploads in `input/`, outputs in `output/`, event media in `events/`) |
| `FALL_MODEL_PATH` | `models/yolo11n-pose.pt` |
| `FALL_IMAGE_SIZE` | `640` |
| `FALL_CLIP_PRE_SECONDS` / `FALL_CLIP_POST_SECONDS` | `5` / `5` |
| `FALL_CORS_ORIGINS` | `*` |
| `FALL_DATABASE_URL` | SQLite file in the data dir |

## Not done yet

- No login or authentication.
- No push alerts (WebSocket); the frontend polls instead.
- No deleting jobs or events.
- Live camera sessions are not recorded in full, only the clips around detected falls.

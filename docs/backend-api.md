# Backend API

FastAPI backend for fall detection. The frontend is a separate app that talks to it over HTTP/JSON.

For a step-by-step guide to running the backend with the dashboard, and testing each feature, see [running-and-testing.md](running-and-testing.md). This page is the API reference.

## Run it

```bash
uv sync
uv run uvicorn backend.main:app --app-dir src --reload
```

To start the backend and the dashboard together, run `uv run run.py` instead.

- API base URL: `http://127.0.0.1:8000`
- Interactive docs (try every endpoint in the browser): `http://127.0.0.1:8000/docs`
- CORS allows any origin during development, so a frontend dev server on any port can call it.

Run the tests with `uv run pytest`.

## How it works

1. A **job** is one processing run: an uploaded video or a live camera.
2. Each frame goes through YOLO Pose + ByteTrack (`src/detectors/yolo_pose.py`, unchanged), then a fall detector.
3. When a fall is detected, the backend creates a **fall event** with status `unacknowledged`. The event stores the time, camera, location, coordinates and confidence. The backend also saves a snapshot and records a short clip (5 s before to 5 s after).
4. In the dashboard, someone acknowledges the alert (`needs_review`), checks the footage, and closes it as `resolved` (a real fall, handled) or `false_alarm`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Server is up |
| POST | `/api/videos` | Upload a video (multipart form, field `file`); queues a job |
| POST | `/api/cameras` | Start a live camera: `{"source": "0", "name": "Bay 3", "label": "Loading dock", "location": "Plant 2", "coords": "29.76, -95.37"}`; `source` is a camera index or `rtsp://...` URL. Only `source` is required, and `location` and `coords` default to the site settings |
| GET | `/api/cameras/options` | Scan camera indices 0-4 and return the local cameras currently available to the dashboard picker |
| POST | `/api/cameras/{job_id}/stop` | Stop a live camera |
| GET | `/api/cameras/{job_id}/stream` | Live annotated feed (MJPEG) |
| GET | `/api/jobs` | List jobs, newest first (`?kind=video\|camera`, `?status=...`, `?limit=`) |
| GET | `/api/jobs/{job_id}` | One job (poll this for upload progress) |
| GET | `/api/jobs/{job_id}/video` | Full annotated output video (uploads only) |
| GET | `/api/jobs/{job_id}/json` | Per-frame pose data (same format as `yolo_pose.py`) |
| GET | `/api/events` | Fall log, newest first (`?status=unacknowledged\|needs_review\|resolved\|false_alarm`, `?job_id=`, `?limit=`, `?offset=`) |
| GET | `/api/events/{event_id}` | One event |
| PATCH | `/api/events/{event_id}` | Review: `{"status": "resolved", "reviewer": "Sam", "notes": "..."}`. The dashboard labels (`"Needs review"`) work too |
| GET | `/api/events/{event_id}/snapshot` | JPEG of the frame where the fall was detected |
| GET | `/api/events/{event_id}/clip` | MP4 clip around the fall |
| GET | `/api/stats` | Raw counts per status |
| GET | `/api/dashboard` | **Everything the dashboard shows**, in the UI's own types (see below). `?scope=today\|all` (default `today`), `?status=`, `?limit=` |
| GET | `/api/location` | Current site location `{"name", "coords"}` |
| PUT | `/api/location` | Report where the system is: `{"latitude": 29.76, "longitude": -95.37, "accuracy": 30}`. The dashboard sends this from the browser. The backend looks up the place name (OpenStreetMap) and stamps it on new events |
| GET | `/api/dashboard/events.csv` | Event log as a CSV download (`?scope=all\|today`, `?status=`) |

## Response shapes

**Event** (`GET /api/events` returns `{"items": [Event, ...], "total": 12}`)

```json
{
  "id": 4,
  "job_id": 2,
  "job_name": "site_cam.mp4",
  "event_type": "fall",
  "camera_name": "Camera 0",
  "location": "Plant 2, Houston, TX",
  "coords": "29.7604, -95.3698",
  "track_id": 3,
  "frame_index": 412,
  "video_time": 13.733,
  "detected_at": "2026-10-18T21:34:30Z",
  "confidence": 0.82,
  "reason": "hip dropped 46% of body height in 0.7s; torso 74 deg from vertical",
  "status": "unacknowledged",
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
  "label": null, "location": null, "coords": null,
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
{"total_events": 12, "unacknowledged": 2, "needs_review": 1, "resolved": 7, "false_alarm": 2, "active_cameras": 1, "running_jobs": 1}
```

**Dashboard** (`GET /api/dashboard`). Each part has the same shape as the matching type in `ui/src/types.ts`, so it can replace the mock data directly. Extra fields are additions.

```json
{
  "location": {"name": "Plant 2, Houston, TX", "coords": "29.7604, -95.3698"},
  "stats": [
    {"label": "Active incidents", "value": "2", "note": "1 unacknowledged · 1 need review", "tone": "red", "icon": "alert"},
    {"label": "Cameras online", "value": "1", "note": "Camera 0", "tone": "green", "icon": "cam"},
    {"label": "Detection uptime", "value": "99.2%", "note": "Last 24 hours", "tone": "green", "icon": "pulse"},
    {"label": "Camera health", "value": "Good", "note": "Feed clear", "tone": "green", "icon": "shield"}
  ],
  "camera": {"id": "CAM-00", "name": "Camera 0", "label": "Loading dock", "fps": 14, "online": true,
             "job_id": 3, "health": "good", "stream_url": "/api/cameras/3/stream"},
  "events": [
    {"id": 4, "time": "10:42:11 AM", "name": "Fall detected", "icon": "fall", "color": "var(--red)",
     "confidence": 82, "status": "Unacknowledged",
     "detected_at": "2026-10-19T15:42:11Z", "camera": "Camera 0", "location": "Plant 2, Houston, TX",
     "coords": "29.7604, -95.3698", "track_id": 3, "reason": "hip dropped 46% ...",
     "snapshot_url": "/api/events/4/snapshot", "clip_url": "/api/events/4/clip"}
  ]
}
```

`uploads` lists the 5 most recent uploaded videos: `{"id", "name", "status", "progress", "fall_count", "error", "created_at", "output_video_url"}`. Each event also has `source` (`"camera"` or `"video"`), `job_id`, `video_time` (seconds into the video), `reviewer`, `notes` and `reviewed_at`.

How the stat cards are worked out:

- **Active incidents**: events still `unacknowledged` or `needs_review`, across all days. Red while any are open.
- **Cameras online**: live camera sessions running now.
- **Detection uptime**: share of the last 24 h (or since the first camera session, if that was more recent) with at least one camera running. Green at 95% or above, red below 80%.
- **Camera health**: every 15th raw frame is checked. A nearly uniform image means `Obstructed` (lens covered or too dark). Few edges means `Blurry`. With several cameras, the card shows the worst one. The thresholds are rough and live in `pipeline.py`.
- `time` is in the server's local time zone; `detected_at` is UTC.

Errors return `{"detail": "message"}` with a 4xx status.

## Frontend tips

- URLs in responses are relative (`/api/...`). Prefix them with the API base URL.
- Use the media URLs directly in HTML. Clips and output videos are H.264, so they play in any browser and support seeking.
  ```html
  <video src="http://127.0.0.1:8000/api/events/4/clip" controls></video>
  <img src="http://127.0.0.1:8000/api/events/4/snapshot">
  <img src="http://127.0.0.1:8000/api/cameras/2/stream">  <!-- live feed -->
  ```
- There is no push notification yet. To show new falls, poll `GET /api/dashboard` every few seconds and alert when the newest event `id` changes.
- Timestamps are UTC in ISO 8601 format.

## Code layout (`src/backend/`)

| File | Purpose |
|---|---|
| `main.py` | Creates the FastAPI app, CORS, startup/shutdown |
| `routers/jobs.py` | Video upload, camera start/stop/stream, job endpoints |
| `routers/events.py` | Fall log, review, snapshot/clip, stats |
| `routers/location.py`, `location.py` | Browser-reported site location, place-name lookup, which location events use |
| `routers/dashboard.py` | `/api/dashboard`: stored data in the UI's shapes, stat cards, CSV export |
| `schemas.py` | JSON request/response models |
| `db.py` | Database tables (`Job`, `FallEvent`, `SiteLocation`) in the shared PostgreSQL. Migrations in `migrations/` are applied on startup ([database.md](database.md)) |
| `storage.py` | Uploads snapshots, clips and annotated videos to the shared S3-compatible bucket; links to files recorded by other stations |
| `pipeline.py` | Runs frames through the model and fall detector, records events and clips |
| `manager.py` | Background threads: video queue (one at a time) and live cameras |
| `fall_detector.py` | **Placeholder** rule-based fall detector (see below) |
| `media.py` | Writes browser-playable H.264 video using a bundled ffmpeg |
| `config.py` | Settings from environment variables |

## Fall detector (placeholder)

`fall_detector.py` uses a simple rule so the system works end to end. It flags a fall when a tracked person's hips drop by 30% or more of their height within 1.5 s, and they then stay horizontal for 0.5 s. It has not been tuned on real fall footage. When the team has a trained classifier, replace `FallDetector.update(people, timestamp)`. As long as it returns a list of `FallDetection(track_id, confidence, reason)`, nothing else needs to change.

## Settings

Set these in `.env` in the project root (copy `.env.example`), or as environment variables, which take precedence.

| Env variable | Default |
|---|---|
| `FALL_DATA_DIR` | `data/` (uploads in `input/`, outputs in `output/`, event media in `events/`) |
| `FALL_MODEL_PATH` | `models/pose_pipeline.json` (YOLO26m + RTMPose, needs a GPU live). `models/pose_pipeline_cpu.json` uses YOLO26n for laptops without a GPU; `models/yolo11n-pose.pt` selects the older YOLO-Pose pipeline |
| `FALL_IMAGE_SIZE` | `native`: YOLO runs at the camera's own resolution. A number (e.g. `640`) scales the long side down to it; faster, but people far from the camera are missed |
| `FALL_CLIP_PRE_SECONDS` / `FALL_CLIP_POST_SECONDS` | `5` / `5` |
| `FALL_CORS_ORIGINS` | `*` |
| `FALL_SITE_LOCATION` | `Unspecified location`. Used until the dashboard reports the browser's location, or if location is blocked |
| `FALL_SITE_COORDS` | empty |
| `FALL_REVERSE_GEOCODE` | `1`. Set `0` to not send coordinates to OpenStreetMap; the name then shows the coordinates |
| `FALL_CAMERA_SOURCE` | unset. Cameras can be selected in the dashboard; set this (e.g. `0`) only to start one automatically when the server boots |
| `FALL_CAMERA_NAME` / `FALL_CAMERA_LABEL` | `Camera <source>` / `Live camera` |
| `FALL_DATABASE_URL` | unset. The shared PostgreSQL (Supabase Session pooler URL, starting `postgresql+psycopg://`). See [database.md](database.md) |
| `FALL_DB_HOST` / `FALL_DB_PORT` / `FALL_DB_NAME` / `FALL_DB_USER` / `FALL_DB_PASSWORD` | `localhost` / `5432` / `fall_detection` / `fall` / `fall`. A self-hosted server, used when `FALL_DATABASE_URL` is unset |
| `FALL_STORAGE_ENDPOINT` / `FALL_STORAGE_REGION` / `FALL_STORAGE_BUCKET` / `FALL_STORAGE_KEY_ID` / `FALL_STORAGE_SECRET` | unset / `us-east-1` / unset. S3-compatible bucket for media; without a bucket, media stays on the station |
| `FALL_STATION_NAME` | the computer's network name. This station's name on its jobs and location |

## Not done yet

- No login or authentication.
- No push alerts (WebSocket); the frontend polls instead.
- No deleting jobs or events.
- Live camera sessions are not recorded in full, only the clips around detected falls.

# 10-19-2026 - Dashboard connected to the backend

## Goal

The `UI/Backend` branch added a dashboard (`ui/`) that only showed hard-coded sample data from `ui/src/data/mock.ts`. The backend (`src/backend/`) could already detect falls and save them, but it stored different fields than the dashboard displays and used different status names.

This change connects the two. When a live camera detects a fall, the backend saves the details the dashboard needs, and the dashboard shows the new event, updated stats and the live feed within a few seconds.

How to run and test it: [../running-and-testing.md](../running-and-testing.md).

## Backend changes (`src/backend/`)

- **Each event stores what the dashboard shows** (`db.py`). Fall events now save the camera name, location and coordinates along with the time and confidence. They're copied onto the event when it's recorded, so old log entries stay correct if a camera is later renamed or moved.
- **Statuses match the UI.** `pending / confirmed / false_alarm` became `unacknowledged / needs_review / resolved / false_alarm`, the same four statuses as `EventStatus` in `ui/src/types.ts`. The review endpoint accepts either form (`needs_review` or `"Needs review"`).
- **Older databases are upgraded automatically.** On startup, missing columns are added and old statuses are renamed (`pending` → `unacknowledged`, `confirmed` → `resolved`). No events are lost.
- **New `GET /api/dashboard` endpoint** (`routers/dashboard.py`). Returns everything the page shows in one response, in the same shapes as `ui/src/types.ts` (`Stat`, `CameraInfo`, `FallEvent`, location). The backend formats the data (local time, confidence as a %, status labels, icons, stat card colours) so the frontend can use it unchanged. By default it returns today's events (`?scope=all` for every day).
- **Stat cards are calculated from real data.** Active incidents (open events), cameras online, detection uptime (share of the last 24 h with a camera running) and camera health.
- **Camera health check** (`pipeline.py`). Every 15th frame is checked: a nearly uniform image means *Obstructed*, very few edges means *Blurry*.
- **Accurate uptime after a crash.** Camera sessions record the time of their last frame. When the server restarts, an interrupted session is closed at that time instead of the restart time, so downtime isn't counted as uptime.
- **Camera settings.** The dashboard discovers local cameras and provides start, switch and stop controls, plus a custom network-stream option. `FALL_CAMERA_SOURCE` remains available for optional automatic startup. `FALL_SITE_LOCATION` and `FALL_SITE_COORDS` set the default location.
- **CSV export** at `GET /api/dashboard/events.csv`, ready for the "Export log" button.

## Frontend changes (`ui/`)

Only what's needed to replace the sample data. The layout and styling are unchanged.

- `vite.config.ts`: the dev server forwards `/api/...` to the backend on port 8000, so the app uses relative URLs and has no hard-coded server address.
- `src/api.ts` (new): the `useDashboard()` hook requests `/api/dashboard` every 2.5 s. It shows "Connecting to backend…" at first and "Backend offline" if a request fails.
- `src/App.tsx`: uses `useDashboard()` instead of `data/mock.ts`.
- `Header.tsx`, `EventLog.tsx`, `CameraFeed.tsx`: take the location as a prop instead of importing it from `mock.ts`. The event log shows each event's own camera and location, plus a "No events today" row when it's empty.
- `CameraFeed.tsx`: shows the backend's live MJPEG stream (`<img src={camera.stream_url}>`), with skeletons, IDs and fall alerts drawn on, and `LIVE · N FPS` or `OFFLINE`.
- `types.ts`: added the `Location` and `Dashboard` types, and optional `camera`, `location` and `stream_url` fields.

`ui/src/data/mock.ts` is no longer used and can be deleted.

## Location from this computer

The location is no longer a fixed setting. It's wherever the laptop running the dashboard is.

- **Browser** (`ui/src/api.ts`, `useReportLocation`). On page load and every 10 minutes after, the dashboard asks the browser for its position (one permission prompt) and sends it to `PUT /api/location`. If location is blocked, nothing is sent.
- **Backend** (`src/backend/location.py`, `routers/location.py`). Stores the position in a one-row `SiteLocation` table, so it survives restarts. It looks up "City, State, Country" from OpenStreetMap's Nominatim service. The lookup only runs again after a move of more than 500 m (Nominatim allows 1 request per second), or when the last lookup failed. If the lookup fails, the coordinates are shown instead.
- **Which location an event gets:** the camera's own location (if it was started with one), then the browser-reported location, then `FALL_SITE_LOCATION` from `.env`.
- **Privacy:** coordinates are sent to OpenStreetMap for the name lookup. `FALL_REVERSE_GEOCODE=0` turns that off.

## One-command launcher

- **`run.py`** (`uv run run.py`):
  - creates `.env` from `.env.example` and runs `npm install` on the first run,
  - checks that ports 8000 and 5173 are free,
  - starts the backend, waits for `/api/health`, then starts Vite,
  - opens the browser, and shows both programs' output with `[backend]` / `[dashboard]` labels.
- **Stopping:** Ctrl+C stops both. If either program exits on its own, the other is stopped too. On Windows, `taskkill /T` also ends uvicorn's reload worker. Uses only the Python standard library and runs on Windows and macOS.
- **`.env`**: settings now live in a file instead of terminal variables. `src/backend/config.py` reads it on startup, so the manual `uvicorn` command uses it too. Variables already set in the terminal take precedence. `.env` is gitignored; `.env.example` is committed.

## Recorded video upload and event review (10-20-2026)

- **Upload panel** (`ui/src/components/VideoUpload.tsx`), under the camera feed. Sends the file to the existing `POST /api/videos`. The backend runs it through the same `yolo_pose.process_frame` (YOLO pose + ByteTrack) and fall detector as the live camera. The panel shows queued, progress %, falls found, errors, and a link to the analyzed video. Recent uploads come from the new `uploads` field of `/api/dashboard`.
- **Review panel** (`ui/src/components/EventReview.tsx`). Opens when an event-log row is clicked. It shows the clip (or the snapshot until the clip is ready) and the details: time, source, location, confidence, track ID and reason. It has a notes box and **Confirm real fall** (Resolved) / **False alarm** buttons, which call `PATCH /api/events/{id}`. Opening an Unacknowledged event marks it Needs review.
- **Backend**: dashboard events now include `source` (camera or uploaded video), `job_id`, `video_time`, `notes`, `reviewer` and `reviewed_at`. The CSV export gains `source` and `video_time_seconds` columns.
- **Fall detector fix: tracker ID switches.** Testing with a video showed that ByteTrack gives a falling person a new ID mid-fall, when their box changes from tall to wide. The new ID had no history, so it looked like someone already lying down, and the fall was missed. `FallDetector` now passes a vanished track's history to a new track that appears within 1 s and within one body height of it. This affects live cameras too.
- **Test video**: `data/samples/synthetic_fall.mp4` (H.264, 6 s), made by `scripts/make_synthetic_fall.py` from ultralytics' sample photo.
- `ui/src/api.ts`: `useDashboard()` now also returns `refresh()`, so changes show immediately. A request counter stops a slow older response from overwriting a newer one. Added the `uploadVideo()` and `reviewEvent()` helpers.

## Tests

- `tests/test_dashboard.py`, 15 new tests: response shapes match the UI types, all four stat cards, uptime math, the "today" filter, the status filter, CSV export, review with UI labels, saving camera labels and locations, crash recovery, the database upgrade and the health check.
- `tests/test_api.py` updated for the new statuses, and now checks that location is saved on events.
- `tests/test_location.py`, 9 tests: browser location saved and named, name reused for small moves, retry after a failed lookup, geocoding turned off, invalid coordinates rejected, events stamped with the location, `.env` reading.
- `tests/test_dashboard.py` +3 (uploads, event source, review shown) and `tests/test_fall_detector.py` +3 (ID switch handled; far-away or late new IDs not merged).
- Full suite: `uv run pytest` → 69 passed. `npm run build` in `ui/` passes the type check.

## Not done yet

- The "All events" filter, "Export log" and "Settings" buttons on the dashboard aren't connected. The backend already supports filtering (`?status=`) and CSV export.
- The dashboard shows one camera. The backend can run several.
- Falls are still detected by the placeholder rule in `fall_detector.py`, not the temporal classifier.

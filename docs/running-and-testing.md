# Running and testing the system

This guide explains how to run the whole system (the backend server and the dashboard), what each part does, and how to check that every feature works. Each test lists the output you should see.

- [How the system works](#how-the-system-works)
- [One-time setup](#one-time-setup)
- [Run everything](#run-everything)
- [Test it](#test-it)
- [Troubleshooting](#troubleshooting)
- [What is not built yet](#what-is-not-built-yet)

---

## How the system works

```
 Camera / video ──▶ YOLO Pose + ByteTrack ──▶ Fall detector ──▶ Database ──▶ Dashboard
  (webcam, RTSP,     skeleton + ID for          rule: fast        events,      live feed,
   uploaded file)    each person, per frame     drop + lying       snapshots,   stats,
                                                down               clips        event log
```

There are two programs, and they run in separate terminals:

| Part | Folder | Runs on | What it does |
|---|---|---|---|
| **Backend** | `src/backend/` | http://127.0.0.1:8000 | A Python [FastAPI](https://fastapi.tiangolo.com/) web server. It reads the camera, runs the AI models, detects falls, saves them to a database, and serves that data to the dashboard over HTTP. |
| **Dashboard** | `ui/` | http://localhost:5173 | A React web page (built with [Vite](https://vitejs.dev/)). It requests the backend's data every 2.5 seconds and shows it: stat cards, the live camera feed, and the event log. |

### What happens to each camera frame

1. **Pose estimation** (`src/yolo_pose.py`). The YOLO11 pose model finds every person in the frame and places 17 body keypoints on each one (shoulders, hips, knees, and so on).
2. **Tracking** (ByteTrack, inside the same call). Each person gets an ID number that stays the same from frame to frame ("ID 3"), so the system can follow one person's movement over time.
3. **Fall detection** (`src/backend/fall_detector.py`). For each tracked person, the detector checks whether their **hips dropped by at least 30% of their body height within 1.5 seconds** and they then **stayed roughly horizontal for 0.5 seconds**. If both are true, it reports a fall with a confidence score. ByteTrack often gives a person a new ID in the middle of a fall, because their box suddenly turns from tall to wide. So when a new ID appears where another vanished less than a second earlier, the detector carries the old ID's history over. This is a simple placeholder rule. Setting `FALL_CLASSIFIER_PATH` in `.env` swaps it for the temporal classifier (`src/temporal/`), which scores each person's last 30 frames of joint positions, velocities and accelerations and fires when the probability stays above 0.8 for 5 frames. The only checkpoint so far was trained on synthetic skeletons, so the rule stays the default (see [What is not built yet](#what-is-not-built-yet)).
4. **Recording the event** (`src/backend/pipeline.py`). When a fall is detected, the backend:
   - saves a row in the PostgreSQL database with the time, camera, location, coordinates, confidence, and the reason ("hip dropped 46% of body height in 0.7s; torso 74 deg from vertical"),
   - saves a **snapshot** JPEG of that moment,
   - records a **clip** from 5 seconds before to 5 seconds after the fall,
   - draws a red **FALL DETECTED** box on the live video for 3 seconds.
5. **Camera health check**. Every 15th frame, the backend checks the image itself. A nearly uniform image (lens covered, or too dark) is reported as **Obstructed**. An image with very few sharp edges (out of focus, dirty lens) is reported as **Blurry**.

### What the dashboard shows

| Dashboard part | Where the data comes from |
|---|---|
| Header location and coordinates | **This computer's location.** The browser asks once for permission, then sends its coordinates to the backend. The backend looks up the city name with OpenStreetMap and stamps it on every new event. If location is blocked, `FALL_SITE_LOCATION` / `FALL_SITE_COORDS` from `.env` are used instead. |
| **Active incidents** | Events still *Unacknowledged* or *Needs review*, across all days. Red while any are open. |
| **Cameras online** | Live cameras running now |
| **Detection uptime** | Share of the last 24 hours that at least one camera was running. Green at 95% or above, red below 80%. |
| **Camera health** | The image check described above: Good / Blurry / Obstructed / Offline |
| **Camera feed** | The live video, with skeletons, IDs and fall alerts drawn by the backend |
| **Event log** | Today's events, newest first: time, event, camera and location, confidence, status. Click a row to review it. |
| **Analyze a recorded video** (under the camera) | Upload a video file. The backend runs the same pose, tracking and fall detection as for the live camera, shows the progress, then lists how many falls it found and links to the analyzed video. Falls found go into the event log. |

Every event moves through these review statuses:

```
Unacknowledged ──▶ Needs review ──▶ Resolved      (a real fall, handled)
  (new alert)       (someone saw it) └▶ False alarm (not a real fall)
```

Click any event in the log to open its **review panel**. It shows the clip, the details, and buttons to confirm the fall or mark it a false alarm (see [Test 7](#test-7-review-an-event)). Opening an *Unacknowledged* event moves it to *Needs review* automatically.

---

## One-time setup

You need:

- **uv**, which manages Python and the Python packages.
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Node.js 18 or newer**, which runs the dashboard. Get it from https://nodejs.org. Check it with `node -v`.
- **The pose model** at `models/yolo11n-pose.pt` (already in the repo).
- **PostgreSQL 14 or newer**, which stores the events. Install it and create the app's database as described in [database.md: First-time setup](database.md#first-time-setup).

Then, from the project root:

```bash
uv sync               # installs Python 3.12 and the backend packages into .venv
cd ui
npm install           # installs the dashboard packages into ui/node_modules
cd ..
uv run scripts/create_db.py   # asks for the postgres password; creates the database and tables
```

**Expected:** `uv sync` ends with `Installed N packages` (or `Checked N packages` if they're already installed). `npm install` ends with `added N packages`. Warnings about funding or deprecations are fine. `create_db.py` ends with `Tables are up to date.`

If you have events from before the switch to PostgreSQL in `data/falls.db`, copy them over once with `uv run scripts/sqlite_to_postgres.py`.

---

## Run everything

### One command

From the project root:

```bash
uv run run.py
```

`run.py` does everything:
1. On the first run, it creates your settings file `.env` from `.env.example` and installs the dashboard packages if needed.
2. It checks that ports 8000 and 5173 are free.
3. It starts the backend, waits until it responds, then starts the dashboard.
4. It opens the dashboard in your browser.
5. It shows both programs' output in one terminal, labelled `[backend]` and `[dashboard]`.

**Expected output:**
```
[run] Starting backend...
[backend] INFO:     Started server process [53336]
[backend] INFO:     Waiting for application startup.
[backend] INFO:     Application startup complete.
[backend] INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
[run] Starting dashboard...
[dashboard]   VITE v5.4.21  ready in 580 ms
[dashboard]   ➜  Local:   http://localhost:5173/
[run]
[run] Dashboard:  http://localhost:5173
[run] API docs:   http://127.0.0.1:8000/docs
[run] Press Ctrl+C to stop both.
[backend] 2026-10-19 18:19:25,210 INFO backend.manager: Pose model warmed up
```

After this, a line like `"GET /api/dashboard HTTP/1.1" 200 OK` appears every 2.5 seconds while the dashboard is open. That's normal: it's the dashboard requesting new data.

Options:
- `uv run run.py --no-browser` doesn't open the browser.
- `uv run run.py --reload` restarts the backend when you edit Python files (for development). Leave it off for demos, so the camera isn't restarted by accident.

**To stop:** press **Ctrl+C** once in that terminal. Both programs shut down, and you should see `[run] Stopped.` Events, snapshots and clips stay in `data/`, and they come back the next time you start.

- The first time, your OS may ask whether Terminal or Python can use the camera. Allow it.
- Only one program can use a webcam at a time. Close Zoom, Teams, and `yolo_pose.py` first.

### Settings (`.env`)

Settings live in `.env` in the project root. `run.py` creates it from `.env.example` on the first run. `.env` is not committed to git, so everyone keeps their own. Restart after editing it.

| Setting | What it does | Default in `.env.example` |
|---|---|---|
| `FALL_CAMERA_SOURCE` | Camera to start automatically: `0` is the built-in webcam, `1` the next camera, or an `rtsp://...` URL for an IP camera. Leave it empty for no camera. | `0` |
| `FALL_CAMERA_NAME` / `FALL_CAMERA_LABEL` | Camera name and description on the dashboard | `Camera 01` / `Webcam` |
| `FALL_SITE_LOCATION` / `FALL_SITE_COORDS` | Location used **only** until the browser reports this computer's location, or if location is blocked | `Unspecified location` / empty |
| `FALL_CLASSIFIER_PATH` | Path to a temporal classifier checkpoint (for example `models/fall_gru.pt`). Empty keeps the rule-based fall detector. | empty |
| `FALL_DB_*` | How to reach the PostgreSQL database ([database.md](database.md#2-check-the-settings-in-env)) | `localhost`, `5432`, `fall_detection`, user `fall`, password `fall` |
| `FALL_REVERSE_GEOCODE` | `1` looks up the city name for the browser's coordinates with OpenStreetMap. `0` keeps the coordinates on this computer and shows them instead of a city name. | `1` |

The rest of the settings are in [backend-api.md](backend-api.md#settings). A variable set in the terminal overrides the same variable in `.env`.

### Running the two parts separately

This is useful when you're only working on one part, or want each part's output in its own terminal. Both commands read `.env` too.

```bash
# Terminal 1: backend
uv run uvicorn backend.main:app --app-dir src --reload

# Terminal 2: dashboard
cd ui
npm run dev
```

Then open **http://localhost:5173**. The dashboard sends every request that starts with `/api` to the backend on port 8000. That forwarding is set up in `ui/vite.config.ts`, so both programs must be running. Stop each with **Ctrl+C**.

---

## Test it

### Test 1: automated tests

```bash
uv run pytest -v
```

**Expected:** `110 passed`. The first run takes about 20–40 seconds, because some tests run the real YOLO model. PostgreSQL must be running: the tests use their own `fall_detection_test` database (created by `scripts/create_db.py`) and empty it before each test, so your real events are never touched.

| Test file | Tests | What it checks |
|---|---|---|
| `tests/test_api.py` | 9 | Core API: health, video upload processed end to end, fall events saved with snapshot, clip and location, review flow, error handling |
| `tests/test_dashboard.py` | 17 | Dashboard data matches the UI's types, the four stat cards, uptime math, "today" filter, CSV export, camera health check, upload progress, events marked as coming from uploads, review shown on the dashboard |
| `tests/test_db.py` | 2 | The migrations create exactly the tables in `db.py`; times keep their timezone |
| `tests/test_fall_detector.py` | 8 | The fall rule: detects a sudden fall only once, including when the tracker switches the person's ID mid-fall; ignores standing still, someone already lying down, a brief dip they recover from, people without tracking IDs, and new IDs that appear far away or long after |
| `tests/test_location.py` | 9 | Browser-reported location: city lookup, reusing the name for small moves, falling back to coordinates or `.env`, location saved on new events, reading `.env` |
| `tests/test_sliding_window.py` | 25 | The sliding window buffer (see [features/10-18-2026-sliding-window.md](features/10-18-2026-sliding-window.md)) |

To check the dashboard compiles with no TypeScript errors:

```bash
cd ui
npm run build
```

**Expected:** `✓ built in 1.06s` (the time varies), with no red errors.

### Test 2: the dashboard loads real data

With both programs running (`uv run run.py`), open http://localhost:5173.

**Expected:**
- The browser asks to **use your location**. Click **Allow**. Within a few seconds the header shows your city, for example `Houston, Texas, United States`, followed by your coordinates. See [Test 3](#test-3-location-from-this-computer).
- **Cameras online:** `1`, green, with the note "Camera 01".
- **Detection uptime:** "Just started", then `100.0%` with "Last N min".
- **Camera health:** `Good`, green, "Feed clear".
- **Active incidents:** `0`, green, "No open incidents". (If falls are already saved, you'll see the count of open ones instead.)
- **Event log:** "No events today", or today's saved events.

If the cards say **"Connecting to backend…"** or **"Backend offline"**, the backend isn't running or failed to start. Check the `[backend]` lines in the terminal.

### Test 3: location from this computer

1. Open the dashboard and click **Allow** when the browser asks for your location.
   **Expected:** within about 3 seconds the header changes from `Unspecified location` (or your `.env` value) to your city, state and country, with coordinates to 4 decimal places. Terminal shows `"PUT /api/location HTTP/1.1" 200 OK`.
2. Open http://127.0.0.1:8000/api/location.
   **Expected:** the same `{"name": "...", "coords": "..."}`.
3. After a fall (Test 5), the new event's **Camera / Location** column shows this location. It's saved on the event, so it stays the same even if you later move the laptop.
4. To test the fallback, block location for the site (click the icon to the left of the address bar, then **Site settings**, set **Location** to **Block**), then restart the backend.
   **Expected:** the header shows the last location that was saved. The location is kept across restarts. If no location was ever saved, it shows `FALL_SITE_LOCATION` from `.env`.

How it works: the browser finds the position, usually from nearby Wi-Fi networks, and the dashboard sends it to the backend when the page loads and every 10 minutes after that. The backend asks OpenStreetMap for the city name only when you've moved more than 500 m, so it doesn't look it up on every page load. Accuracy is typically 20–100 m on a laptop with Wi-Fi.

### Test 4: live camera feed

**Expected:** the camera panel shows your webcam with:
- a coloured skeleton over each person,
- an `ID n` label above each person (the tracking ID),
- `LIVE · N FPS` in the top-right corner. About 5 FPS is normal on a laptop without a GPU, and 15–30 with one.
- the camera name and `CAM-00 · Webcam` underneath, with a green **Online** label.

You can also open the raw feed directly at http://127.0.0.1:8000/api/cameras/1/stream (if you've restarted the camera, change the `1` to the camera's `job_id`).

### Test 5: detect a fall

1. Step back until your whole body (head to feet) is in the frame, and wait for an `ID` label to appear.
2. Drop to the floor quickly, within about 1 second, and **stay lying down for at least 1 second**.

**Expected:**
- **Video:** a red box and the text **FALL DETECTED** on you for 3 seconds.
- **Terminal** (`[backend]` lines): a line like
  ```
  WARNING backend.pipeline: Fall detected: job 1 track 3 at 42.7s (hip dropped 46% of body height in 0.7s; torso 74 deg from vertical)
  ```
- **Dashboard, within about 3 seconds:**
  - a new row at the top of the event log: `Fall detected` · `Camera 01` with your location · a confidence such as `78%` · a red **Unacknowledged** label,
  - **Active incidents** changes to `1` in red, with the note "1 unacknowledged · 0 need review".
- The same person can't trigger another fall for 10 seconds.

**If nothing happens:**
- Your whole body must be visible. Keep the hips and shoulders in the frame.
- A slow sit-down is ignored on purpose. The drop has to be fast.
- A side view works better than walking straight toward the camera.
- The rule is a placeholder that hasn't been tuned. Missed falls and false alarms are expected until the trained classifier replaces it.

### Test 6: view the evidence for an event

Open http://127.0.0.1:8000/api/dashboard?scope=all to see the raw data. Each event has a `snapshot_url` and a `clip_url`. Put `http://127.0.0.1:8000` in front of them and open them:

- `http://127.0.0.1:8000/api/events/1/snapshot`: **expected** a JPEG of the moment the fall was detected, with the red box.
- `http://127.0.0.1:8000/api/events/1/clip`: **expected** a playable video from about 5 s before to 5 s after the fall. `clip_url` is `null` for the first 5 seconds after the fall, while the "after" footage is still being recorded.

### Test 7: review an event

1. Click an event row in the event log.
   **Expected:** a review panel opens with:
   - the **clip** playing on a loop (from 5 s before to 5 s after the fall). Within the first 5 seconds after a live fall, it shows the snapshot instead, with a note that the clip is still being recorded.
   - **Detected** (date and time), **Source** ("Live camera: Camera 01", or "Uploaded video: name.mp4" and the time in the video), **Location**, **Confidence**, **Person** (track ID), and **Why flagged** (e.g. "hip dropped 84% of body height in 0.7s; torso 94 deg from vertical").
   - If the event was *Unacknowledged*, its label in the log behind the panel changes to orange **Needs review** as soon as the panel opens.
2. Optionally type a note, then click **Confirm real fall**.
   **Expected:** the panel closes and the row shows a green **Resolved** label. Active incidents goes down by 1.
3. Open the event again and click **False alarm**.
   **Expected:** a blue **False alarm** label, and the row's icon turns into an amber warning sign. You can change the decision at any time this way. Your note is saved and shown the next time you open the event.
4. Press **Esc**, click outside the panel, or click **×** to close it without changing anything.

The same review can be done through the API's test page at http://127.0.0.1:8000/docs (**PATCH /api/events/{event_id}**), for example `{"status": "Resolved", "notes": "..."}`.

### Test 8: camera health

Cover the webcam lens with your finger or a piece of paper for about 3 seconds.

**Expected:** **Camera health** changes to `Obstructed` in red, with the note "Camera 01: view blocked or too dark". Uncover the lens and it returns to `Good`.

A very out-of-focus or smudged view shows `Blurry`. The thresholds are rough. They're `OBSTRUCTED_MAX_STD` and `BLURRY_MAX_SHARPNESS` in `src/backend/pipeline.py`.

### Test 9: export the log

Open http://127.0.0.1:8000/api/dashboard/events.csv.

**Expected:** a file named `fall-events-YYYY-MM-DD.csv` downloads. It opens in Excel with one row per event and these columns: `id, detected_at_utc, local_time, event, camera, location, coords, confidence_percent, status, track_id, reason, reviewer, notes, reviewed_at_utc`.

### Test 10: data is kept after a restart, and the offline state works

This test needs the two parts running separately (see [Running the two parts separately](#running-the-two-parts-separately)), because Ctrl+C in `run.py` stops both.

1. Stop the backend (Ctrl+C in its terminal) and leave the dashboard open.
   **Expected:** within about 3 seconds the stat cards show `-` and "Backend offline", and the camera panel shows **OFFLINE**. Events already on screen stay visible.
2. Start the backend again.
   **Expected:** everything comes back on its own, including all earlier events (they're saved in PostgreSQL). **Detection uptime** now drops below 100%, because the time the backend was stopped counts as downtime.

### Test 11: analyze a recorded video (no webcam needed)

The repo includes a test video, **`data/samples/synthetic_fall.mp4`**: a person cut out of a sample photo stands for 2 s, falls over in 0.6 s, then lies still (6 s in total). It isn't realistic footage, but it runs through the whole pipeline. Recreate it with `uv run scripts/make_synthetic_fall.py`.

1. In the **Analyze a recorded video** box under the camera feed, click **Upload video** and choose `data/samples/synthetic_fall.mp4` (or any `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v` or `.webm`).
   **Expected:** the file appears in the list with "Waiting to start…", then "Analyzing… N%" and a progress bar.
2. Wait for it to finish. The sample takes about 10 seconds on a laptop; longer videos take longer.
   **Expected:** "Done · 1 fall found (see event log)", and a **Watch analyzed video** link. A video with no falls says "Done · no falls found".
3. Look at the event log.
   **Expected:** a new **Fall detected** row. The Camera / Location column shows `synthetic_fall.mp4` and "Uploaded video · at 0:02". Active incidents goes up by 1.
4. Click **Watch analyzed video**.
   **Expected:** the whole video in a new tab, with skeletons, IDs, and the red FALL DETECTED box when the person falls.
5. Click the new row and review it as in [Test 7](#test-7-review-an-event).

Notes:
- Videos are analyzed one at a time; others wait in the list as "Waiting to start…".
- Analyzing a video while the live camera is running slows down both, because they share the same computer.
- Events from uploads appear in the log at the time the video was *analyzed* (the time in the video is shown separately), and use the current site location.

### Test 12: the original stand-alone script

`src/yolo_pose.py` still works on its own. It doesn't use the backend, the dashboard or the fall detector.

```bash
uv run src/yolo_pose.py --model models/yolo11n-pose.pt                  # live camera, Q or Esc to quit
uv run src/yolo_pose.py --input data/input/YourVideo.mp4 --model models/yolo11n-pose.pt
```

**Expected:** a window with skeletons, IDs, FPS and a person count. For a video, the results are saved as `data/output/YourVideo.mp4` and `.json`. Stop the backend's camera first, because only one program can use the webcam.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard says "Backend offline" | The backend isn't running, or it crashed. Check the `[backend]` lines for an error, and that it printed `Application startup complete`. |
| "Cameras online 0" / camera panel says OFFLINE | `FALL_CAMERA_SOURCE` is empty in `.env`, or another app is using the webcam. Look for `Could not open camera 0` in the terminal. Try `1` if you have more than one camera. |
| Camera starts, then goes offline | The terminal shows `Lost connection to camera`. Unplug and replug the camera, then restart the backend. |
| Very low FPS | Normal on CPU (about 5 FPS). Close other heavy programs. A computer with an NVIDIA GPU runs much faster. |
| `npm` / `node` not found | Install Node.js and open a new terminal. |
| `Port 8000 (backend) is already in use` (or 5173) | Another copy is still running, maybe in another terminal or VS Code. Stop it with Ctrl+C there, or end `python` / `node` in Task Manager or Activity Monitor. |
| Header shows `Unspecified location` | Location is blocked for the dashboard. Click the icon to the left of the address bar, then **Site settings**, set **Location** to **Allow**, and reload. On Windows, also check **Settings → Privacy & security → Location** is on. On macOS, check **System Settings → Privacy & Security → Location Services** for your browser. |
| Header shows coordinates instead of a city | The OpenStreetMap lookup failed (no internet) or `FALL_REVERSE_GEOCODE=0`. The lookup is retried the next time the dashboard reports the location: on page reload, or every 10 minutes. |
| `[run] ERROR: Could not connect to the database` | PostgreSQL isn't running, or `.env` has the wrong `FALL_DB_*` settings. See [database.md: Troubleshooting](database.md#troubleshooting). |
| Start over with an empty event log | Back up first if needed (`uv run scripts/backup.py backup`). Stop the backend, run `psql -h localhost -U fall -d fall_detection -c "TRUNCATE fallevent, job, sitelocation RESTART IDENTITY"`, and delete `data/events/`. |
| The browser console shows `[AssignmentCache]` or other messages from `all.js` | Those come from a browser extension, not this project. Ignore them. |

---

## What is not built yet

- **Temporal classifier (stage 6): built, not yet trained on real falls.** `src/temporal/` has the frame sampler, feature extraction, GRU/TCN model, threshold gate and a drop-in `TemporalFallDetector`. `models/fall_gru.pt` was trained on synthetic skeletons only, so the dashboard keeps the rule in `fall_detector.py` unless `FALL_CLASSIFIER_PATH` is set. Next step: collect labeled footage (OmniFall plus site clips), run it through `yolo_pose.py`, list it in a manifest and train with `scripts/train_temporal.py --manifest`. Details and the plotting tool: [features/10-23-2026-temporal-classifier.md](features/10-23-2026-temporal-classifier.md).
- **Dashboard buttons.** "All events" (filter), "Export log" and "Settings" aren't hooked up yet. The backend already supports filtering and CSV export; the buttons just need to be connected.
- **Alerts beyond the dashboard.** There are no sounds, push notifications, SMS or email yet. The dashboard checks for new events every 2.5 s.
- **Several cameras on the dashboard.** The backend can run several, but the dashboard shows one.
- **Login.** There is no authentication. Only run it on a trusted network.
- **X-CLIP verifier.** Not started.

For every endpoint and response format, see [backend-api.md](backend-api.md).

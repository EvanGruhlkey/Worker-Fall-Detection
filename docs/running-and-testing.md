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
3. **Fall detection** (`src/backend/fall_detector.py`). For each tracked person, the detector checks whether their **hips dropped by at least 30% of their body height within 1.5 seconds** and they then **stayed roughly horizontal for 0.5 seconds**. If both are true, it reports a fall with a confidence score. This is a simple placeholder rule. It will be replaced by the trained temporal classifier (see [What is not built yet](#what-is-not-built-yet)).
4. **Recording the event** (`src/backend/pipeline.py`). When a fall is detected, the backend:
   - saves a row in the database (`data/falls.db`) with the time, camera, location, coordinates, confidence, and the reason ("hip dropped 46% of body height in 0.7s; torso 74 deg from vertical"),
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
| **Event log** | Today's events, newest first: time, event, camera and location, confidence, status |

Every event moves through these review statuses:

```
Unacknowledged ──▶ Needs review ──▶ Resolved      (a real fall, handled)
  (new alert)       (someone saw it) └▶ False alarm (not a real fall)
```

The dashboard has no review buttons yet. Statuses are changed through the API for now (see [Test 7](#test-7-review-an-event)).

---

## One-time setup

You need:

- **uv**, which manages Python and the Python packages.
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Node.js 18 or newer**, which runs the dashboard. Get it from https://nodejs.org. Check it with `node -v`.
- **The pose model** at `models/yolo11n-pose.pt` (already in the repo).

Then, from the project root:

```bash
uv sync               # installs Python 3.12 and the backend packages into .venv
cd ui
npm install           # installs the dashboard packages into ui/node_modules
cd ..
```

**Expected:** `uv sync` ends with `Installed N packages` (or `Checked N packages` if they're already installed). `npm install` ends with `added N packages`. Warnings about funding or deprecations are fine.

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

**Expected:** `63 passed`. The first run takes about 20–40 seconds, because some tests run the real YOLO model.

| Test file | Tests | What it checks |
|---|---|---|
| `tests/test_api.py` | 9 | Core API: health, video upload processed end to end, fall events saved with snapshot, clip and location, review flow, error handling |
| `tests/test_dashboard.py` | 15 | Dashboard data matches the UI's types, the four stat cards, uptime math, "today" filter, CSV export, camera health check, upgrading an old database |
| `tests/test_fall_detector.py` | 5 | The fall rule: detects a sudden fall only once; ignores standing still, someone already lying down, a brief dip they recover from, and people without tracking IDs |
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

The dashboard has no review buttons yet, so use the API's built-in test page:

1. Open **http://127.0.0.1:8000/docs**. This page lists every backend endpoint, and you can try each one in the browser.
2. Find **PATCH /api/events/{event_id}**, click **Try it out**, and enter event ID `1`.
3. Set the request body to `{"status": "Needs review", "reviewer": "Sam"}` and click **Execute**.

**Expected:** response code `200`. Within about 3 seconds the row's label on the dashboard changes to an orange **Needs review**, and Active incidents shows "0 unacknowledged · 1 need review".

4. Repeat with `{"status": "Resolved", "notes": "Worker OK, slipped on oil"}`.

**Expected:** a green **Resolved** label, and Active incidents goes to `0` (green). `False alarm` gives a blue label, and the row's icon turns into an amber warning sign.

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
   **Expected:** everything comes back on its own, including all earlier events (they're saved in `data/falls.db`). **Detection uptime** now drops below 100%, because the time the backend was stopped counts as downtime.

### Test 11: process a recorded video (no webcam needed)

This also works with no camera, for example to test with fall footage.

1. Open http://127.0.0.1:8000/docs, then **POST /api/videos**, then **Try it out**. Choose an `.mp4` file and click **Execute**.
   **Expected:** response `201` with a job `id` and `"status": "queued"`.
2. Run **GET /api/jobs/{job_id}** a few times.
   **Expected:** `status` goes from `running` to `completed`, and `progress` climbs to `1.0`.
3. When it's `completed`, open `output_video_url` (e.g. `http://127.0.0.1:8000/api/jobs/2/video`).
   **Expected:** the whole video with skeletons, IDs and fall alerts drawn on it.

Any falls found in the video appear in the dashboard's event log, named after the video file in the Camera column.

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
| Start over with an empty event log | Stop the backend and delete `data/falls.db*` and `data/events/`. They're recreated on the next start. |
| The browser console shows `[AssignmentCache]` or other messages from `all.js` | Those come from a browser extension, not this project. Ignore them. |

---

## What is not built yet

- **Temporal classifier (stage 6).** `src/sliding_window.py` already turns each person's last 30 frames into the classifier's input, but no trained model exists yet. Until then, falls come from the rule in `fall_detector.py`. When the model is ready, only `FallDetector.update()` needs to change; the events, database and dashboard stay the same. Train the model at the same FPS the live system runs at. 30 frames is 6 seconds at 5 FPS, but only 1 second at 30 FPS.
- **Dashboard buttons.** "All events" (filter), "Export log", "Settings", and changing an event's status from the dashboard aren't hooked up yet. The backend already supports filtering, CSV export and review; the buttons just need to be connected.
- **Alerts beyond the dashboard.** There are no sounds, push notifications, SMS or email yet. The dashboard checks for new events every 2.5 s.
- **Several cameras on the dashboard.** The backend can run several, but the dashboard shows one.
- **Login.** There is no authentication. Only run it on a trusted network.
- **X-CLIP verifier.** Not started.

For every endpoint and response format, see [backend-api.md](backend-api.md).

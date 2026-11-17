# User study

A standalone two-condition worker-safety study. Participants receive an anonymous ID, enter background information, consent, monitor six raw camera feeds for two minutes, then complete arithmetic while automated alerts monitor a different footage set for two minutes. A final survey follows. The footage sets alternate between participants; manual monitoring is always first.

## Requirements

- Git, Node.js with npm, and uv.
- Python 3.12. `uv sync` installs the project's Python environment.
- Prepared study videos and their matching manifest. These files are not included in Git.

## Fresh checkout

In PowerShell:

```powershell
git clone --branch user-study https://github.com/EvanGruhlkey/Worker-Fall-Detection.git
cd Worker-Fall-Detection
uv sync
```

For an existing checkout, switch to `user-study` and pull the latest changes instead.

Ask the study team for the prepared `rig-wall-two` folder. Copy it to `data/study/rig-wall-two/` inside the repository. Keep the filenames unchanged. A fresh collection needs its `media/` directory and `calm-manifest.json`; do not copy another collection's `study.sqlite3`. The server creates a new database automatically. Keep the database when continuing an existing collection.

If prepared footage is unavailable, use the preparation commands below. They require the source videos, labels, and saved poses.

## Run locally

From the repository root in PowerShell:

```powershell
npm.cmd ci --prefix user_study/frontend
npm.cmd run build --prefix user_study/frontend
$env:FALL_STUDY_DIR = Join-Path (Get-Location) 'data/study/rig-wall-two'
uv run uvicorn user_study.backend.app:app --host 127.0.0.1 --port 8765
```

Open http://localhost:8765/. No facilitator key is required. Study setup provides results, CSV/JSON downloads, and footage configuration. Participant sessions use scoped HttpOnly cookies and recover after a reload. The app is separate from the production dashboard and database. For frontend development, run `npm.cmd run dev --prefix user_study/frontend`; its API proxy targets port 8765.

Keep the server terminal open while using the study. Press **Ctrl+C** to stop it. Restart with the same `FALL_STUDY_DIR` to keep existing responses. Rebuild the frontend after source changes.

On macOS or Linux, use `npm` instead of `npm.cmd`, and replace the PowerShell environment assignment and server command with:

```sh
FALL_STUDY_DIR="$PWD/data/study/rig-wall-two" uv run uvicorn user_study.backend.app:app --host 127.0.0.1 --port 8765
```

## First-time configuration

1. Open **Study setup**, then **Settings**.
2. Under **Replace study footage**, choose the copied `calm-manifest.json` and select **Save footage**. This is required for a fresh database. For newly generated footage, choose its `rig-wall-manifest.json` instead.
3. Select **Back to study**.
4. Have the participant enter their group, age range, and monitoring experience, then select **Continue**. The app assigns an anonymous ID.
5. Complete consent, both monitoring conditions, and the final survey.
6. Select **New participant** for the next person. Download responses from **Study setup → Results**.

Use the same hostname throughout a session. `localhost` and `127.0.0.1` keep separate cookies.

## Troubleshooting

- **No footage configured:** import the matching manifest in Settings.
- **Media missing:** check that the study folder's `media/` directory contains each filename in the manifest. Importing a manifest does not copy videos.
- **Port 8765 is in use:** stop the previous study server before restarting.
- **Old screen after an update:** rebuild the frontend and reload the browser.

## Prepare footage

Run these scripts from the repository root, using the project's labeled input footage and verified saved poses:

```powershell
uv run user_study/scripts/prepare_study_rig_wall.py --directory data/study/rig-wall-calm --cache-directory data/study/rig-wall/preparation --classifier user_study/models/fall_gru_trial2.pt
uv run user_study/scripts/render_study_pipeline.py --directory data/study/rig-wall-calm --classifier user_study/models/fall_gru_trial2.pt
uv run user_study/scripts/build_study_reviews.py --directory data/study/rig-wall-calm
```

Each set has six falls, spaced approximately 17 seconds apart across the six feeds. Around 90% of combined camera time shows normal work. Manual mode uses raw footage. Automated mode replays verified pipeline boxes, poses, track IDs, probabilities, and actual detector alerts. Incident review shows the flagged person from half a second before to half a second after detection. These are prerecorded pipeline outputs, not live inference. Source and classifier hashes are recorded during preparation; rendering verifies its detector events against the saved outputs. Classifier state resets between source clips. False alarms and missed detections are retained.

The rig clips are AI-generated, not human-recorded. Ground truth comes from `data/labels/clips.csv` independently of detector events. The trial-2 GRU checkpoint is from `feature/modelTraining2` commit `5e3df49`, SHA-256 `d35ae9f3b4c71914d3e581d647329c08ac3760123438c0260dc35e2bda1f881f`. It uses 30 frames, threshold 0.99, minimum drop 0.15, gap 5, and duplicate merging. Optional production configuration: `FALL_CLASSIFIER_PATH=user_study/models/fall_gru_trial2.pt`.

Set `FALL_STUDY_DIR` to the prepared folder, or copy its media into your active study directory and import the manifest through Settings. Existing sessions keep their original manifests. Local media, participant SQLite records, and exports are not committed. The current preview uses prefixed media and `calm-manifest.json` under `data/study/rig-wall-two`.

## Verify

```powershell
npm.cmd run build --prefix user_study/frontend
uv run python -m pytest user_study/tests --confcutdir=user_study/tests -q
```

Completed study-mode sessions enter summaries; pilot and demo sessions do not. Exports retain raw events and incomplete sessions, but exclude authentication tokens. Matching is one-to-one by camera and timestamp; duplicate reports count as false alarms. Video-time detection latency excludes model processing and network delivery. Keep the study on a trusted local machine and handle records according to the team's consent and retention arrangements. Withdrawal deletes the session locally; downloaded copies require separate handling.

The Oxy logo is from https://www.oxy.com/siteassets/images/logos/logo.png and is bundled for offline use.

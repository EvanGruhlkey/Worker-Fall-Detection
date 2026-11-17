# User study

A standalone two-condition worker-safety study. Participants receive an anonymous ID, enter background information, consent, monitor six raw camera feeds for two minutes, then complete arithmetic while automated alerts monitor a different footage set for two minutes. A final survey follows. The footage sets alternate between participants; manual monitoring is always first.

## Run locally

From the repository root in PowerShell:

```powershell
npm.cmd ci --prefix user_study/frontend
npm.cmd run build --prefix user_study/frontend
$env:FALL_STUDY_DIR = Join-Path (Get-Location) 'data/study/rig-wall-two'
uv run uvicorn user_study.backend.app:app --host 127.0.0.1 --port 8765
```

Open http://localhost:8765/. No facilitator key is required. Study setup provides results, CSV/JSON downloads, and footage configuration. Participant sessions use scoped HttpOnly cookies and recover after a reload. The app is separate from the production dashboard and database. For frontend development, run `npm.cmd run dev --prefix user_study/frontend`; its API proxy targets port 8765.

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

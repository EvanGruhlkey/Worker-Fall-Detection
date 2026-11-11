# Adding CAUCAFall to training

CAUCAFall (CC BY 4.0, see `DATA_SOURCES.md`) adds 50 falls and 50 daily
activities (walk, hop, kneel, sit down, pick up object) filmed by a CCTV-style
camera high in a room corner, in daylight and night IR, by 10 people. The rig
clips stay the test set: CAUCAFall is added to training and every result is
still scored on rig clips, so runs stay comparable with batch 1.

## Status

| Step | Status |
|---|---|
| 1. Download videos + labels | done (`data/external/caucafall/`, not in git) |
| 2. Label sheet with fall times | done: `data/labels/external_caucafall.csv`, times checked by eye on all 50 falls |
| 3. Pose extraction | `scripts/extract_poses_rtmpose.py` (see step 2) |
| 4. Faller track IDs | automatic after step 3 |
| 5. Manifest | automatic after step 3 |
| 6. Training batch 2 | after step 5 |

## Steps

### 1. Get the videos (only if you don't have `data/external/caucafall/`)

```
uv run scripts/download_caucafall.py     # ~0.15 GB, 10-20 min, checksums verified
uv run scripts/import_caucafall.py       # copies the videos to data/external/caucafall/videos/
```

`import_caucafall.py` keeps the reviewed fall times already in
`external_caucafall.csv`; it never overwrites them.

### 2. Extract poses

```
uv run scripts/extract_poses_rtmpose.py data/external/caucafall/videos --out data/output/caucafall
```

This is a re-implementation of the team's YOLO26n + ByteTrack + RTMPose-t
batch pipeline (whose code is not in the repo). It uses the same RTMPose-t
model (SHA-256 checked) and settings, and was checked against the team's rig
files with `scripts/compare_pose_files.py`: matched keypoints differ by a
median 0.4-0.6% of body height, keypoint confidences correlate 0.95-0.99 with
the same mean. Models download once to `data/external/models/`. About 30-60 min
on a laptop CPU; reruns skip finished videos.

**Keep every frame (20 fps); don't pass `--fps 12`.** 20 is not a multiple of
12, so sampling would give 10 fps and the clips would be refused for not
matching the rig clips. The training loader resamples 20 fps poses to exactly
12 fps itself (by interpolating between frames).

Commit the 100 JSON files in `data/output/caucafall/` (the videos stay out of git).

### 3. Fill in the faller track IDs

```
uv run scripts/import_caucafall.py
```

It matches the dataset's labeled box to the pose tracks and writes
`faller_track_id` for every fall clip. Any clip it cannot match is set to
`quality=review` with a note; check those few by opening the annotated video.

### 4. Build the manifest

```
uv run scripts/build_manifest.py --keep-only
```

It should report 131 clips: rig 21 falls + 10 normal, caucafall 50 + 50. Fix
any error it prints before training.

### 5. Train batch 2

```
powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Batch 2
```

| Run | What it answers |
|---|---|
| `rig-only` | the reference (same data as batch 1's `margin`) |
| `plus-cauca` | does adding CAUCAFall to training catch more rig falls? |
| `plus-cauca-aug3` | CAUCAFall + augmentation |
| `rig-only-win18` | does a 1.5 s window work as well as 2.5 s? |
| `cauca-in-folds` | how well does the model do on CAUCAFall itself (sanity check) |

Compare them in `runs/README.md`. Keep CAUCAFall only if it catches more rig
falls without more false alerts.

### 6. Commit

```
git add data/output/caucafall data/labels/external_caucafall.csv runs
git commit -m "Add CAUCAFall poses and training batch 2 results"
```

## Notes on the labels

- The dataset labels every frame from mid-fall to the end of the clip as
  "fall". We use our own rule instead: start = loss of balance, end = body down
  on the floor. All 50 were set by eye from frame grids.
- 12 falls are slow and controlled (the person kneels or crawls down); they are
  flagged "slow, controlled descent" in `notes`. They are kept, but if they
  cause kneeling to be flagged as a fall, set them to `quality=drop`.
- The "kneel", "sit down" and "pick up object" clips are the most useful part:
  they are exactly the motions that cause false alarms.

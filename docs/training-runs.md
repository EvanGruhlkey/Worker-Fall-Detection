# Running the temporal-model training experiments

This trains the fall classifier on the labeled rig clips and writes a report
for each experiment. You don't need to change any code or labels; just run it
and push the results.

**Time:** about 1 hour, mostly unattended. **Computer:** any laptop (CPU only),
plugged in.

## What it does

Five experiments run side by side. Each one trains and tests the model 10
times (5-fold cross-validation × 2 random seeds), so every clip is tested by a
model that never saw its scene. Each experiment changes one thing:

| Run | What it tests |
|---|---|
| `baseline` | current settings |
| `margin` | leave out ambiguous windows (part of a fall) |
| `margin-aug3` | + 3 augmented copies of each training clip |
| `margin-synth200` | + 200 synthetic stick-figure clips |
| `margin-tcn` | a TCN model instead of the GRU |

## Steps

### 1. Get the branch

```
git fetch
git checkout featureModelTraining
git pull
```

### 2. Install dependencies

Needs [uv](https://docs.astral.sh/uv/) (`pip install uv` if you don't have it).

```
uv sync
```

### 3. Check that everything works (about 1 minute)

```
uv run pytest -q
```

All tests should pass. If any fail, stop and send the output to the team.

### 4. Keep the laptop awake

Plug it in. If you'll close the lid, set the lid to do nothing when plugged in
(Windows: Control Panel > Power Options > "Choose what closing the lid does";
Mac: System Settings > Battery > Options > "Prevent automatic sleeping when the
display is off"). Leave it on a desk, not in a bag; it runs at full CPU.
Wi-Fi is not needed once step 2 is done.

### 5. Start the runs

**Windows** (PowerShell, from the project folder):

```
powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1
```

Five windows open, one per run, each showing its progress. Each window says
"Run ... finished" in green when done. Don't close a window before then, or that
run stops.

**Mac / Linux** (terminal, from the project folder):

```bash
export PYTHONUNBUFFERED=1
C="--seeds 0 1 --plots summary --threads 4"
uv run scripts/train_cv.py --name baseline --external off $C > /dev/null &
uv run scripts/train_cv.py --name margin --external off --ignore-partial $C > /dev/null &
uv run scripts/train_cv.py --name margin-aug3 --external off --ignore-partial --augment 3 $C > /dev/null &
uv run scripts/train_cv.py --name margin-synth200 --external off --ignore-partial --synthetic 200 $C > /dev/null &
uv run scripts/train_cv.py --name margin-tcn --external off --ignore-partial --arch tcn $C > /dev/null &
sleep 15; tail -f runs/logs/*.log
```

`tail` shows every run's progress; press Ctrl+C to stop watching (the runs keep
going). Keep the terminal open until all five logs end with a `Report:` line.

Progress lines look like:

```
  seed 0 fold 2/5: epoch  25/40  train loss 0.0120  val loss 0.84 ...
  seed 0 fold 2/5: 7120 train windows (390 falls), best epoch 18, 6 test clips, 64s
```

Each run does seed 0 folds 1-5, then seed 1 folds 1-5. The `margin-aug3` run is
the slowest.

### 6. Check the results

When all five are done you should have:

- `runs/2026-xx-xx_<name>/report.md` for each run (open it in VS Code with
  the Markdown preview, or on GitHub after pushing)
- `runs/README.md`: one line per run, to compare them
- `runs/logs/`: the full logs

### 7. Commit and push the results

```
git add runs
git commit -m "Training batch 1 results"
git push
```

Then tell the team it's pushed. The next batch (combining what helped) is
chosen from these reports.

## Trial 2 runs (batch 3)

Trial 2 adds the night rig clips (`data/input/rig_falls_night`), the team's gym
falls (`data/input/man_made_fall_videos`, label sheet
`data/labels/external_gym.csv`, used for training only) and takes the 12 slow
CAUCAFall falls out of training. Batch 3 runs the new baseline plus one change
at a time, with 3 seeds:

All Trial 2 poses come from **YOLO26m** (the live dashboard's detector since
2026-11-15), not Trial 1's YOLO26n, and the faller IDs in the label sheets were
redone for them. YOLO26m sees the faller in 97% of rig fall frames (YOLO26n: 73%),
so several clips Trial 1 could not use are back in. Trial 1's YOLO26n poses are
not in this repo's working copy any more; git history has the rig and CAUCAFall
ones. To re-extract, see `scripts/extract_poses_rtmpose.py` (GPU:
`uv run --no-group cpu --group gpu ...`).

```
powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Batch 3 -Parallel 3
```

`-Parallel 3` keeps three runs going at once and starts the next as one
finishes (ten at once would fight for the CPU). Leave that first window open.

| Run | What it changes |
|---|---|
| t2-base | Trial 1's best setup on the new data |
| t2-no-gym | without the gym falls (do they help?) |
| t2-with-slow | slow CAUCAFall falls back in training (does leaving them out help?) |
| t2-win18 | 1.5 s window instead of 2.5 s |
| t2-drop015 | fast-drop check: no alert without a fast body drop |
| t2-smooth01 | label smoothing 0.1 (less overconfident scores) |
| t2-low-x3 | kneeling / crouching / lying normal windows count 3x in training |
| t2-gap12 | a lost track may be handed to a new ID up to 1 s later (was 0.4 s) |
| t2-dedupe | a second, smaller box on the same body is removed |
| t2-mirror | one mirrored copy of each training clip |

Compare runs on the **Honest** and **On Trial 1 clips** columns of
`runs/README.md`: Honest picks each fold's threshold without that fold, and
On Trial 1 clips scores the same 21 falls as Trial 1. Then put the changes
that helped together in batch 4 of `scripts/run_experiments.ps1`.

## Rules

- **Only one person runs a batch.** Two people pushing results for the same
  batch will conflict in `runs/README.md`.
- Don't edit `data/labels/clips.csv` or `manifest.json` as part of this; the
  results must match the committed labels.
- Rerunning on the same day replaces that day's run of the same name.

## If something goes wrong

| Problem | Fix |
|---|---|
| `uv` not found | `pip install uv`, then reopen the terminal |
| PowerShell says scripts are disabled | use the exact command in step 5 (it includes `-ExecutionPolicy Bypass`) |
| A window closed early or a run stopped | rerun just that one, e.g. `uv run scripts/train_cv.py --name margin --external off --ignore-partial --seeds 0 1 --plots summary --threads 4` |
| The laptop is very slow | lower `--threads` (Mac/Linux: edit `C=`; Windows: `-Threads 2` on the script) |
| An error mentioning `manifest.json` or a pose file | run `git pull` again; the pose files and manifest are in the branch |

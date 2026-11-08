# 10-23-2026 - Temporal classifier (stages 2, 6 and 7) and motion-signal plots

## Goal

Stage 5 (the sliding window) was the last stage built. This change adds the
rest of the skeleton path from the system guide
(`docs/fall_detection_system_guide.pdf`):

```
video -> 2. frame sampler -> 3. YOLO pose -> 4. ByteTrack -> 5. TrackWindows
      -> features (positions + velocities + accelerations)
      -> 6. GRU / TCN classifier -> fall probability
      -> 7. threshold gate (p > 0.8 for 5 frames, 30 s cooldown) -> fall event
```

It also adds what we need to *see* what the classifier sees: for every
tracked person, plots of hip height, torso angle, vertical velocity
and fall probability over time, both as a scrolling panel next to the
annotated video and as full-length PNGs. These are the "joint height and
velocity over a 2 second window" curves from the guide's figure, computed from
real footage.

There is no labeled site footage yet, so the first checkpoint
(`models/fall_gru.pt`) is a **bootstrap model trained on procedurally
generated skeletons**. It proves the stages work together and gives the
dashboard something to run; it is not a validated detector. See
[Known limitations](#known-limitations).

## Implementation

All new code is in `src/temporal/`, plus one offline runner and one training
script. Stage 5 (`src/sliding_window.py`) gained three small additions and is
otherwise unchanged: each `Window` now also carries its boxes in the same
normalized frame as the keypoints, and `TrackWindows` has `rename()`,
`last_seen()`, `last_box()` and `track_ids()` for tracker ID handoffs.

### Stage 2: frame sampler (`temporal/sampler.py`)

`FrameSampler(source_fps, target_fps=15)` keeps every `stride`-th frame, with
`stride = round(source_fps / target_fps)`: 30 fps -> every 2nd frame (15 fps),
25 fps -> every 2nd frame (12.5 fps), 15 fps -> every frame. It also reports
the resulting frame rate, which the features need. `TimeSampler` does the
same for live cameras by timestamp, since their frame rate is uneven, and
measures the rate it achieves. The 640 px resize is already done by YOLO.

This fixes the window-length problem noted in the sliding-window doc: 30
frames is now about 2 s at 15 fps (2.4 s at 12.5 fps) instead of 1 s at the
camera's native 30 fps.

### Features (`temporal/features.py`)

The guide (section 5.1) says to feed joint velocities and accelerations as
explicit inputs, not only positions. `extract_features(window, fps)` turns a
`Window` into a `(30, 91)` matrix per frame:

| block         | size | meaning                                                      |
|---------------|------|--------------------------------------------------------------|
| positions     | 36   | the window's normalized x, y, confidence for 12 body joints       |
| velocities    | 24   | per-joint dx, dy in **body heights per second**              |
| accelerations | 24   | per-joint change of velocity per second                      |
| box           | 2    | box height / reference height, width / height (from stage 5) |
| box velocity  | 2    | change of the two box features per second                    |
| signals       | 3    | the interpretable signals below                              |

- **Per-second units.** Velocities are multiplied by the frame rate, so the
  same motion gives the same number at 10, 12.5 or 15 fps.
- **Padding.** Padded frames are all zero and no velocity is computed across
  a padding boundary or a tracker gap, so padding never looks like motion.
- **Missing joints.** YOLO still reports a position for joints it cannot see
  (checked on `data/output/001.json`: 16 % of joints have confidence below
  0.3 and jittery coordinates). Joints below confidence 0.3 take the last good
  position (the guide's carry-forward rule), the raw confidence is kept so the
  model can learn to distrust them, and no velocity spike results.
- **Signals.** Three values the plots show and the model also receives:
  hip height above the box bottom (in body heights), torso angle
  (0 upright, 1 horizontal), hip vertical velocity (upward positive,
  so a fall is a negative spike as in the guide's figure). Heights are
  measured from the person's own box bottom, so they do not depend on the
  window's origin and read the same across a whole video.
- `FEATURE_VERSION` is stored in every checkpoint; a checkpoint from another
  feature layout is refused on load.

### Stage 6: the model (`temporal/model.py`)

Both architectures from the guide, selected by name:

| model     | structure                                                        | parameters |
|-----------|------------------------------------------------------------------|-----------:|
| `FallGRU` | GRU (hidden 64) over the 30 frames, final state -> linear -> logit | 37,313     |
| `FallTCN` | causal dilated 1-D convolutions (dilations 1, 2, 4, 8, receptive field 61 frames), last step -> linear | 107,137 |

Feature standardisation (mean and std per column, from the training set) is
stored inside the model as buffers, so a checkpoint is self-contained.
`save_checkpoint` / `load_checkpoint` write and read one `.pt` file holding
the architecture, config, weights, feature version, training frame rate,
window size, gate threshold and test metrics. Loading uses
`weights_only=True`.

### Stage 7: the gate (`temporal/gate.py`)

`ThresholdGate(threshold=0.8, debounce=5, cooldown_seconds=30)`: a track
fires when its probability has been above the threshold for 5 consecutive
processed frames, then not again for 30 s. 25 lines, as the guide predicted.

### Tracker ID handoff (`temporal/detector.py`)

The dashboard work found that ByteTrack gives a falling person a new ID
mid-fall (their box turns from tall to wide). A new window would start empty
and never contain the fall. `handoff_lost_tracks()` applies the same rule the
rule-based detector uses: a new ID appearing within `max_gap` frames of a
lost one, within one body height of it, takes over that window. The gate's
debounce and cooldown move with it, so one fall fires once even across an ID
switch (the first run of `synthetic_fall.mp4` fired twice before that was
added).

### Drop-in detector (`temporal/detector.py`)

`TemporalFallDetector(checkpoint)` has the same `update(people, timestamp)`
-> `[FallDetection]` interface as `backend.fall_detector.FallDetector`, so
events, clips and the dashboard need no change. It runs the time sampler,
the window, the features, one batched model call for every full window in
the frame, and the gate. Enable it with `FALL_CLASSIFIER_PATH=models/fall_gru.pt`
in `.env`; the rule-based detector stays the default.

### Training (`temporal/synthetic.py`, `temporal/dataset.py`, `temporal/train.py`, `scripts/train_temporal.py`)

Section 4 of the guide, as code:

- **Windows** are cut with a stride of 5 by replaying each sequence through
  the real `TrackWindows`, so training and inference share stage 5. A window
  is positive when more than half of a fall interval lies inside it, which
  means windows ending mid-fall are positive and windows where the fall has
  slid out (person just lying) are negative.
- **Split by source**, never by window.
- **Loss** is class-weighted binary cross-entropy (`pos_weight` = negatives /
  positives), AdamW, cosine learning rate, best epoch by validation F1.
- **Metrics**: precision, recall, F1 and the confusion matrix on the test
  split at 0.5 and at the gate threshold, written next to the checkpoint as
  `models/fall_gru.json`.
- **Real footage** goes through a manifest of `yolo_pose.py` JSON files with
  fall intervals (source frame indices or seconds); the trainer resamples
  them to the target frame rate. Format in `temporal/dataset.py`.
- **Synthetic data** (`temporal/synthetic.py`): a 17-joint stick figure
  built from five parameters (torso angle, hip height, position, arm swing,
  foreshortening) is animated through ten activities: stand, walk, crouch,
  bend, sit, lie still, lie down slowly, trip and recover, fall sideways, fall
  toward the camera. Falls accelerate (0.3 to 0.8 s), the hard negatives are
  slow or recover. Each sequence gets a random camera scale and position,
  mirror, 10 to 15 fps, pose jitter, runs of low-confidence joints and short
  tracker drop-outs, and is rendered in the `process_frame` output format.

### Offline runner and plots (`src/fall_pipeline.py`, `temporal/visualize.py`)

```bash
uv run src/fall_pipeline.py --input data/input/test.mp4 --classifier models/fall_gru.pt
```

runs stages 2 to 7 over a video (or `--pose-json` to replay an existing pose
file without YOLO) and writes `data/output/<name>_temporal/`:

- `signals.mp4`: the annotated video with a 640 px panel beside it showing
  the focused track's last 6 s of heights, torso angle, vertical velocity and
  probability, scrolling with the video. The focus is the most alarming track
  of the last 2 s, else the longest-lived. Each person gets a `p 0.12` label
  and a red outline while above the threshold.
- `track_<id>.png`: the same four panels over the whole video per track,
  with `FALL` markers where the gate fired.
- `signals.json`: every track's per-frame signals, probabilities and events.

Without `--classifier` it still produces the signals and plots, so the
visualisation is useful before a model exists. The classifier scores all
full windows in a frame in one batch.

Chart style follows the data-viz guidance: 2 px lines, hairline solid
gridlines, legend wherever a panel has two or more series, text in ink
colours only, fall events as a labelled line, not a colour alone.

## Verification

- **Tests**: `tests/test_temporal.py`, 41 tests (109 in the full suite).
  Feature layout and padding rules, velocity units (10 px/frame on a 400 px
  body at 15 fps = 0.375 heights/s), upright vs lying signals, missing-joint
  carry-forward, both models' sizes and outputs, checkpoint round trip and
  feature-version refusal, gate debounce / cooldown / per-track isolation,
  sampler strides, synthetic sequences for all ten activities, window
  labelling rule, split with no leakage, a short training run that learns,
  track handoff, panel and PNG rendering, and two end-to-end tests through
  `TemporalFallDetector` with the bootstrap checkpoint: a synthetic fall
  fires exactly once (also with a forced ID switch mid-fall); crouch, walk
  and lie-still never fire.
- **Bootstrap training**: `uv run scripts/train_temporal.py --synthetic 1500`.
  5,315 / 1,091 / 1,141 windows (train / val / test, split by sequence), 45 s
  on a laptop CPU. Test split at the 0.8 gate threshold: precision 0.995,
  recall 0.986, F1 0.991 (2 false positives, 5 misses of 368 falls). These
  numbers describe synthetic skeletons only.
- **`data/input/test.mp4`** (two workers building a stone wall, 25 fps,
  10 s, no fall): sampled to 12.5 fps, 126 frames in 51 s on CPU. Four
  tracks, highest probability on any frame 0.006, no events. The plots show
  the pose jitter of real footage: hip velocity flickers by +/-0.5 heights/s
  at 12.5 fps while the person is still, and the foreground worker's box ends
  at his waist (the wall hides his legs), so his "hip height above the box
  bottom" sits near zero. Both are worth knowing before training on real
  data.
- **`data/samples/synthetic_fall.mp4`** (stands 2 s, falls in 0.6 s, lies
  3 s): the probability rises from 0 to 0.65 on the first falling frame and
  to 1.0 on the next, the gate fires once at 2.33 s (5 frames after the fall
  starts), and the hip velocity trace dips to -1.9 heights/s while the torso
  angle goes from 0 to 95 degrees. ByteTrack switched the ID from 1 to 2 at
  frame 40; the handoff kept one window and one event.

## Known limitations

- **The checkpoint is a bootstrap, not a detector.** It has only seen
  idealised 2-D stick figures. Expect both misses and false alarms on real
  footage; it fires at the *start* of the synthetic fall because synthetic
  falls accelerate from the first frame, which real falls may not. Train on
  OmniFall and site clips before trusting it (the manifest path exists for
  this).
- **Threshold.** The checkpoint stores the guide's initial 0.8. The trainer
  also reports the threshold the validation set would suggest (0.5 for the
  synthetic set, which only says the synthetic task is easy). Tune on real
  validation data.
- **Heights are relative to the box and the window.** When the legs are
  hidden (worker behind a wall) the box bottom is not the feet. The reference
  height is the tallest box *in the window*, so about 2 s after a fall, when
  the standing frames have slid out, the heights of a person lying still
  jump back to standing-like values (visible at 4.3 s in the synthetic plot)
  and a track only ever seen lying down reads as "standing". The guide's
  floor calibration and per-track standing baseline (section 5.2) are the
  real fixes; the classifier is trained on the transition, not the posture,
  which softens this.
- **Frame rate.** The bootstrap model was trained at 10 to 15 fps with
  per-second velocities. The runner warns when the sampled rate differs from
  the checkpoint's by more than 35 %.
- **Speed.** YOLO dominates (about 300 ms per frame on this CPU); the
  sampler, window, features, model and gate together cost 1.8 ms per frame
  for one track (`TemporalFallDetector.update`, measured). Rendering the
  matplotlib panel costs about 180 ms per frame because the axes are redrawn
  from scratch, so the offline runner with video runs at 2 to 3 fps on CPU;
  `--no-video` skips it.
- **Two alert tiers** ("person down, please check" after 60 s motionless) and
  the X-CLIP verifier (stage 8) are not started.
- `torch` and `numpy` are still only installed through `ultralytics`;
  `matplotlib` is now a direct dependency.

## How to use going forward

```bash
uv sync                                                      # matplotlib was added
uv run pytest tests -v                                       # 109 passed

# plots only, no classifier needed
uv run src/fall_pipeline.py --input data/input/test.mp4

# with the bootstrap classifier
uv run src/fall_pipeline.py --input data/samples/synthetic_fall.mp4 --classifier models/fall_gru.pt
uv run src/fall_pipeline.py --pose-json data/output/001.json --classifier models/fall_gru.pt --no-video

# retrain the bootstrap (45 s), or train a TCN
uv run scripts/train_temporal.py --synthetic 1500 --arch gru
uv run scripts/train_temporal.py --synthetic 1500 --arch tcn

# real footage, once labeled
uv run scripts/train_temporal.py --manifest data/labels/manifest.json

# use it in the dashboard
echo FALL_CLASSIFIER_PATH=models/fall_gru.pt >> .env
uv run run.py
```


## Body-only feature version 2

The pipeline now outputs 12 body keypoints, excluding the nose, eyes, and ears. Temporal windows contain 91 features per frame. The three motion signals are hip height, torso angle, and hip vertical velocity. Plots and saved signal JSON contain no head signals.

Existing COCO-17 pose JSON files are converted to the 12 body points when loaded for training or offline replay. Files declaring an unknown keypoint layout are rejected.

The committed `models/fall_gru.pt` was trained with feature version 1 and is incompatible. Loading it reports that retraining is required. Leave `FALL_CLASSIFIER_PATH` empty to use the body-only rule detector until a version 2 classifier is ready. The model is not silently replaced or retrained by this change.

To train a new bootstrap model without overwriting the old checkpoint:

```sh
uv run scripts/train_temporal.py --synthetic 1500 --output models/fall_gru_body12.pt
```

Use labeled site footage for the final model and validate missed falls and false alarms before deployment.

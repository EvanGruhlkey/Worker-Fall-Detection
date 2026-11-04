# 11-04-2026 - Run YOLO pose at native resolution (people far from the camera were missed)

## Problem

On the "Large Group Setting" clips (1280 x 704, 24 fps, 8 to 12 workers each)
people far from the camera were not detected, or flickered in and out, so
they never got a stable track or a 30-frame window.

## Root cause

There is no cap on detections (Ultralytics' default `max_det` is 300). The
problem is the inference size. `process_frame` passed `imgsz=640`, YOLO's
default, so every 1280 px frame was scaled to half size before the model saw
it. A worker who is 90 px tall in the frame became 45 px, which is below what
`yolo11n-pose` reliably finds. A synthetic test (copies of `bus.jpg` shrunk
onto a 1920 x 1080 canvas) shows the cliff:

| person height in frame | found at `imgsz=640` | found at native size |
|---|---|---|
| 150 px | 4 of 12 | 9 of 12 |
| 100 px | 1 of 12 | 9 of 12 |
| 70 px | 0 of 12 | 6 of 12 |
| 50 px | 0 of 12 | 0 of 12 |

Note the last row: even at native resolution the nano model stops finding
people below about 60 px. That is a model limit, not a resize limit (see
"Beyond this fix").

## Fix

The inference size is now a setting, and the default is the frame's own
resolution (no downscaling).

- **`src/yolo_pose.py`**: `parse_image_size()` turns `"native"` / a number
  into `None` / an int; `resolve_image_size()` turns `None` into the frame's
  `[height, width]` rounded up to the model stride (32), so YOLO letterboxes
  with a few pixels of padding instead of shrinking. `process_frame`,
  `process_video` and `process_camera` default to native. `--image-size`
  accepts `native` or a number. The pose JSON metadata records the size used.
- **`src/backend/config.py`**: `FALL_IMAGE_SIZE=native` (default) or a
  number; documented in `.env.example` and `docs/backend-api.md`.
- **`src/fall_pipeline.py`** and **`src/yolo_pose_live.py`**: same option.
- **`scripts/bench_image_size.py`**: reproduces the table below.
- **`tests/test_yolo_pose.py`**: 20 tests for the parsing, the stride
  rounding and the setting. Full suite: 129 passed.

## Verification

`uv run scripts/bench_image_size.py "data/input/large_group/**/*.mp4"`,
every 6th frame, `yolo11n-pose.pt`, predict only (no tracker), Intel Mac CPU:

| video | people/frame at 640 | people/frame native | gain | mean conf 640 | mean conf native | smallest person 640 | smallest person native | ms/frame 640 | ms/frame native |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| backward_fall_near_equipment | 9.5 | 10.9 | 1.15x | 0.66 | 0.79 | 147 px | 80 px | 111 | 290 |
| backward_slip_while_stepping_back | 5.5 | 7.4 | 1.33x | 0.71 | 0.79 | 117 px | 88 px | 88 | 315 |
| fall_from_ladder | 5.0 | 8.5 | 1.70x | 0.76 | 0.72 | 147 px | 56 px | 86 | 334 |
| fall_while_working_on_pipe | 4.0 | 6.3 | 1.57x | 0.55 | 0.64 | 90 px | 91 px | 120 | 335 |
| forward_fall_while_carrying_a_tool | 5.3 | 10.0 | 1.88x | 0.59 | 0.75 | 113 px | 89 px | 103 | 462 |
| forward_trip_while_walking | 7.8 | 9.5 | 1.21x | 0.66 | 0.79 | 102 px | 86 px | 198 | 399 |
| slip_behind_person | 4.5 | 9.6 | 2.14x | 0.55 | 0.69 | 93 px | 54 px | 106 | 334 |
| slip_between_coworkers | 5.9 | 7.0 | 1.17x | 0.59 | 0.78 | 84 px | 67 px | 100 | 290 |
| slip_in_the_open | 3.3 | 7.3 | 2.23x | 0.59 | 0.68 | 124 px | 58 px | 87 | 328 |
| sudden_collapse | 3.4 | 7.2 | 2.11x | 0.68 | 0.68 | 97 px | 66 px | 92 | 286 |
| **all ten** | 5.4 | 8.4 | 1.54x | | | | | 109 | 337 |

Native finds 1.5x as many people per frame on average (up to 2.2x on the
wide shots), with higher mean confidence on 8 of 10 clips, and reaches
people down to about 55 px tall instead of about 85 px. The cost is 3x the
CPU time per frame: about 3 fps instead of 9 fps on this laptop. The frame
sampler (stage 2) already runs the classifier at 10 to 15 fps, so with a
GPU this is not a concern; on a CPU-only machine set `FALL_IMAGE_SIZE=960`
as a middle ground, or `640` to get the old speed back.

## Beyond this fix

- **Model size.** `yolo11n-pose` is the smallest model. `yolo11s-pose` or
  `yolo11m-pose` find smaller and more occluded people at the same
  resolution; worth benchmarking with the same script once a GPU is
  available (`--model models/yolo11s-pose.pt`).
- **Tiling (SAHI).** For fixed wide-angle site cameras at 4K, slicing the
  frame into overlapping tiles is the standard way to find very small
  people; Ultralytics documents it at
  https://docs.ultralytics.com/guides/sahi-tiled-inference. Not needed for
  1280 px clips.
- **ByteTrack thresholds.** Distant people have lower confidence; the tracker
  only starts a track at `new_track_thresh` 0.25 and shows it from its second
  frame. If far tracks still flicker, lower `track_high_thresh` /
  `new_track_thresh` to about 0.2 in a project tracker YAML.

## Next steps (plan as of 11-04-2026)

The ten "Large Group Setting" clips (`data/input/large_group/`, 1280 x 704,
24 fps, 15 s each, 8 to 12 workers, one staged fall per clip) are now the
working test set for the perception stages. Observed on them so far:

- the nano model still drops a person for one or two frames on occluded or
  unusual poses, and its boxes jitter;
- ByteTrack creates far more IDs than there are people (23 IDs for about 11
  workers on `backward_fall_near_equipment`, 9 of them within a second and a
  body height of a vanished ID, i.e. likely switches);
- pipes and equipment are sometimes detected as people.

Planned, in order:

1. **Pose model upgrade (Evan, in progress).** Benchmark `yolo11s-pose` and
   `yolo11m-pose` against the nano with `scripts/bench_image_size.py` at
   native resolution, then switch `FALL_MODEL_PATH` to the best one the
   target hardware can run at 10 to 15 fps. The weights are not committed;
   Ultralytics downloads them by name.
2. **Site-specific fine-tuning with model-assisted labels.** Do not hand-draw
   skeletons from scratch. Run `yolo11x-pose` at native size over sampled
   frames, export the predictions as YOLO pose labels, and *correct* them in
   CVAT / Roboflow / Label Studio, spending the human time on falling,
   lying, occluded and distant workers and on pipe false positives (frames
   with no person label are hard negatives). Train from the pretrained
   weights with the Ultralytics small-dataset recipe (`freeze=10`,
   `optimizer=AdamW`, `lr0=0.001`, `mosaic=0.5`, `epochs=50`, `patience=20`,
   `imgsz=1280`), split by video, mixed with a slice of COCO-Pose, and
   validate with `model.val()` pose mAP plus missed-frames-per-track on
   held-out clips. Real site footage must be in the validation set; the AI
   clips teach the generator's look as much as real worksites.
3. **Tracker.** Lower `conf` on the track call to about 0.15 and raise
   `track_buffer` in a project tracker YAML so low-confidence distant
   workers stay in their tracks. If switches between look-alike workers
   remain, try BoT-SORT with `with_reid: True`. Fine-tuning the pose model
   does not fix the switch during a fall (tall box to wide box fails the IoU
   match); the ID handoff in `fall_detector.py` / `temporal/detector.py`
   is the mitigation there.
4. **Temporal classifier on real labels.** Each clip has one fall; label the
   fall interval and train through `scripts/train_temporal.py --manifest`
   so the dashboard can move off the rule-based detector.
5. **From the system guide, still open:** floor calibration per camera,
   two alert tiers, X-CLIP verifier.

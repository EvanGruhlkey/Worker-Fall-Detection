# Training labels

`clips.csv` is the hand-edited label sheet for the temporal fall classifier.
`scripts/build_manifest.py` checks it and writes `manifest.json`, which
`scripts/train_temporal.py --manifest` reads. Edit the CSV, never the manifest.

Videos live in `data/input/rig_falls/` (every clip has a fall) and
`data/input/normal_rig_work/` (no clip has a fall). They are gitignored; this
folder is not, so labels are versioned.

## Columns

| column | who fills it | meaning |
|---|---|---|
| clip_id | prefilled | file name without `.mp4`; must match the pose JSON name |
| folder, file | prefilled | where the video is |
| has_fall | prefilled | 1 for `rig_falls`, 0 for `normal_rig_work` |
| **fall_start_s** | you | seconds, first moment of uncontrolled descent (balance is lost) |
| **fall_end_s** | you | seconds, moment of impact with the ground or a structure |
| **faller_track_id** | you, after pose extraction | track ID of the person who falls (see below); `3;7` if the ID changes mid-fall |
| **quality** | you | `keep`, `review` (not yet checked) or `drop` (bad AI physics, morphing bodies, fall not visible) |
| **group** | you | clips generated from the same prompt or scene get the same group, so they are never split between train and test. Defaults to the clip's own name. |
| fall_type | prefilled from file name where possible | slip, trip, ladder, stairs, fall_from_height, loss_of_balance, collapse; `none` for normal clips |
| fall_direction | you, optional | forward, backward, sideways |
| camera_view | you, optional | e.g. overhead, eye_level, far, close |
| people_in_frame | you, optional | 1, 2, 3+ |
| conditions | prefilled where possible | night, rain, wet, mist, ...; separate several with `;` |
| notes | you, optional | anything odd |
| duration_s, fps, frames, resolution | prefilled | measured from the file; used to catch typos |

**Two falls in one clip:** add a second row with the same clip_id and the
second fall's times. Fill in quality and group the same way on both rows.

## Labeling rules

- Times are in seconds from the start of the clip, e.g. `6.4`. Scrub in a
  player that shows tenths (VLC: Ctrl+T; or step frames with `E`). One frame
  at 24 fps is about 0.04 s, so tenths are precise enough.
- **Start** = the person is no longer in control (foot slips, trip catches,
  ladder rung missed). **End** = impact. Lying on the ground afterwards is not
  part of the fall. A typical fall is 0.5 to 1.5 s long.
- Apply the same rule to every clip. Inconsistent start/end points become
  label noise.
- A clip with no clear fall (it is cut off, hidden, or looks physically wrong):
  set `quality=drop` and say why in notes.

## faller_track_id

Pose extraction assigns each person a track ID. In a clip with more than one
person, every person's windows inside the fall interval would be labeled as a
fall unless the faller's ID is given, so it is required for multi-person
clips. To find it, run `yolo_pose.py` on the clip and read the ID drawn over
the person in `data/output/<clip_id>.mp4`. Pose extraction can produce
different IDs if it is rerun with other settings, so fill this column after
the final extraction.

## Build the manifest

```
uv run scripts/build_manifest.py              # includes review rows
uv run scripts/build_manifest.py --keep-only  # only rows marked keep
```

The build stops on any error in the sheet, skips clips that have no pose JSON
yet, and warns about fall clips with no faller_track_id.

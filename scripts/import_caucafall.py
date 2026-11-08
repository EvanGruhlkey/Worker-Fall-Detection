"""Turn the CAUCAFall download into label rows and pose-ready videos.

CAUCAFall (Eraso, Muñoz, Muñoz, Pinto; Universidad del Cauca; CC BY 4.0;
doi:10.17632/7w7fccy7ky.4) has 10 subjects x 10 activities: five falls and five
daily activities (hop, kneel, pick up object, sit down, walk), one .avi each
(20 fps, 720x480) with a YOLO label file per frame: class 0 = no fall,
1 = fall, plus the person's box.

Their "fall" label starts during the fall and runs to the end of the clip
(lying included), while our labels run from the loss of balance to impact. So
both points are taken from the labeled box instead:

- impact: the first frame, after the box has dropped past halfway from its
  standing height to its lying height, where it is within 15% of the lying height
- start:  the last frame before that drop where the box was still within 15%
  of its standing height

    uv run scripts/import_caucafall.py                 # writes the label CSV + videos
    uv run scripts/import_caucafall.py --sheets DIR    # also frame grids to check by eye

Writes data/labels/external_caucafall.csv (same columns as clips.csv plus
`source`) and copies each video to data/external/caucafall/videos/<clip_id>.avi
so the pose pipeline can be run on one flat folder, writing
data/output/caucafall/<clip_id>.json.
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "caucafall" / "raw" / "CAUCAFall"
VIDEOS = ROOT / "data" / "external" / "caucafall" / "videos"
OUTPUT = ROOT / "data" / "labels" / "external_caucafall.csv"
POSES = ROOT / "data" / "output" / "caucafall"

COLUMNS = ["clip_id", "folder", "file", "has_fall", "fall_start_s", "fall_end_s", "faller_track_id",
           "quality", "group", "fall_type", "fall_direction", "camera_view", "people_in_frame",
           "conditions", "notes", "duration_s", "fps", "frames", "resolution", "source"]
ACTIVITIES = {
    "Fall backwards": ("fall", "backward"), "Fall forward": ("fall", "forward"),
    "Fall left": ("fall", "sideways"), "Fall right": ("fall", "sideways"),
    "Fall sitting": ("fall_from_sitting", ""),
    "Hop": ("none", ""), "Kneel": ("none", ""), "Pick up object": ("none", ""),
    "Sit down": ("none", ""), "Walk": ("none", ""),
}
NEAR = 0.15  # "within 15%" of the standing / lying box height


def read_labels(folder):
    """{frame number: (class, box height, box width)} from the per-frame YOLO files."""
    labels, suffixed = {}, {}
    for path in folder.glob("*.txt"):
        if path.name == "classes.txt":
            continue
        match = re.search(r"(\d+)([a-z]?)\.txt$", path.name)
        if not match:
            continue
        parts = path.read_text(encoding="utf-8", errors="replace").split()
        if len(parts) < 5:
            continue
        entry = (int(parts[0]), float(parts[4]), float(parts[3]))
        frame = int(match.group(1)) % 100000
        # A file with a letter after its number is a second save of that frame;
        # the plain file wins when both exist.
        (suffixed if match.group(2) else labels)[frame] = entry
    for frame, entry in suffixed.items():
        labels.setdefault(frame, entry)
    return labels


def fall_interval(frames, labels):
    """(start, impact) frame positions from the box heights, or None, plus a note."""
    classes = np.array([labels[f][0] for f in frames])
    heights = np.array([labels[f][1] for f in frames])
    fallen = np.flatnonzero(classes == 1)
    if len(fallen) == 0:
        return None, "no frame labeled fall"
    first = int(fallen[0])
    smooth = np.convolve(np.pad(heights, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    standing = float(np.median(smooth[max(0, first - 30):max(1, first - 5)]))
    lying = float(np.median(smooth[-max(5, len(smooth) // 10):]))
    drop = standing - lying
    if drop < 0.1 * standing:
        # The box barely shrinks (e.g. a fall toward the camera): fall back to
        # the dataset's own fall label around its first fall frame.
        return (max(0, first - 10), min(len(frames) - 1, first + 10)), "box height barely changes; times from label"
    middle = standing - drop / 2
    crossing = next((i for i in range(max(0, first - 30), len(smooth)) if smooth[i] <= middle), first)
    impact = next((i for i in range(crossing, len(smooth)) if smooth[i] <= lying + NEAR * drop), crossing)
    start = next((i for i in range(crossing, -1, -1) if smooth[i] >= standing - NEAR * drop), 0)
    return (start, max(impact, start + 1)), ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheets", help="Write frame grids around each fall to this folder, for checking by eye")
    args = parser.parse_args(argv)
    if not RAW.exists():
        print(f"Error: {RAW} not found; download CAUCAFall first (see docs/external-data.md)", file=sys.stderr)
        return 1
    VIDEOS.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject_dir in sorted(RAW.glob("Subject.*"), key=lambda p: int(p.name.split(".")[1])):
        subject = int(subject_dir.name.split(".")[1])
        for activity, (fall_type, direction) in ACTIVITIES.items():
            folder = subject_dir / activity
            videos = sorted(folder.glob("*.avi"))
            if not videos:
                print(f"warning: no video in {folder}", file=sys.stderr)
                continue
            capture = cv2.VideoCapture(str(videos[0]))
            fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            size = f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            capture.release()

            clip_id = f"cauca_s{subject:02d}_{activity.lower().replace(' ', '_')}"
            shutil.copyfile(videos[0], VIDEOS / f"{clip_id}.avi")
            labels = read_labels(folder)
            frames = sorted(labels)
            notes, quality = [], "keep"
            if abs(len(frames) - count) > 2:
                notes.append(f"{len(frames)} label files for {count} video frames; times may be off")
                quality = "review"
            has_fall = fall_type != "none"
            start_s = end_s = ""
            if has_fall:
                interval, note = fall_interval(frames, labels)
                if note:
                    notes.append(note)
                if interval is None:
                    quality = "review"
                else:
                    # Label files count from 1; video frame k is at k / fps.
                    start_s, end_s = f"{interval[0] / fps:.2f}", f"{interval[1] / fps:.2f}"
                first_fall = next((i for i, f in enumerate(frames) if labels[f][0] == 1), None)
                if first_fall is not None:
                    notes.append(f"dataset fall label from {first_fall / fps:.2f} s")
            elif any(labels[f][0] == 1 for f in frames):
                notes.append("daily-activity clip has frames labeled fall")
                quality = "review"
            rows.append({
                "clip_id": clip_id, "folder": "caucafall", "file": f"{clip_id}.avi", "has_fall": int(has_fall),
                "fall_start_s": start_s, "fall_end_s": end_s, "faller_track_id": "", "quality": quality,
                "group": f"cauca_s{subject:02d}", "fall_type": fall_type, "fall_direction": direction,
                "camera_view": "high corner", "people_in_frame": "", "conditions": "",
                "notes": "; ".join(notes), "duration_s": f"{count / fps:.2f}", "fps": f"{fps:g}",
                "frames": count, "resolution": size, "source": "caucafall",
                "_folder": folder,  # not written to the CSV
            })
    kept = keep_reviewed_fields(rows)
    if kept:
        print(f"  kept the reviewed times, quality and notes of {kept} clip(s) already in {OUTPUT.name}")
    assigned = assign_faller_tracks(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    falls = sum(int(r["has_fall"]) for r in rows)
    review = [r["clip_id"] for r in rows if r["quality"] == "review"]
    print(f"Wrote {OUTPUT}: {len(rows)} clips ({falls} falls), videos in {VIDEOS}")
    if review:
        print(f"  {len(review)} marked quality=review: {', '.join(review)}")
    print(f"  faller_track_id filled for {assigned} fall clip(s) from pose files in {POSES}"
          if assigned else f"  no pose files in {POSES} yet; rerun after pose extraction to fill faller_track_id")
    if args.sheets:
        write_sheets(rows, Path(args.sheets))
    return 0


def keep_reviewed_fields(rows):
    """Carry fall times, quality and notes over from an existing CSV.

    The computed times are only a first guess; once the sheet has been checked
    by eye (and edited), rerunning this script, e.g. to fill faller_track_id
    after pose extraction, must not overwrite that work.
    """
    if not OUTPUT.exists():
        return 0
    with open(OUTPUT, newline="", encoding="utf-8-sig") as handle:
        existing = {r["clip_id"]: r for r in csv.DictReader(handle)}
    kept = 0
    for row in rows:
        old = existing.get(row["clip_id"])
        if old is None:
            continue
        for field in ("fall_start_s", "fall_end_s", "quality", "notes", "people_in_frame", "conditions"):
            row[field] = old.get(field, row[field])
        kept += 1
    return kept


def assign_faller_tracks(rows):
    """Fill faller_track_id by matching the dataset's labeled box to the pose tracks.

    For every fall clip with a pose file, each labeled frame's box (YOLO,
    normalized) is compared with the tracked boxes at the same time; a track
    that overlaps it (IoU >= 0.3) in at least 20% of the labeled frames is the
    faller. Several IDs are kept when the tracker switched mid-clip.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from temporal.pose_io import load_pose_json

    assigned = 0
    for row in rows:
        pose_path = POSES / f"{row['clip_id']}.json"
        if row["has_fall"] != 1 or not pose_path.exists():
            continue
        poses = load_pose_json(pose_path)
        width, height = (int(v) for v in row["resolution"].split("x"))
        folder = row["_folder"]
        labels = read_labels(folder)
        boxes = _label_boxes(folder)
        fps = float(row["fps"])
        hits, seen = {}, {}
        for position, frame in enumerate(sorted(labels)):
            if frame not in boxes:
                continue
            cx, cy, w, h = boxes[frame]
            label_box = ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)
            t = position / fps
            nearest = min(range(len(poses.times)), key=lambda i: abs(poses.times[i] - t))
            if abs(poses.times[nearest] - t) > 0.6 / poses.fps:
                continue
            for person in poses.people[nearest]:
                track = person.get("track_id")
                if track is None:
                    continue
                seen[track] = seen.get(track, 0) + 1
                if _iou(label_box, person["bbox"]) >= 0.3:
                    hits[track] = hits.get(track, 0) + 1
        # The faller is every track that is mostly on the labeled person: a
        # tracker that loses them mid-fall gives one ID before and another
        # after, and both belong to the faller however short either is.
        ids = sorted(t for t, n in hits.items() if n >= 3 and n >= 0.5 * seen[t])
        if ids:
            row["faller_track_id"] = ";".join(str(i) for i in ids)
            assigned += 1
        else:
            row["quality"] = "review"
            row["notes"] = "; ".join(filter(None, [row["notes"], "no pose track matches the labeled person"]))
    return assigned


def _label_boxes(folder):
    """{frame number: (cx, cy, w, h)} from the per-frame YOLO files (normalized)."""
    boxes = {}
    for path in folder.glob("*.txt"):
        match = re.search(r"(\d+)\.txt$", path.name)
        parts = path.read_text(encoding="utf-8", errors="replace").split() if match else []
        if match and len(parts) >= 5:
            boxes[int(match.group(1)) % 100000] = tuple(float(v) for v in parts[1:5])
    return boxes


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def write_sheets(rows, out):
    """One grid per fall clip: frames every 0.1 s around the computed fall, marked S/I."""
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not row["fall_start_s"]:
            continue
        start, end = float(row["fall_start_s"]), float(row["fall_end_s"])
        capture = cv2.VideoCapture(str(VIDEOS / row["file"]))
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        wanted = {round((start - 0.6 + 0.1 * k) * fps) for k in range(int((end - start + 1.4) / 0.1) + 1)}
        tiles, index = [], 0
        while True:
            ok, frame = capture.read()
            if not ok or index > max(wanted):
                break
            if index in wanted:
                t = index / fps
                tag = " S" if abs(t - start) < 0.051 else (" I" if abs(t - end) < 0.051 else "")
                frame = cv2.resize(frame, (240, 160))
                cv2.rectangle(frame, (0, 0), (110, 22), (0, 0, 0), -1)
                cv2.putText(frame, f"{t:.2f}{tag}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                tiles.append(frame)
            index += 1
        if not tiles:
            continue
        columns = 8
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
        cv2.imwrite(str(out / f"{row['clip_id']}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare person detectors on the falls where the current one loses the faller.

    uv run scripts/compare_detectors.py                        # YOLO26n / s / m, default clip list
    uv run scripts/compare_detectors.py --models yolo26n yolo26s --image-size 960

For every fall clip in the list, frames are sampled at 12 fps from 1 s before
the labeled fall start to 1 s after impact. Each detector (person class,
confidence 0.25, plain detection: no tracking, no pose) counts the people it
finds in each frame. The score per clip is

    kept = mean people found during the fall / people found in the second before it

so a detector that keeps seeing the faller while they fall scores ~100%, and
one that loses them drops below (e.g. 2 of 3 people kept = 67%). In clips
where nobody is found before the fall the "before" count is taken as the
number of people the label sheet says are in frame (or 1).

Writes runs/detectors/<date>_compare.csv and .md, and prints a summary. This
measures detection only; it does not retrain anything.
"""

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_poses_rtmpose import MODELS, ROTATIONS  # noqa: E402

OUT = ROOT / "runs" / "detectors"

# (label sheet row clip_id, why it is in the test)
DEFAULT_CLIPS = [
    # Trial 1 rig falls the temporal model missed because detection failed
    ("ILLOvByCSJmu2U4QL4R7q_output", "Trial 1: drill floor fall D, not visible to the detector"),
    ("miller_group_fall_08_fall_from_low_platform", "Trial 1: fall not visible to the detector"),
    ("slip_behind_person", "Trial 1: faller hidden, lost"),
    ("miller_group_fall_03_slip_oily_patch_drizzle", "Trial 1: faller partly lost"),
    ("backward_slip_while_stepping_back", "Trial 1: faller partly lost"),
    ("miller_group_fall_01_slip_wet_grating", "Trial 1: missed, visible"),
    ("miller_group_fall_10_slip_carrying_toolbox_night_rain", "Trial 1: missed, night rain"),
    # New night-vision clips: YOLO26n finds (almost) nobody
    ("miller_night_fall_01_nv_slip_wet_grating", "night vision"),
    ("miller_night_fall_02_nv_trip_over_hose", "night vision, nobody detected"),
    ("miller_night_fall_03_nv_ladder_fall", "night vision, nobody detected"),
    ("miller_night_fall_04_nv_stair_tumble_alone", "night vision, nobody detected"),
    ("miller_night_fall_05_nv_stumble_carrying_pipe", "night vision"),
    # Clean catches, to check a bigger detector does not make them worse
    ("sudden_collapse", "control: caught in Trial 1"),
    ("fall_from_ladder", "control: caught in Trial 1"),
]
CAUCA_COUNT = 8  # CAUCAFall falls with the fewest detections in Trial 1's pose files


def read_sheets():
    rows = {}
    for sheet in [ROOT / "data" / "labels" / "clips.csv", *sorted((ROOT / "data" / "labels").glob("external_*.csv"))]:
        with open(sheet, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("has_fall") == "1" and row.get("fall_start_s"):
                    rows.setdefault(row["clip_id"], row)
    return rows


def video_path(row):
    for base in (ROOT / "data" / "input", ROOT / "data" / "external" / "caucafall" / "videos", ROOT / "data" / "external"):
        candidate = base / row.get("folder", "") / row["file"]
        if candidate.exists():
            return candidate
        candidate = base / row["file"]
        if candidate.exists():
            return candidate
    hits = list((ROOT / "data").rglob(row["file"]))
    return hits[0] if hits else None


def worst_cauca(rows, count):
    """CAUCAFall falls whose Trial 1 pose files have the fewest detections during the fall."""
    import json

    scored = []
    for clip_id, row in rows.items():
        if row.get("source") != "caucafall":
            continue
        pose = ROOT / "data" / "output" / "caucafall" / f"{clip_id}.json"
        if not pose.exists():
            continue
        data = json.loads(pose.read_text(encoding="utf-8"))
        a, b = float(row["fall_start_s"]), float(row["fall_end_s"])
        inside = [len(f["people"]) for f in data["frames"] if a <= f["timestamp"] <= b + 0.5]
        if inside:
            scored.append((float(np.mean([n > 0 for n in inside])), clip_id))
    return [(clip_id, f"CAUCAFall: person found in {share:.0%} of fall frames (YOLO26n)")
            for share, clip_id in sorted(scored)[:count]]


def sample_frames(video, start, end, rotate="none"):
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        from extract_poses_rtmpose import true_timing

        real, skip = true_timing(video)
        fps = real or fps
    except Exception:  # noqa: BLE001
        skip = 0
    wanted = {}
    t = max(0.0, start - 1.0)
    while t <= end + 1.0:
        wanted[skip + int(round(t * fps))] = t
        t += 1 / 12
    frames, index = [], 0
    while index <= max(wanted):
        ok = capture.grab()
        if not ok:
            break
        if index in wanted:
            ok, image = capture.retrieve()
            if ok:
                if ROTATIONS.get(rotate) is not None:
                    image = cv2.rotate(image, ROTATIONS[rotate])
                frames.append((wanted[index], image))
        index += 1
    capture.release()
    return frames


def detector(name, image_size):
    from ultralytics import YOLO
    import shutil

    path = MODELS / f"{name}.pt"
    if not path.exists():
        YOLO(f"{name}.pt")
        shutil.move(f"{name}.pt", path)
    model = YOLO(str(path))
    return lambda image: len(model.predict(image, imgsz=image_size, conf=0.25, classes=[0], verbose=False)[0].boxes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s", "yolo26m"])
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--cauca", type=int, default=CAUCA_COUNT, help="How many of the worst CAUCAFall falls to add")
    args = parser.parse_args(argv)

    rows = read_sheets()
    clips = [(c, why) for c, why in DEFAULT_CLIPS if c in rows] + worst_cauca(rows, args.cauca)
    print(f"{len(clips)} fall clips, detectors {', '.join(args.models)} at {args.image_size} px")
    samples = {}
    for clip_id, _ in clips:
        row = rows[clip_id]
        video = video_path(row)
        if video is None:
            print(f"  {clip_id}: video not found, skipped")
            continue
        samples[clip_id] = sample_frames(video, float(row["fall_start_s"]), float(row["fall_end_s"]))

    results = {}
    for name in args.models:
        count = detector(name, args.image_size)
        started, n = time.perf_counter(), 0
        for clip_id, frames in samples.items():
            row = rows[clip_id]
            a, b = float(row["fall_start_s"]), float(row["fall_end_s"])
            before = [count(im) for t, im in frames if a - 1.0 <= t < a]
            during = [count(im) for t, im in frames if a <= t <= b + 0.5]
            n += len(before) + len(during)
            expected = float(np.median(before)) if before and np.median(before) > 0 else \
                float(row.get("people_in_frame") or 1)
            results[(name, clip_id)] = {"before": float(np.mean(before)) if before else 0.0,
                                        "during": float(np.mean(during)) if during else 0.0,
                                        "kept": min(1.0, float(np.mean(during)) / expected) if during else 0.0}
        ms = (time.perf_counter() - started) / max(n, 1) * 1000
        results[(name, "_ms")] = ms
        mean_kept = np.mean([results[(name, c)]["kept"] for c in samples])
        print(f"  {name}: people kept during falls {mean_kept:.0%} (mean over clips), {ms:.0f} ms per frame here")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{dt.date.today().isoformat()}_compare"
    why = dict(clips)
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", "why"] + [f"{m}_{k}" for m in args.models for k in ("before", "during", "kept")])
        for clip_id in samples:
            writer.writerow([clip_id, why[clip_id]] + [round(results[(m, clip_id)][k], 3)
                                                       for m in args.models for k in ("before", "during", "kept")])
    lines = [f"# Detector comparison ({dt.date.today().isoformat()}, {args.image_size} px, confidence 0.25)", "",
             "Kept = people found during the fall / people found in the second before it (100% = nobody lost).", "",
             "| Clip | Why | " + " | ".join(args.models) + " |", "|---|---|" + "---:|" * len(args.models)]
    for clip_id in samples:
        lines.append(f"| {clip_id} | {why[clip_id]} | " +
                     " | ".join(f"{results[(m, clip_id)]['kept']:.0%} ({results[(m, clip_id)]['during']:.1f})"
                                for m in args.models) + " |")
    lines.append("| **Mean** | | " + " | ".join(f"**{np.mean([results[(m, c)]['kept'] for c in samples]):.0%}**"
                                                for m in args.models) + " |")
    lines.append("| Time per frame on this computer | | " + " | ".join(f"{results[(m, '_ms')]:.0f} ms" for m in args.models) + " |")
    lines += ["", "Numbers in brackets: mean people found per frame during the fall."]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {stem.with_suffix('.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

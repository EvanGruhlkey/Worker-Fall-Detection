"""Turn the hand-edited label sheet into the manifest that training reads.

    uv run scripts/build_manifest.py                      # data/labels/clips.csv -> manifest.json
    uv run scripts/build_manifest.py --keep-only          # only rows marked quality=keep

One row per fall event: a clip with two falls gets two rows with the same
clip_id. Clips without a fall have one row with empty fall times. Rows marked
quality=drop are left out. Pose JSON is looked for at
data/output/<folder>/<clip_id>.json (the RTMPose batch layout), then
data/output/<clip_id>.json (yolo_pose.py); clips without one are reported and
skipped. Each pose file is also checked: that it covers the whole clip, that
the faller's track ID is actually seen during the fall, and that every clip
ends up at the same frame rate.

The sheet is checked before anything is written, so a typo (end before start,
a time past the end of the clip, a fall time on a no-fall clip) stops the
build instead of becoming a silently wrong label.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal.pose_io import load_pose_json  # noqa: E402

QUALITIES = {"keep", "review", "drop"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=str(ROOT / "data" / "labels" / "clips.csv"))
    parser.add_argument("--output", default=str(ROOT / "data" / "labels" / "manifest.json"))
    parser.add_argument("--pose-dir", default=str(ROOT / "data" / "output"))
    parser.add_argument(
        "--target-fps", type=float, default=12.0,
        help="Rate the windows are cut at (default 12, the rig pose files' rate; frames are never invented)",
    )
    parser.add_argument("--keep-only", action="store_true", help="Leave out rows still marked review")
    return parser.parse_args(argv)


def _number(text, field, where, errors):
    try:
        return float(text)
    except ValueError:
        errors.append(f"{where}: {field} {text!r} is not a number")
        return None


def read_clips(path):
    """Group the sheet's rows by clip, validating as it goes; return (clips, errors)."""
    clips = {}
    errors = []
    with open(path, newline="", encoding="utf-8-sig") as handle:  # -sig: Excel adds a BOM
        for line, row in enumerate(csv.DictReader(handle), start=2):
            row = {key: (value or "").strip() for key, value in row.items() if key}
            clip_id = row.get("clip_id", "")
            where = f"line {line} ({clip_id or 'no clip_id'})"
            if not clip_id:
                errors.append(f"{where}: clip_id is empty")
                continue

            quality = row.get("quality", "").lower()
            if quality not in QUALITIES:
                errors.append(f"{where}: quality must be one of {sorted(QUALITIES)}, got {quality!r}")
            has_fall = row.get("has_fall", "")
            if has_fall not in ("0", "1"):
                errors.append(f"{where}: has_fall must be 0 or 1, got {has_fall!r}")
                continue

            clip = clips.setdefault(clip_id, {
                "has_fall": has_fall, "quality": quality, "group": row.get("group") or clip_id,
                "folder": row.get("folder", ""),
                "duration": _number(row.get("duration_s") or "0", "duration_s", where, errors),
                "falls": [], "track_ids": set(), "lines": [],
            })
            clip["lines"].append(line)
            if clip["has_fall"] != has_fall or clip["quality"] != quality:
                errors.append(f"{where}: rows for the same clip disagree on has_fall or quality")

            start, end = row.get("fall_start_s", ""), row.get("fall_end_s", "")
            if has_fall == "0":
                if start or end:
                    errors.append(f"{where}: has_fall is 0 but fall times are filled in")
            elif quality != "drop":
                if not start or not end:
                    errors.append(f"{where}: fall clip needs fall_start_s and fall_end_s")
                else:
                    a = _number(start, "fall_start_s", where, errors)
                    b = _number(end, "fall_end_s", where, errors)
                    if a is not None and b is not None:
                        if not 0 <= a < b:
                            errors.append(f"{where}: need 0 <= fall_start_s < fall_end_s, got {a} and {b}")
                        elif clip["duration"] and b > clip["duration"] + 0.05:
                            errors.append(f"{where}: fall_end_s {b} is past the clip's end ({clip['duration']} s)")
                        else:
                            clip["falls"].append([a, b])

            for part in row.get("faller_track_id", "").replace(",", ";").split(";"):
                if part.strip():
                    if part.strip().isdigit():
                        clip["track_ids"].add(int(part))
                    else:
                        errors.append(f"{where}: faller_track_id {part!r} is not a whole number")
            if row.get("faller_track_id") and has_fall == "0":
                errors.append(f"{where}: faller_track_id is only for fall clips")
    return clips, errors


def main(argv=None):
    args = parse_args(argv)
    clips, errors = read_clips(args.csv)
    if errors:
        print(f"{len(errors)} problem(s) in {args.csv}:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    output = Path(args.output)
    pose_dir = Path(args.pose_dir)
    videos, skipped, warnings, rates = [], defaultdict(list), [], {}
    for clip_id, clip in sorted(clips.items()):
        if clip["quality"] == "drop":
            skipped["quality=drop"].append(clip_id)
            continue
        if clip["quality"] == "review" and args.keep_only:
            skipped["quality=review"].append(clip_id)
            continue
        pose_json = find_pose_json(pose_dir, clip["folder"], clip_id)
        if pose_json is None:
            skipped["no pose JSON yet"].append(clip_id)
            continue
        try:
            poses = load_pose_json(pose_json)
        except (OSError, ValueError, KeyError) as error:
            print(f"Error: {clip_id}: cannot read {pose_json}: {error}", file=sys.stderr)
            return 1
        warnings += check_pose_file(clip_id, clip, poses)
        rates[clip_id] = poses.resample(args.target_fps).fps

        entry = {"name": clip_id, "group": clip["group"], "pose_json": _relative(pose_json, output.parent)}
        if clip["has_fall"] == "1":
            entry["falls_seconds"] = sorted(clip["falls"])
            if clip["track_ids"]:
                entry["track_ids"] = sorted(clip["track_ids"])
            else:
                # Every person's windows overlapping the fall would be labeled a fall.
                warnings.append(f"{clip_id}: no faller_track_id, so every person in the clip is "
                                "labeled as falling during the fall interval")
        videos.append(entry)

    manifest = {"target_fps": args.target_fps, "source": str(Path(args.csv).name), "videos": videos}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    falls = sum(1 for v in videos if "falls_seconds" in v)
    groups = len({v["group"] for v in videos})
    distinct = sorted({round(rate, 2) for rate in rates.values()})
    rate_note = f", {distinct[0]:g} fps" if len(distinct) == 1 else ""
    print(f"Wrote {output}: {len(videos)} clips ({falls} with falls, {len(videos) - falls} without), "
          f"{groups} groups{rate_note}")
    if len(distinct) > 1:
        by_rate = defaultdict(list)
        for clip_id, rate in rates.items():
            by_rate[round(rate, 2)].append(clip_id)
        for rate, ids in sorted(by_rate.items()):
            warnings.append(f"{len(ids)} clip(s) end up at {rate:g} fps: {', '.join(sorted(ids))}")
        warnings.append("clips end up at different frame rates; training will refuse this manifest "
                        "until they are re-extracted at one rate or --target-fps is lowered")
    for reason, ids in skipped.items():
        print(f"  skipped {len(ids)} ({reason}): {', '.join(ids)}")
    for warning in warnings:
        print(f"  warning: {warning}")
    return 0


def find_pose_json(pose_dir, folder, clip_id):
    """data/output/<folder>/<clip>.json (RTMPose batch layout), else data/output/<clip>.json."""
    candidates = [pose_dir / folder / f"{clip_id}.json"] if folder else []
    candidates.append(pose_dir / f"{clip_id}.json")
    return next((path for path in candidates if path.exists()), None)


def check_pose_file(clip_id, clip, poses):
    """Warnings about a pose file that would quietly weaken training."""
    warnings = []
    duration = clip["duration"] or 0
    covered = poses.times[-1] + 1 / poses.fps
    if duration and abs(covered - duration) > 0.25:
        warnings.append(f"{clip_id}: pose file covers {covered:.2f} s but the clip is {duration:.2f} s; "
                        "was it extracted from a different version of the video?")
    if clip["has_fall"] != "1" or not clip["track_ids"]:
        return warnings
    for start, end in clip["falls"]:
        first, last = poses.interval_indices(start, end)
        seen = sum(
            any(person.get("track_id") in clip["track_ids"] for person in poses.people[i])
            for i in range(first, last + 1)
        )
        total = last - first + 1
        if seen == 0:
            warnings.append(f"{clip_id}: faller track {_ids(clip)} is never seen during the fall "
                            f"{start}-{end} s; check the ID (the fall may yield no positive windows)")
        elif seen < total / 2:
            warnings.append(f"{clip_id}: faller track {_ids(clip)} is seen in only {seen} of {total} frames "
                            f"of the fall {start}-{end} s (tracker lost them; the handoff may recover it)")
    return warnings


def _ids(clip):
    return ";".join(str(i) for i in sorted(clip["track_ids"]))


def _relative(path, start):
    """Path relative to the manifest's folder (how dataset.py resolves it), else absolute."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()
    except ValueError:  # different drive on Windows
        return str(Path(path).resolve())


if __name__ == "__main__":
    raise SystemExit(main())

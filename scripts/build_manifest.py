"""Turn the hand-edited label sheet into the manifest that training reads.

    uv run scripts/build_manifest.py                      # data/labels/clips.csv -> manifest.json
    uv run scripts/build_manifest.py --keep-only          # only rows marked quality=keep

One row per fall event: a clip with two falls gets two rows with the same
clip_id. Clips without a fall have one row with empty fall times. Rows marked
quality=drop are left out. Pose JSON is expected where yolo_pose.py writes it,
data/output/<clip_id>.json; clips without one are reported and skipped.

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
QUALITIES = {"keep", "review", "drop"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=str(ROOT / "data" / "labels" / "clips.csv"))
    parser.add_argument("--output", default=str(ROOT / "data" / "labels" / "manifest.json"))
    parser.add_argument("--pose-dir", default=str(ROOT / "data" / "output"))
    parser.add_argument("--target-fps", type=float, default=15.0)
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
    videos, skipped, warnings = [], defaultdict(list), []
    for clip_id, clip in sorted(clips.items()):
        if clip["quality"] == "drop":
            skipped["quality=drop"].append(clip_id)
            continue
        if clip["quality"] == "review" and args.keep_only:
            skipped["quality=review"].append(clip_id)
            continue
        pose_json = pose_dir / f"{clip_id}.json"
        if not pose_json.exists():
            skipped["no pose JSON yet"].append(clip_id)
            continue

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
    print(f"Wrote {output}: {len(videos)} clips ({falls} with falls, {len(videos) - falls} without), "
          f"{groups} groups")
    for reason, ids in skipped.items():
        print(f"  skipped {len(ids)} ({reason}): {', '.join(ids)}")
    for warning in warnings:
        print(f"  warning: {warning}")
    return 0


def _relative(path, start):
    """Path relative to the manifest's folder (how dataset.py resolves it), else absolute."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()
    except ValueError:  # different drive on Windows
        return str(Path(path).resolve())


if __name__ == "__main__":
    raise SystemExit(main())

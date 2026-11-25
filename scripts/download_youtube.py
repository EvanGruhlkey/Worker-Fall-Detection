"""Download the Trial 4 YouTube fall clips and draft their label sheet.

    uv run --no-sync scripts/download_youtube.py

Reads data/labels/youtube_sources.csv (one row per video, with its URL and
title), downloads each video into data/external/youtube/videos/<clip_id>.mp4
(not versioned), measures each file's frame rate, length and size, and writes
data/labels/youtube_draft.csv in the label sheet's columns with the fall times,
faller ID and quality left blank. Safe to rerun: videos already downloaded are
skipped.

The draft is deliberately not named external_*.csv: build_manifest.py reads
every external_*.csv and stops on fall clips without fall times. The labeled
sheet is data/labels/external_youtube.csv (Trial 4); this script never writes
it. To add new videos, rerun and copy the new draft rows into it.

yt-dlp runs through uvx so it never touches the project environment (a plain
uv sync would swap torch to the CPU build). Only the video stream is kept, as
up to 1080 on the short side, preferring H.264: no audio, so no ffmpeg merge
is needed, and OpenCV can read it. Sources and attribution: DATA_SOURCES.md.
"""

import csv
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "labels" / "youtube_sources.csv"
DRAFT = ROOT / "data" / "labels" / "youtube_draft.csv"
OUT = ROOT / "data" / "external" / "youtube" / "videos"
# Best video-only stream up to 1080 on the short side (so vertical clips keep
# their full size), preferring H.264; AV1 is skipped as OpenCV may not decode it.
FORMAT = "bv*[vcodec!^=av01]/b"
SORT = "res:1080,vcodec:avc1"
COLUMNS = [
    "clip_id", "folder", "file", "has_fall", "fall_start_s", "fall_end_s", "faller_track_id",
    "quality", "group", "fall_type", "fall_direction", "camera_view", "people_in_frame",
    "conditions", "notes", "duration_s", "fps", "frames", "resolution", "source", "batch",
]


def download(row):
    path = OUT / f"{row['clip_id']}.mp4"
    if path.exists():
        return path, "already downloaded"
    command = [
        "uvx", "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", FORMAT, "-S", SORT, "-o", str(path), row["url"],
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not path.exists():
        return None, (result.stderr.strip().splitlines() or ["unknown error"])[-1]
    return path, "downloaded"


def measure(path):
    video = cv2.VideoCapture(str(path))
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        readable = video.read()[0]
    finally:
        video.release()
    if not readable or fps <= 0:
        return None
    return {"fps": round(fps, 3), "frames": frames, "duration_s": f"{frames / fps:.2f}",
            "resolution": f"{width}x{height}"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SOURCES, newline="", encoding="utf-8") as handle:
        sources = list(csv.DictReader(handle))

    drafts, failed = [], []
    for row in sources:
        path, status = download(row)
        info = measure(path) if path else None
        if info is None:
            failed.append((row["clip_id"], status if path is None else "OpenCV cannot read the file"))
            print(f"FAILED  {row['clip_id']}: {failed[-1][1]}")
            continue
        print(f"{status:18s}  {row['clip_id']:18s} {info['resolution']:>9s} {info['fps']:>6} fps "
              f"{info['duration_s']:>6} s")
        notes = f"YouTube: {row['title']} ({row['uploader']}), {row['url']}"
        if row["flags"]:
            notes += f"; check: {row['flags']}"
        drafts.append({
            "clip_id": row["clip_id"], "folder": "youtube", "file": path.name, "has_fall": "1",
            "quality": "review", "group": row["clip_id"], "notes": notes,
            "source": "youtube", "batch": "trial4", **info,
        })

    with open(DRAFT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(drafts)

    print(f"\n{len(drafts)} of {len(sources)} videos ready in {OUT.relative_to(ROOT)}")
    print(f"Draft label sheet: {DRAFT.relative_to(ROOT)}")
    for clip_id, reason in failed:
        print(f"  not downloaded: {clip_id} ({reason})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

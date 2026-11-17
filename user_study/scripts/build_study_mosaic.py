"""Encode the nine existing camera reels onto one perfectly synchronized source."""
import argparse
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg


def build(directory, set_id):
    path = directory / "rig-wall-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = next(s for s in manifest["sets"] if s["id"] == set_id)
    if len(config["feeds"]) != 9:
        raise ValueError("A 3x3 source requires nine camera feeds")
    output = directory / "media" / f"rig_{set_id}_wall.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    for feed in config["feeds"]:
        command.extend(["-threads", "1", "-i", str(directory / "media" / feed["file"])])
    command.extend(["-filter_complex", "".join(f"[{i}:v]" for i in range(9)) + "xstack=inputs=9:layout=0_0|640_0|1280_0|0_360|640_360|1280_360|0_720|640_720|1280_720[v]",
                    "-map", "[v]", "-an", "-t", str(manifest["duration"]), "-r", "12", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
    subprocess.run(command, check=True)
    print(f"Ready: {output}", flush=True)
    return output.name


def prepare(directory):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = dict(zip(("A", "B"), executor.map(lambda key: build(directory, key), ("A", "B"))))
    path = directory / "rig-wall-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for config in manifest["sets"]:
        config["wall_file"] = outputs[config["id"]]
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/study/rig-wall-two"))
    prepare(parser.parse_args().directory)

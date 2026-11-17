"""Build short person-focused detector replays from verified pipeline media."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from temporal.pose_io import load_pose_json


def prepare(directory):
    manifest_path = directory / 'rig-wall-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    segments = json.loads((directory / 'preparation/provenance.json').read_text())
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for vs in manifest['sets']:
        for index, alert in enumerate(vs['alerts']):
            segment = next(s for s in segments if s['set'] == vs['id'] and s['camera'] == alert['camera'] and s['offset'] <= alert['time'] < s['offset'] + s['duration'])
            time = alert['time'] - segment['offset']
            events = json.loads(Path(segment['signals']).read_text())['events']
            event = min(events, key=lambda e: abs(e['time'] - time))
            assert abs(event['time'] - time) < .002
            start, end = max(0, time - .5), min(segment['duration'], time + .5)
            poses = load_pose_json(segment['pose_source']).resample(12)
            metadata = json.loads(Path(segment['pose_source']).read_text())['metadata']
            boxes = [p['bbox'] for t, people in zip(poses.times, poses.people) if start <= t <= end for p in people if p['track_id'] == event['track_id']]
            if not boxes:
                raise ValueError('Flagged person has no pose boxes')
            scale = min(640 / metadata['width'], 360 / metadata['height'])
            dx, dy = (640 - metadata['width'] * scale) / 2, (360 - metadata['height'] * scale) / 2
            left, top = min(b[0] for b in boxes)*scale+dx, min(b[1] for b in boxes)*scale+dy
            right, bottom = max(b[2] for b in boxes)*scale+dx, max(b[3] for b in boxes)*scale+dy
            pad = max(24, (bottom-top)*.2)
            x, y = max(0, math.floor((left-pad)/2)*2), max(0, math.floor((top-pad)/2)*2)
            width = min(640-x, math.ceil((right+pad-x)/2)*2)
            height = min(338-y, math.ceil((bottom+pad-y)/2)*2)
            name = f"incident_{vs['id']}_{index:03}.mp4"
            source = directory / 'preparation/pipeline-sources' / (Path(segment['source']).stem + '.mp4')
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-ss', str(start), '-i', str(source), '-t', str(end-start), '-vf', f'crop={width}:{height}:{x}:{y},scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(directory/'media'/name)], check=True)
            alert['review_file'] = name
            print(f"Ready {name}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('data/study/rig-wall-trial2'))
    prepare(parser.parse_args().directory)

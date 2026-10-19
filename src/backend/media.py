"""Video writing that browsers can play back (H.264 via a bundled ffmpeg)."""

import logging
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

log = logging.getLogger(__name__)


class VideoWriter:
    """Write BGR frames to an H.264 MP4, falling back to OpenCV's mp4v.

    OpenCV's default mp4v codec does not play in browsers, so frames are piped
    to the ffmpeg binary shipped with imageio-ffmpeg and encoded as H.264.
    """

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        fps = fps if fps and fps > 0 else 30.0
        self._process = None
        self._cv_writer = None

        try:
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                # libx264 with yuv420p needs even dimensions.
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(self.path),
            ]
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except (OSError, RuntimeError) as error:
            log.warning("ffmpeg unavailable (%s); falling back to mp4v", error)
            self._cv_writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not self._cv_writer.isOpened():
                raise RuntimeError(f"Could not create video: {self.path}")

    def write(self, frame):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if self._process:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self._cv_writer.write(frame)

    def close(self):
        if self._process:
            self._process.stdin.close()
            stderr = self._process.stderr.read().decode(errors="replace")
            self._process.stderr.close()
            if self._process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}: {stderr.strip()}")
        elif self._cv_writer:
            self._cv_writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def encode_jpeg(frame, quality=80):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG")
    return buffer.tobytes()


def write_clip(path, jpeg_frames, fps):
    """Write a list of JPEG-encoded frames to an H.264 clip."""
    if not jpeg_frames:
        return
    first = cv2.imdecode(np.frombuffer(jpeg_frames[0], np.uint8), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    with VideoWriter(path, width, height, fps) as writer:
        writer.write(first)
        for data in jpeg_frames[1:]:
            writer.write(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))

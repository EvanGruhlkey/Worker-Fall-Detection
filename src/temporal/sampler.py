"""Stage 2: drop frames so the classifier sees 10 to 15 frames per second.

A fall takes half a second to a second, so 10 to 15 fps captures it, and the
window of 30 frames then spans about two seconds as the guide assumes. Any
resize is done by YOLO (``imgsz``; by default the frame's native size), so
only the frame rate is handled here.

``FrameSampler`` is for files with a known, steady frame rate; it keeps every
``stride``-th frame. ``TimeSampler`` is for live sources, where frames arrive
unevenly; it keeps a frame once roughly ``1 / target_fps`` seconds have passed
and measures the frame rate it actually achieves.
"""


class FrameSampler:
    def __init__(self, source_fps, target_fps=15.0):
        if source_fps <= 0 or target_fps <= 0:
            raise ValueError("Frame rates must be greater than zero")
        self.stride = max(1, round(source_fps / target_fps))
        # The rate the kept frames actually have, e.g. 25 fps / 2 = 12.5 fps.
        self.fps = source_fps / self.stride

    def keep(self, frame_index):
        return frame_index % self.stride == 0

    def processed_index(self, frame_index):
        """Index among the kept frames, which is what TrackWindows must get."""
        return frame_index // self.stride


class TimeSampler:
    def __init__(self, target_fps=15.0, tolerance=0.25):
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        self.interval = 1.0 / target_fps
        self._minimum_gap = self.interval * (1.0 - tolerance)
        self._last = None
        self.count = 0
        # Measured rate of the kept frames, used to scale velocities.
        self.fps = float(target_fps)

    def keep(self, timestamp):
        if self._last is not None:
            gap = timestamp - self._last
            if gap < self._minimum_gap:
                return False
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / gap)
        self._last = timestamp
        self.count += 1
        return True

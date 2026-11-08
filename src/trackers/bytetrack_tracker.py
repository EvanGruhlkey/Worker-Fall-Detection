"""ByteTrack association for the YOLO detector's person boxes."""

from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


class PersonTracker:
    """Assign persistent IDs to detected people across frames."""

    def __init__(self):
        settings = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=12, match_thresh=0.8, fuse_score=True,)
        self.tracker = BYTETracker(settings)

    def update(self, boxes, scores, frame):
        """Return rows of ``x1, y1, x2, y2, track_id, confidence, class, detection_index``."""
        classes = np.zeros(len(boxes), dtype=np.float32)
        detections = np.column_stack((boxes, scores, classes))
        return self.tracker.update(Boxes(detections, frame.shape[:2]), frame)

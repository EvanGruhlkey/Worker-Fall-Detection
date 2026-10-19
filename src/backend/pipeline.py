"""Frame processing: pose tracking -> fall detection -> fall events and clips.

Each job (uploaded video or live camera) gets its own ``FrameProcessor`` and
its own YOLO model, because ``model.track(persist=True)`` keeps tracker state
on the model and sharing one model would mix up track IDs between sources.
"""

import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from sqlmodel import Session

from yolo_pose import load_pose_model, process_frame

from .db import FallEvent, Job, JobStatus, get_engine, update_row, utc_now
from .fall_detector import FallDetector
from .media import VideoWriter, encode_jpeg, write_clip

log = logging.getLogger(__name__)

ALERT_SECONDS = 3.0
FALL_COLOR = (0, 0, 255)


class _PendingClip:
    def __init__(self, event_id, frames, end_time):
        self.event_id = event_id
        self.frames = frames
        self.end_time = end_time


class FrameProcessor:
    """Runs one source's frames through the model and fall detector."""

    def __init__(self, job_id, settings, detector=None):
        self.job_id = job_id
        self.settings = settings
        self.model = load_pose_model(settings.model_path)
        self.detector = detector or FallDetector()
        self.event_dir = settings.events_dir / str(job_id)
        self.event_dir.mkdir(parents=True, exist_ok=True)

        self._buffer = deque()  # (timestamp, jpeg) pre-roll for clips
        self._pending = []
        self._alerts = {}  # track_id -> timestamp the on-screen alert ends
        self._clip_writer = ThreadPoolExecutor(max_workers=1)
        self.fall_count = 0
        self.latest_jpeg = None

    def process(self, frame, frame_index, timestamp):
        """Process one frame; returns (annotated_frame, people, jpeg)."""
        annotated, people = process_frame(self.model, frame, self.settings.image_size)
        detections = self.detector.update(people, timestamp)
        for detection in detections:
            self._alerts[detection.track_id] = timestamp + ALERT_SECONDS
        self._draw_alerts(annotated, people, timestamp)

        jpeg = encode_jpeg(annotated)
        self.latest_jpeg = jpeg

        for detection in detections:
            self._record_event(detection, frame_index, timestamp, jpeg)

        self._buffer.append((timestamp, jpeg))
        while self._buffer and timestamp - self._buffer[0][0] > self.settings.clip_pre_seconds:
            self._buffer.popleft()

        still_pending = []
        for clip in self._pending:
            clip.frames.append((timestamp, jpeg))
            if timestamp >= clip.end_time:
                self._submit_clip(clip)
            else:
                still_pending.append(clip)
        self._pending = still_pending

        return annotated, people, jpeg

    def finish(self):
        """Write any clips still collecting post-roll and wait for all clips."""
        for clip in self._pending:
            self._submit_clip(clip)
        self._pending = []
        self._clip_writer.shutdown(wait=True)

    def _draw_alerts(self, annotated, people, timestamp):
        for person in people:
            alert_until = self._alerts.get(person["track_id"])
            if alert_until is None or timestamp > alert_until:
                continue
            x1, y1, x2, y2 = (int(v) for v in person["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), FALL_COLOR, 4)
            cv2.putText(
                annotated, "FALL DETECTED", (x1, min(y2 + 30, annotated.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, FALL_COLOR, 3, cv2.LINE_AA,
            )

    def _record_event(self, detection, frame_index, timestamp, jpeg):
        with Session(get_engine()) as session:
            event = FallEvent(
                job_id=self.job_id,
                track_id=detection.track_id,
                frame_index=frame_index,
                video_time=round(timestamp, 3),
                confidence=detection.confidence,
                reason=detection.reason,
            )
            session.add(event)
            session.commit()
            session.refresh(event)

            snapshot = self.event_dir / f"event_{event.id}.jpg"
            snapshot.write_bytes(jpeg)
            event.snapshot_path = self._relative(snapshot)
            session.add(event)

            job = session.get(Job, self.job_id)
            job.fall_count += 1
            session.add(job)
            session.commit()
            event_id = event.id

        self.fall_count += 1
        log.warning(
            "Fall detected: job %s track %s at %.1fs (%s)",
            self.job_id, detection.track_id, timestamp, detection.reason,
        )
        self._pending.append(
            _PendingClip(
                event_id,
                list(self._buffer),
                timestamp + self.settings.clip_post_seconds,
            )
        )

    def _submit_clip(self, clip):
        self._clip_writer.submit(self._write_clip, clip)

    def _write_clip(self, clip):
        path = self.event_dir / f"event_{clip.event_id}.mp4"
        times = [t for t, _ in clip.frames]
        duration = times[-1] - times[0] if len(times) > 1 else 0
        # Use the real frame rate the frames arrived at so clips play in real time.
        fps = (len(times) - 1) / duration if duration > 0 else 10.0
        try:
            write_clip(path, [jpeg for _, jpeg in clip.frames], fps)
            update_row(FallEvent, clip.event_id, clip_path=self._relative(path))
        except Exception:
            log.exception("Could not write clip for event %s", clip.event_id)

    def _relative(self, path):
        return Path(path).relative_to(self.settings.data_dir).as_posix()


def run_video_job(job_id, settings):
    """Process an uploaded video file end to end (runs in a worker thread)."""
    job = update_row(Job, job_id, status=JobStatus.running, started_at=utc_now())
    input_path = Path(job.source)
    stem = f"job_{job_id}_{input_path.stem}"
    output_video = settings.output_dir / f"{stem}.mp4"
    output_json = settings.output_dir / f"{stem}.json"

    processor = None
    video = cv2.VideoCapture(str(input_path))
    try:
        if not video.isOpened():
            raise RuntimeError(f"Could not open video: {input_path.name}")
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        update_row(Job, job_id, fps=fps, total_frames=total)

        processor = FrameProcessor(job_id, settings)
        frames = []
        frame_index = 0
        with VideoWriter(output_video, width, height, fps) as writer:
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                annotated, people, _ = processor.process(frame, frame_index, frame_index / fps)
                writer.write(annotated)
                frames.append({"frame_index": frame_index, "people": people})
                frame_index += 1
                if frame_index % 15 == 0:
                    update_row(Job, job_id, frames_processed=frame_index)

        if frame_index == 0:
            raise RuntimeError("Video contains no decodable frames")

        output_json.write_text(
            json.dumps(
                {
                    "metadata": {
                        "source": input_path.name,
                        "model": settings.model_path.name,
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "frame_count": frame_index,
                        "keypoint_format": "COCO-17",
                    },
                    "frames": frames,
                }
            ),
            encoding="utf-8",
        )
        processor.finish()
        update_row(
            Job, job_id,
            status=JobStatus.completed,
            frames_processed=frame_index,
            total_frames=frame_index,
            output_video=output_video.relative_to(settings.data_dir).as_posix(),
            output_json=output_json.relative_to(settings.data_dir).as_posix(),
            finished_at=utc_now(),
        )
    except Exception as error:
        log.exception("Video job %s failed", job_id)
        if processor:
            processor.finish()
        update_row(
            Job, job_id, status=JobStatus.failed, error=str(error), finished_at=utc_now()
        )
    finally:
        video.release()


class CameraWorker(threading.Thread):
    """Processes a live camera or stream URL until stopped."""

    def __init__(self, job_id, source, settings, width=None, height=None):
        super().__init__(name=f"camera-{job_id}", daemon=True)
        self.job_id = job_id
        self.source = source
        self.settings = settings
        self.width, self.height = width, height
        self.fps = 0.0
        self._stop_event = threading.Event()
        self._frame_ready = threading.Condition()
        self._processor = None
        self._frame_number = 0

    def stop(self):
        self._stop_event.set()

    @property
    def latest_jpeg(self):
        return self._processor.latest_jpeg if self._processor else None

    def wait_for_frame(self, last_number, timeout=1.0):
        """Block until a frame newer than ``last_number``; returns (number, jpeg)."""
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._frame_number != last_number or not self.is_alive(),
                timeout=timeout,
            )
            return self._frame_number, self.latest_jpeg

    def run(self):
        update_row(Job, self.job_id, status=JobStatus.running, started_at=utc_now())
        capture_source = int(self.source) if self.source.isdigit() else self.source
        camera = cv2.VideoCapture(capture_source)
        status, error = JobStatus.stopped, None
        try:
            if not camera.isOpened():
                raise RuntimeError(f"Could not open camera {self.source}")
            if self.width:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self._processor = FrameProcessor(self.job_id, self.settings)
            start = previous = time.monotonic()
            frame_index = 0
            while not self._stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError(f"Lost connection to camera {self.source}")
                now = time.monotonic()
                self._processor.process(frame, frame_index, now - start)

                elapsed = now - previous
                previous = now
                if elapsed > 0:
                    instant = 1.0 / elapsed
                    self.fps = instant if self.fps == 0 else 0.9 * self.fps + 0.1 * instant

                frame_index += 1
                with self._frame_ready:
                    self._frame_number = frame_index
                    self._frame_ready.notify_all()
                if frame_index % 30 == 0:
                    update_row(
                        Job, self.job_id,
                        frames_processed=frame_index, fps=round(self.fps, 1),
                    )
        except Exception as exc:
            log.exception("Camera job %s failed", self.job_id)
            status, error = JobStatus.failed, str(exc)
        finally:
            camera.release()
            if self._processor:
                self._processor.finish()
            update_row(
                Job, self.job_id,
                status=status, error=error, finished_at=utc_now(),
                frames_processed=self._frame_number,
            )
            with self._frame_ready:
                self._frame_ready.notify_all()

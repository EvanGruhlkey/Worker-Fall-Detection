"""Owns the background workers: a queue for uploaded videos and live cameras."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from sqlmodel import Session, select

from yolo_pose import load_pose_model, process_frame

from .db import Job, JobKind, JobStatus, get_engine, utc_now
from .pipeline import CameraWorker, run_video_job

log = logging.getLogger(__name__)

CAMERA_SCAN_LIMIT = 5


class JobManager:
    def __init__(self, settings, video_workers=1):
        self.settings = settings
        # Inference is CPU/GPU bound, so videos are processed one at a time
        # by default; raise video_workers on a machine with spare capacity.
        self._videos = ThreadPoolExecutor(
            max_workers=video_workers, thread_name_prefix="video-job"
        )
        self._cameras = {}

    def warm_up(self):
        """Run one throwaway inference in the background.

        PyTorch lazily imports a large part of itself on the first inference
        (10+ seconds on some machines), which would otherwise stall the first
        video or camera started after the server boots.
        """

        def run():
            try:
                model = load_pose_model(self.settings.model_path)
                process_frame(model, np.zeros((480, 640, 3), np.uint8), self.settings.image_size)
                log.info("Pose model warmed up")
            except Exception:
                log.exception("Model warm-up failed")

        threading.Thread(target=run, name="model-warm-up", daemon=True).start()

    def recover_interrupted_jobs(self):
        """Jobs left running when the server stopped can't resume; mark them."""
        with Session(get_engine()) as session:
            jobs = session.exec(
                select(Job).where(Job.status.in_([JobStatus.queued, JobStatus.running]))
            ).all()
            for job in jobs:
                job.status = JobStatus.failed
                job.error = "Interrupted by server restart"
                # The real stop time is the last frame seen, not now; using now
                # would count the time the server was down as uptime.
                job.finished_at = job.last_frame_at or job.started_at or utc_now()
                session.add(job)
            session.commit()

    def submit_video(self, job_id):
        self._videos.submit(run_video_job, job_id, self.settings)

    def create_camera(
        self, source, name=None, label=None, location=None, coords=None,
        width=None, height=None,
    ):
        """Record a new camera session and start processing it; returns the job."""
        source = source.strip()
        for worker in self.active_cameras().values():
            if worker.source == source:
                raise ValueError(f"Camera {source} is already running")

        with Session(get_engine()) as session:
            job = Job(
                kind=JobKind.camera,
                name=name or (f"Camera {source}" if source.isdigit() else source),
                source=source,
                label=label,
                location=location,
                coords=coords,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
        self.start_camera(job.id, source, width, height)
        return job

    def start_configured_camera(self):
        """Start the camera named by FALL_CAMERA_SOURCE, if any."""
        if not self.settings.camera_source:
            return None
        try:
            return self.create_camera(
                self.settings.camera_source,
                name=self.settings.camera_name,
                label=self.settings.camera_label,
            )
        except Exception:
            log.exception("Could not start camera %s", self.settings.camera_source)
            return None

    def start_camera(self, job_id, source, width=None, height=None):
        worker = CameraWorker(job_id, source, self.settings, width, height)
        self._cameras[job_id] = worker
        worker.start()
        return worker

    def get_camera(self, job_id):
        return self._cameras.get(job_id)

    def active_cameras(self):
        return {job_id: w for job_id, w in self._cameras.items() if w.is_alive()}

    def camera_options(self):
        """Return local camera indices OpenCV can currently open.

        A running camera is included without opening it a second time. Network
        streams cannot be discovered, so the dashboard accepts those as a
        custom source instead.
        """
        active_sources = {worker.source for worker in self.active_cameras().values()}
        available = []
        for index in range(CAMERA_SCAN_LIMIT):
            source = str(index)
            if source in active_sources:
                available.append(source)
                continue

            camera = cv2.VideoCapture(index)
            try:
                if camera.isOpened():
                    available.append(source)
            finally:
                camera.release()
        return available

    def stop_camera(self, job_id, wait=True):
        worker = self._cameras.get(job_id)
        if worker is None:
            return False
        worker.stop()
        if wait:
            worker.join(timeout=15)
        return True

    def shutdown(self):
        for job_id in list(self._cameras):
            self.stop_camera(job_id)
        self._videos.shutdown(wait=False, cancel_futures=True)

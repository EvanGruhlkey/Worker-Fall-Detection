"""Owns the background workers: a queue for uploaded videos and live cameras."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sqlmodel import Session, select

from yolo_pose import load_pose_model, process_frame

from .db import Job, JobStatus, get_engine, utc_now
from .pipeline import CameraWorker, run_video_job

log = logging.getLogger(__name__)


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
                job.finished_at = utc_now()
                session.add(job)
            session.commit()

    def submit_video(self, job_id):
        self._videos.submit(run_video_job, job_id, self.settings)

    def start_camera(self, job_id, source, width=None, height=None):
        worker = CameraWorker(job_id, source, self.settings, width, height)
        self._cameras[job_id] = worker
        worker.start()
        return worker

    def get_camera(self, job_id):
        return self._cameras.get(job_id)

    def active_cameras(self):
        return {job_id: w for job_id, w in self._cameras.items() if w.is_alive()}

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

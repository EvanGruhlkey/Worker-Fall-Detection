"""Shared storage for snapshots, clips and annotated videos (S3-compatible).

Media is always written to the local data folder first. With a bucket
configured (FALL_STORAGE_*), each file is then uploaded in the background
under the same relative path (e.g. ``events/12/event_40.jpg``), so a station
can show footage another station recorded. Without one, files stay local.

Works with any S3-compatible service: Supabase Storage, AWS S3, Cloudflare R2,
MinIO. See docs/database.md.
"""

import logging
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

# Folders under the data dir whose files are shared. Uploaded source videos
# (input/) stay on the station that processed them.
SHARED_FOLDERS = ("events", "output")
LINK_SECONDS = 3600

_media = None


class MediaStore:
    def __init__(self, settings, client=None):
        self.data_dir = Path(settings.data_dir).resolve()
        self.bucket = settings.storage_bucket
        self._client = client
        if self.bucket and client is None:
            self._client = _s3_client(settings)
        self._uploads = ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-upload")

    @property
    def enabled(self):
        return self._client is not None and bool(self.bucket)

    def local_path(self, relative_path):
        """The file inside the data folder, or None for a path that escapes it."""
        path = (self.data_dir / relative_path).resolve()
        return path if path.is_relative_to(self.data_dir) else None

    def upload(self, relative_path):
        """Upload a file from the data folder in the background (no-op if local only)."""
        if self.enabled and relative_path:
            self._uploads.submit(self._upload_now, relative_path)

    def _upload_now(self, relative_path):
        path = self.local_path(relative_path)
        if path is None or not path.is_file():
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            self._client.upload_file(
                str(path), self.bucket, relative_path, ExtraArgs={"ContentType": content_type}
            )
        except Exception as error:
            # Kept locally; sync() uploads it on the next start.
            log.warning("Upload of %s failed, will retry on next start: %s", relative_path, error)

    def remote_keys(self):
        keys = set()
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            keys.update(item["Key"] for item in page.get("Contents", []))
        return keys

    def local_files(self):
        """Relative paths of every shared file in the data folder."""
        for folder in SHARED_FOLDERS:
            for path in sorted((self.data_dir / folder).rglob("*")):
                if path.is_file():
                    yield path.relative_to(self.data_dir).as_posix()

    def sync(self):
        """Upload local media missing from the bucket (e.g. recorded while offline)."""
        if not self.enabled:
            return 0
        try:
            remote = self.remote_keys()
        except Exception as error:
            log.warning("Could not list the media bucket, skipping sync: %s", error)
            return 0
        missing = [key for key in self.local_files() if key not in remote]
        for key in missing:
            self._upload_now(key)
        if missing:
            log.info("Uploaded %d media files recorded while offline", len(missing))
        return len(missing)

    def sync_in_background(self):
        threading.Thread(target=self.sync, name="media-sync", daemon=True).start()

    def download(self, relative_path, destination):
        self._client.download_file(self.bucket, relative_path, str(destination))

    def link(self, relative_path, filename=None):
        """A temporary URL the browser can load the file from directly."""
        params = {"Bucket": self.bucket, "Key": relative_path}
        if filename:
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        return self._client.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=LINK_SECONDS
        )

    def shutdown(self):
        """Wait for uploads in progress (so a clip written at shutdown is shared)."""
        self._uploads.shutdown(wait=True)


def _s3_client(settings):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.storage_endpoint,
        region_name=settings.storage_region,
        aws_access_key_id=settings.storage_key_id,
        aws_secret_access_key=settings.storage_secret,
        # Path-style URLs work with every S3-compatible service.
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def init_media(settings, client=None):
    global _media
    _media = MediaStore(settings, client)
    return _media


def get_media():
    if _media is None:
        raise RuntimeError("Media storage not initialised; call init_media() first")
    return _media


def share(relative_path):
    """Upload a newly written file from the data folder, if storage is configured."""
    if _media is not None:
        _media.upload(relative_path)

"""Shared FastAPI dependencies and helpers for routers."""

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from .storage import get_media


def get_settings(request: Request):
    return request.app.state.settings


def get_manager(request: Request):
    return request.app.state.manager


def data_file_response(settings, relative_path, media_type=None, filename=None):
    """Serve a stored file: this station's copy, else a link to shared storage.

    Paths are relative to the data directory; anything outside it is refused.
    """
    if not relative_path:
        raise HTTPException(404, "File not available yet")
    media = get_media()
    path = media.local_path(relative_path)
    if path is None:
        raise HTTPException(404, "File not found")
    if path.is_file():
        return FileResponse(path, media_type=media_type, filename=filename)
    if media.enabled:
        # Recorded on another station: the browser loads it from the bucket.
        return RedirectResponse(media.link(relative_path, filename), status_code=307)
    raise HTTPException(404, "File not found")

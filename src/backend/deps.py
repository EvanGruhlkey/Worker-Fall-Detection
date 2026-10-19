"""Shared FastAPI dependencies and helpers for routers."""

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse


def get_settings(request: Request):
    return request.app.state.settings


def get_manager(request: Request):
    return request.app.state.manager


def data_file_response(settings, relative_path, media_type=None, filename=None):
    """Serve a file stored under the data directory, refusing anything outside it."""
    if not relative_path:
        raise HTTPException(404, "File not available yet")
    data_dir = settings.data_dir.resolve()
    path = (data_dir / relative_path).resolve()
    if not path.is_relative_to(data_dir) or not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type=media_type, filename=filename)

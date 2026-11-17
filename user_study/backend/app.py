"""Standalone study server: no production database, camera, or model startup."""
from fastapi import FastAPI
from .router import router

app = FastAPI(title="Oxy user study")
app.include_router(router)

from pathlib import Path
from fastapi.staticfiles import StaticFiles

frontend = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend.is_dir():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="study-ui")

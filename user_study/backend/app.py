"""Standalone study server and participant-only entry point."""
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from .router import router

frontend = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def create_app(*, participant_only=False):
    study_app = FastAPI(title="Oxy user study", openapi_url=None if participant_only else "/openapi.json")
    study_app.state.participant_only = participant_only
    routes = router
    if participant_only:
        routes = APIRouter()
        for route in router.routes:
            path = route.path
            if (path == "/api/study/sessions" and route.methods == {"POST"}
                    or path == "/api/study/session"
                    or path.startswith("/api/study/session/")):
                routes.routes.append(route)
    study_app.include_router(routes)
    if frontend.is_dir():
        study_app.mount("/", StaticFiles(directory=frontend, html=True), name="study-ui")
    return study_app


app = create_app()
participant_app = create_app(participant_only=True)

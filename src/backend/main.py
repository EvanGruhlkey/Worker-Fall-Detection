"""FastAPI application entry point.

Run from the project root:

    uv run uvicorn backend.main:app --app-dir src --reload

Interactive API docs: http://127.0.0.1:8000/docs
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .db import init_db
from .manager import JobManager
from .routers import dashboard, events, jobs, location


def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        settings.ensure_dirs()
        init_db(settings.database_url)
        manager = JobManager(settings)
        manager.recover_interrupted_jobs()
        manager.warm_up()
        app.state.settings = settings
        app.state.manager = manager
        manager.start_configured_camera()
        yield
        manager.shutdown()

    app = FastAPI(
        title="Oxy Worker Fall Detection API",
        description="Upload videos or connect cameras, detect worker falls, and review them.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # The frontend runs as a separate app. Any origin is allowed during early
    # development; set FALL_CORS_ORIGINS (comma separated) to lock it down.
    origins = os.environ.get("FALL_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(jobs.router)
    app.include_router(events.router)
    app.include_router(dashboard.router)
    app.include_router(location.router)

    @app.get("/api/health", tags=["system"])
    def health():
        return {"status": "ok", "model": settings.model_path.name}

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()

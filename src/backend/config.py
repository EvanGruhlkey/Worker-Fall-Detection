"""Backend settings, read from environment variables with sensible defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: _env_path("FALL_DATA_DIR", REPO_ROOT / "data")
    )
    model_path: Path = field(
        default_factory=lambda: _env_path(
            "FALL_MODEL_PATH", REPO_ROOT / "models" / "yolo11n-pose.pt"
        )
    )
    image_size: int = field(
        default_factory=lambda: int(os.environ.get("FALL_IMAGE_SIZE", "640"))
    )
    # Seconds of footage kept before and after a fall for the review clip.
    clip_pre_seconds: float = field(
        default_factory=lambda: float(os.environ.get("FALL_CLIP_PRE_SECONDS", "5"))
    )
    clip_post_seconds: float = field(
        default_factory=lambda: float(os.environ.get("FALL_CLIP_POST_SECONDS", "5"))
    )

    @property
    def input_dir(self):
        return self.data_dir / "input"

    @property
    def output_dir(self):
        return self.data_dir / "output"

    @property
    def events_dir(self):
        return self.data_dir / "events"

    @property
    def database_url(self):
        return os.environ.get(
            "FALL_DATABASE_URL", f"sqlite:///{(self.data_dir / 'falls.db').as_posix()}"
        )

    def ensure_dirs(self):
        for directory in (self.input_dir, self.output_dir, self.events_dir):
            directory.mkdir(parents=True, exist_ok=True)

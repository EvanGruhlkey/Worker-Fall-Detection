"""Backend settings, read from environment variables with sensible defaults.

Variables can also be put in a ``.env`` file in the project root (see
``.env.example``). Variables already set in the shell take precedence.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from yolo_pose import DEFAULT_IMAGE_SIZE, parse_image_size

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path):
    """Copy KEY=value lines from a .env file into os.environ, without overriding."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file(REPO_ROOT / ".env")


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
    # YOLO inference size: None runs at the camera's native resolution (best
    # for people far from the camera); an int such as 640 scales the long side
    # down to it (faster). FALL_IMAGE_SIZE=native or a number.
    image_size: int | None = field(
        default_factory=lambda: parse_image_size(
            os.environ.get("FALL_IMAGE_SIZE", DEFAULT_IMAGE_SIZE)
        )
    )
    # Seconds of footage kept before and after a fall for the review clip.
    clip_pre_seconds: float = field(
        default_factory=lambda: float(os.environ.get("FALL_CLIP_PRE_SECONDS", "5"))
    )
    clip_post_seconds: float = field(
        default_factory=lambda: float(os.environ.get("FALL_CLIP_POST_SECONDS", "5"))
    )
    # Where this installation is; shown in the dashboard header and stored on
    # every fall event. A camera can override it with its own location.
    site_location: str = field(
        default_factory=lambda: os.environ.get("FALL_SITE_LOCATION", "Unspecified location")
    )
    site_coords: str | None = field(
        default_factory=lambda: os.environ.get("FALL_SITE_COORDS") or None
    )
    # Look up a place name (via OpenStreetMap) for the location the dashboard's
    # browser reports. Set FALL_REVERSE_GEOCODE=0 to keep coordinates local.
    reverse_geocode: bool = field(
        default_factory=lambda: os.environ.get("FALL_REVERSE_GEOCODE", "1") != "0"
    )
    # Set FALL_CAMERA_SOURCE (e.g. "0" or an rtsp:// URL) to start a live
    # camera when the server boots, so the dashboard has a feed without any
    # extra API call.
    camera_source: str | None = field(
        default_factory=lambda: os.environ.get("FALL_CAMERA_SOURCE") or None
    )
    camera_name: str | None = field(
        default_factory=lambda: os.environ.get("FALL_CAMERA_NAME") or None
    )
    camera_label: str | None = field(
        default_factory=lambda: os.environ.get("FALL_CAMERA_LABEL") or None
    )
    # Temporal classifier checkpoint (see docs/features/10-23-2026-temporal-classifier.md).
    # Leave empty to keep the rule-based detector in fall_detector.py.
    classifier_path: Path | None = field(
        default_factory=lambda: _env_path("FALL_CLASSIFIER_PATH", None)
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

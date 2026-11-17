"""Study manifests are facilitator-only; participant views are constructed separately."""
from pathlib import PurePosixPath
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Feed(Model):
    id: str = Field(min_length=1, max_length=60, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=100)
    file: str = Field(min_length=1, max_length=200)
    duration: float = Field(gt=0)
    pipeline_file: str | None = None

    @model_validator(mode="after")
    def safe_file(self):
        for filename in (self.file, self.pipeline_file):
            if filename is None:
                continue
            path = PurePosixPath(filename)
            if path.is_absolute() or ".." in path.parts or "\\" in filename or ":" in filename:
                raise ValueError("Media files must be relative paths inside the study media directory")
            if path.suffix.lower() not in {".mp4", ".webm"}:
                raise ValueError("Use browser-compatible MP4 or WebM footage")
        return self


class Incident(Model):
    camera: str
    time: float = Field(ge=0)
    review_file: str | None = None

    @field_validator("review_file")
    @classmethod
    def safe_review(cls, value):
        if value is not None:
            Feed(id="review", name="Review", file=value, duration=1)
        return value


class VideoSet(Model):
    id: Literal["A", "B"]
    feeds: list[Feed] = Field(min_length=6, max_length=6)
    wall_file: str | None = None
    pipeline_wall_file: str | None = None

    @field_validator("wall_file", "pipeline_wall_file")
    @classmethod
    def safe_wall(cls, value):
        if value is not None:
            Feed(id="wall", name="Wall", file=value, duration=1)
        return value
    truth: list[Incident] = Field(max_length=1000)
    alerts: list[Incident] = Field(max_length=1000)


class Manifest(Model):
    title: str = Field(min_length=1, max_length=100)
    mode: Literal["study", "pilot", "demo"] = "study"
    duration: float = Field(default=120, ge=10, le=120)
    early_tolerance: float = Field(default=1, ge=0, le=30)
    late_tolerance: float = Field(default=15, ge=0, le=60)
    detector_version: str = Field(min_length=1, max_length=200)
    alert_source: str = Field(min_length=1, max_length=300)
    sets: list[VideoSet] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def coherent(self):
        if self.mode == "study" and self.duration != 120:
            raise ValueError("Real study conditions must last 120 seconds; shorter runs must be labeled pilot or demo")
        if {s.id for s in self.sets} != {"A", "B"}:
            raise ValueError("Provide distinct A and B sets")
        files = []
        for video_set in self.sets:
            ids = [f.id for f in video_set.feeds]
            if len(set(ids)) != len(ids):
                raise ValueError("Camera IDs must be unique within a set")
            for feed in video_set.feeds:
                if feed.duration < self.duration:
                    raise ValueError("Every feed must cover the entire condition")
                files.append(feed.file)
            for event in [*video_set.truth, *video_set.alerts]:
                if event.camera not in ids or event.time >= self.duration:
                    raise ValueError("Incident references an unknown camera or out-of-range timestamp")
        if len(set(files)) != len(files):
            raise ValueError("Use different footage across cameras and conditions")
        if self.mode == "study" and abs(len(self.sets[0].truth) - len(self.sets[1].truth)) > 1:
            raise ValueError("Balance fall counts across the two sets")
        return self


class Enrollment(Model):
    code: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,40}$")
    group: Literal["oxy", "student"]
    age_range: Literal["18-24", "25-34", "35-44", "45-54", "55+"]
    experience: Literal["none", "some", "regular"]


class Consent(Model):
    adult: Literal[True]
    vision: Literal[True]
    eligible: Literal[True]
    voluntary: Literal[True]


class Report(Model):
    camera: str
    time: float = Field(ge=0)
    notes: str = Field(default="", max_length=500)


class ReportNote(Model):
    text: str = Field(max_length=500)


class Decision(Model):
    action: Literal["dispatch", "investigate", "false_alarm"]


class Survey(Model):
    manual_confidence: int = Field(ge=1, le=5)
    manual_effort: int = Field(ge=1, le=5)
    manual_ease: int = Field(ge=1, le=5)
    automated_confidence: int = Field(ge=1, le=5)
    automated_effort: int = Field(ge=1, le=5)
    automated_ease: int = Field(ge=1, le=5)
    trust: int = Field(ge=1, le=5)
    feedback: str = Field(max_length=3000)


class TaskAnswer(Model):
    index: int = Field(ge=0)
    answer: int = Field(ge=-10000, le=10000)


class Note(Model):
    text: str = Field(max_length=3000)


class Pause(Model):
    reason: Literal["hidden", "buffering", "media_error", "operator", "drift"]

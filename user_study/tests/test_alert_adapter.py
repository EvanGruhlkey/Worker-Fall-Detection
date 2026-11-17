import json
import runpy
from pathlib import Path
import pytest

convert = runpy.run_path(str(Path(__file__).resolve().parents[2] / "user_study/scripts" / "build_study_alerts.py"))["convert"]


def test_pipeline_adapter_preserves_real_events_and_provenance(tmp_path):
    path = tmp_path / "signals.json"
    path.write_text(json.dumps({"classifier": "trained-checkpoint.pt", "events": [{"time": 8, "track_id": 1}, {"time": 3, "track_id": 2}, {"time": 3, "track_id": 3}]}))
    result = convert(path, "camera1", 2)
    assert result["alerts"] == [{"camera": "camera1", "time": 5}, {"camera": "camera1", "time": 5}, {"camera": "camera1", "time": 10}]
    assert len(result["sha256"]) == 64
    path.write_text(json.dumps({"events": []}))
    with pytest.raises(ValueError, match="no classifier"):
        convert(path, "camera1")

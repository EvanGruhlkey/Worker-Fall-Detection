import pytest
from user_study.backend.scoring import score


def test_matching_maximizes_count_before_distance():
    result = score([{"camera": "c", "time": 10}, {"camera": "c", "time": 14}],
                   [{"camera": "c", "time": 13}, {"camera": "c", "time": 17}], 0, 4)
    assert result["detected"] == 2
    assert result["latencies"] == [3, 3]


def test_duplicates_wrong_camera_and_empty_denominators():
    result = score([{"camera": "a", "time": 5}],
                   [{"camera": "a", "time": 6}, {"camera": "a", "time": 7}, {"camera": "b", "time": 5}], 0, 3)
    assert result["detected"] == 1
    assert result["false_alarms"] == 2
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 1
    assert score([], [], 0, 3)["recall"] is None
    assert score([], [], 0, 3)["mean_latency"] is None

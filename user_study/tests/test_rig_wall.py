import runpy
from pathlib import Path

module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "user_study/scripts/prepare_study_rig_wall.py"))


def test_six_camera_cycle_runs_two_minutes_with_six_falls():
    normal = [{"frames": 180, "id": f"normal{i}"} for i in range(10)]
    falls = [{"frames": 180, "id": f"fall{i}", "onset": 7.5} for i in range(11)]
    timelines = [module["timeline"](normal, falls, camera, 0) for camera in range(6)]
    assert len(timelines) == 6
    assert len({schedule[0]["clip"]["id"] for schedule in timelines}) == 6
    assert sum(part["kind"] == "fall" for schedule in timelines for part in schedule) == 6
    event_times = []
    for schedule in timelines:
        assert schedule[0]["start_frame"] == 0
        assert schedule[-1]["start_frame"] + schedule[-1]["frames"] == 1440
        for previous, following in zip(schedule, schedule[1:]):
            assert previous["start_frame"] + previous["frames"] == following["start_frame"]
        event_times.extend(part["start_frame"] / 12 + part["clip"]["onset"] for part in schedule if part["kind"] == "fall")
    assert len(set(event_times)) == 6
    assert min(event_times) >= 5 and max(event_times) < 118
    intervals = [b-a for a,b in zip(sorted(event_times), sorted(event_times)[1:])]
    assert len(set(intervals)) > 1
    assert min(intervals) >= 15
    normal_frames = sum(part["frames"] for schedule in timelines for part in schedule if part["kind"] == "normal")
    assert normal_frames / (6 * 1440) >= .85


def test_conditions_do_not_repeat_the_same_fall_schedule():
    normal = [{"frames": 180, "id": f"normal{i}"} for i in range(10)]
    falls = [{"frames": 180, "id": f"fall{i}", "onset": 7.5} for i in range(11)]
    schedules = []
    for set_index in (0, 1):
        times = sorted(part["start_frame"] for camera in range(6) for part in module["timeline"](normal, falls, camera, set_index) if part["kind"] == "fall")
        schedules.append(times)
    assert schedules[0] != schedules[1]

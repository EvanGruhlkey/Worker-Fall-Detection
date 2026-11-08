from backend.fall_detector import FallDetector

FPS = 30


def person(track_id, hip_y, torso_angle_horizontal=False, height=400):
    """A synthetic tracked person: upright, or lying with the torso horizontal."""
    keypoints = [{"x": 0.0, "y": 0.0, "confidence": 0.0} for _ in range(12)]
    if torso_angle_horizontal:
        shoulders, hips = (300.0, hip_y), (450.0, hip_y)
        bbox = [250.0, hip_y - 60, 650.0, hip_y + 60]
    else:
        shoulders, hips = (400.0, hip_y - 150), (400.0, hip_y)
        bbox = [350.0, hip_y - 200, 450.0, hip_y - 200 + height]
    for index in (0, 1):
        keypoints[index] = {"x": shoulders[0], "y": shoulders[1], "confidence": 0.9}
    for index in (6, 7):
        keypoints[index] = {"x": hips[0], "y": hips[1], "confidence": 0.9}
    return {"track_id": track_id, "bbox": bbox, "confidence": 0.9, "keypoints": keypoints}


def run(detector, frames):
    detections = []
    for index, people in enumerate(frames):
        detections += detector.update(people, index / FPS)
    return detections


def test_sudden_fall_is_detected_once():
    standing = [[person(1, 300)] for _ in range(FPS)]
    falling = [[person(1, 300 + i * 25)] for i in range(10)]
    lying = [[person(1, 560, torso_angle_horizontal=True)] for _ in range(FPS * 2)]

    detections = run(FallDetector(), standing + falling + lying)

    assert len(detections) == 1
    assert detections[0].track_id == 1
    assert 0 < detections[0].confidence <= 1
    assert "hip dropped" in detections[0].reason


def test_standing_person_is_not_a_fall():
    assert run(FallDetector(), [[person(1, 300)] for _ in range(FPS * 3)]) == []


def test_lying_without_descent_is_not_a_fall():
    # Someone already lying down when first seen (e.g. working under equipment).
    lying = [[person(1, 560, torso_angle_horizontal=True)] for _ in range(FPS * 3)]
    assert run(FallDetector(), lying) == []


def test_brief_dip_that_recovers_is_not_a_fall():
    standing = [[person(1, 300)] for _ in range(FPS)]
    down = [[person(1, 560, torso_angle_horizontal=True)] for _ in range(5)]  # < 0.5s
    back_up = [[person(1, 300)] for _ in range(FPS)]
    assert run(FallDetector(), standing + down + back_up) == []


def test_untracked_people_are_ignored():
    frames = [[{**person(1, 300 + i * 10), "track_id": None}] for i in range(FPS * 2)]
    assert run(FallDetector(), frames) == []


def test_fall_is_detected_when_the_tracker_switches_id():
    # ByteTrack often gives a falling person a new ID when their box turns wide.
    standing = [[person(1, 300)] for _ in range(FPS)]
    falling = [[person(1, 300 + i * 25)] for i in range(6)]
    lying = [[person(7, 560, torso_angle_horizontal=True)] for _ in range(FPS * 2)]

    detections = run(FallDetector(), standing + falling + lying)

    assert len(detections) == 1
    assert detections[0].track_id == 7


def test_new_id_far_away_does_not_inherit_history():
    standing = [[person(1, 300)] for _ in range(FPS)]
    # A different person, already lying down, appears across the frame.
    far = person(7, 560, torso_angle_horizontal=True)
    far["bbox"] = [far["bbox"][0] + 2000, far["bbox"][1], far["bbox"][2] + 2000, far["bbox"][3]]
    for kp in far["keypoints"]:
        kp["x"] += 2000
    lying = [[far] for _ in range(FPS * 2)]
    assert run(FallDetector(), standing + lying) == []


def test_new_id_long_after_does_not_inherit_history():
    standing = [[person(1, 300)] for _ in range(FPS)]
    gap = [[] for _ in range(FPS * 2)]  # nobody seen for 2 s
    lying = [[person(7, 560, torso_angle_horizontal=True)] for _ in range(FPS * 2)]
    assert run(FallDetector(), standing + gap + lying) == []

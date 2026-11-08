"""Compare two pose JSON files of the same video (e.g. two pose pipelines).

    uv run scripts/compare_pose_files.py REFERENCE.json CANDIDATE.json [...more pairs]

Per frame (matched by source frame), people are paired by box IoU >= 0.5.
Reports: share of people matched both ways, box IoU, keypoint error in body
heights (box height) for joints both sides see (confidence >= 0.3), keypoint
confidence correlation, and how stable track IDs are relative to each other.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temporal.pose_io import load_pose_json  # noqa: E402


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(reference_path, candidate_path):
    ref, cand = load_pose_json(reference_path), load_pose_json(candidate_path)
    cand_at = {round(t, 3): people for t, people in zip(cand.times, cand.people)}
    ref_count = cand_count = matched = 0
    ious, errors, conf_pairs, id_pairs = [], [], [], Counter()
    for t, ref_people in zip(ref.times, ref.people):
        cand_people = cand_at.get(round(t, 3))
        if cand_people is None:
            continue
        ref_count += len(ref_people)
        cand_count += len(cand_people)
        used = set()
        for person in ref_people:
            best, best_iou = None, 0.5
            for j, other in enumerate(cand_people):
                value = iou(person["bbox"], other["bbox"])
                if j not in used and value >= best_iou:
                    best, best_iou = j, value
            if best is None:
                continue
            used.add(best)
            matched += 1
            other = cand_people[best]
            ious.append(best_iou)
            id_pairs[(person["track_id"], other["track_id"])] += 1
            height = max(person["bbox"][3] - person["bbox"][1], 1.0)
            for a, b in zip(person["keypoints"], other["keypoints"]):
                conf_pairs.append((a["confidence"], b["confidence"]))
                if a["confidence"] >= 0.3 and b["confidence"] >= 0.3:
                    errors.append(np.hypot(a["x"] - b["x"], a["y"] - b["y"]) / height)
    confs = np.array(conf_pairs)
    # Track agreement: for each reference ID, the share of its matches that go to
    # its most common candidate ID (1.0 = the two trackers split people identically).
    by_ref = {}
    for (r, c), n in id_pairs.items():
        by_ref.setdefault(r, Counter())[c] += n
    consistency = sum(max(c.values()) for c in by_ref.values()) / max(sum(id_pairs.values()), 1)
    errors = np.array(errors)
    return {
        "frames": len(ref.times),
        "people ref/cand": f"{ref_count}/{cand_count}",
        "matched (of ref)": f"{matched / max(ref_count, 1):.1%}",
        "matched (of cand)": f"{matched / max(cand_count, 1):.1%}",
        "box IoU median": f"{np.median(ious):.3f}" if ious else "-",
        "kp error median": f"{np.median(errors):.4f} body heights" if len(errors) else "-",
        "kp error p90": f"{np.percentile(errors, 90):.4f}" if len(errors) else "-",
        "kp conf corr": f"{np.corrcoef(confs[:, 0], confs[:, 1])[0, 1]:.3f}" if len(confs) > 2 else "-",
        "kp conf mean ref/cand": f"{confs[:, 0].mean():.3f}/{confs[:, 1].mean():.3f}" if len(confs) else "-",
        "track consistency": f"{consistency:.1%}",
    }


def main(argv):
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 1
    for reference, candidate in zip(argv[::2], argv[1::2]):
        print(Path(reference).stem)
        for key, value in compare(reference, candidate).items():
            print(f"  {key:22s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

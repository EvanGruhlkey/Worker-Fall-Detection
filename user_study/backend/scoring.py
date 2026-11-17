"""Deterministic maximum-cardinality, minimum-distance temporal matching."""
from functools import lru_cache


def score(truth, detections, early, late):
    pairs = []
    for camera in sorted({e["camera"] for e in truth}):
        targets = sorted((e["time"], i) for i, e in enumerate(truth) if e["camera"] == camera)
        reports = sorted((e["time"], i) for i, e in enumerate(detections) if e["camera"] == camera)

        @lru_cache(None)
        def match(i, j):
            if i == len(targets) or j == len(reports):
                return ()
            options = [match(i + 1, j), match(i, j + 1)]
            delta = reports[j][0] - targets[i][0]
            if -early <= delta <= late:
                options.append(((targets[i][1], reports[j][1]),) + match(i + 1, j + 1))
            return min(options, key=lambda p: (-len(p), sum(abs(detections[b]["time"] - truth[a]["time"]) for a, b in p), p))

        pairs.extend(match(0, 0))
    pairs.sort()
    latencies = [detections[b].get("reported_at", detections[b]["time"]) - truth[a]["time"] for a, b in pairs]
    return {
        "detected": len(pairs), "missed": len(truth) - len(pairs),
        "false_alarms": len(detections) - len(pairs),
        "precision": len(pairs) / len(detections) if detections else None,
        "recall": len(pairs) / len(truth) if truth else None,
        "latencies": latencies,
        "mean_latency": sum(latencies) / len(latencies) if latencies else None,
        "matches": [{"truth_index": a, "detection_index": b} for a, b in pairs],
    }

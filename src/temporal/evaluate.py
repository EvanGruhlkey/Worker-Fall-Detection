"""Event-level evaluation: does the whole detector raise the right alerts?

Window scores say how well single 2.5 s windows are classified. What matters
on a rig is per fall: did an alert fire for the person who fell, how soon after
impact, and how many alerts fired when nobody fell. This module replays a clip
through the same pieces live detection uses (TrackWindows, track handoff,
score_tracks, ThresholdGate) and scores the alerts.

Scoring is split in two so a threshold sweep is cheap: ``score_clip`` runs the
model once and records every track's probability per frame; ``alerts_for``
then replays only the gate at any threshold.

Rules, for one clip:

- caught:      an alert on the faller (following track handoffs) between
               ``early`` seconds before the fall starts and ``catch_after``
               seconds after impact (the labeled fall end)
- delay:       alert time minus impact time (negative = before impact)
- late:        a faller alert after that window (counted apart, not caught)
- false alert: any other alert (another person, or the faller before the fall)
"""

from dataclasses import dataclass, field

import numpy as np

from utils.sliding_window import TrackWindows

from .detector import drop_checked, handoff_lost_tracks, score_tracks, track_drops
from .features import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER
from .gate import ThresholdGate


@dataclass
class ClipScores:
    clip: object  # dataset.Clip
    probabilities: list = field(default_factory=list)  # per frame: {track_id: p}
    renames: list = field(default_factory=list)  # per frame: [(old, new)]
    alive: list = field(default_factory=list)  # per frame: track IDs still held
    faller: list = field(default_factory=list)  # per frame: set of the faller's IDs
    drops: list = field(default_factory=list)  # per frame: {track_id: fastest body drop in its window}

    def faller_curve(self):
        """Highest faller probability per frame (NaN where none was scored)."""
        return np.array([
            max((p for t, p in probs.items() if t in ids), default=np.nan)
            for probs, ids in zip(self.probabilities, self.faller)
        ])

    def others_curve(self):
        """Highest probability among everyone else per frame (NaN if nobody scored)."""
        return np.array([
            max((p for t, p in probs.items() if t not in ids), default=np.nan)
            for probs, ids in zip(self.probabilities, self.faller)
        ])


@dataclass
class Alert:
    time: float
    frame: int
    track_id: int
    probability: float
    is_faller: bool


def score_clip(model, clip, window_size, max_gap=5):
    """Run the model over a clip the way live detection would."""
    windows = TrackWindows(window_size=window_size, max_gap=max_gap)
    everyone = clip.has_fall and not clip.track_ids  # no ID given: anyone may be the faller
    faller = set(clip.track_ids or ()) if clip.has_fall else set()
    scores = ClipScores(clip)
    for index, people in enumerate(clip.frames):
        renames = handoff_lost_tracks(windows, people, index)
        for old_id, new_id in renames:
            if old_id in faller:
                faller.add(new_id)
        updated = windows.update(index, people)
        if everyone:
            faller.update(updated)
        scored = score_tracks(model, windows, updated, clip.fps)
        scores.probabilities.append({t: p for t, (p, _) in scored.items()})
        scores.drops.append(track_drops(windows, scored, clip.fps))
        scores.renames.append(renames)
        scores.alive.append(set(windows.track_ids()))
        scores.faller.append(set(faller))
    return scores


def alerts_for(scores, threshold, debounce=5, cooldown_seconds=30.0, min_drop=0.0):
    """The alerts the gate raises over a scored clip at ``threshold``.

    With ``min_drop`` the fast-drop check runs first, as in live detection.
    """
    gate = ThresholdGate(threshold, debounce, cooldown_seconds)
    alerts, known = [], set()
    times = scores.clip.times
    for index, probs in enumerate(scores.probabilities):
        for old_id, new_id in scores.renames[index]:
            gate.rename(old_id, new_id)
        if min_drop > 0:
            probs = drop_checked(probs, scores.drops[index], min_drop)
        for track_id, probability in probs.items():
            known.add(track_id)
            if gate.update(track_id, probability, times[index]):
                alerts.append(Alert(times[index], index, track_id, probability,
                                    track_id in scores.faller[index]))
        for track_id in known - scores.alive[index]:
            gate.forget(track_id)
        known &= scores.alive[index]
    return alerts


UPRIGHT_TORSO = 1 / 3  # torso within 30 degrees of vertical
UPRIGHT_HIP = 0.4  # hips at least 0.4 standing heights above the box bottom


def _posture(person, standing_height):
    """(torso angle 0-1 from vertical, hip height in standing heights) of one detection, or None."""
    kp = person["keypoints"]

    def mid(a, b):
        pts = [kp[i] for i in (a, b) if kp[i]["confidence"] >= 0.3]
        return (np.mean([p["x"] for p in pts]), np.mean([p["y"] for p in pts])) if pts else None

    shoulder, hip = mid(LEFT_SHOULDER, RIGHT_SHOULDER), mid(LEFT_HIP, RIGHT_HIP)  # the 12 body joints
    if shoulder is None or hip is None or standing_height <= 0:
        return None
    dx, dy = shoulder[0] - hip[0], hip[1] - shoulder[1]
    torso = float(np.degrees(np.arctan2(abs(dx), dy)) / 90.0)
    return torso, float((person["bbox"][3] - hip[1]) / standing_height)


def stays_down(scores, alerts, seconds, upright_share=0.5, lookback=4.0):
    """Keep only alerts after which the person is not back on their feet within ``seconds``.

    A fall ends with the worker down; most false alerts are workers who drop
    fast and get straight back up (a quick crouch, a jump down, ducking under a
    pipe). Each alert is held for ``seconds`` and dropped when, over that time,
    at least ``upright_share`` of the frames the person is seen in show them
    upright (torso near vertical and hips at standing height, measured against
    their own box height over the ``lookback`` seconds before the alert). A
    person the detector loses during the check counts as down, so a fall the
    tracker drops is not filtered out. Kept alerts move to the end of the check,
    as they would in live use. Approximation: the gate's cooldown still starts
    at the original alert, so a dropped alert can hide a second one on the same
    person within the cooldown (rare in 10-15 s clips; errs towards fewer alerts).
    """
    if seconds <= 0:
        return alerts
    clip = scores.clip
    times = clip.times
    kept = []
    for alert in alerts:
        ids = {alert.track_id}
        before = [p["bbox"][3] - p["bbox"][1]
                  for i in range(len(times)) if alert.time - lookback <= times[i] <= alert.time
                  for p in clip.frames[i] if p["track_id"] in ids]
        standing = float(np.percentile(before, 90)) if before else 0.0
        upright = seen = 0
        end_index = alert.frame
        for i in range(alert.frame, len(times)):
            if times[i] > alert.time + seconds:
                break
            end_index = i
            for old_id, new_id in scores.renames[i]:
                if old_id in ids:
                    ids.add(new_id)
            for person in clip.frames[i]:
                if person["track_id"] not in ids:
                    continue
                posture = _posture(person, standing)
                if posture is None:
                    continue
                seen += 1
                torso, hip = posture
                upright += torso < UPRIGHT_TORSO and hip >= UPRIGHT_HIP
                break
        if seen and upright / seen >= upright_share:
            continue
        kept.append(Alert(times[end_index], end_index, alert.track_id, alert.probability, alert.is_faller))
    return kept


def clip_outcome(scores, alerts, catch_after=1.5, early=0.5):
    """Score one clip's alerts against its labeled falls."""
    clip = scores.clip
    falls = clip.fall_seconds
    caught, delays, used = [], [], set()
    for start, end in falls:
        hit = next(
            (i for i, a in enumerate(alerts)
             if a.is_faller and i not in used and start - early <= a.time <= end + catch_after),
            None,
        )
        caught.append(hit is not None)
        if hit is not None:
            used.add(hit)
            delays.append(alerts[hit].time - end)
    late = sum(
        1 for i, a in enumerate(alerts)
        if i not in used and a.is_faller and any(a.time > end + catch_after for _, end in falls)
    )
    false_alerts = sum(1 for i, a in enumerate(alerts) if i not in used) - late
    # Time in which an alert would count as false: the clip minus each fall's
    # catch window and the time the faller then spends on the ground.
    busy = sum(min(clip.duration, end + catch_after) - max(0.0, start - early) for start, end in falls)
    faller_curve, others_curve = scores.faller_curve(), scores.others_curve()
    return {
        "clip": clip.name,
        "group": clip.group,
        "has_fall": clip.has_fall,
        "falls": len(falls),
        "caught": sum(caught),
        "delay": float(np.mean(delays)) if delays else None,
        "late": late,
        "false_alerts": false_alerts,
        "non_fall_seconds": max(0.0, clip.duration - busy),
        "peak_faller": _peak(faller_curve),
        "peak_others": _peak(others_curve),
    }


def _peak(curve):
    finite = curve[np.isfinite(curve)] if len(curve) else curve
    return float(finite.max()) if len(finite) else None


def summarize(outcomes):
    """Totals over many clip outcomes (one threshold)."""
    falls = sum(o["falls"] for o in outcomes)
    caught = sum(o["caught"] for o in outcomes)
    false_alerts = sum(o["false_alerts"] for o in outcomes)
    seconds = sum(o["non_fall_seconds"] for o in outcomes)
    normal = [o for o in outcomes if not o["has_fall"]]
    delays = [o["delay"] for o in outcomes if o["delay"] is not None]
    return {
        "falls": falls,
        "caught": caught,
        "recall": caught / falls if falls else None,
        "late": sum(o["late"] for o in outcomes),
        "false_alerts": false_alerts,
        "non_fall_minutes": seconds / 60,
        "false_alerts_per_hour": false_alerts / (seconds / 3600) if seconds else None,
        "normal_clip_false_alerts": sum(o["false_alerts"] for o in normal),
        "median_delay": float(np.median(delays)) if delays else None,
    }

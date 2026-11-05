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

from sliding_window import TrackWindows

from .detector import handoff_lost_tracks, score_tracks
from .gate import ThresholdGate


@dataclass
class ClipScores:
    clip: object  # dataset.Clip
    probabilities: list = field(default_factory=list)  # per frame: {track_id: p}
    renames: list = field(default_factory=list)  # per frame: [(old, new)]
    alive: list = field(default_factory=list)  # per frame: track IDs still held
    faller: list = field(default_factory=list)  # per frame: set of the faller's IDs

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
        scores.renames.append(renames)
        scores.alive.append(set(windows.track_ids()))
        scores.faller.append(set(faller))
    return scores


def alerts_for(scores, threshold, debounce=5, cooldown_seconds=30.0):
    """The alerts the gate raises over a scored clip at ``threshold``."""
    gate = ThresholdGate(threshold, debounce, cooldown_seconds)
    alerts, known = [], set()
    times = scores.clip.times
    for index, probs in enumerate(scores.probabilities):
        for old_id, new_id in scores.renames[index]:
            gate.rename(old_id, new_id)
        for track_id, probability in probs.items():
            known.add(track_id)
            if gate.update(track_id, probability, times[index]):
                alerts.append(Alert(times[index], index, track_id, probability,
                                    track_id in scores.faller[index]))
        for track_id in known - scores.alive[index]:
            gate.forget(track_id)
        known &= scores.alive[index]
    return alerts


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

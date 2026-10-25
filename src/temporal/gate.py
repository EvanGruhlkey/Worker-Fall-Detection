"""Stage 7: turn a noisy per-frame probability stream into fall events.

Three cheap rules from the system guide:

- threshold: the probability must exceed a cutoff (initially 0.8);
- debounce:  it must stay above the cutoff for N consecutive frames (5), so a
             single bad pose estimate never fires;
- cooldown:  after an event, the same person fires nothing for 30 s, so one
             fall does not raise ten alerts.
"""


class ThresholdGate:
    def __init__(self, threshold=0.8, debounce=5, cooldown_seconds=30.0):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if debounce < 1:
            raise ValueError("debounce must be at least 1 frame")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be zero or greater")
        self.threshold = threshold
        self.debounce = debounce
        self.cooldown_seconds = cooldown_seconds
        self._streak = {}
        self._last_fired = {}

    def update(self, track_id, probability, timestamp):
        """Record one probability for a track; return True when an event fires."""
        if probability >= self.threshold:
            self._streak[track_id] = self._streak.get(track_id, 0) + 1
        else:
            self._streak[track_id] = 0
            return False

        if self._streak[track_id] < self.debounce:
            return False
        last = self._last_fired.get(track_id)
        if last is not None and timestamp - last < self.cooldown_seconds:
            return False
        self._last_fired[track_id] = timestamp
        return True

    def streak(self, track_id):
        """How many consecutive frames the track has been above the threshold."""
        return self._streak.get(track_id, 0)

    def is_above(self, track_id):
        return self.streak(track_id) > 0

    def rename(self, old_id, new_id):
        """Carry the debounce streak and cooldown over when a track changes ID."""
        if old_id in self._streak:
            self._streak[new_id] = self._streak.pop(old_id)
        if old_id in self._last_fired:
            self._last_fired[new_id] = self._last_fired.pop(old_id)

    def forget(self, track_id):
        self._streak.pop(track_id, None)
        self._last_fired.pop(track_id, None)

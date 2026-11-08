"""Plots of the motion signals the classifier sees: heights, torso angle,
vertical velocity and fall probability over time.

Two outputs share one style:

- ``SignalPanel`` renders a rolling view for one track into an image that is
  placed next to the annotated video frame, so the plots scroll with the video.
- ``plot_track_timeline`` saves a full-length PNG per track after a run.

Lines are 2 px, gridlines are solid hairlines, text uses ink colours (never a
series colour), a legend is present wherever a panel holds two or more series,
and fall events are marked with a labelled line rather than colour alone.
"""

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .features import SIGNAL_INDEX  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
CRITICAL = "#d03b3b"

PANELS = (
    # (title, y label, [(series key, legend label)], y limits)
    ("Height above the feet", "body heights", [("hip_height", "hip"), ("box_height", "box height")], (0.0, 1.2)),
    ("Torso angle", "degrees from upright", [("torso_angle", "torso")], (0.0, 100.0)),
    ("Vertical velocity", "body heights / s", [("hip_velocity", "hip")], None),
    ("Fall probability", "p(fall)", [("probability", "classifier")], (0.0, 1.05)),
)


class TrackSignals:
    """Per-frame signals of one track, appended while a video is processed."""

    def __init__(self, track_id):
        self.track_id = track_id
        self.time, self.frame = [], []
        self.hip_height, self.box_height = [], []
        self.torso_angle, self.hip_velocity = [], []
        self.probability, self.above = [], []
        self.events = []  # times at which the threshold gate fired
        self.previous_ids = []  # tracker IDs this person had before a handoff

    def rename(self, new_id):
        self.previous_ids.append(self.track_id)
        self.track_id = new_id

    def label(self):
        if self.previous_ids:
            return f"Track {self.track_id} (was {', '.join(map(str, self.previous_ids))})"
        return f"Track {self.track_id}"

    def append(self, time, frame, signals, box_height, probability=np.nan, above=False):
        self.time.append(float(time))
        self.frame.append(int(frame))
        self.hip_height.append(float(signals[SIGNAL_INDEX["hip_height"]]))
        self.box_height.append(float(box_height))
        self.torso_angle.append(float(signals[SIGNAL_INDEX["torso_angle"]]) * 90.0)
        self.hip_velocity.append(float(signals[SIGNAL_INDEX["hip_velocity"]]))
        self.probability.append(float(probability))
        self.above.append(bool(above))

    def __len__(self):
        return len(self.time)

    @property
    def last_time(self):
        return self.time[-1] if self.time else None

    def series(self, key):
        return np.asarray(getattr(self, key), dtype=np.float64)

    def max_probability(self):
        values = self.series("probability")
        return float(np.nanmax(values)) if len(values) and not np.all(np.isnan(values)) else None

    def to_dict(self):
        return {
            "track_id": self.track_id,
            "previous_ids": list(self.previous_ids),
            "frames": len(self),
            "max_probability": self.max_probability(),
            "events": [round(t, 3) for t in self.events],
            "time": [round(t, 3) for t in self.time],
            "frame": list(self.frame),
            "hip_height": [round(v, 4) for v in self.hip_height],
            "box_height": [round(v, 4) for v in self.box_height],
            "torso_angle": [round(v, 2) for v in self.torso_angle],
            "hip_velocity": [round(v, 4) for v in self.hip_velocity],
            "probability": [None if np.isnan(v) else round(v, 4) for v in self.probability],
        }


def _style(ax, title, ylabel, limits):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(True, color=GRIDLINE, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.set_title(title, loc="left", fontsize=9, color=INK_SECONDARY, pad=4)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_SECONDARY)
    if limits is not None:
        ax.set_ylim(*limits)


def _draw_panels(fig, axes, signals, threshold, start, end, rolling):
    """Fill four axes with one track's signals between `start` and `end`."""
    t = signals.series("time") if signals is not None else np.zeros(0)
    for ax, (title, ylabel, series, limits) in zip(axes, PANELS):
        ax.clear()
        _style(ax, title, ylabel, limits)
        for color, (key, label) in zip(SERIES, series):
            values = signals.series(key) if signals is not None else np.zeros(0)
            ax.plot(t, values, color=color, linewidth=2, solid_capstyle="round",
                    solid_joinstyle="round", label=label)
        if title == "Vertical velocity":
            ax.axhline(0.0, color=AXIS, linewidth=1)
            in_view = (t >= start) & (t <= end)
            peak = 2.0
            if signals is not None and in_view.any():
                peak = max(peak, 1.15 * float(np.nanmax(np.abs(
                    signals.series("hip_velocity")[in_view]))))
            ax.set_ylim(-peak, peak)
        if title == "Fall probability":
            ax.axhline(threshold, color=INK_MUTED, linewidth=1)
            ax.text(end, threshold, f" {threshold:.2f} threshold", fontsize=7, color=INK_MUTED,
                    va="bottom", ha="right")
            if signals is not None and len(t):
                p = signals.series("probability")
                ax.fill_between(t, threshold, p, where=p >= threshold, color=CRITICAL, alpha=0.15,
                                linewidth=0, interpolate=True)
        if len(series) > 1:
            ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=INK_SECONDARY, ncol=len(series))
        if signals is not None:
            for event in signals.events:
                if start <= event <= end:
                    ax.axvline(event, color=CRITICAL, linewidth=1.5)
                    if title == "Fall probability":
                        ax.text(event, 1.0, " FALL", fontsize=8, color=INK, va="top", ha="left")
        ax.set_xlim(start, end)
    axes[-1].set_xlabel("time (s)", fontsize=8, color=INK_SECONDARY)
    if rolling:
        fig.align_ylabels(axes)


class SignalPanel:
    """Rolling plot of one track's signals, rendered to a BGR image."""

    def __init__(self, width=640, height=720, seconds=6.0, threshold=0.8):
        self.width, self.height, self.seconds, self.threshold = width, height, seconds, threshold
        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=SURFACE)
        self.canvas = FigureCanvasAgg(self.fig)
        self.axes = self.fig.subplots(4, 1, sharex=True)
        self.fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.07, hspace=0.45)
        self.title = self.fig.suptitle("", fontsize=11, color=INK, x=0.12, ha="left")

    def render(self, signals, now, label=""):
        self.title.set_text(label)
        end = max(now, self.seconds)
        _draw_panels(self.fig, self.axes, signals, self.threshold, end - self.seconds, end + 0.1, True)
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        return cv2.cvtColor(rgba[..., :3].copy(), cv2.COLOR_RGB2BGR)


def compose(frame, panel):
    """Place the signal panel to the right of the frame, matching its height."""
    height = panel.shape[0]
    width = max(1, round(frame.shape[1] * height / frame.shape[0]))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.hstack([resized, panel])


def plot_track_timeline(signals, path, threshold=0.8, title=None):
    """Save a full-length four-panel plot of one track's signals."""
    fig = Figure(figsize=(12, 9), dpi=110, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 1, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.06, hspace=0.4)
    fig.suptitle(title or signals.label(), fontsize=12, color=INK, x=0.08, ha="left")
    t = signals.series("time")
    start, end = (float(t[0]), float(t[-1]) + 0.05) if len(t) else (0.0, 1.0)
    _draw_panels(fig, axes, signals, threshold, start, end, False)
    fig.savefig(path, facecolor=SURFACE)
    return path


def draw_track_status(frame, person, probability=None, above=False, highlight=False):
    """Write the probability under a person's box; outline it when above threshold."""
    x1, y1, x2, y2 = (int(v) for v in person["bbox"])
    if highlight or above:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (59, 59, 208), 4)  # CRITICAL in BGR
    if probability is None or np.isnan(probability):
        return
    label = f"p {probability:.2f}"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (w, h), baseline = cv2.getTextSize(label, font, scale, thickness)
    x = max(0, min(x1, frame.shape[1] - w - 8))
    y = min(frame.shape[0] - baseline - 4, y2 + h + 10)
    cv2.rectangle(frame, (x, y - h - 6), (x + w + 8, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, label, (x + 4, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

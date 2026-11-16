"""Build a write-up package for a training trial: charts, example videos, stills, rundown.

    uv run scripts/build_trial_report.py                       # trial 1 (runs of 2026-11-05)
    uv run scripts/build_trial_report.py --runs-date 2026-11-05 --out reports/2026-11-05_trial1
    # trial 2: runs over two days, best = the combined batch 4 run
    uv run scripts/build_trial_report.py --trial 2 --runs-date 2026-11-15 2026-11-16 --best t2-best

What it does:

1. Re-runs the best setup's cross-validation (Rig + CAUCAFall, ambiguous
   windows left out; one seed) to get, for every rig clip, the fall scores
   from the fold model that never saw that clip. Training is deterministic for
   a seed, so this is seed 0 of the reported run.
2. Renders every rig clip as an annotated video: skeletons and boxes from the
   pose file, each person's fall score, the labeled fall, alerts (true or false),
   and a timeline strip of the scores under the video.
3. Saves stills at the moments that matter (alerts, impacts of missed falls).
4. Draws summary charts across all runs of the trial from runs/<date>_*/metrics.json.
5. Writes data/examples.csv and data/examples.json: every rig clip's outcome,
   alert times and the paths of its video and stills (the rundown README is
   written from these).

Output folder layout:
    charts/    summary charts (PNG, paste into any document)
    plots/     the best run's per-run plots (copied)
    videos/    caught/ missed/ not_visible/ normal_work/  one annotated MP4 per rig clip
    stills/    key frames as JPG
    data/      scores_seed0.json (per-frame scores, to re-render without retraining), per_clip.csv
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from backend.media import VideoWriter  # noqa: E402
from temporal.crossval import RunConfig, run_cv  # noqa: E402
from temporal.dataset import load_clips  # noqa: E402
from temporal.evaluate import alerts_for, clip_outcome  # noqa: E402

# Palette (light, for documents): ink text, one blue accent, grey for the rest.
INK, INK2, INK3, GRID = "#14181c", "#4a525b", "#7c858e", "#e3e6e9"
BLUE, ORANGE, GREY, RED, GREEN = "#2a78d6", "#eb6834", "#9aa3ad", "#c62828", "#1a7f4b"
BGR = lambda hexcolor: tuple(int(hexcolor[i:i + 2], 16) for i in (5, 3, 1))  # noqa: E731

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
              (12, 14), (14, 16), (0, 5), (0, 6)]

# Readable names for clips whose file names are generated IDs.
NAMES = {
    "_4RXOScjeGcGW1uLah2Cb_output": "Drill floor fall A", "YdQeOIiCjBFNpPjdTE641_output": "Drill floor fall B",
    "oO8L7gEH7O6hbAxYdXVHD_output": "Drill floor fall C", "ILLOvByCSJmu2U4QL4R7q_output": "Drill floor fall D",
    "miller_group_fall_01_slip_wet_grating": "Slip on wet grating",
    "miller_group_fall_02_trip_over_hose": "Trip over hose",
    "miller_group_fall_03_slip_oily_patch_drizzle": "Slip on oily patch, drizzle",
    "miller_group_fall_04_ladder_fall_pipe_rack": "Ladder fall at pipe rack",
    "miller_group_fall_05_stair_misstep_night": "Stair misstep at night",
    "miller_group_fall_08_fall_from_low_platform": "Fall from low platform",
    "miller_group_fall_10_slip_carrying_toolbox_night_rain": "Slip carrying toolbox, night rain",
    "slip_behind_person": "Slip behind another worker",
    "miller_night_fall_01_nv_slip_wet_grating": "Night vision: slip on wet grating",
    "miller_night_fall_02_nv_trip_over_hose": "Night vision: trip over hose",
    "miller_night_fall_05_nv_stumble_carrying_pipe": "Night vision: stumble carrying pipe",
    "miller_night_fall_06_oh_slip_oily_patch": "Floodlit night: slip on oily patch",
    "miller_night_fall_07_oh_trip_over_flange": "Floodlit night: trip over flange",
    "miller_night_fall_08_oh_fall_from_low_platform": "Floodlit night: fall from low platform",
    "miller_night_fall_09_oh_slip_stepping_off_ladder": "Floodlit night: slip stepping off ladder",
    "miller_night_fall_10_oh_slip_carrying_toolbox_alone": "Floodlit night: slip carrying toolbox",
}
RUN_NAMES = {
    "baseline": "Rig only, first version", "margin": "Rig only, cleaner labels",
    "rig-only": "Rig only, cleaner labels (batch 2 copy)", "margin-aug3": "Rig + variations",
    "margin-synth200": "Rig + synthetic figures", "margin-tcn": "TCN model (rig only)",
    "rig-only-win18": "1.5 s window (rig only)", "plus-cauca": "Rig + CAUCAFall",
    "plus-cauca-aug3": "Rig + CAUCAFall + variations", "cauca-in-folds": "Rig + CAUCAFall, mixed folds",
    # Trial 2
    "t2-base": "Trial 2 baseline", "t2-no-gym": "Without gym falls",
    "t2-with-slow": "Slow CAUCAFall falls in training", "t2-win18": "1.5 s window",
    "t2-drop015": "Fast-drop check", "t2-smooth01": "Label smoothing 0.1",
    "t2-low-x3": "Low poses count 3x", "t2-gap12": "Wider ID handoff (1 s)",
    "t2-dedupe": "Duplicate boxes removed", "t2-mirror": "Mirrored copies",
    "t2-best": "Mirror + fast drop + dedupe",
}
# Points labeled on chart 1, per trial: run name -> (dx, dy, alignment).
SCATTER_LABELS = {
    1: {"plus-cauca": (-12, -4, "right"), "margin": (10, -4, "left"), "margin-tcn": (0, 10, "center"),
        "plus-cauca-aug3": (-10, 10, "right"), "baseline": (10, -4, "left"), "rig-only-win18": (0, -18, "center")},
    2: {"t2-best": (-12, -4, "right"), "t2-mirror": (0, 10, "center"), "t2-base": (10, -4, "left"), "t2-smooth01": (10, -4, "left"),
        "t2-win18": (-10, -4, "right"), "t2-gap12": (0, 10, "center"), "t2-no-gym": (10, -4, "left")},
}


def run_name(name):
    return RUN_NAMES.get(name, name)


def clip_title(name, normal_index):
    if name in NAMES:
        return NAMES[name]
    if name in normal_index:
        return f"Normal work {normal_index[name]}"
    return name.replace("_", " ").capitalize()


# Re-running the cross-validation


def rerun_best(manifest, log, best_config, epochs=40, folds=5):
    """Seed 0 of the best run, with exactly that run's settings (from its metrics.json)."""
    from dataclasses import fields

    clips, fps = load_clips(manifest)
    known = {f.name for f in fields(RunConfig)}
    settings = {k: v for k, v in best_config.items() if k in known}
    settings.update(name=f"{best_config['name']}-seed0", seeds=(0,), epochs=epochs, folds=folds, final=False)
    if "drop_sources" in settings:
        settings["drop_sources"] = tuple(settings["drop_sources"])
    config = RunConfig(**settings)
    log(f"Re-running {best_config['name']} cross-validation (seed 0) on {len(clips)} clips...")
    result = run_cv(clips, fps, config, log=log)
    return result


# Rendering


def find_video(name):
    for folder in ("rig_falls", "normal_rig_work", "rig_falls_night"):
        for suffix in (".mp4", ".avi"):
            path = ROOT / "data" / "input" / folder / f"{name}{suffix}"
            if path.exists():
                return path
    return None


def render_clip(clip, scores, alerts, threshold, out_video, title, subtitle, stills_dir, still_times):
    """Write an annotated MP4 of one clip and save JPG stills at `still_times` (seconds)."""
    video = cv2.VideoCapture(str(find_video(clip.name)))
    source_fps = video.get(cv2.CAP_PROP_FPS) or 24.0
    width = 960
    ok, first = video.read()
    if not ok:
        raise RuntimeError(f"cannot read video for {clip.name}")
    scale = width / first.shape[1]
    video_h = int(round(first.shape[0] * scale))
    strip_h, header_h = 190, 54
    frame_h = header_h + video_h + strip_h
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    faller_curve, others_curve = scores.faller_curve(), scores.others_curve()
    times = np.array(clip.times)
    duration = max(clip.duration, times[-1] + 1 / clip.fps)
    plot_left, plot_right = 70, width - 20
    plot_top, plot_bottom = header_h + video_h + 34, frame_h - 30
    tx = lambda t: int(plot_left + t / duration * (plot_right - plot_left))  # noqa: E731
    py = lambda p: int(plot_bottom - p * (plot_bottom - plot_top))  # noqa: E731

    def strip(canvas, upto):
        cv2.rectangle(canvas, (0, header_h + video_h), (width, frame_h), (255, 255, 255), -1)
        cv2.putText(canvas, "Fall score over time", (plot_left, header_h + video_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BGR(INK), 1, cv2.LINE_AA)
        legend = [("person who falls", BLUE), ("everyone else (highest)", ORANGE)] if clip.has_fall else \
            [("highest score of anyone", ORANGE)]
        lx = plot_left + 200
        for text, colour in legend:
            cv2.line(canvas, (lx, header_h + video_h + 17), (lx + 22, header_h + video_h + 17), BGR(colour), 3)
            cv2.putText(canvas, text, (lx + 28, header_h + video_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        BGR(INK2), 1, cv2.LINE_AA)
            lx += 40 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        for start, end in clip.fall_seconds:
            cv2.rectangle(canvas, (tx(start), plot_top), (tx(end), plot_bottom), (224, 224, 224), -1)
            cv2.putText(canvas, "labeled fall", (tx(start), plot_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        BGR(INK3), 1, cv2.LINE_AA)
        for p in (0.0, 0.5, 1.0):
            cv2.line(canvas, (plot_left, py(p)), (plot_right, py(p)), BGR(GRID), 1)
            cv2.putText(canvas, f"{p:.1f}", (plot_left - 34, py(p) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        BGR(INK3), 1, cv2.LINE_AA)
        for x in range(plot_left, plot_right, 12):
            cv2.line(canvas, (x, py(threshold)), (min(x + 6, plot_right), py(threshold)), BGR(INK3), 1)
        cv2.putText(canvas, f"alert threshold {threshold:.2f}", (plot_right - 170, py(threshold) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, BGR(INK3), 1, cv2.LINE_AA)
        for t in range(0, int(duration) + 1, 2):
            cv2.putText(canvas, f"{t}s", (tx(t) - 8, plot_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        BGR(INK3), 1, cv2.LINE_AA)
        series = [(others_curve, ORANGE)] + ([(faller_curve, BLUE)] if clip.has_fall else [])
        for curve, colour in series:
            points = [(tx(t), py(p)) for t, p in zip(times, curve) if t <= upto and np.isfinite(p)]
            for a, b in zip(points, points[1:]):
                if b[0] - a[0] <= 3 * (plot_right - plot_left) / (duration * clip.fps):
                    cv2.line(canvas, a, b, BGR(colour), 2, cv2.LINE_AA)
        for alert in alerts:
            if alert.time <= upto:
                colour = GREEN if alert.is_faller and clip.has_fall else RED
                x, y = tx(alert.time), py(alert.probability)
                cv2.drawMarker(canvas, (x, y - 10), BGR(colour), cv2.MARKER_TRIANGLE_DOWN, 14, 3)
        cv2.line(canvas, (tx(upto), plot_top), (tx(upto), plot_bottom), BGR(INK), 1)

    writer = VideoWriter(out_video, width, frame_h, clip.fps)
    still_frames = {round(t, 2): None for t in still_times}
    source_index, current = 0, None
    for index, t in enumerate(times):
        wanted = int(round(t * source_fps))
        while source_index <= wanted:
            ok, image = video.read()
            if not ok:
                break
            current = image
            source_index += 1
        if current is None:
            continue
        frame = np.full((frame_h, width, 3), 255, np.uint8)
        frame[header_h:header_h + video_h] = cv2.resize(current, (width, video_h))
        probabilities = scores.probabilities[index]
        faller_ids = scores.faller[index]
        for person in clip.frames[index]:
            track = person.get("track_id")
            p = probabilities.get(track)
            box = [int(v * scale) for v in person["bbox"]]
            box[1] += header_h
            box[3] += header_h
            is_faller = clip.has_fall and track in faller_ids
            above = p is not None and p >= threshold
            colour = RED if above else (BLUE if is_faller else (230, 230, 230))
            colour = colour if isinstance(colour, tuple) else BGR(colour)
            kp = [(int(k["x"] * scale), int(k["y"] * scale) + header_h, k["confidence"]) for k in person["keypoints"]]
            # 17 COCO points, or the 12 body points (face removed) the pose loader now keeps
            offset = 17 - len(kp)
            for a, b in COCO_EDGES:
                a, b = a - offset, b - offset
                if min(a, b) < 0:
                    continue
                if kp[a][2] >= 0.3 and kp[b][2] >= 0.3:
                    cv2.line(frame, kp[a][:2], kp[b][:2], colour, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colour, 3 if above or is_faller else 1)
            label = (f"{'faller ' if is_faller else ''}ID {track}" + (f"  {p:.2f}" if p is not None else ""))
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y = max(header_h + h + 4, box[1] - 4)
            cv2.rectangle(frame, (box[0], y - h - 4), (box[0] + w + 6, y + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (box[0] + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (width, header_h), (255, 255, 255), -1)
        cv2.putText(frame, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BGR(INK), 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.47, BGR(INK2), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{t:5.2f} s", (width - 92, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR(INK), 2, cv2.LINE_AA)
        recent = [a for a in alerts if 0 <= t - a.time <= 2.0]
        if recent:
            a = recent[-1]
            good = a.is_faller and clip.has_fall
            text = f"FALL ALERT  ID {a.track_id}" + ("  (correct)" if good else "  (FALSE ALARM)")
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (12, header_h + 10), (12 + w + 16, header_h + 20 + h), BGR(GREEN if good else RED), -1)
            cv2.putText(frame, text, (20, header_h + 14 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
        strip(frame, t)
        writer.write(frame)
        for key in still_frames:
            if still_frames[key] is None and t >= key - 1e-6:
                still_frames[key] = frame.copy()
    writer.close()
    video.release()
    saved = []
    for key, image in still_frames.items():
        if image is not None:
            path = stills_dir / f"{clip.name}_{key:05.2f}s.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved.append(path)
    return saved


# Charts


def _fig(w, h):
    figure = Figure(figsize=(w, h), dpi=150, facecolor="white")
    return figure


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK3)
    ax.tick_params(colors=INK2, labelsize=9)


def chart_scatter(runs, path, best, ctx):
    figure = _fig(9, 5.4)
    ax = figure.subplots()
    _style(ax)
    for r in runs:
        is_best = r["name"] == best
        ax.scatter(r["false"], r["caught"], s=140 if is_best else 70, color=BLUE if is_best else GREY,
                   edgecolor="white", linewidth=1.5, zorder=3)
    # Labels on the points the story rests on; the rest are named in chart 2.
    offsets = SCATTER_LABELS.get(ctx["trial"], {best: (10, -4, "left")})
    falls, xmax = ctx["falls"], max(r["false"] for r in runs) * 1.15 + 0.5
    for r in runs:
        if r["name"] in offsets:
            dx, dy, ha = offsets[r["name"]]
            ax.annotate(run_name(r["name"]) + (" (best)" if r["name"] == best else ""), (r["false"], r["caught"]),
                        textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8.5,
                        color=INK, fontweight="bold" if r["name"] == best else "normal")
    if ctx.get("ceiling") is not None:
        ax.axhline(ctx["ceiling"], color=INK3, linestyle="--", linewidth=1)
        ax.text(xmax - 0.2, ctx["ceiling"] + 0.3,
                f"Ceiling: {ctx['ceiling']} of {falls} falls are visible to the person detector",
                ha="right", fontsize=8.5, color=INK2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, falls)
    ax.set_yticks(range(0, falls + 1, 3))
    ax.set_xlabel(f"False alarms in {ctx['minutes']:.1f} min of rig footage with no fall (fewer is better)",
                  color=INK2)
    ax.set_ylabel(f"Rig falls caught (of {falls})", color=INK2)
    ax.set_title(f"{run_name(best)} caught the most falls for its false alarms", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    figure.text(0.01, 0.01, f"Grouped 5-fold cross-validation on rig clips, mean of {ctx['seeds']} training runs "
                "per setup. Unlabeled points are named in chart 2.",
                fontsize=7.5, color=INK3)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(path, facecolor="white")


def chart_bars(runs, path, best, ctx):
    order = sorted(runs, key=lambda r: (r["caught"], -r["false"]))
    labels = [run_name(r["name"]) for r in order]
    figure = _fig(10, max(5.2, 0.42 * len(order) + 1.2))
    max_false = max(r["false"] for r in runs)
    left, right = figure.subplots(1, 2, sharey=True)
    for ax, key, title, limit in ((left, "caught", f"Rig falls caught (of {ctx['falls']})", ctx["falls"]),
                                  (right, "false", f"False alarms in {ctx['minutes']:.1f} min",
                                   max(12, max_false * 1.1))):
        _style(ax)
        values = [r[key] for r in order]
        colours = [BLUE if r["name"] == best else GREY for r in order]
        ax.barh(range(len(order)), values, color=colours, height=0.62)
        for i, v in enumerate(values):
            ax.text(v + limit * 0.01, i, f"{v:.3g}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, limit * 1.08)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", visible=False)
    left.set_yticks(range(len(order)), labels, fontsize=9)
    figure.suptitle("Every setup, both measures", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    figure.tight_layout()
    figure.savefig(path, facecolor="white")


def chart_threshold(sweep, path, used, ctx, best):
    xs = [p["threshold"] for p in sweep]
    figure = _fig(10, 3.8)
    left, right = figure.subplots(1, 2)
    falls = ctx["falls"]
    top = max(falls, max(p["false_alerts"] for p in sweep) + 1)
    for ax, key, title in ((left, "caught", f"Rig falls caught (of {falls})"),
                           (right, "false_alerts", f"False alarms in {ctx['minutes']:.1f} min")):
        _style(ax)
        ys = [p[key] for p in sweep]
        ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=3.5)
        ax.set_ylim(0, top)
        ax.set_yticks(range(0, int(top) + 1, 3))
        ax.axvline(used, color=INK3, linestyle="--", linewidth=1)
        ax.text(used - 0.01, top * 0.93, f"used: {used:.2f}", ha="right", fontsize=8.5, color=INK2)
        ax.annotate(f"{ys[0]:.3g}", (xs[0], ys[0]), textcoords="offset points", xytext=(4, 8), fontsize=8.5, color=INK)
        ax.annotate(f"{ys[-1]:.3g}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(-4, -14), ha="right",
                    fontsize=8.5, color=INK, fontweight="bold")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Alert threshold", color=INK2)
    figure.suptitle(f"Falls caught and false alarms at each alert threshold ({run_name(best)})", x=0.01,
                    ha="left", fontsize=12, fontweight="bold", color=INK)
    figure.tight_layout()
    figure.savefig(path, facecolor="white")


def chart_outcomes(rows, path, best):
    """One row per rig fall: alert time after impact, or why it was missed."""
    rows = sorted(rows, key=lambda r: (r["delay"] is None, r["delay"] if r["delay"] is not None else 0, r["title"]))
    figure = _fig(9, 0.32 * len(rows) + 1.4)
    ax = figure.subplots()
    _style(ax)
    ax.axvline(0, color=INK3, linewidth=1)
    ax.axvspan(-0.5, 1.5, color="#eef3fb", zorder=0)
    for i, r in enumerate(rows):
        if r["delay"] is not None:
            ax.scatter(r["delay"], i, color=BLUE, s=55, zorder=3)
            ax.text(r["delay"] + 0.06, i, f"{r['delay']:+.2f} s", va="center", fontsize=8, color=INK)
        else:
            ax.text(-0.45, i, r["reason"], va="center", fontsize=8, color=RED)
    ax.set_yticks(range(len(rows)), [r["title"] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 1.6)
    ax.set_xlabel("Alert time relative to impact, seconds (shaded = counts as caught)", color=INK2)
    ax.set_title(f"When the alert fired for each rig fall\n({run_name(best)}, one training run)", loc="left",
                 fontsize=11, fontweight="bold", color=INK)
    figure.tight_layout()
    figure.savefig(path, facecolor="white")


# Main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-date", nargs="+", default=["2026-11-05"],
                        help="Date prefix(es) of the trial's run folders in runs/")
    parser.add_argument("--trial", type=int, default=1, help="Trial number (output folder name)")
    parser.add_argument("--best", default="plus-cauca", help="Run name of the best setup")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "labels" / "manifest.json"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=40, help="Only for a quick test")
    parser.add_argument("--folds", type=int, default=5, help="Only for a quick test")
    parser.add_argument("--limit", type=int, help="Render only this many clips (quick test)")
    args = parser.parse_args(argv)
    out = Path(args.out or ROOT / "reports" / f"{args.runs_date[-1]}_trial{args.trial}")
    for sub in ("charts", "plots", "videos/caught", "videos/missed", "videos/not_visible", "videos/normal_work",
                "stills", "data"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    runs = []
    for path in sorted(p for date in args.runs_date for p in (ROOT / "runs").glob(f"{date}_*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        c = m["at_chosen_threshold"]
        runs.append({"name": m["name"], "caught": c["caught"], "false": c["false_alerts"],
                     "normal": c["normal_clip_false_alerts"], "delay": c["median_delay"], "threshold": m["chosen_threshold"],
                     "ap": m["window_average_precision"], "metrics": m, "dir": path.parent})
    best = next(r for r in runs if r["name"] == args.best)
    chart_runs = [r for r in runs if r["name"] != "rig-only"]  # identical to "margin"
    chosen = best["metrics"]["at_chosen_threshold"]
    n_seeds = len(best["metrics"]["config"]["seeds"])
    ctx = {"trial": args.trial, "falls": chosen["falls"], "minutes": chosen["non_fall_minutes"],
           "seeds": {1: "one", 2: "two", 3: "three"}.get(n_seeds, str(n_seeds)),
           "ceiling": 18 if args.trial == 1 else None}  # later trials: measured after the rerun below
    chart_scatter(chart_runs, out / "charts" / "1_setups_caught_vs_false_alarms.png", args.best, ctx)
    chart_bars(chart_runs, out / "charts" / "2_setups_both_measures.png", args.best, ctx)
    chart_threshold(best["metrics"]["threshold_sweep"], out / "charts" / "3_threshold_tradeoff_best.png",
                    best["threshold"], ctx, args.best)
    for plot in (best["dir"] / "plots").glob("*.png"):
        shutil.copyfile(plot, out / "plots" / f"best_{plot.name}")
    shutil.copyfile(best["dir"] / "report.md", out / "data" / "best_run_report.md")
    shutil.copyfile(best["dir"] / "per_clip.csv", out / "data" / "best_run_per_clip.csv")
    log(f"Charts done ({time.perf_counter() - started:.0f}s)")

    result = rerun_best(args.manifest, log, best["metrics"]["config"], args.epochs, args.folds)
    threshold = result.threshold
    clips = result.clips
    fold_of = {c.name: next(i + 1 for i, g in enumerate(result.folds) if c.group in g) for c in clips}
    if ctx["ceiling"] is None:  # falls in which the detector sees the faller at all
        ctx["ceiling"] = sum(1 for c in clips if c.has_fall and result.positive_windows[c.name] > 0)
        chart_scatter(chart_runs, out / "charts" / "1_setups_caught_vs_false_alarms.png", args.best, ctx)
    normal_names = sorted((c.name for c in clips if not c.has_fall), key=str.lower)
    normal_index = {n: i + 1 for i, n in enumerate(normal_names)}
    labels = {r["clip_id"]: r for r in csv.DictReader(open(ROOT / "data/labels/clips.csv", encoding="utf-8-sig"))}

    rows, dump = [], {"threshold": threshold, "fps": result.fps, "clips": {}}
    for clip in clips[:args.limit]:
        scores = result.scores[(0, clip.name)]
        alerts = alerts_for(scores, threshold, result.config.debounce, result.config.cooldown,
                            result.config.min_drop)
        outcome = clip_outcome(scores, alerts, result.config.catch_after)
        title = clip_title(clip.name, normal_index)
        if clip.has_fall:
            caught = outcome["caught"] == outcome["falls"]
            not_visible = result.positive_windows[clip.name] == 0
            category = "caught" if caught else ("not_visible" if not_visible else "missed")
            reason = "" if caught else ("worker not detected during the fall" if not_visible else
                                        "worker partly lost by the detector" if result.positive_windows[clip.name] < 10
                                        else "model scored the fall too low")
        else:
            category, reason = "normal_work", ""
        impact = clip.fall_seconds[0][1] if clip.has_fall else None
        still_times = sorted({round(a.time, 2) for a in alerts} | ({round(impact, 2)} if impact is not None and
                                                                   category != "caught" else set()))
        result_text = {"caught": "CAUGHT", "missed": "MISSED", "not_visible": "MISSED (not visible)",
                       "normal_work": f"{outcome['false_alerts']} false alarm(s)" if outcome["false_alerts"] else
                       "no alarms"}[category]
        subtitle = (f"Fold {fold_of[clip.name]} model (never saw this clip)  |  result: {result_text}"
                    + (f"  |  labeled fall {clip.fall_seconds[0][0]:.2f}-{clip.fall_seconds[0][1]:.2f} s"
                       if clip.has_fall else ""))
        video_path = out / "videos" / category / f"{clip.name}.mp4"
        stills = render_clip(clip, scores, alerts, threshold, video_path, title, subtitle, out / "stills", still_times)
        source = find_video(clip.name)
        rows.append({
            "title": title, "clip": clip.name, "category": category, "fold": fold_of[clip.name],
            "fall": f"{clip.fall_seconds[0][0]:.2f}-{clip.fall_seconds[0][1]:.2f}" if clip.has_fall else "",
            "delay": outcome["delay"], "false_alerts": outcome["false_alerts"], "reason": reason,
            "fall_windows": result.positive_windows[clip.name],
            "peak_faller": outcome["peak_faller"], "peak_others": outcome["peak_others"],
            "video": video_path.relative_to(out).as_posix(), "stills": [s.relative_to(out).as_posix() for s in stills],
            "source_video": source.relative_to(ROOT).as_posix(),
            "pose_video": f"data/output/{source.parent.name}/{clip.name}.mp4",
            "alerts": [{"time": round(a.time, 2), "track": a.track_id, "p": round(a.probability, 3),
                        "faller": a.is_faller} for a in alerts],
            "notes": labels.get(clip.name, {}).get("notes", ""),
        })
        dump["clips"][clip.name] = {
            "times": [round(t, 3) for t in clip.times],
            "probabilities": [{str(k): round(v, 4) for k, v in p.items()} for p in scores.probabilities],
            "faller_ids": [sorted(ids) for ids in scores.faller],
        }
        log(f"  rendered {category:11s} {clip.name}")
    (out / "data" / "scores_seed0.json").write_text(json.dumps(dump), encoding="utf-8")
    with open(out / "data" / "examples.csv", "w", newline="", encoding="utf-8") as handle:
        fields = ["title", "clip", "category", "fold", "fall", "delay", "false_alerts", "reason", "fall_windows",
                  "peak_faller", "peak_others", "video", "stills", "source_video", "pose_video"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "stills": ";".join(r["stills"])})
    chart_outcomes([r for r in rows if r["category"] != "normal_work"], out / "charts" / "4_alert_timing_per_fall.png",
                   args.best)
    (out / "data" / "examples.json").write_text(json.dumps({"threshold": threshold, "rows": rows,
                                                            "seed0": {"folds": result.folds}}, indent=1),
                                                encoding="utf-8")
    log(f"Done in {time.perf_counter() - started:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

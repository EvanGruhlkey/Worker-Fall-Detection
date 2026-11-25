"""Rendered videos, stills, charts and data for a saved model on the YouTube falls.

    uv run --no-sync scripts/build_youtube_report.py
    uv run --no-sync scripts/build_youtube_report.py --checkpoint models/fall_gru_trial3.pt \
        --out reports/2026-11-23_trial4/trial3_model_on_youtube

Scores every clip in data/labels/manifest_youtube.json with the checkpoint's
own alert rule (as scripts/score_checkpoint.py does), then:

- videos/<outcome>/   one annotated MP4 per clip, drawn by the same renderer as
  the trial reports (scripts/build_trial_report.py): skeletons and boxes, each
  person's fall score, the labeled fall, alerts, and a score timeline
- stills/             the alert frame of each caught fall, the impact frame of each miss
- charts/             PNG charts for the write-up
- data/per_clip.csv   one row per clip: outcome, why it was missed, timing, tracking

Each miss is put in one of three groups, from the faller's scores inside the
catch window (0.5 s before the fall to 1.5 s after impact):
  detector lost the person  the faller has a score in under 60% of the window
  score too brief           the faller reached the threshold, but for fewer
                            frames in a row than the alert rule needs
  score too low             the faller was scored but stayed below the threshold
"""

import argparse
import csv
import json
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_trial_report as trial_report  # noqa: E402
from build_trial_report import BLUE, GREY, INK, INK2, INK3, GRID, ORANGE  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from temporal.dataset import load_clips  # noqa: E402
from temporal.detector import drop_duplicate_people  # noqa: E402
from temporal.evaluate import alerts_for, clip_outcome, score_clip, summarize  # noqa: E402
from temporal.model import load_checkpoint  # noqa: E402

VIDEOS = ROOT / "data" / "external" / "youtube" / "videos"
SHEET = ROOT / "data" / "labels" / "external_youtube.csv"
SOURCES = ROOT / "data" / "labels" / "youtube_sources.csv"
EARLY, CATCH_AFTER, COVERAGE_MIN = 0.5, 1.5, 0.6
OUTCOMES = {  # key: (folder, label, colour)
    "caught": ("caught", "Caught", BLUE),
    "brief": ("missed_score_too_brief", "Missed: score too brief", ORANGE),
    "low": ("missed_score_too_low", "Missed: score too low", ORANGE),
    "lost": ("missed_detector_lost_person", "Missed: detector lost the person", GREY),
}
FALL_TYPES = {"slip": "Slip", "trip": "Trip", "loss_of_balance": "Loss of balance",
              "fall_from_height": "Fall from height", "faint": "Faint"}


def short_title(title):
    """'Man Slips on Icy Steps || ViralHog' -> 'Man Slips on Icy Steps'."""
    title = title.split("||")[0].strip()
    head, sep, tail = title.rpartition(" - ")
    return head if sep and tail.strip().isdigit() else title


def classify(scores, outcome, threshold, debounce):
    """(outcome key, share of the catch window the faller is scored, longest run at/above threshold)."""
    clip = scores.clip
    start, end = clip.fall_seconds[0]
    times = np.array(clip.times)
    window = (times >= start - EARLY) & (times <= end + CATCH_AFTER)
    curve = scores.faller_curve()[window]
    coverage = float(np.isfinite(curve).mean()) if window.any() else 0.0
    run = longest = 0
    for above in np.nan_to_num(curve) >= threshold:
        run = run + 1 if above else 0
        longest = max(longest, run)
    if outcome["caught"]:
        return "caught", coverage, longest
    if coverage < COVERAGE_MIN:
        return "lost", coverage, longest
    return ("brief" if longest > 0 else "low"), coverage, longest


# Charts (static PNGs for documents; same look as the trial reports)


def _fig(w=8.0, h=4.2):
    fig = Figure(figsize=(w, h), dpi=150, facecolor="white")
    return fig, fig.add_subplot(111)


def _style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(fig, title, subtitle):
    fig.text(0.02, 0.97, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    width_chars = int(fig.get_figwidth() * 13.5)
    fig.text(0.02, 0.905, textwrap.fill(subtitle, width_chars), ha="left", va="top", fontsize=9, color=INK2,
             linespacing=1.3)


def chart_outcomes(rows, path, rule):
    order = ["caught", "lost", "brief", "low"]
    counts = [sum(r["outcome"] == k for r in rows) for k in order]
    fig, ax = _fig(8, 3.4)
    y = np.arange(len(order))[::-1]
    ax.barh(y, counts, height=0.55, color=[OUTCOMES[k][2] for k in order], edgecolor="white", linewidth=2)
    for yi, n in zip(y, counts):
        ax.text(n + 0.3, yi, f"{n}", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(y, [OUTCOMES[k][1] for k in order], color=INK, fontsize=10)
    ax.set_xlim(0, max(counts) + 3)
    ax.set_xlabel("falls (of 30)", color=INK2, fontsize=9)
    _style(ax)
    total = len(rows)
    caught = counts[0]
    _title(fig, f"The Trial 3 model caught {caught} of {total} real falls found online",
           f"Most misses come from the person detector losing the faller, not from the fall model. "
           f"Alert rule: {rule}.")
    fig.subplots_adjust(left=0.32, right=0.97, top=0.74, bottom=0.17)
    fig.savefig(path)


def chart_per_clip(rows, path, threshold):
    fig, ax = _fig(8, 5)
    for key in ("lost", "low", "brief", "caught"):
        pts = [r for r in rows if r["outcome"] == key]
        if not pts:
            continue
        x = [r["faller_scored_share"] * 100 for r in pts]
        yv = [r["peak_faller_score_in_window"] if r["peak_faller_score_in_window"] is not None else 0 for r in pts]
        marker = "o" if key == "caught" else ("s" if key == "lost" else "D")
        ax.scatter(x, yv, s=60, color=OUTCOMES[key][2], marker=marker, edgecolors="white", linewidths=1.5,
                   label=OUTCOMES[key][1], zorder=3)
    ax.axhline(threshold, color=GRID, linewidth=1, zorder=1)
    ax.text(1, threshold - 0.03, f"alert threshold {threshold:g}", fontsize=8, color=INK3, va="top")
    ax.axvline(COVERAGE_MIN * 100, color=INK3, linewidth=1, zorder=1)
    ax.text(COVERAGE_MIN * 100 + 1, 0.42, "under 60%: counted as\n'detector lost the person'",
            fontsize=8, color=INK3, va="center")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("share of the catch window in which the faller has a box and a score (%)", color=INK2, fontsize=9)
    ax.set_ylabel("highest fall score for the faller\nin the catch window", color=INK2, fontsize=9)
    _style(ax, "both")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    _title(fig, "Where the faller is tracked, the model mostly reacts",
           "Each dot is one clip. Catch window: 0.5 s before the fall starts to 1.5 s after impact.")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.84, bottom=0.12)
    fig.savefig(path)


def chart_alert_rule(scored, path, threshold, debounce, min_drop):
    frames = list(range(1, 11))
    caught = [summarize([clip_outcome(s, alerts_for(s, threshold, d, min_drop=min_drop)) for s in scored])["caught"]
              for d in frames]
    fig, ax = _fig(8, 3.8)
    ax.plot(frames, caught, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor="white", zorder=3)
    i = frames.index(debounce)
    ax.annotate(f"saved rule: {debounce} frames ({debounce / 12:.2f} s)\n{caught[i]} caught",
                (debounce, caught[i]), xytext=(debounce + 0.4, caught[i] + 2.2), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.8))
    ax.annotate(f"1 frame: {caught[0]} caught", (1, caught[0]), xytext=(1.4, caught[0] + 1.6), fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.8))
    ax.set_xticks(frames)
    ax.set_xlabel(f"frames in a row the score must stay at or above {threshold:g} before an alert (12 per second)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("falls caught (of 30)", color=INK2, fontsize=9)
    ax.set_ylim(0, 30)
    _style(ax, "y")
    _title(fig, "A shorter hold would catch more of these falls",
           "Same scores, alert rule changed after the fact. On the rig clips the 7-frame hold is what "
           "cut false alarms, so this is a trade-off.")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.15)
    fig.savefig(path)


def chart_by_type(rows, path):
    types = [t for t in FALL_TYPES if any(r["fall_type"] == t for r in rows)]
    types.sort(key=lambda t: -sum(r["fall_type"] == t for r in rows))
    fig, ax = _fig(8, 3.4)
    y = np.arange(len(types))[::-1]
    for yi, t in zip(y, types):
        n = sum(r["fall_type"] == t for r in rows)
        c = sum(r["fall_type"] == t and r["outcome"] == "caught" for r in rows)
        ax.barh(yi, n, height=0.55, color=GRID)
        ax.barh(yi, c, height=0.55, color=BLUE)
        ax.text(n + 0.3, yi, f"{c} of {n}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [FALL_TYPES[t] for t in types], color=INK, fontsize=10)
    ax.set_xlabel("falls (blue: caught, grey: missed)", color=INK2, fontsize=9)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(0, max(sum(r["fall_type"] == t for r in rows) for t in types) + 3)
    _style(ax)
    _title(fig, "Falls caught by kind of fall", "Few falls per kind outside slips, so treat these as examples, "
                                                 "not rates.")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.17)
    fig.savefig(path)


def chart_timing(rows, path):
    caught = sorted((r for r in rows if r["outcome"] == "caught"), key=lambda r: r["alert_delay_s"])
    fig, ax = _fig(8, 0.9 + 0.32 * len(caught))
    y = np.arange(len(caught))[::-1]
    ax.axvspan(-EARLY, CATCH_AFTER, color="#f1f3f5", zorder=0)
    ax.axvline(0, color=INK3, linewidth=1)
    ax.scatter([r["alert_delay_s"] for r in caught], y, s=55, color=BLUE, edgecolors="white", linewidths=1.5,
               zorder=3)
    ax.set_yticks(y, [textwrap.shorten(r["title"], 40, placeholder="...") for r in caught], fontsize=8.5,
                  color=INK)
    ax.set_xlim(-EARLY - 0.1, CATCH_AFTER + 0.1)
    ax.set_xlabel("alert time relative to impact (s)", color=INK2, fontsize=9)
    _style(ax)
    median = float(np.median([r["alert_delay_s"] for r in caught]))
    _title(fig, f"Caught falls alerted a median {median:.2f} s after impact", "One dot per caught fall. Shaded: the window in which an alert counts as caught.")
    fig.subplots_adjust(left=0.38, right=0.97, top=1 - 0.75 / (0.9 + 0.32 * len(caught)) - 0.04,
                        bottom=0.5 / (0.9 + 0.32 * len(caught)) + 0.04)
    fig.savefig(path)


def chart_rig_vs_youtube(rig, youtube, path):
    labels = ["Rig clips (AI-generated)\nTrial 3 cross-validation", "Real falls from YouTube\nTrial 3 final model"]
    rates = [rig[0] / rig[1] * 100, youtube[0] / youtube[1] * 100]
    fig, ax = _fig(8, 3.0)
    y = np.array([1, 0])
    ax.barh(y, rates, height=0.5, color=[GREY, BLUE], edgecolor="white", linewidth=2)
    for yi, rate, (c, n) in zip(y, rates, (rig, youtube)):
        ax.text(rate + 1, yi, f"{rate:.0f}%  ({c:g} of {n})", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, 110)
    ax.set_xlabel("falls caught (%), same alert rule (0.995 held for 7 frames)", color=INK2, fontsize=9)
    _style(ax)
    _title(fig, "Real footage is much harder than the rig clips",
           "Rig number: Trial 3 baseline scored with the final alert rule (gate sweep, run t3-thr).")
    fig.subplots_adjust(left=0.3, right=0.97, top=0.76, bottom=0.2)
    fig.savefig(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "fall_gru_trial3.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "labels" / "manifest_youtube.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "2026-11-23_trial4" / "trial3_model_on_youtube")
    parser.add_argument("--rig-caught", type=float, default=31.3, help="Rig falls caught with the same rule")
    parser.add_argument("--rig-falls", type=int, default=42)
    parser.add_argument("--charts-only", action="store_true", help="Skip rendering the videos")
    args = parser.parse_args(argv)

    model, meta = load_checkpoint(args.checkpoint)
    threshold, debounce = float(meta["threshold"]), int(meta.get("debounce", 5))
    min_drop, max_gap, window = float(meta.get("min_drop", 0.0)), int(meta.get("max_gap", 5)), int(meta["window_size"])
    rule = f"score {threshold:g} or higher for {debounce} frames ({debounce / 12:.2f} s), fast-drop check {min_drop:g}"
    clips, _ = load_clips(args.manifest)
    if meta.get("merge_duplicates"):
        clips = [replace(c, frames=[drop_duplicate_people(f) for f in c.frames]) for c in clips]
    sheet = {r["clip_id"]: r for r in csv.DictReader(open(SHEET, encoding="utf-8"))}
    sources = {r["clip_id"]: r for r in csv.DictReader(open(SOURCES, encoding="utf-8"))}
    trial_report.find_video = lambda name: VIDEOS / sheet[name]["file"]

    for sub in ("charts", "stills", "data", *(f"videos/{v[0]}" for v in OUTCOMES.values())):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(clips)} clips with {args.checkpoint.name} ({rule})")
    scored = [score_clip(model, clip, window, max_gap) for clip in clips]
    rows = []
    for number, scores in enumerate(scored, 1):
        clip = scores.clip
        alerts = alerts_for(scores, threshold, debounce, min_drop=min_drop)
        outcome = clip_outcome(scores, alerts)
        key, coverage, longest = classify(scores, outcome, threshold, debounce)
        start, end = clip.fall_seconds[0]
        times = np.array(clip.times)
        in_window = (times >= start - EARLY) & (times <= end + CATCH_AFTER)
        curve = scores.faller_curve()[in_window]
        peak = float(np.nanmax(curve)) if np.isfinite(curve).any() else None
        label = sheet[clip.name]
        title = short_title(sources[clip.name]["title"])
        row = {
            "clip": clip.name, "title": title, "url": sources[clip.name]["url"],
            "fall_type": label["fall_type"], "fall_start_s": start, "fall_end_s": end,
            "outcome": key, "outcome_label": OUTCOMES[key][1],
            "alert_delay_s": outcome["delay"],
            "faller_scored_share": round(coverage, 3), "longest_run_at_threshold": longest,
            "peak_faller_score_in_window": None if peak is None else round(peak, 4),
            "false_alerts": outcome["false_alerts"], "late_alerts": outcome["late"],
            "faller_track_ids": label["faller_track_id"], "resolution": label["resolution"],
        }
        if not args.charts_only:
            folder = args.out / "videos" / OUTCOMES[key][0]
            video = folder / f"{clip.name}.mp4"
            if key == "caught":
                status = f"CAUGHT: alert {outcome['delay']:+.2f} s after impact"
                still = [next(a.time for a in alerts if a.is_faller)]
            else:
                status = f"MISSED ({OUTCOMES[key][1][8:]})"
                still = [end]
            subtitle = (f"{status}  |  {FALL_TYPES.get(label['fall_type'], label['fall_type'])}  |  "
                        f"Trial 3 model, never trained on this clip  |  YouTube: {sources[clip.name]['uploader']}")
            stills = trial_report.render_clip(clip, scores, alerts, threshold, video, title[:70], subtitle,
                                              args.out / "stills", still)
            row["video"] = str(video.relative_to(args.out)).replace("\\", "/")
            row["still"] = str(stills[0].relative_to(args.out)).replace("\\", "/") if stills else ""
            print(f"  [{number}/{len(scored)}] {clip.name}: {OUTCOMES[key][1]}", flush=True)
        rows.append(row)

    rows.sort(key=lambda r: (list(OUTCOMES).index(r["outcome"]), r["clip"]))
    with open(args.out / "data" / "per_clip.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize([clip_outcome(s, alerts_for(s, threshold, debounce, min_drop=min_drop)) for s in scored])
    by_outcome = {k: sum(r["outcome"] == k for r in rows) for k in OUTCOMES}
    (args.out / "data" / "summary.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint.name), "alert_rule": rule, "summary": summary,
         "by_outcome": by_outcome}, indent=2) + "\n", encoding="utf-8")

    chart_outcomes(rows, args.out / "charts" / "1_outcomes.png", rule)
    chart_per_clip(rows, args.out / "charts" / "2_tracking_vs_score.png", threshold)
    chart_alert_rule(scored, args.out / "charts" / "3_alert_rule_hold.png", threshold, debounce, min_drop)
    chart_by_type(rows, args.out / "charts" / "4_by_fall_type.png")
    chart_timing(rows, args.out / "charts" / "5_alert_timing.png")
    chart_rig_vs_youtube((args.rig_caught, args.rig_falls), (summary["caught"], summary["falls"]),
                         args.out / "charts" / "6_rig_vs_youtube.png")
    print(f"Caught {summary['caught']}/{summary['falls']}; {by_outcome}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

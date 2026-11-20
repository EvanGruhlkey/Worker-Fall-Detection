"""Write a cross-validation run to ``runs/<date>_<name>/``.

    report.md      one-page summary: data, settings, headline, threshold table,
                   folds, per-clip results, caveats (renders on GitHub)
    metrics.json   the same numbers, for comparing runs
    per_clip.csv   one row per clip at the chosen threshold
    plots/         training curves, precision-recall, threshold trade-off, and
                   (with plots="all") probability timelines of every clip
    model.pt       the final model, when one was trained

Plots follow the house style of ``temporal.visualize``: 2 px lines, solid
hairline grids, ink-coloured text, a legend whenever a panel has two or more
series, and labels (not colour alone) for the fall band and threshold.
"""

import csv
import datetime as dt
import json
import subprocess

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .crossval import (  # noqa: E402
    average_precision, config_dict, drop_sweep, fold_summaries, pooled_windows, subset_summary,
)
from .evaluate import summarize  # noqa: E402
from .model import save_checkpoint  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
FALL_BAND = "#e1e0d9"
SERIES_1 = "#2a78d6"  # blue: the faller / main series
SERIES_2 = "#eb6834"  # orange: everyone else / second series


def write_run(run_dir, result, *, plots="all", data_note="", extra_notes=()):
    run_dir.mkdir(parents=True, exist_ok=True)
    per_clip = per_clip_rows(result)
    metrics = build_metrics(result, data_note)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with open(run_dir / "per_clip.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(per_clip[0].keys()))
        writer.writeheader()
        writer.writerows(per_clip)

    images = []
    if plots in ("all", "summary"):
        (run_dir / "plots").mkdir(exist_ok=True)
        images += [plot_curves(result, run_dir / "plots" / "curves.png"),
                   plot_precision_recall(result, run_dir / "plots" / "precision_recall.png"),
                   plot_thresholds(result, run_dir / "plots" / "thresholds.png")]
    if plots == "all":
        images += [plot_timelines(result, run_dir / "plots" / "timelines_falls.png", falls=True),
                   plot_timelines(result, run_dir / "plots" / "timelines_normal.png", falls=False)]
    if result.final_model is not None:
        save_checkpoint(
            run_dir / "model.pt", result.final_model, fps=result.fps,
            window_size=result.config.window_size, threshold=result.threshold,
            min_drop=result.config.min_drop, max_gap=result.config.max_gap, debounce=result.config.debounce,
            merge_duplicates=result.config.merge_duplicates, metrics=metrics, notes=f"{result.config.name}: {data_note}",
        )
    (run_dir / "report.md").write_text(
        render_markdown(result, metrics, per_clip, [p for p in images if p], extra_notes), encoding="utf-8"
    )
    return metrics


# Numbers


def build_metrics(result, data_note):
    labels, probs = pooled_windows(result)
    chosen = next(p for p in result.sweep if p["threshold"] == result.threshold)
    return {
        "name": result.config.name,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data": data_note,
        "fps": result.fps,
        "clips": len(result.clips),
        "fall_clips": sum(c.has_fall for c in result.clips),
        "sources": _source_counts(result.clips),
        "extra_training_clips": _source_counts(result.extra_train),
        "groups": len({c.group for c in result.clips}),
        "config": config_dict(result.config),
        "folds": [sorted(f) for f in result.folds],
        "chosen_threshold": result.threshold,
        "at_chosen_threshold": chosen,
        "window_average_precision": average_precision(labels, probs),
        "nested": result.nested,
        "by_source": {source: subset_summary(result, {c.name for c in result.clips if c.source == source})
                      for source in sorted({c.source for c in result.clips})},
        "by_batch": {batch: subset_summary(result, {c.name for c in result.headline_clips if c.batch == batch})
                     for batch in sorted({c.batch for c in result.headline_clips if c.batch})},
        "by_action": by_action(result),
        "threshold_sweep": result.sweep,
        "drop_sweep": drop_sweep(result),
        "fold_results": fold_summaries(result),
        "final_model_epochs": result.final_epochs,
    }


def by_action(result):
    """False alerts on the rig clips, grouped by the action column; most false alerts first.

    Clips without an action are grouped as Trial 1-2 normal work or fall clips,
    so every rig false alert lands in exactly one row. Empty when no clip has an
    action label (runs before Trial 3).
    """
    clips = result.headline_clips
    if not any(c.action for c in clips):
        return {}

    def key(c):
        if c.action:
            return c.action
        return "fall clips (outside the fall)" if c.has_fall else "normal work, Trials 1-2"

    rows = {k: subset_summary(result, {c.name for c in clips if key(c) == k}) for k in {key(c) for c in clips}}
    return dict(sorted(rows.items(), key=lambda item: (-item[1]["false_alerts"], item[0])))


def per_clip_rows(result):
    seeds = result.config.seeds
    rows = []
    for index, clip in enumerate(result.clips):
        outs = [result.outcomes[s][index] for s in seeds]
        fold = next(i + 1 for i, groups in enumerate(result.folds) if clip.group in groups)
        delays = [o["delay"] for o in outs if o["delay"] is not None]
        rows.append({
            "clip": clip.name,
            "group": clip.group,
            "fold": fold,
            "has_fall": int(clip.has_fall),
            "fall_seconds": ";".join(f"{a:.2f}-{b:.2f}" for a, b in clip.fall_seconds),
            "positive_windows": result.positive_windows[clip.name],
            "caught": f"{sum(o['caught'] for o in outs)}/{sum(o['falls'] for o in outs)}" if clip.has_fall else "",
            "delay_s": f"{np.mean(delays):+.2f}" if delays else "",
            "late_alerts": sum(o["late"] for o in outs),
            "false_alerts": sum(o["false_alerts"] for o in outs),
            "peak_faller_p": _fmt(np.mean([o["peak_faller"] for o in outs if o["peak_faller"] is not None])
                                  if any(o["peak_faller"] is not None for o in outs) else None),
            "peak_others_p": _fmt(np.mean([o["peak_others"] for o in outs if o["peak_others"] is not None])
                                  if any(o["peak_others"] is not None for o in outs) else None),
        })
    return rows


# Markdown


def render_markdown(result, metrics, per_clip, images, extra_notes):
    c = result.config
    chosen = metrics["at_chosen_threshold"]
    seeds = len(c.seeds)
    seeds_note = f" (mean of {seeds} seeds)" if seeds > 1 else ""
    falls = chosen["falls"]
    lines = [
        f"# Temporal model run: {c.name}",
        "",
        f"{metrics['created']} · commit `{metrics['git_commit']}` · {metrics['data']}",
        "",
        "## Headline",
        "",
        f"At the chosen gate threshold **{result.threshold:g}**{seeds_note}, held out by "
        f"{c.folds}-fold grouped cross-validation (every clip tested once, by a model that never saw its scene):",
        "",
        f"- **Falls caught: {chosen['caught']:g} of {falls}** "
        f"({_pct(chosen['recall'])}), alert within {c.catch_after:g} s of impact",
        f"- **False alerts: {chosen['false_alerts']:g}** in {chosen['non_fall_minutes']:.1f} min of non-fall footage"
        f" ({chosen['normal_clip_false_alerts']:g} on the normal-work clips)",
        f"- Median alert time: {_delay(chosen['median_delay'])} relative to impact",
        f"- Late alerts (faller, after the catch window): {chosen['late']:g}",
        f"- Window-level: average precision {_fmt(metrics['window_average_precision'])}, "
        f"precision {chosen['window']['precision']:.2f} / recall {chosen['window']['recall']:.2f} at this threshold",
    ]
    nested = metrics.get("nested")
    if nested:
        thresholds = sorted({t["threshold"] for t in nested["thresholds"]})
        lines += [
            f"- **Honest estimate (threshold chosen without the tested fold): {nested['caught']:g} of "
            f"{nested['falls']} caught, {nested['false_alerts']:g} false alerts.** Each fold's threshold "
            f"({', '.join(f'{t:g}' for t in thresholds)}) was picked from the other folds only, so these "
            "are the numbers to compare between runs.",
        ]
    if len(c.seeds) > 1:
        spread = subset_summary_spread(result)
        lines += [f"- Seed to seed: falls caught {spread['caught']}, false alerts {spread['false']} "
                  "(the run-to-run noise; differences between runs smaller than this are not real)"]
    lines += [
        "",
        f"> One false alert in {chosen['non_fall_minutes']:.1f} min is "
        f"{60 / chosen['non_fall_minutes']:.0f}/hour, so this data cannot show the 1/hour target directly; "
        "the threshold is chosen for the fewest false alerts seen, then the most falls caught.",
        "",
        "## Data and settings",
        "",
        f"- {metrics['clips']} clips ({metrics['fall_clips']} with a fall), {metrics['groups']} scene groups, "
        f"{result.fps:g} fps, windows of {c.window_size} frames ({c.window_size / result.fps:.1f} s), stride {c.stride}",
        f"- Model: {c.arch.upper()}, hidden {c.hidden}, dropout {c.dropout}, up to {c.epochs} epochs "
        f"(best by validation), learning rate {c.learning_rate}",
        f"- Ambiguous windows (0-50% of a fall) left out: {'yes' if c.ignore_partial else 'no'}",
        "- Augmentation: " + ("one mirrored copy per training clip" if c.augment_mode == "mirror"
                              else f"{c.augment} augmented copies per training clip")
        + f"; synthetic clips added: {c.synthetic}",
        f"- Label smoothing: {c.label_smoothing:g}; low-posture normal windows weighted "
        f"x{c.hard_negative_weight:g}",
        f"- Tracking: a lost track continues for up to {c.max_gap} frames ({c.max_gap / result.fps:.2f} s); "
        f"duplicate boxes on one body {'removed' if c.merge_duplicates else 'kept'}",
        f"- Gate: debounce {c.debounce} frames ({c.debounce / result.fps:.2f} s), cooldown {c.cooldown:g} s",
        f"- Fast-drop check: " + (f"alerts need a body drop of at least {c.min_drop:g} body lengths within 0.75 s"
                                  if c.min_drop > 0 else "off"),
        f"- Seeds: {', '.join(str(s) for s in c.seeds)}",
        f"- Tested clips by source: {_source_text(metrics['sources'])}",
        f"- External clips added to every training set (never tested): "
        f"{_source_text(metrics['extra_training_clips']) or 'none'}",
    ]
    if result.final_model is not None:
        lines += [f"- Final model: trained on all clips for {result.final_epochs} epochs, saved as `model.pt` "
                  f"with threshold {result.threshold:g}"]

    lines += ["", "## Threshold trade-off", "",
              "| Threshold | Falls caught | False alerts | Per hour | Normal-clip false | Median alert time | Window precision | Window recall |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in result.sweep:
        mark = " **(chosen)**" if p["threshold"] == result.threshold else ""
        lines.append(
            f"| {p['threshold']:g}{mark} | {p['caught']:g}/{p['falls']} | {p['false_alerts']:g} | "
            f"{_fmt(p['false_alerts_per_hour'], 0)} | {p['normal_clip_false_alerts']:g} | {_delay(p['median_delay'])} | "
            f"{p['window']['precision']:.2f} | {p['window']['recall']:.2f} |"
        )

    lines += ["", "## Fast-drop check (at the chosen threshold)", "",
              "What requiring a fast body drop before an alert would change. The run's own setting is marked.", "",
              "| Minimum drop (body lengths in 0.75 s) | Falls caught | False alerts |", "|---:|---:|---:|"]
    for row in metrics["drop_sweep"]:
        mark = " **(this run)**" if abs(row["min_drop"] - c.min_drop) < 1e-9 else ""
        label = "off" if row["min_drop"] == 0 else f"{row['min_drop']:g}"
        lines.append(f"| {label}{mark} | {row['caught']:g}/{row['falls']} | {row['false_alerts']:g} |")

    if len(metrics["by_source"]) > 1:
        lines += ["", "## By source (at the chosen threshold)", "",
                  "The headline and threshold use the rig clips only.", "",
                  "| Source | Falls caught | False alerts | Non-fall minutes |", "|---|---:|---:|---:|"]
        for source, s in metrics["by_source"].items():
            lines.append(f"| {source} | {s['caught']:g}/{s['falls']} | {s['false_alerts']:g} | "
                         f"{s['non_fall_minutes']:.1f} |")
    if metrics["by_batch"]:
        lines += ["", "## By batch (rig clips, at the chosen threshold)", "",
                  "`trial1` is exactly the clip set Trial 1 was scored on, so it compares directly with Trial 1's "
                  "numbers. Later batches are the clips added since.", "",
                  "| Batch | Clips | Falls caught | False alerts | Non-fall minutes |", "|---|---:|---:|---:|---:|"]
        for batch, s in metrics["by_batch"].items():
            lines.append(f"| {batch} | {s['clips']} | {s['caught']:g}/{s['falls']} | {s['false_alerts']:g} | "
                         f"{s['non_fall_minutes']:.1f} |")
    if metrics.get("by_action"):
        lines += ["", "## False alerts by action (rig clips, at the chosen threshold)", "",
                  "Normal-work clips grouped by what they show (the action column of clips.csv). Clips without an "
                  "action label are the Trial 1-2 normal-work clips; alerts on fall clips are those outside the "
                  "fall (on coworkers, or before or after it). Per hour = false alerts / non-fall time.", "",
                  "| Action | Clips | False alerts | Non-fall minutes | Per hour |", "|---|---:|---:|---:|---:|"]
        for action, s in metrics["by_action"].items():
            per_hour = s["false_alerts"] / (s["non_fall_minutes"] / 60) if s["non_fall_minutes"] else None
            lines.append(f"| {action} | {s['clips']} | {s['false_alerts']:g} | {s['non_fall_minutes']:.1f} | "
                         f"{_fmt(per_hour, 0)} |")
    lines += ["", "## Folds", "",
              "| Seed | Fold | Test clips | Train windows (falls) | Test windows (falls) | Best epoch | Avg precision | F1 @0.5 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in metrics["fold_results"]:
        lines.append(
            f"| {f['seed']} | {f['fold']} | {f['test_clips']} | {f['train_windows']} ({f['train_positives']}) | "
            f"{f['test_windows']} ({f['test_positives']}) | {f['best_epoch']} | {_fmt(f['average_precision'])} | "
            f"{f['f1_at_0.5']:.2f} |"
        )

    lines += ["", "## Per clip (at the chosen threshold)", "",
              "| Clip | Fold | Fall (s) | Fall windows | Caught | Alert time | False alerts | Peak p faller | Peak p others |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in per_clip:
        lines.append(
            f"| {row['clip']} | {row['fold']} | {row['fall_seconds'] or '-'} | {row['positive_windows'] if row['has_fall'] else '-'} | "
            f"{row['caught'] or '-'} | {row['delay_s'] or '-'} | {row['false_alerts']} | "
            f"{row['peak_faller_p'] or '-'} | {row['peak_others_p'] or '-'} |"
        )

    notes = list(extra_notes)
    empty = [r["clip"] for r in per_clip if r["has_fall"] and r["positive_windows"] == 0]
    if empty:
        notes.append(f"{len(empty)} fall clip(s) give no fall windows because the pose detector loses the faller "
                     f"mid-fall: {', '.join(empty)}. They can only be caught if the model fires on the onset.")
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    if images:
        lines += ["", "## Plots", ""]
        for path in images:
            lines += [f"![{path.stem}](plots/{path.name})", ""]
    return "\n".join(lines) + "\n"


# Plots


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def _figure(width, height):
    figure = Figure(figsize=(width, height), dpi=110, facecolor=SURFACE)
    return figure


def plot_curves(result, path):
    """Train and validation loss per fold (first seed), small multiples."""
    seed = result.config.seeds[0]
    folds = [f for f in result.fold_results if f.seed == seed]
    figure = _figure(3.0 * len(folds), 2.8)
    axes = figure.subplots(1, len(folds), sharey=True, squeeze=False)[0]
    for ax, fold in zip(axes, folds):
        epochs = [h["epoch"] for h in fold.history]
        ax.plot(epochs, [h["train_loss"] for h in fold.history], color=SERIES_1, linewidth=2, label="train loss")
        ax.plot(epochs, [h.get("val_loss") for h in fold.history], color=SERIES_2, linewidth=2, label="validation loss")
        ax.axvline(fold.best_epoch, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.text(fold.best_epoch, ax.get_ylim()[1], " best", color=INK_SECONDARY, fontsize=7, va="top")
        ax.set_title(f"Fold {fold.fold + 1}", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        _axes_style(ax)
    axes[0].set_ylabel("loss (weighted BCE)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK_SECONDARY)
    figure.suptitle("Training curves", color=INK, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    return path


def plot_precision_recall(result, path):
    labels, probs = pooled_windows(result)
    order = np.argsort(-probs)
    hits = labels[order] >= 0.5
    if not hits.any():
        return None
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    recall = np.cumsum(hits) / hits.sum()
    figure = _figure(4.6, 3.6)
    ax = figure.subplots()
    ax.plot(recall, precision, color=SERIES_1, linewidth=2)
    point = next(p for p in result.sweep if p["threshold"] == result.threshold)["window"]
    ax.plot([point["recall"]], [point["precision"]], marker="o", markersize=8, color=SERIES_1,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"threshold {result.threshold:g}", (point["recall"], point["precision"]),
                textcoords="offset points", xytext=(-8, -16), ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall (fall windows found)", fontsize=8)
    ax.set_ylabel("precision (alarmed windows that are falls)", fontsize=8)
    ax.set_title(f"Window precision-recall, held out (AP {_fmt(average_precision(labels, probs))})", fontsize=9)
    _axes_style(ax)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    return path


def plot_thresholds(result, path):
    """Falls caught and false alerts against the threshold: two panels, one axis each."""
    xs = [p["threshold"] for p in result.sweep]
    figure = _figure(8.0, 3.0)
    left, right = figure.subplots(1, 2, sharex=True)
    falls = result.sweep[0]["falls"]
    left.plot(xs, [p["caught"] for p in result.sweep], color=SERIES_1, linewidth=2, marker="o", markersize=4)
    left.set_ylim(0, falls + 0.5)
    left.set_title(f"Falls caught (of {falls})", fontsize=9)
    right.plot(xs, [p["false_alerts"] for p in result.sweep], color=SERIES_1, linewidth=2, marker="o", markersize=4)
    right.set_ylim(bottom=0)
    right.set_title(f"False alerts (in {result.sweep[0]['non_fall_minutes']:.1f} min of non-fall footage)", fontsize=9)
    for ax in (left, right):
        ax.axvline(result.threshold, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.text(result.threshold, ax.get_ylim()[1], " chosen", color=INK_SECONDARY, fontsize=7, va="top")
        ax.set_xlabel("gate threshold", fontsize=8)
        _axes_style(ax)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    return path


def plot_timelines(result, path, falls):
    """Held-out probability over time for every clip (first seed), as a grid."""
    seed = result.config.seeds[0]
    clips = [c for c in result.clips if c.has_fall == falls]
    if not clips:
        return None
    columns = 3 if falls else 2
    rows = int(np.ceil(len(clips) / columns))
    figure = _figure(4.2 * columns, 1.9 * rows + 0.6)
    axes = figure.subplots(rows, columns, sharey=True, squeeze=False).ravel()
    outcome_by_name = {o["clip"]: o for o in result.outcomes[seed]}
    for ax, clip in zip(axes, clips):
        scores = result.scores[(seed, clip.name)]
        times = np.array(clip.times)
        for start, end in clip.fall_seconds:
            ax.axvspan(start, end, color=FALL_BAND, alpha=0.9, linewidth=0)
            ax.text((start + end) / 2, 1.04, "fall", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
        if falls:
            ax.plot(times, scores.faller_curve(), color=SERIES_1, linewidth=2, label="faller")
        ax.plot(times, scores.others_curve(), color=SERIES_2 if falls else SERIES_1, linewidth=2,
                label="others (max)" if falls else "anyone (max)")
        ax.axhline(result.threshold, color=INK_MUTED, linewidth=1, linestyle="--")
        for alert in alerts_at(result, scores):
            ax.plot([alert.time], [alert.probability], marker="v", markersize=8,
                    color=SERIES_1 if alert.is_faller else SERIES_2, markeredgecolor=SURFACE, markeredgewidth=1.5)
        o = outcome_by_name[clip.name]
        status = (f"caught {o['caught']}/{o['falls']}" if falls else "") + \
                 (f", {o['false_alerts']} false" if o["false_alerts"] else ("" if falls else "no alerts"))
        ax.set_title(f"{_short(clip.name)}  ({status.strip(', ')})", fontsize=8, loc="left")
        ax.set_ylim(0, 1.12)
        ax.set_xlim(0, max(clip.duration, 1.0))
        _axes_style(ax)
    for ax in axes[len(clips):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
    title = "Fall clips" if falls else "Normal-work clips"
    figure.suptitle(f"{title}: held-out fall probability over time (s); dashed = threshold "
                    f"{result.threshold:g}, triangles = alerts", color=INK, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    return path


def alerts_at(result, scores):
    from .evaluate import alerts_for

    c = result.config
    return alerts_for(scores, result.threshold, c.debounce, c.cooldown, c.min_drop)


# Helpers


def subset_summary_spread(result):
    """Lowest to highest falls caught and false alerts across seeds, at the chosen threshold."""
    per_seed = [summarize(result.outcomes[s]) for s in result.config.seeds]
    caught = [p["caught"] for p in per_seed]
    false = [p["false_alerts"] for p in per_seed]
    return {"caught": f"{min(caught)} to {max(caught)}", "false": f"{min(false)} to {max(false)}"}


def _source_counts(clips):
    counts = {}
    for clip in clips:
        entry = counts.setdefault(clip.source, {"clips": 0, "falls": 0})
        entry["clips"] += 1
        entry["falls"] += int(clip.has_fall)
    return counts


def _source_text(counts):
    return ", ".join(f"{s} {v['clips']} ({v['falls']} falls)" for s, v in sorted(counts.items()))


def _short(name, limit=34):
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _fmt(value, digits=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def _pct(value):
    return "" if value is None else f"{100 * value:.0f}%"


def _delay(value):
    return "-" if value is None else f"{value:+.2f} s"


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+changes" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"

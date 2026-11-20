"""Try alert rules on a run's saved scores, without retraining.

    uv run scripts/gate_sweep.py runs/2026-11-19_t3-thr        # needs a run made with --save-scores

The model's per-frame fall scores stay fixed; only the rule that turns scores
into alerts changes. Three dials, every combination:

- threshold: the score a person must stay above (0.97 ... 0.999)
- time above it: how long the score must stay above the threshold before an
  alert (the "debounce", in frames at 12 fps: 5 = 0.4 s today, 7 = 0.6 s, 10 = 0.8 s)
- stays-down check: hold each alert this many seconds and drop it if the person
  is back on their feet (0 = off; see temporal.evaluate.stays_down)

An alert held by the stays-down check comes that much later, so the window in
which it still counts as catching the fall is widened by the same time (an
alert 1 s after impact plus a 1 s check still counts). Results are the mean of
the run's seeds on the rig clips, like every run report, plus the honest
estimate (each fold's threshold chosen on the other folds only) and the
Trial 1 clip set. Writes gate_sweep.md and gate_sweep.json into the run folder.
"""

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal.crossval import choose_threshold  # noqa: E402
from temporal.dataset import load_clips  # noqa: E402
from temporal.evaluate import ClipScores, alerts_for, clip_outcome, stays_down, summarize  # noqa: E402

THRESHOLDS = (0.97, 0.98, 0.99, 0.995, 0.999)
DEBOUNCES = (5, 7, 10)
STAYS_DOWN = (0.0, 1.0, 1.5, 2.0)


def _down(row):
    return "off" if not row["stays_down_seconds"] else f"{row['stays_down_seconds']:g} s"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="Run folder holding scores.pkl.gz")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "labels" / "manifest.json"))
    args = parser.parse_args(argv)
    run = Path(args.run)
    with gzip.open(run / "scores.pkl.gz", "rb") as handle:
        saved = pickle.load(handle)
    cfg = saved["config"]
    clips_all, _ = load_clips(args.manifest)
    names = {name for _, name in saved["scores"]}
    clips = [c for c in clips_all if c.name in names]
    fold_of = {c.name: next(i for i, groups in enumerate(saved["folds"]) if c.group in groups) for c in clips}
    seeds = saved["seeds"]
    scores = {key: ClipScores(next(c for c in clips if c.name == key[1]), **value)
              for key, value in saved["scores"].items()}
    print(f"{run.name}: {len(clips)} rig clips, seeds {seeds}, min_drop {cfg['min_drop']}, "
          f"cooldown {cfg['cooldown']} s")

    cache = {}

    def outcomes(seed, threshold, debounce, down):
        key = (seed, threshold, debounce, down)
        if key not in cache:
            outs = {}
            for clip in clips:
                s = scores[(seed, clip.name)]
                alerts = alerts_for(s, threshold, debounce, cfg["cooldown"], cfg["min_drop"])
                alerts = stays_down(s, alerts, down)
                outs[clip.name] = clip_outcome(s, alerts, cfg["catch_after"] + down)
            cache[key] = outs
        return cache[key]

    def mean_summary(threshold, debounce, down, subset=None):
        per_seed = [summarize([o for n, o in outcomes(seed, threshold, debounce, down).items()
                               if subset is None or n in subset]) for seed in seeds]
        out = {k: float(np.mean([p[k] for p in per_seed])) for k in ("caught", "false_alerts", "non_fall_minutes")}
        out["falls"] = per_seed[0]["falls"]
        delays = [p["median_delay"] for p in per_seed if p["median_delay"] is not None]
        out["median_delay"] = float(np.mean(delays)) if delays else None
        out["spread_false_alerts"] = [p["false_alerts"] for p in per_seed]
        return out

    def honest(debounce, down):
        """Each fold's threshold chosen (fewest false alerts, then most caught) on the other folds only."""
        per_seed = []
        for seed in seeds:
            outs = []
            for fold in range(len(saved["folds"])):
                others = [c.name for c in clips if fold_of[c.name] != fold]
                tested = [c.name for c in clips if fold_of[c.name] == fold]
                if not tested:
                    continue
                sweep = []
                for threshold in THRESHOLDS:
                    s = summarize([outcomes(seed, threshold, debounce, down)[n] for n in others])
                    sweep.append({"threshold": threshold, "caught": s["caught"], "false_alerts": s["false_alerts"]})
                chosen = choose_threshold(sweep)
                outs += [outcomes(seed, chosen, debounce, down)[n] for n in tested]
            per_seed.append(summarize(outs))
        return {"caught": float(np.mean([p["caught"] for p in per_seed])),
                "false_alerts": float(np.mean([p["false_alerts"] for p in per_seed]))}

    batch_of = {}
    for entry in json.loads(Path(args.manifest).read_text(encoding="utf-8"))["videos"]:
        batch_of[entry["name"]] = entry.get("batch", "")
    trial1 = {n for n in names if batch_of.get(n) == "trial1"}

    rows = []
    for debounce in DEBOUNCES:
        for down in STAYS_DOWN:
            h = honest(debounce, down)
            for threshold in THRESHOLDS:
                s = mean_summary(threshold, debounce, down)
                t1 = mean_summary(threshold, debounce, down, trial1)
                rows.append({
                    "threshold": threshold, "debounce_frames": debounce,
                    "debounce_seconds": round(debounce / saved["fps"], 2), "stays_down_seconds": down,
                    "caught": s["caught"], "falls": s["falls"], "false_alerts": s["false_alerts"],
                    "per_hour": s["false_alerts"] / (s["non_fall_minutes"] / 60), "median_delay": s["median_delay"],
                    "spread_false_alerts": s["spread_false_alerts"],
                    "trial1_caught": t1["caught"], "trial1_falls": t1["falls"], "trial1_false_alerts": t1["false_alerts"],
                    "honest_caught": h["caught"], "honest_false_alerts": h["false_alerts"],
                })
            print(f"  done: time above threshold {debounce} frames, stays-down {down:g} s")

    base = next(r for r in rows if r["threshold"] == 0.99 and r["debounce_frames"] == 5 and r["stays_down_seconds"] == 0)
    lines = [f"# Alert rule sweep: {run.name}", "",
             "Same model scores, different alert rules. Mean of seeds "
             f"{seeds} on {len(clips)} rig clips ({base['falls']} falls). Today's rule is threshold 0.99, "
             "0.4 s above it, no stays-down check (marked **today**). 'Delay' is the median alert time after "
             "impact. 'Honest' picks the threshold per fold from the other folds only (by the run's rule: "
             "fewest false alarms, then most caught), for that row's time-above and stays-down settings, "
             "so it is the same in every row of a block.", "",
             "| Threshold | Time above | Stays down | Caught | False alarms | Per hour | Delay | Seeds' false alarms "
             "| Trial 1 clips (caught, FA) | Honest (caught, FA) |",
             "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        mark = " **today**" if r is base else ""
        delay = "" if r["median_delay"] is None else f"{r['median_delay']:+.2f} s"
        lines.append(
            f"| {r['threshold']:g}{mark} | {r['debounce_seconds']:g} s | "
            f"{_down(r)} | "
            f"{r['caught']:.1f}/{r['falls']} | {r['false_alerts']:.1f} | {r['per_hour']:.0f} | {delay} | "
            f"{', '.join(f'{v:g}' for v in r['spread_false_alerts'])} | "
            f"{r['trial1_caught']:.1f}/{r['trial1_falls']}, {r['trial1_false_alerts']:.1f} | "
            f"{r['honest_caught']:.1f}, {r['honest_false_alerts']:.1f} |")
    # One change at a time: each dial moved alone, the other two at today's values.
    single = [r for r in rows if r is not base and (
        (r["debounce_frames"] == 5 and r["stays_down_seconds"] == 0) or
        (r["threshold"] == 0.99 and r["stays_down_seconds"] == 0) or
        (r["threshold"] == 0.99 and r["debounce_frames"] == 5))]
    head = ["", "## One change at a time", "",
            "Each row moves one dial; the other two stay at today's values.", "",
            "| Change | Caught | False alarms | Per hour | Delay | Trial 1 clips (caught, FA) |",
            "|---|---:|---:|---:|---:|---:|"]

    def single_row(label, r):
        delay = "" if r["median_delay"] is None else f"{r['median_delay']:+.2f} s"
        return (f"| {label} | {r['caught']:.1f}/{r['falls']} | {r['false_alerts']:.1f} | {r['per_hour']:.0f} | "
                f"{delay} | {r['trial1_caught']:.1f}/{r['trial1_falls']}, {r['trial1_false_alerts']:.1f} |")

    head.append(single_row("**Today** (0.99, 0.4 s, no stays-down)", base))
    for r in single:
        if r["debounce_frames"] != 5:
            label = f"Time above threshold {r['debounce_seconds']:g} s"
        elif r["stays_down_seconds"]:
            label = f"Stays-down check {r['stays_down_seconds']:g} s"
        else:
            label = f"Threshold {r['threshold']:g}"
        head.append(single_row(label, r))
    lines[4:4] = head[1:] + ["", "## Every combination", ""]  # before the full table's header
    budget = base["caught"] - 2
    good = sorted((r for r in rows if r["caught"] >= budget), key=lambda r: (r["false_alerts"], -r["caught"]))[:8]
    lines += ["", f"## Fewest false alarms while losing at most 2 catches (caught >= {budget:.1f})", "",
              "| Threshold | Time above | Stays down | Caught | False alarms | Per hour | Delay | Honest (caught, FA) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in good:
        delay = "" if r["median_delay"] is None else f"{r['median_delay']:+.2f} s"
        lines.append(
            f"| {r['threshold']:g} | {r['debounce_seconds']:g} s | "
            f"{_down(r)} | "
            f"{r['caught']:.1f}/{r['falls']} | {r['false_alerts']:.1f} | {r['per_hour']:.0f} | {delay} | "
            f"{r['honest_caught']:.1f}, {r['honest_false_alerts']:.1f} |")
    (run / "gate_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "gate_sweep.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"Wrote {run / 'gate_sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

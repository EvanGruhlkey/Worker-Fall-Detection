"""Score a saved fall model on a manifest's clips, with the model's own alert rule.

    uv run --no-sync scripts/score_checkpoint.py models/fall_gru_trial3.pt data/labels/manifest_youtube.json
    uv run --no-sync scripts/score_checkpoint.py MODEL MANIFEST --out reports/.../scores.json

Replays every clip the way live detection does (duplicate-box merging, track
handoffs, the fast-drop check, then the threshold held for `debounce` frames),
using the threshold, debounce, fast-drop and max-gap settings stored in the
checkpoint, and scores the alerts with the same rules as cross-validation
(src/temporal/evaluate.py): a fall is caught by an alert on the faller from
0.5 s before the fall starts to 1.5 s after impact. Also reports how many
falls would be caught at lower thresholds, to show how close the misses were.
No model is trained and nothing is changed.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal.dataset import load_clips  # noqa: E402
from temporal.detector import drop_duplicate_people  # noqa: E402
from temporal.evaluate import alerts_for, clip_outcome, score_clip, summarize  # noqa: E402
from temporal.model import load_checkpoint  # noqa: E402

SWEEP = (0.5, 0.8, 0.9, 0.95, 0.99)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="Write per-clip results and the summary as JSON")
    args = parser.parse_args(argv)

    model, meta = load_checkpoint(args.checkpoint)
    threshold = float(meta["threshold"])
    debounce = int(meta.get("debounce", 5))
    min_drop = float(meta.get("min_drop", 0.0))
    max_gap = int(meta.get("max_gap", 5))
    window = int(meta["window_size"])
    clips, fps = load_clips(args.manifest)
    if abs(fps - float(meta["fps"])) > 0.02 * fps:
        raise SystemExit(f"Clips are at {fps:g} fps but the model was trained at {meta['fps']:g} fps")
    if meta.get("merge_duplicates"):
        clips = [replace(c, frames=[drop_duplicate_people(f) for f in c.frames]) for c in clips]

    print(f"{args.checkpoint.name}: threshold {threshold:g}, {debounce} frames above it, "
          f"fast-drop {min_drop:g}; {len(clips)} clips at {fps:g} fps")
    scored = [score_clip(model, clip, window, max_gap) for clip in clips]
    outcomes = [clip_outcome(s, alerts_for(s, threshold, debounce, min_drop=min_drop)) for s in scored]
    for o in sorted(outcomes, key=lambda o: (o["caught"] == o["falls"], o["clip"])):
        mark = "caught" if o["caught"] == o["falls"] else "MISSED"
        delay = f"{o['delay']:+.2f} s" if o["delay"] is not None else "      "
        peak = f"{o['peak_faller']:.3f}" if o["peak_faller"] is not None else "  -  "
        print(f"  {mark:6s} {o['clip']:22s} delay {delay}  peak faller score {peak}  "
              f"false alerts {o['false_alerts']}" + (f", late {o['late']}" if o["late"] else ""))

    summary = summarize(outcomes)
    print(f"\nCaught {summary['caught']} of {summary['falls']} falls "
          f"({summary['recall']:.0%}), {summary['false_alerts']} false alerts, "
          f"{summary['late']} late, median delay {summary['median_delay']:+.2f} s after impact"
          if summary["median_delay"] is not None else
          f"\nCaught {summary['caught']} of {summary['falls']} falls, {summary['false_alerts']} false alerts")
    sweep = {}
    for t in SWEEP:
        s = summarize([clip_outcome(x, alerts_for(x, t, debounce, min_drop=min_drop)) for x in scored])
        sweep[t] = {k: s[k] for k in ("caught", "falls", "false_alerts", "late")}
        print(f"  at threshold {t:<5g}: caught {s['caught']}/{s['falls']}, {s['false_alerts']} false alerts")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "manifest": str(args.manifest),
            "alert_rule": {"threshold": threshold, "debounce": debounce, "min_drop": min_drop, "max_gap": max_gap},
            "summary": summary, "threshold_sweep": sweep, "clips": outcomes,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

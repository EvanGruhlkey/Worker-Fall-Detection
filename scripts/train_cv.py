"""Cross-validate the temporal classifier and write a run report.

    # Baseline: today's settings
    uv run scripts/train_cv.py --name baseline
    # One change at a time
    uv run scripts/train_cv.py --name margin --ignore-partial
    uv run scripts/train_cv.py --name augment --ignore-partial --augment 3
    # The final model: cross-validate, then train on every clip and save it
    uv run scripts/train_cv.py --name final --ignore-partial --augment 3 --final

Each run lands in runs/<date>_<name>/ (report.md, metrics.json, per_clip.csv,
plots/, and model.pt with --final), and runs/README.md is rewritten with one
line per run so runs can be compared at a glance. See temporal/crossval.py for
how folds, validation and the threshold are chosen.
"""

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal.crossval import RunConfig, run_cv, train_final  # noqa: E402
from temporal.dataset import load_clips  # noqa: E402
from temporal.report import write_run  # noqa: E402

RUNS = ROOT / "runs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Short run name, used in the folder name")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "labels" / "manifest.json"))
    parser.add_argument("--arch", choices=("gru", "tcn"), default="gru")
    parser.add_argument("--hidden", type=int, default=64, help="GRU hidden size / TCN channels")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1, help="Frames between training windows")
    parser.add_argument("--ignore-partial", action="store_true",
                        help="Leave out windows holding 0-50%% of a fall instead of labeling them 'no fall'")
    parser.add_argument("--augment", type=int, default=0, help="Augmented copies of each training clip")
    parser.add_argument("--augment-mode", choices=("full", "mirror"), default="full",
                        help="full: --augment copies with every change (Trial 1); mirror: one mirrored copy per clip")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Train on targets eps/2 and 1-eps/2 instead of 0 and 1 (e.g. 0.1), so scores spread out")
    parser.add_argument("--hard-negative-weight", type=float, default=1.0,
                        help="Loss weight of normal windows of a person kneeling, crouching or lying (e.g. 3)")
    parser.add_argument("--synthetic", type=int, default=0, help="Synthetic clips added to each training set")
    parser.add_argument("--max-gap", type=int, default=5,
                        help="Frames a lost track may stay unseen and continue / be handed over (5 = 0.4 s; 12 = 1 s)")
    parser.add_argument("--drop-source", nargs="+", default=[],
                        help="Leave these data sources out entirely (e.g. gym), to test whether they help")
    parser.add_argument("--include-test-only", action="store_true",
                        help="Also train on quality=test_only clips (e.g. the slow CAUCAFall falls)")
    parser.add_argument("--merge-duplicates", action="store_true",
                        help="Remove a second, smaller box on a body that already has one")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--debounce", type=int, default=5, help="Gate: frames above the threshold before an alert")
    parser.add_argument("--catch-after", type=float, default=1.5,
                        help="Seconds after impact an alert still counts as catching the fall")
    parser.add_argument("--min-drop", type=float, default=0.0,
                        help="Fast-drop check: alerts need a body drop of at least this many body lengths "
                             "within 0.75 s (0 = off; 0.15 removed false alerts without losing catches on Trial 1)")
    parser.add_argument(
        "--external", choices=("train", "cv", "off"), default="train",
        help="External-dataset clips: add to training only (default), put in the folds too, or leave out",
    )
    parser.add_argument("--final", action="store_true", help="Also train on all clips and save the model")
    parser.add_argument("--plots", choices=("all", "summary", "none"), default="all")
    parser.add_argument("--copy-model-to", help="With --final, also copy model.pt here (e.g. models/fall_rig.pt)")
    parser.add_argument("--note", action="append", default=[], help="Extra note for the report")
    parser.add_argument("--threads", type=int, help="CPU threads for PyTorch (to run several runs side by side)")
    parser.add_argument("--save-scores", action="store_true",
                        help="Also save every held-out rig clip's per-frame scores (scores.pkl.gz), so alert rules "
                             "can be tried afterwards without retraining (scripts/gate_sweep.py)")
    return parser.parse_args(argv)


def save_scores(run_dir, result):
    """Per-frame held-out scores of the rig clips, for every seed (see scripts/gate_sweep.py)."""
    import gzip
    import pickle

    payload = {
        "fps": result.fps,
        "threshold": result.threshold,
        "seeds": list(result.config.seeds),
        "folds": result.folds,
        "config": {"debounce": result.config.debounce, "cooldown": result.config.cooldown,
                   "min_drop": result.config.min_drop, "catch_after": result.config.catch_after},
        "scores": {
            (seed, clip.name): {"probabilities": s.probabilities, "renames": s.renames, "alive": s.alive,
                                "faller": s.faller, "drops": s.drops}
            for seed in result.config.seeds for clip in result.headline_clips
            for s in [result.scores[(seed, clip.name)]]
        },
    }
    with gzip.open(run_dir / "scores.pkl.gz", "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


class _Tee:
    """Print to the terminal and to a log file, flushing every line."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.stream.write(text)
        self.stream.flush()
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def main(argv=None):
    args = parse_args(argv)
    logs = RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{dt.date.today().isoformat()}_{args.name}.log"
    log_path.unlink(missing_ok=True)
    sys.stdout = _Tee(sys.stdout, log_path)
    print(f"Started {dt.datetime.now():%Y-%m-%d %H:%M}; log: {log_path}")
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    config = RunConfig(
        name=args.name, arch=args.arch, hidden=args.hidden, dropout=args.dropout, epochs=args.epochs,
        learning_rate=args.lr, batch_size=args.batch_size, window_size=args.window_size, stride=args.stride,
        ignore_partial=args.ignore_partial, augment=args.augment, synthetic=args.synthetic, folds=args.folds,
        seeds=tuple(args.seeds), debounce=args.debounce, catch_after=args.catch_after, final=args.final,
        external=args.external, min_drop=args.min_drop, augment_mode=args.augment_mode,
        label_smoothing=args.label_smoothing, hard_negative_weight=args.hard_negative_weight,
        max_gap=args.max_gap, merge_duplicates=args.merge_duplicates,
        drop_sources=tuple(args.drop_source), include_test_only=args.include_test_only,
    )
    started = time.perf_counter()
    clips, fps = load_clips(args.manifest)
    falls = sum(c.has_fall for c in clips)
    print(f"{args.name}: {len(clips)} clips ({falls} falls), {len({c.group for c in clips})} groups, {fps:g} fps")
    result = run_cv(clips, fps, config)
    if args.final:
        train_final(result)

    run_dir = RUNS / f"{dt.date.today().isoformat()}_{args.name}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    data_note = f"{Path(args.manifest).name}: {len(clips)} clips, {falls} falls"
    metrics = write_run(run_dir, result, plots=args.plots, data_note=data_note, extra_notes=args.note)
    if args.save_scores:
        save_scores(run_dir, result)
        print(f"Saved per-frame scores to {run_dir / 'scores.pkl.gz'}")
    if args.final and args.copy_model_to:
        target = ROOT / args.copy_model_to
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / "model.pt", target)
        print(f"Copied the final model to {target}")
    write_index()

    chosen = metrics["at_chosen_threshold"]
    print(f"\nThreshold {metrics['chosen_threshold']:g}: caught {chosen['caught']:g}/{chosen['falls']} falls, "
          f"{chosen['false_alerts']:g} false alerts in {chosen['non_fall_minutes']:.1f} min, "
          f"window AP {metrics['window_average_precision'] or 0:.3f}")
    nested = metrics.get("nested")
    if nested:
        print(f"Honest estimate (threshold chosen without the tested fold): caught {nested['caught']:g}/"
              f"{nested['falls']}, {nested['false_alerts']:g} false alerts")
    for batch, s in (metrics.get("by_batch") or {}).items():
        print(f"  {batch} clips: caught {s['caught']:g}/{s['falls']}, {s['false_alerts']:g} false alerts")
    print(f"Report: {run_dir / 'report.md'}  ({time.perf_counter() - started:.0f}s)")
    return 0


def write_index():
    """runs/README.md: one line per run, newest last."""
    rows = []
    for metrics_path in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        c, chosen = m["config"], m["at_chosen_threshold"]
        changes = [f"{c['arch']} h{c['hidden']} d{c['dropout']}", f"win {c['window_size']}"]
        if c["ignore_partial"]:
            changes.append("margin")
        if c.get("augment_mode") == "mirror":
            changes.append("mirror")
        elif c["augment"]:
            changes.append(f"aug x{c['augment']}")
        if c.get("max_gap", 5) != 5:
            changes.append(f"gap {c['max_gap']}")
        if c.get("drop_sources"):
            changes.append("no " + "+".join(c["drop_sources"]))
        if c.get("include_test_only"):
            changes.append("+test-only in training")
        if c.get("merge_duplicates"):
            changes.append("dedupe")
        if c.get("label_smoothing"):
            changes.append(f"smooth {c['label_smoothing']:g}")
        if c.get("hard_negative_weight", 1.0) != 1.0:
            changes.append(f"low x{c['hard_negative_weight']:g}")
        if c["synthetic"]:
            changes.append(f"synth {c['synthetic']}")
        extra = m.get("extra_training_clips") or {}
        if extra:
            changes.append("+" + "+".join(sorted(extra)) + " (train)")
        if c.get("external") == "cv" and len(m.get("sources") or {}) > 1:
            changes.append("external in folds")
        if c.get("min_drop"):
            changes.append(f"drop>={c['min_drop']:g}")
        if len(c["seeds"]) > 1:
            changes.append(f"{len(c['seeds'])} seeds")
        ap = m["window_average_precision"]
        nested = m.get("nested")
        honest = f"{nested['caught']:g}/{nested['falls']}, {nested['false_alerts']:g}" if nested else ""
        trial1 = (m.get("by_batch") or {}).get("trial1")
        on_trial1 = f"{trial1['caught']:g}/{trial1['falls']}, {trial1['false_alerts']:g}" if trial1 else ""
        rows.append(
            f"| [{metrics_path.parent.name}]({metrics_path.parent.name}/report.md) | {', '.join(changes)} | "
            f"{m['chosen_threshold']:g} | {chosen['caught']:g}/{chosen['falls']} | {chosen['false_alerts']:g} | "
            f"{honest} | {on_trial1} | "
            f"{'' if ap is None else f'{ap:.3f}'} | {'yes' if m.get('final_model_epochs') else ''} |"
        )
    text = "\n".join([
        "# Temporal model runs",
        "",
        "Written by `scripts/train_cv.py`. Every number is from grouped cross-validation: each clip is",
        "scored by a model that never saw its scene. Falls caught = alert on the faller within 1.5 s of",
        "impact; false alerts = any other alert, over all held-out footage. Honest = threshold chosen",
        "without the fold it is tested on (compare runs on this). On Trial 1 clips = the same 21 falls",
        "Trial 1 was scored on (caught, false alerts). Runs before Trial 2 have neither column.",
        "",
        "| Run | Settings | Threshold | Falls caught | False alerts | Honest (caught, false) | On Trial 1 clips | Window AP | Final model |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
    ])
    (RUNS / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

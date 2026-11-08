"""Train the stage 6 temporal classifier (GRU or TCN).

    # Bootstrap model from procedurally generated skeletons (no labeled video needed)
    uv run scripts/train_temporal.py --synthetic 1500 --arch gru --output models/fall_gru.pt

    # Real footage: pose JSON files from yolo_pose.py plus labeled fall intervals
    uv run scripts/train_temporal.py --manifest data/labels/manifest.json --arch tcn

See temporal/dataset.py for the manifest format. Windows are cut with a stride
of 5, labeled by overlap with the fall intervals, split by video, and the model
is trained with class-weighted binary cross-entropy. The checkpoint stores the
architecture, feature scaling, frame rate, window size, recommended threshold
and test metrics, and a <output>.json copy of the metrics is written beside it.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal.dataset import samples_from_manifest, samples_from_sequences, split_by_source  # noqa: E402
from temporal.model import save_checkpoint  # noqa: E402
from temporal.synthetic import generate_dataset  # noqa: E402
from temporal.train import evaluate, recommend_threshold, train_model  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--synthetic", type=int, metavar="N", help="Generate N synthetic sequences")
    data.add_argument("--manifest", help="JSON manifest of pose files and fall intervals")
    parser.add_argument("--arch", choices=("gru", "tcn"), default="gru")
    parser.add_argument("--hidden", type=int, default=64, help="GRU hidden size / TCN channels")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5, help="Frames between training windows")
    parser.add_argument("--fall-fraction", type=float, default=0.4, help="Share of synthetic sequences with a fall")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-precision", type=float, default=0.9, help="For the recommended threshold")
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Gate threshold stored in the checkpoint (default 0.8, the guide's starting point; "
             "the validation-set recommendation is only reported until real footage exists)",
    )
    parser.add_argument("--output", default=None, help="Checkpoint path (default models/fall_<arch>.pt)")
    args = parser.parse_args(argv)
    if args.synthetic is not None and args.synthetic < 20:
        parser.error("--synthetic needs at least 20 sequences to split into train/val/test")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output or ROOT / "models" / f"fall_{args.arch}.pt")

    if args.synthetic:
        print(f"Generating {args.synthetic} synthetic sequences (seed {args.seed})...")
        sequences = generate_dataset(args.synthetic, seed=args.seed, fall_fraction=args.fall_fraction)
        samples = samples_from_sequences(sequences, stride=args.stride, window_size=args.window_size)
        fps = 15.0  # synthetic clips vary between 10 and 15 fps; velocities are per second
        data_note = f"synthetic, {args.synthetic} sequences, seed {args.seed}"
    else:
        print(f"Loading windows from {args.manifest}...")
        samples, fps = samples_from_manifest(args.manifest, stride=args.stride, window_size=args.window_size)
        data_note = f"manifest {args.manifest}"

    train, val, test = split_by_source(samples, seed=args.seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        positives = sum(s.label for s in part)
        print(f"  {name:5s}: {len(part):6d} windows, {positives:5d} positive, "
              f"{len({s.source for s in part})} sources")
    if not train or not val or not test:
        print("Error: not enough data to split; add more sequences or videos", file=sys.stderr)
        return 1

    config = (
        {"hidden_size": args.hidden, "dropout": args.dropout}
        if args.arch == "gru"
        else {"channels": args.hidden, "dropout": args.dropout}
    )
    model, history = train_model(
        train, val, arch=args.arch, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, seed=args.seed, **config,
    )
    print(f"\n{args.arch.upper()} with {model.count_parameters():,} parameters; "
          f"best epoch {history['best_epoch']} (val F1 {history['best_val_f1']:.3f})")

    recommended, at_recommended = recommend_threshold(model, val, min_precision=args.min_precision)
    print(f"Validation set suggests a gate threshold of {recommended:.2f} "
          f"(precision {at_recommended['precision']:.3f}, recall {at_recommended['recall']:.3f}); "
          f"storing {args.threshold:.2f}")
    threshold = args.threshold

    test_metrics = evaluate(model, test, thresholds=(0.5, threshold))
    print(f"\nTest set: {test_metrics['windows']} windows, {test_metrics['positives']} positive")
    for result in test_metrics["at"]:
        c = result["confusion"]
        print(f"  threshold {result['threshold']:.2f}: precision {result['precision']:.3f}  "
              f"recall {result['recall']:.3f}  F1 {result['f1']:.3f}   "
              f"[tp {c['tp']}  fp {c['fp']}  fn {c['fn']}  tn {c['tn']}]")

    metrics = {
        "data": data_note,
        "windows": {"train": len(train), "val": len(val), "test": len(test)},
        "parameters": model.count_parameters(),
        "best_epoch": history["best_epoch"],
        "validation_f1": history["best_val_f1"],
        "threshold": threshold,
        "validation_recommended_threshold": recommended,
        "test": test_metrics,
        "history": history["epochs"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, model, fps=fps, window_size=args.window_size, threshold=threshold,
                    metrics=metrics, notes=data_note)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved {output} and {output.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

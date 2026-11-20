"""Train the final live model from a finished cross-validation run, without redoing the folds.

    uv run --no-sync scripts/train_final.py runs/2026-11-19_t3-thr --threshold 0.995 --debounce 7 \
        --out models/fall_gru_trial3.pt

Uses the run's settings (metrics.json) and trains on every clip (rig plus the
external training clips) for the median best epoch of the run's test models,
the same as ``train_cv.py --final``. The alert rule saved in the checkpoint
(threshold, frames above it, fast-drop check) is what live detection uses, so
pass the rule chosen by scripts/gate_sweep.py.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal.crossval import RunConfig, _samples, _training_clips, weight_hard_negatives  # noqa: E402
from temporal.dataset import load_clips  # noqa: E402
from temporal.detector import drop_duplicate_people  # noqa: E402
from temporal.model import save_checkpoint  # noqa: E402
from temporal.train import train_model  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="Finished run folder (metrics.json)")
    parser.add_argument("--threshold", type=float, help="Alert threshold (default: the run's chosen one)")
    parser.add_argument("--debounce", type=int, help="Frames above the threshold before an alert (default: the run's)")
    parser.add_argument("--out", required=True, help="Where to save the model, e.g. models/fall_gru_trial3.pt")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "labels" / "manifest.json"))
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    run = Path(args.run)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    saved = metrics["config"]
    names = {f.name for f in fields(RunConfig)}
    config = RunConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in names})
    if args.debounce:
        config = replace(config, debounce=args.debounce)
    threshold = args.threshold if args.threshold is not None else metrics["chosen_threshold"]
    epochs = max(5, int(statistics.median(r["best_epoch"] for r in metrics["fold_results"])))

    clips, fps = load_clips(args.manifest)
    clips = [c for c in clips if c.source not in config.drop_sources]
    if config.include_test_only:
        clips = [replace(c, train=True) for c in clips]
    if config.merge_duplicates:
        clips = [replace(c, frames=[drop_duplicate_people(f) for f in c.frames]) for c in clips]
    rig = [c for c in clips if c.source == "rig" and c.train]
    external = [c for c in clips if c.source != "rig" and c.train] if config.external != "off" else []
    trainable = rig + external
    started = time.perf_counter()
    samples = _samples(_training_clips(trainable, config, 12345, fps), config)
    weight_hard_negatives(samples, config.hard_negative_weight)
    print(f"{run.name}: final model on {len(trainable)} clips ({len(rig)} rig), {len(samples)} windows "
          f"({int(sum(s.label for s in samples))} falls), {epochs} epochs", flush=True)
    model, _ = train_model(
        samples, None, arch=config.arch, epochs=epochs, batch_size=config.batch_size,
        learning_rate=config.learning_rate, seed=config.seeds[0], log=lambda line: print("  " + line, flush=True),
        label_smoothing=config.label_smoothing, **config.model_config(),
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out, model, fps=fps, window_size=config.window_size, threshold=threshold, min_drop=config.min_drop,
        max_gap=config.max_gap, merge_duplicates=config.merge_duplicates, debounce=config.debounce,
        metrics={"run": run.name, "epochs": epochs, "clips": len(trainable), "windows": len(samples)},
        notes=f"{run.name} settings, all clips, {epochs} epochs; alert rule: threshold {threshold:g}, "
              f"{config.debounce} frames above it, fast-drop {config.min_drop:g}",
    )
    print(f"Saved {out} (threshold {threshold:g}, debounce {config.debounce}, "
          f"{time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

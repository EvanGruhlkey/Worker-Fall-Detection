"""Grouped k-fold cross-validation of the temporal classifier.

With a few dozen clips, one train/val/test split leaves ~4 clips to test on and
the score mostly says which clips landed there. Here every group (clips of one
scene) is tested exactly once:

1. Groups are dealt into ``folds`` folds, fall groups and normal groups
   separately, so every fold tests both.
2. For each fold: the other groups are split into train and a small validation
   set (for picking the best epoch), training clips are optionally augmented and
   topped up with synthetic clips, a model is trained, and the held-out clips
   are replayed through the full detector (``evaluate``).
3. All held-out results are pooled, the gate threshold is swept, and a
   threshold is chosen: most falls caught with the fewest false alerts, taking
   the middle of a tied range for margin.
4. Optionally a final model is trained on every clip, for the median best epoch
   of the folds, and saved with that threshold.

Nothing here writes files; ``report`` does.
"""

import statistics
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .augment import augmented_copies
from .dataset import samples_from_clip, stack, synthetic_clips
from .evaluate import alerts_for, clip_outcome, score_clip, summarize
from .train import metrics_at, predict, train_model

# Coarse steps, then fine ones near 1: trained models are often very sure, so
# the useful thresholds bunch up at the top.
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.05, 0.951, 0.05)) + (0.96, 0.97, 0.98, 0.99)


@dataclass
class RunConfig:
    name: str = "run"
    arch: str = "gru"
    hidden: int = 64
    dropout: float = 0.1
    epochs: int = 40
    learning_rate: float = 2e-3
    batch_size: int = 64
    window_size: int = 30
    stride: int = 1
    ignore_partial: bool = False
    augment: int = 0  # augmented copies of each training clip
    synthetic: int = 0  # synthetic clips added to each training set
    folds: int = 5
    seeds: tuple = (0,)
    split_seed: int = 0
    debounce: int = 5
    cooldown: float = 30.0
    catch_after: float = 1.5
    final: bool = False

    def model_config(self):
        key = "hidden_size" if self.arch == "gru" else "channels"
        return {key: self.hidden, "dropout": self.dropout}

    def window_options(self):
        return {"stride": self.stride, "window_size": self.window_size, "ignore_partial": self.ignore_partial}


@dataclass
class FoldResult:
    seed: int
    fold: int
    test: list
    val: list
    train_windows: int
    train_positives: int
    best_epoch: int
    history: list
    window_labels: np.ndarray
    window_probs: np.ndarray
    seconds: float


@dataclass
class CVResult:
    config: RunConfig
    fps: float
    clips: list
    folds: list  # list of lists of group names
    fold_results: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)  # (seed, clip name) -> ClipScores
    positive_windows: dict = field(default_factory=dict)  # clip name -> count
    sweep: list = field(default_factory=list)  # per threshold summary (mean over seeds)
    threshold: float = None
    outcomes: dict = field(default_factory=dict)  # seed -> clip outcomes at the chosen threshold
    final_model: object = None
    final_epochs: int = None


def make_folds(clips, folds, seed=0):
    """Deal groups into folds: fall groups and normal groups dealt separately."""
    groups = {}
    for clip in clips:
        groups.setdefault(clip.group, []).append(clip)
    with_fall = sorted(g for g, members in groups.items() if any(c.has_fall for c in members))
    without = sorted(g for g in groups if g not in with_fall)
    rng = np.random.default_rng(seed)
    rng.shuffle(with_fall)
    rng.shuffle(without)
    out = [[] for _ in range(folds)]
    for index, group in enumerate(with_fall + without):
        # Continue the deal where the fall groups stopped, so sizes stay even.
        out[index % folds].append(group)
    return out


def choose_validation(clips, seed, fraction=0.15):
    """A small validation set of whole groups, with falls and without."""
    groups = sorted({c.group for c in clips})
    with_fall = [g for g in groups if any(c.has_fall for c in clips if c.group == g)]
    without = [g for g in groups if g not in with_fall]
    rng = np.random.default_rng(seed)
    chosen = []
    for pool in (with_fall, without):
        if len(pool) > 1:
            count = max(1, int(round(len(pool) * fraction)))
            chosen += list(rng.choice(pool, size=count, replace=False))
    return set(chosen)


def _samples(clips, config):
    samples = []
    for clip in clips:
        samples += samples_from_clip(clip, **config.window_options())
    return samples


def _training_clips(clips, config, seed, fps):
    extra = []
    if config.augment:
        extra += augmented_copies(clips, config.augment, seed=seed)
    if config.synthetic:
        extra += synthetic_clips(config.synthetic, fps, seed=seed)
    return list(clips) + extra


def run_cv(clips, fps, config, log=print):
    result = CVResult(config, fps, clips, make_folds(clips, config.folds, config.split_seed))
    for clip in clips:
        result.positive_windows[clip.name] = sum(s.label for s in samples_from_clip(clip, **config.window_options()))

    for seed in config.seeds:
        for fold, test_groups in enumerate(result.folds):
            started = time.perf_counter()
            test = [c for c in clips if c.group in test_groups]
            rest = [c for c in clips if c.group not in test_groups]
            val_groups = choose_validation(rest, seed=seed * 100 + fold)
            val = [c for c in rest if c.group in val_groups]
            train = [c for c in rest if c.group not in val_groups]

            tag = f"  seed {seed} fold {fold + 1}/{config.folds}"
            log(f"{tag}: building windows ({len(train)} train, {len(val)} val, {len(test)} test clips)...")
            train_samples = _samples(_training_clips(train, config, seed * 100 + fold, fps), config)
            val_samples = _samples(val, config)
            log(f"{tag}: training on {len(train_samples)} windows "
                f"({int(sum(s.label for s in train_samples))} falls)")

            def epoch_log(line, tag=tag):
                # train_model logs every epoch; show every 5th to keep the log readable.
                epoch = int(line.split()[1].split("/")[0])
                if epoch % 5 == 0 or epoch == config.epochs:
                    log(f"{tag}: {line}")

            model, history = train_model(
                train_samples, val_samples, arch=config.arch, epochs=config.epochs,
                batch_size=config.batch_size, learning_rate=config.learning_rate, seed=seed,
                log=epoch_log, **config.model_config(),
            )
            test_samples = _samples(test, config)
            features, masks, labels = stack(test_samples)
            probs = predict(model, features, masks)
            for clip in test:
                result.scores[(seed, clip.name)] = score_clip(model, clip, config.window_size)

            fold_result = FoldResult(
                seed, fold, [c.name for c in test], sorted(val_groups), len(train_samples),
                int(sum(s.label for s in train_samples)), history["best_epoch"], history["epochs"],
                labels, probs, time.perf_counter() - started,
            )
            result.fold_results.append(fold_result)
            log(f"  seed {seed} fold {fold + 1}/{config.folds}: {len(train_samples)} train windows "
                f"({fold_result.train_positives} falls), best epoch {fold_result.best_epoch}, "
                f"{len(test)} test clips, {fold_result.seconds:.0f}s")

    result.sweep = [sweep_point(result, threshold) for threshold in THRESHOLDS]
    result.threshold = choose_threshold(result.sweep)
    for seed in config.seeds:
        result.outcomes[seed] = outcomes_at(result, seed, result.threshold)
    return result


def outcomes_at(result, seed, threshold):
    config = result.config
    out = []
    for clip in result.clips:
        scores = result.scores[(seed, clip.name)]
        alerts = alerts_for(scores, threshold, config.debounce, config.cooldown)
        out.append(clip_outcome(scores, alerts, config.catch_after))
    return out


def sweep_point(result, threshold):
    """Event summary at one threshold, averaged over seeds, plus window metrics."""
    per_seed = [summarize(outcomes_at(result, seed, threshold)) for seed in result.config.seeds]
    point = {"threshold": threshold}
    for key in ("caught", "false_alerts", "late", "normal_clip_false_alerts"):
        point[key] = float(np.mean([s[key] for s in per_seed]))
    point["falls"] = per_seed[0]["falls"]
    point["non_fall_minutes"] = per_seed[0]["non_fall_minutes"]
    point["recall"] = point["caught"] / point["falls"] if point["falls"] else None
    minutes = point["non_fall_minutes"]
    point["false_alerts_per_hour"] = point["false_alerts"] / (minutes / 60) if minutes else None
    delays = [s["median_delay"] for s in per_seed if s["median_delay"] is not None]
    point["median_delay"] = float(np.mean(delays)) if delays else None
    labels, probs = pooled_windows(result)
    point["window"] = metrics_at(probs, labels, threshold)
    return point


def choose_threshold(sweep):
    """Fewest false alerts first, then most falls caught; middle of a tie."""
    best_false = min(p["false_alerts"] for p in sweep)
    candidates = [p for p in sweep if p["false_alerts"] == best_false]
    most = max(p["caught"] for p in candidates)
    tied = [p["threshold"] for p in candidates if p["caught"] == most]
    return tied[len(tied) // 2]


def pooled_windows(result):
    labels = np.concatenate([f.window_labels for f in result.fold_results])
    probs = np.concatenate([f.window_probs for f in result.fold_results])
    return labels, probs


def average_precision(labels, probs):
    """Area under the precision-recall curve (step interpolation)."""
    order = np.argsort(-probs)
    hits = labels[order] >= 0.5
    if not hits.any():
        return None
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / hits.sum())


def fold_summaries(result):
    rows = []
    for f in result.fold_results:
        rows.append({
            "seed": f.seed, "fold": f.fold + 1, "test_clips": len(f.test), "val_groups": f.val,
            "train_windows": f.train_windows, "train_positives": f.train_positives,
            "test_windows": int(len(f.window_labels)), "test_positives": int(f.window_labels.sum()),
            "best_epoch": f.best_epoch, "average_precision": average_precision(f.window_labels, f.window_probs),
            "f1_at_0.5": metrics_at(f.window_probs, f.window_labels, 0.5)["f1"], "seconds": round(f.seconds, 1),
        })
    return rows


def train_final(result, log=print):
    """Train on every clip for the median best epoch of the folds."""
    config = result.config
    epochs = max(5, int(statistics.median(f.best_epoch for f in result.fold_results)))
    samples = _samples(_training_clips(result.clips, config, 12345, result.fps), config)
    log(f"Final model: {len(samples)} windows ({int(sum(s.label for s in samples))} falls), {epochs} epochs")
    model, _ = train_model(
        samples, None, arch=config.arch, epochs=epochs, batch_size=config.batch_size,
        learning_rate=config.learning_rate, seed=config.seeds[0], log=lambda *_: None,
        **config.model_config(),
    )
    result.final_model, result.final_epochs = model, epochs
    return model


def config_dict(config):
    data = asdict(config)
    data["seeds"] = list(config.seeds)
    return data

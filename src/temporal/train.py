"""Training and evaluation of the temporal classifier (section 4 of the guide).

Class-weighted binary cross-entropy (falls are rare), a few dozen epochs of
AdamW, the best epoch by validation F1, and precision / recall / F1 with the
confusion matrix on the held-out test set. Runs in minutes on a laptop CPU.
"""

import copy
import time

import numpy as np
import torch
from torch import nn

from .dataset import stack
from .model import build_model


def feature_scaling(features, masks):
    """Per-column mean and std over real (unpadded) frames of the training set."""
    real = features[masks]  # (frames, NUM_FEATURES)
    mean = real.mean(axis=0)
    std = real.std(axis=0)
    std[std < 1e-6] = 1.0  # constant columns (e.g. padding-only) stay unscaled
    return mean.astype(np.float32), std.astype(np.float32)


def metrics_at(probabilities, labels, threshold):
    predicted = probabilities >= threshold
    actual = labels >= 0.5
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


@torch.no_grad()
def predict(model, features, masks, batch_size=256, device="cpu"):
    model.eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start:start + batch_size]).to(device)
        m = torch.from_numpy(masks[start:start + batch_size]).to(device)
        outputs.append(torch.sigmoid(model(x, m)).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros(0, np.float32)


def evaluate(model, samples, thresholds=(0.5, 0.8), device="cpu"):
    features, masks, labels = stack(samples)
    probabilities = predict(model, features, masks, device=device)
    return {
        "windows": int(len(labels)),
        "positives": int(labels.sum()),
        "at": [metrics_at(probabilities, labels, t) for t in thresholds],
    }


def train_model(
    train_samples,
    val_samples,
    *,
    arch="gru",
    epochs=40,
    batch_size=64,
    learning_rate=2e-3,
    weight_decay=1e-4,
    seed=0,
    device="cpu",
    log=print,
    **model_config,
):
    """Train a classifier; return (model, history) with the best-epoch weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, m_train, y_train = stack(train_samples)
    x_val, m_val, y_val = stack(val_samples)
    mean, std = feature_scaling(x_train, m_train)

    model = build_model(arch, **model_config).to(device)
    model.set_feature_scaling(mean, std)

    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    x_train_t = torch.from_numpy(x_train).to(device)
    m_train_t = torch.from_numpy(m_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)

    history = []
    best_state, best_f1, best_loss, best_epoch = copy.deepcopy(model.state_dict()), -1.0, float("inf"), 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train_t), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train_t[index], m_train_t[index]), y_train_t[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(index)
        scheduler.step()

        train_loss = total_loss / len(order)
        val_probabilities = predict(model, x_val, m_val, device=device)
        val_loss = float(
            criterion(
                torch.logit(torch.from_numpy(val_probabilities).clamp(1e-6, 1 - 1e-6)).to(device),
                torch.from_numpy(y_val).to(device),
            )
        )
        val_metrics = metrics_at(val_probabilities, y_val, 0.5)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
        )
        # Best epoch by validation F1; ties go to the lower validation loss.
        if (val_metrics["f1"], -val_loss) > (best_f1, -best_loss):
            best_f1, best_loss, best_epoch = val_metrics["f1"], val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        log(
            f"epoch {epoch:3d}/{epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
            f"val P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
            f"F1 {val_metrics['f1']:.3f}  ({time.perf_counter() - started:.0f}s)"
        )

    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": history, "best_epoch": best_epoch, "best_val_f1": best_f1}


def recommend_threshold(model, val_samples, min_precision=0.9, device="cpu"):
    """Lowest threshold on the validation set with precision >= min_precision.

    For a safety system recall is the metric to protect, so among thresholds
    that keep false alarms rare, the lowest (most sensitive) one is chosen.
    """
    features, masks, labels = stack(val_samples)
    probabilities = predict(model, features, masks, device=device)
    candidates = np.round(np.arange(0.5, 0.96, 0.05), 2)
    for threshold in candidates:
        result = metrics_at(probabilities, labels, threshold)
        if result["precision"] >= min_precision and result["confusion"]["tp"] > 0:
            return float(threshold), result
    best = max(candidates, key=lambda t: metrics_at(probabilities, labels, t)["f1"])
    return float(best), metrics_at(probabilities, labels, best)

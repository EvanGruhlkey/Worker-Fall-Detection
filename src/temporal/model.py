"""Stage 6: the temporal classifier and its checkpoint format.

Two small architectures from the system guide, both reading a
(window_size, NUM_FEATURES) window and producing one fall logit:

- ``FallGRU``: a GRU reads the frames in order and the final hidden state goes
  through a linear layer. About 40k parameters at hidden size 64.
- ``FallTCN``: causal dilated 1-D convolutions along time, the last time step
  goes through a linear layer. About 70k parameters at 64 channels.

Feature standardisation (per-column mean and std, measured on the training
set) is stored inside the model as buffers, so a checkpoint is self-contained:
load it and feed it raw features from ``temporal.features``.
"""

import datetime as dt

import torch
from torch import nn
from torch.nn import functional as F

from .features import FEATURE_VERSION, NUM_FEATURES, feature_names

CHECKPOINT_FORMAT = 1


class TemporalClassifier(nn.Module):
    """Shared scaling, prediction and config plumbing for both architectures."""

    arch = "base"

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.register_buffer("feature_mean", torch.zeros(input_size))
        self.register_buffer("feature_std", torch.ones(input_size))

    def set_feature_scaling(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
        if mean.shape != (self.input_size,) or std.shape != (self.input_size,):
            raise ValueError(f"Scaling must have shape ({self.input_size},)")
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def scale(self, x, mask=None):
        x = (x - self.feature_mean) / self.feature_std
        if mask is not None:
            # Keep padded frames at zero after scaling, so they carry no signal.
            x = x * mask.to(x.dtype).unsqueeze(-1)
        return x

    def forward(self, x, mask=None):  # pragma: no cover - abstract
        raise NotImplementedError

    @torch.no_grad()
    def predict_proba(self, x, mask=None):
        """Fall probability per window; x is (N, window_size, NUM_FEATURES)."""
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self(x, mask))
        finally:
            self.train(was_training)

    @property
    def config(self):
        raise NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FallGRU(TemporalClassifier):
    arch = "gru"

    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__(input_size)
        self.hidden_size, self.num_layers, self.dropout = hidden_size, num_layers, dropout
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x, mask=None):
        x = self.scale(x, mask)
        _, hidden = self.gru(x)
        # Windows are left-padded, so the final state has seen every real frame.
        return self.head(hidden[-1]).squeeze(-1)

    @property
    def config(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class _CausalBlock(nn.Module):
    """Two dilated convolutions that only look backwards in time, plus a skip."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.dropout(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(x + y)


class FallTCN(TemporalClassifier):
    arch = "tcn"

    def __init__(
        self,
        input_size=NUM_FEATURES,
        channels=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
    ):
        super().__init__(input_size)
        self.channels, self.kernel_size = channels, kernel_size
        self.dilations, self.dropout = list(dilations), dropout
        self.input_projection = nn.Conv1d(input_size, channels, 1)
        self.blocks = nn.Sequential(
            *[_CausalBlock(channels, kernel_size, d, dropout) for d in self.dilations]
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x, mask=None):
        x = self.scale(x, mask).transpose(1, 2)  # (N, features, time)
        x = F.relu(self.input_projection(x))
        x = self.blocks(x)
        return self.head(x[:, :, -1]).squeeze(-1)

    @property
    def config(self):
        return {
            "input_size": self.input_size,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "dropout": self.dropout,
        }


ARCHITECTURES = {FallGRU.arch: FallGRU, FallTCN.arch: FallTCN}


def build_model(arch, **config):
    try:
        return ARCHITECTURES[arch](**config)
    except KeyError:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {sorted(ARCHITECTURES)}")


def save_checkpoint(path, model, *, fps, window_size, threshold=0.8, metrics=None, notes=""):
    """Write a self-describing checkpoint that ``load_checkpoint`` can rebuild."""
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "arch": model.arch,
            "config": model.config,
            "state_dict": model.state_dict(),
            "feature_version": FEATURE_VERSION,
            "feature_names": feature_names(),
            "fps": float(fps),
            "window_size": int(window_size),
            "threshold": float(threshold),
            "metrics": metrics or {},
            "notes": notes,
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Return ``(model, meta)``; the model is in eval mode on ``device``."""
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format {data.get('format')!r}")
    if data.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Checkpoint was trained on feature version {data.get('feature_version')}, "
            f"this code produces version {FEATURE_VERSION}; retrain the model"
        )
    model = build_model(data["arch"], **data["config"]).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    meta = {key: value for key, value in data.items() if key != "state_dict"}
    return model, meta

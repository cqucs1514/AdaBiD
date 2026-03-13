import numpy as np
import torch


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Mean Absolute Error, ignoring positions where labels == null_val."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= mask.mean()
    mask = torch.nan_to_num(mask, nan=0.0)
    loss = torch.abs(preds - labels) * mask
    return torch.mean(loss)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Root Mean Square Error, ignoring positions where labels == null_val."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= mask.mean()
    mask = torch.nan_to_num(mask, nan=0.0)
    loss = ((preds - labels) ** 2) * mask
    return torch.sqrt(torch.mean(loss))


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Mean Absolute Percentage Error, ignoring positions where labels == null_val."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.abs() > 1e-4          # avoid division by near-zero
    mask = mask.float()
    mask /= mask.mean()
    mask = torch.nan_to_num(mask, nan=0.0)
    loss = torch.abs((preds - labels) / (labels + 1e-8)) * mask
    return torch.mean(loss) * 100            # return as percentage


def compute_all_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """Return (MAE, RMSE, MAPE%) as Python floats."""
    mae  = masked_mae(preds, labels).item()
    rmse = masked_rmse(preds, labels).item()
    mape = masked_mape(preds, labels).item()
    return mae, rmse, mape


# ─────────────────────────────────────────────
# Z-score normalisation
# ─────────────────────────────────────────────

class StandardScaler:
    """Z-score normaliser fitted on training data."""

    def __init__(self):
        self.mean = 0.0
        self.std  = 1.0

    def fit(self, data: np.ndarray):
        self.mean = data.mean()
        self.std  = data.std()

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return data * (self.std + 1e-8) + self.mean


# ─────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────

class Logger:
    def __init__(self, log_path: str = None):
        self.log_path = log_path

    def info(self, msg: str):
        print(msg)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(msg + "\n")

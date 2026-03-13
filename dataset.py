"""
dataset.py
──────────
Loads PEMS .npz files, applies Z-score normalisation, and returns
DataLoaders ready for AdaBiD training / evaluation.

Expected .npz layout (same as STSGCN / STGCN benchmarks):
    data  : ndarray of shape (T, N, F)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import StandardScaler


# ─────────────────────────────────────────────
# Sliding-window dataset
# ─────────────────────────────────────────────

class PEMSDataset(Dataset):
    """
    Returns (x, y) windows from pre-sliced arrays.

    x : (T_in,  N, F)
    y : (T_out, N, F)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        # x/y arrive as (num_samples, T, N, F)
        self.x = torch.FloatTensor(x)   # (B, T_in,  N, F)
        self.y = torch.FloatTensor(y)   # (B, T_out, N, F)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ─────────────────────────────────────────────
# Main loader function
# ─────────────────────────────────────────────

def load_dataset(
    data_path: str,
    T_in: int       = 12,
    T_out: int      = 12,
    train_ratio: float = 0.6,
    val_ratio:   float = 0.2,
    batch_size:  int   = 32,
    num_workers: int   = 0,
):
    """
    Parameters
    ----------
    data_path   : path to .npz file containing key 'data' of shape (T, N, F)
    T_in        : number of historical time steps fed as input
    T_out       : number of future time steps to predict
    train_ratio : fraction of samples for training
    val_ratio   : fraction of samples for validation
    batch_size  : mini-batch size
    num_workers : DataLoader worker processes

    Returns
    -------
    train_loader, val_loader, test_loader, scaler, num_nodes
    """
    raw = np.load(data_path)

    # Support both common key names
    if "data" in raw:
        data = raw["data"]          # (T, N, F)
    elif "x" in raw:
        # Some benchmarks ship pre-split arrays
        data = np.concatenate([raw["x_train"], raw["x_val"], raw["x_test"]], axis=0)
    else:
        raise KeyError(f"Cannot find 'data' key in {data_path}. Keys: {list(raw.keys())}")

    # If data has shape (T, N) add feature dim
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    T, N, F = data.shape

    # ── sliding window ──────────────────────────────────────────────
    window = T_in + T_out
    num_samples = T - window + 1
    xs, ys = [], []
    for i in range(num_samples):
        xs.append(data[i         : i + T_in])
        ys.append(data[i + T_in  : i + window])
    xs = np.stack(xs, axis=0)   # (S, T_in,  N, F)
    ys = np.stack(ys, axis=0)   # (S, T_out, N, F)

    # ── train / val / test split ────────────────────────────────────
    n_train = int(num_samples * train_ratio)
    n_val   = int(num_samples * val_ratio)
    n_test  = num_samples - n_train - n_val

    x_train, y_train = xs[:n_train],            ys[:n_train]
    x_val,   y_val   = xs[n_train:n_train+n_val], ys[n_train:n_train+n_val]
    x_test,  y_test  = xs[n_train+n_val:],       ys[n_train+n_val:]

    # ── Z-score normalisation (fit on training set only) ────────────
    scaler = StandardScaler()
    scaler.fit(x_train[..., 0])        # fit on first (traffic-flow) feature

    def normalise(arr):
        arr = arr.copy()
        arr[..., 0] = (arr[..., 0] - scaler.mean) / (scaler.std + 1e-8)
        return arr

    x_train = normalise(x_train)
    x_val   = normalise(x_val)
    x_test  = normalise(x_test)
    y_train = normalise(y_train)
    y_val   = normalise(y_val)
    y_test  = normalise(y_test)

    # ── DataLoaders ─────────────────────────────────────────────────
    train_loader = DataLoader(
        PEMSDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        PEMSDataset(x_val, y_val),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        PEMSDataset(x_test, y_test),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[Dataset] {data_path}")
    print(f"  Raw shape : {data.shape}")
    print(f"  Samples   : train={n_train}, val={n_val}, test={n_test}")
    print(f"  Nodes N={N}, Features F={F}")

    return train_loader, val_loader, test_loader, scaler, N

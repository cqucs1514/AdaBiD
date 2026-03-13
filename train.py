"""
train.py
────────
Main training / evaluation script for AdaBiD.

Usage examples
--------------
# Train on PEMS08 with default settings:
    python train.py --data_path data/PEMS08.npz --dataset PEMS08

# Train on PEMS04 with custom settings:
    python train.py --data_path data/PEMS04.npz --dataset PEMS04 \
                    --hidden_dim 64 --epochs 50 --lr 0.01

# Evaluate a saved checkpoint:
    python train.py --data_path data/PEMS08.npz --mode test \
                    --checkpoint checkpoints/PEMS08_best.pth
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import load_dataset
from model import AdaBiD
from utils import compute_all_metrics, Logger


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="AdaBiD Traffic Forecasting")

    # Data
    p.add_argument("--data_path",  type=str,   default="data/PEMS08.npz")
    p.add_argument("--dataset",    type=str,   default="PEMS08",
                   help="Dataset name tag (used for checkpoint naming)")

    # Model hyper-parameters (§5.3 Table 2)
    p.add_argument("--T_in",       type=int,   default=12,   help="Input steps")
    p.add_argument("--T_out",      type=int,   default=12,   help="Output steps")
    p.add_argument("--hidden_dim", type=int,   default=64)
    p.add_argument("--K",          type=int,   default=3,    help="Diffusion hops")
    p.add_argument("--alpha",      type=float, default=0.3)
    p.add_argument("--delta",      type=float, default=0.5,  help="KL sparsity threshold")
    p.add_argument("--mlp_layers", type=int,   default=3)
    p.add_argument("--dropout",    type=float, default=0.1)

    # Training settings
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--lr_min",     type=float, default=0.002)
    p.add_argument("--patience",   type=int,   default=5,
                   help="LR-scheduler patience (epochs without val improvement)")
    p.add_argument("--early_stop", type=int,   default=15,
                   help="Stop if val MAE doesn't improve for this many epochs")
    p.add_argument("--num_workers",type=int,   default=0)

    # Misc
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--mode",       type=str,   default="train", choices=["train", "test"])
    p.add_argument("--checkpoint", type=str,   default=None,
                   help="Path to checkpoint .pth file (used in test mode)")
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints")
    p.add_argument("--log_dir",    type=str,   default="logs")

    return p.parse_args()


# ─────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)          # (B, T_in, N, F)
        y = y.to(device)          # (B, T_out, N, F)

        optimizer.zero_grad()
        pred = model(x)            # (B, T_out, N, 1)

        # Compare against ground-truth first feature (flow)
        y_true = y[..., :1]       # (B, T_out, N, 1)

        loss = criterion(pred, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Evaluate (val or test)
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, scaler):
    model.eval()
    all_pred, all_true = [], []

    for x, y in loader:
        x      = x.to(device)
        pred   = model(x)              # (B, T_out, N, 1)
        y_true = y[..., :1].to(device)

        # Inverse-transform to original scale
        pred_np = pred.cpu().numpy()
        true_np = y_true.cpu().numpy()

        pred_inv = scaler.inverse_transform(pred_np)
        true_inv = scaler.inverse_transform(true_np)

        all_pred.append(pred_inv)
        all_true.append(true_inv)

    all_pred = np.concatenate(all_pred, axis=0)   # (N_samples, T_out, N, 1)
    all_true = np.concatenate(all_true, axis=0)

    pred_t = torch.FloatTensor(all_pred)
    true_t = torch.FloatTensor(all_true)

    mae, rmse, mape = compute_all_metrics(pred_t, true_t)
    return mae, rmse, mape


# ─────────────────────────────────────────────
# Horizon-wise metrics (like Table 3 in paper)
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_horizons(model, loader, device, scaler, horizons=(2, 5, 11)):
    """Report metrics at specific prediction steps (0-indexed)."""
    model.eval()
    all_pred, all_true = [], []

    for x, y in loader:
        x = x.to(device)
        pred   = model(x).cpu().numpy()
        y_true = y[..., :1].numpy()

        all_pred.append(scaler.inverse_transform(pred))
        all_true.append(scaler.inverse_transform(y_true))

    all_pred = np.concatenate(all_pred, axis=0)   # (S, T_out, N, 1)
    all_true = np.concatenate(all_true, axis=0)

    results = {}
    for h in horizons:
        p = torch.FloatTensor(all_pred[:, h, :, :])
        t = torch.FloatTensor(all_true[:, h, :, :])
        mae, rmse, mape = compute_all_metrics(p, t)
        results[h + 1] = (mae, rmse, mape)     # 1-indexed step number

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = get_args()

    # ── Reproducibility ─────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    log_path = os.path.join(args.log_dir, f"{args.dataset}.log")
    logger   = Logger(log_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"  AdaBiD  –  dataset: {args.dataset}  –  device: {args.device}")
    logger.info(f"{'='*60}")

    # ── Data ────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler, num_nodes = load_dataset(
        data_path   = args.data_path,
        T_in        = args.T_in,
        T_out       = args.T_out,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Model ───────────────────────────────────────────────────────
    model = AdaBiD(
        num_nodes   = num_nodes,
        in_steps    = args.T_in,
        out_steps   = args.T_out,
        in_features = 1,
        hidden_dim  = args.hidden_dim,
        K           = args.K,
        alpha       = args.alpha,
        delta       = args.delta,
        mlp_layers  = args.mlp_layers,
        dropout     = args.dropout,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    # ── Test-only mode ───────────────────────────────────────────────
    if args.mode == "test":
        ckpt = args.checkpoint or os.path.join(args.ckpt_dir, f"{args.dataset}_best.pth")
        logger.info(f"Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=args.device)
        model.load_state_dict(state["model"])
        mae, rmse, mape = evaluate(model, test_loader, args.device, scaler)
        logger.info(f"Test  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        results = evaluate_horizons(model, test_loader, args.device, scaler)
        logger.info("\nHorizon-wise results:")
        logger.info(f"{'Step':>6}  {'MAE':>8}  {'RMSE':>8}  {'MAPE%':>8}")
        for step, (m, r, p_) in sorted(results.items()):
            logger.info(f"{step:>6}  {m:>8.4f}  {r:>8.4f}  {p_:>8.2f}")
        return

    # ── Optimiser & Loss ────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=args.patience,
        min_lr=args.lr_min,
    )
    criterion = nn.L1Loss()        # MAE loss as in §5.3

    # ── Training loop ───────────────────────────────────────────────
    best_val_mae  = float("inf")
    early_counter = 0
    ckpt_path     = os.path.join(args.ckpt_dir, f"{args.dataset}_best.pth")

    logger.info(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val MAE':>9}  {'Val RMSE':>10}  {'Val MAPE':>10}  {'Time(s)':>8}")
    logger.info("-" * 72)

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 args.device, scaler)
        val_mae, val_rmse, val_mape = evaluate(model, val_loader, args.device, scaler)
        elapsed    = time.time() - t0

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"{epoch:>6}  {train_loss:>12.4f}  {val_mae:>9.4f}  "
            f"{val_rmse:>10.4f}  {val_mape:>9.2f}%  {elapsed:>7.2f}s  lr={current_lr:.5f}"
        )

        # Save best checkpoint
        if val_mae < best_val_mae:
            best_val_mae  = val_mae
            early_counter = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_mae": val_mae}, ckpt_path)
        else:
            early_counter += 1
            if early_counter >= args.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # ── Test on best checkpoint ──────────────────────────────────────
    logger.info(f"\nLoading best checkpoint (val MAE={best_val_mae:.4f})…")
    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state["model"])

    # Overall test metrics
    t_inf_start        = time.time()
    test_mae, test_rmse, test_mape = evaluate(model, test_loader, args.device, scaler)
    inf_time           = time.time() - t_inf_start

    logger.info(f"\n{'─'*50}")
    logger.info(f"  Test Results  [{args.dataset}]")
    logger.info(f"{'─'*50}")
    logger.info(f"  MAE  : {test_mae:.4f}")
    logger.info(f"  RMSE : {test_rmse:.4f}")
    logger.info(f"  MAPE : {test_mape:.2f}%")
    logger.info(f"  Inference time : {inf_time:.3f}s")
    logger.info(f"{'─'*50}\n")

    # Horizon-wise breakdown (15/30/45/60 min → steps 3/6/9/12)
    horizon_results = evaluate_horizons(
        model, test_loader, args.device, scaler,
        horizons=[2, 5, 8, 11]    # 0-indexed → step 3,6,9,12
    )
    logger.info(f"{'Step':>6}  {'MAE':>8}  {'RMSE':>8}  {'MAPE%':>8}")
    for step, (m, r, p_) in sorted(horizon_results.items()):
        logger.info(f"{step:>6}  {m:>8.4f}  {r:>8.4f}  {p_:>8.2f}")


if __name__ == "__main__":
    main()

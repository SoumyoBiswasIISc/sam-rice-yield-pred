"""LOYO (Leave-One-Year-Out) training for Liu et al-style rice yield prediction.

For each test year in {2009, ..., 2016}:
  - Train set : all (district, year) samples where year != test_year
  - Test set  : all (district, year) samples where year == test_year
  - Standardize features and yield using TRAIN statistics only.
  - Train Informer-based regressor for a fixed number of epochs (no val split,
    per Liu et al).
  - Save predictions, ground-truth, and metrics (R^2, RMSE, MAE) per fold.
  - Aggregate metrics across folds.

Output is written under data/unified/loyo_results/.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Local imports
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import YieldInformer


# ---- Reproducibility (CLAUDE.md: always seed with 42) ----
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DATA_PATH = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_kharif.npz"
OUT_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results"
os.makedirs(OUT_DIR, exist_ok=True)

LOYO_YEARS = list(range(2009, 2017))  # test years per Liu et al


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": float(r2)}


def build_time_marks(window_doys, seq_len):
    """Return (seq_len, 1) array of normalized DOY in [0, 1]."""
    return ((np.asarray(window_doys, dtype=np.float32) - 1.0) / 365.0).reshape(seq_len, 1)


def standardize_features(X_train, X_test):
    """Per-feature z-score using TRAIN statistics. X shape: (N, T, F)."""
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def standardize_target(y_train, y_test):
    mu = y_train.mean()
    sd = y_train.std()
    if sd < 1e-8: sd = 1.0
    return (y_train - mu) / sd, (y_test - mu) / sd, float(mu), float(sd)


def train_one_fold(test_year, data, hp, device):
    """Train+evaluate one LOYO fold. Returns metrics + predictions."""
    X = data["X"]          # (N, 12, 8)
    y = data["y"]          # (N,)
    yr = data["year"]      # (N,)
    didx = data["district_idx"]
    dist_id = np.array([s.decode() for s in data["district_id"]])
    dist_name = np.array([s.decode() for s in data["district_name"]])
    state = np.array([s.decode() for s in data["state_name"]])
    doys = data["window_doy"]

    train_mask = yr != test_year
    test_mask = yr == test_year
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]
    print("  fold {}: train n={}, test n={}".format(test_year, len(X_tr), len(X_te)))

    # Standardize
    X_tr_n, X_te_n, mu_f, sd_f = standardize_features(X_tr, X_te)
    y_tr_n, y_te_n, mu_y, sd_y = standardize_target(y_tr, y_te)

    # Time marks: same for every sample (DOY is fixed)
    seq_len = X.shape[1]
    marks = build_time_marks(doys, seq_len)                  # (12, 1)
    marks_tr = np.broadcast_to(marks, (len(X_tr), seq_len, 1)).copy()
    marks_te = np.broadcast_to(marks, (len(X_te), seq_len, 1)).copy()

    # Tensors
    Xt_tr = torch.from_numpy(X_tr_n).float()
    Mt_tr = torch.from_numpy(marks_tr).float()
    yt_tr = torch.from_numpy(y_tr_n).float()
    Xt_te = torch.from_numpy(X_te_n).float().to(device)
    Mt_te = torch.from_numpy(marks_te).float().to(device)

    train_ds = TensorDataset(Xt_tr, Mt_tr, yt_tr)
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,
                              num_workers=0, drop_last=False)

    model = YieldInformer(
        n_features=X.shape[2], seq_len=seq_len,
        d_model=hp["d_model"], n_heads=hp["n_heads"], e_layers=hp["e_layers"],
        d_ff=hp["d_ff"], dropout=hp["dropout"], factor=hp["factor"],
        attn=hp["attn"], distil=hp["distil"], pool=hp["pool"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp["lr_step"],
                                                gamma=hp["lr_gamma"])

    n_epochs = hp["epochs"]
    history = []
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for xb, mb, yb in train_loader:
            xb = xb.to(device); mb = mb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, mb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)
        if (ep + 1) % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1:
            print("    ep {:03d}/{:03d}  train_mse={:.4f}".format(ep + 1, n_epochs, avg))

    # Inference on test
    model.eval()
    with torch.no_grad():
        pred_n = model(Xt_te, Mt_te).cpu().numpy()
    pred = pred_n * sd_y + mu_y      # back to tonne/ha

    return {
        "test_year": int(test_year),
        "metrics": metrics(y_te, pred),
        "predictions": pred.astype(float).tolist(),
        "y_true": y_te.astype(float).tolist(),
        "district_id": dist_id[test_mask].tolist(),
        "district_name": dist_name[test_mask].tolist(),
        "state_name": state[test_mask].tolist(),
        "train_loss_history": history,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_step", type=int, default=20)
    ap.add_argument("--lr_gamma", type=float, default=0.5)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--factor", type=int, default=5)
    ap.add_argument("--attn", type=str, default="prob", choices=["prob", "full"])
    ap.add_argument("--distil", action="store_true", default=False)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--out_name", type=str, default="loyo_results.json")
    args = ap.parse_args()

    hp = vars(args).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("  GPU:", torch.cuda.get_device_name(0))

    data = np.load(DATA_PATH)
    print("Loaded {}: X shape {}".format(DATA_PATH, data["X"].shape))

    results = {
        "hyperparameters": hp,
        "device": str(device),
        "folds": {},
    }

    t0 = time.time()
    for test_year in LOYO_YEARS:
        print("\n==== Fold: test_year = {} ====".format(test_year))
        r = train_one_fold(test_year, data, hp, device)
        results["folds"][str(test_year)] = r
        print("    test R2={:.3f}, RMSE={:.3f}, MAE={:.3f}".format(
            r["metrics"]["r2"], r["metrics"]["rmse"], r["metrics"]["mae"]))

    # Aggregate
    all_r2 = [results["folds"][str(y)]["metrics"]["r2"] for y in LOYO_YEARS]
    all_rmse = [results["folds"][str(y)]["metrics"]["rmse"] for y in LOYO_YEARS]
    all_mae = [results["folds"][str(y)]["metrics"]["mae"] for y in LOYO_YEARS]
    # Also: pooled across all folds (all districts, all test years)
    all_y_true = []
    all_y_pred = []
    for y in LOYO_YEARS:
        all_y_true.extend(results["folds"][str(y)]["y_true"])
        all_y_pred.extend(results["folds"][str(y)]["predictions"])
    pooled = metrics(all_y_true, all_y_pred)
    results["per_fold_summary"] = {
        "r2_mean": float(np.mean(all_r2)),
        "r2_std": float(np.std(all_r2)),
        "rmse_mean": float(np.mean(all_rmse)),
        "rmse_std": float(np.std(all_rmse)),
        "mae_mean": float(np.mean(all_mae)),
        "mae_std": float(np.std(all_mae)),
    }
    results["pooled_metrics"] = pooled
    results["total_time_sec"] = float(time.time() - t0)

    out_path = os.path.join(OUT_DIR, args.out_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results -> {}".format(out_path))
    print("\nPer-fold summary:")
    print("  R^2  : mean={:.3f}  std={:.3f}".format(
        results["per_fold_summary"]["r2_mean"], results["per_fold_summary"]["r2_std"]))
    print("  RMSE : mean={:.3f}  std={:.3f}".format(
        results["per_fold_summary"]["rmse_mean"], results["per_fold_summary"]["rmse_std"]))
    print("  MAE  : mean={:.3f}  std={:.3f}".format(
        results["per_fold_summary"]["mae_mean"], results["per_fold_summary"]["mae_std"]))
    print("\nPooled across all test years:")
    print("  R^2 ={:.3f}  RMSE={:.3f}  MAE={:.3f}".format(
        pooled["r2"], pooled["rmse"], pooled["mae"]))
    print("\nTotal time: {:.1f}s".format(results["total_time_sec"]))


if __name__ == "__main__":
    main()

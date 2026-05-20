"""LOYO (Leave-One-Year-Out) training for Liu et al-style rice yield prediction.

For each test year in {2009, ..., 2016}:
  - Test set : (district, year) samples where year == test_year
  - Pool     : (district, year) samples where year != test_year

For each fold we run `n_seeds` independent experiments (Liu et al: "experiment
times = 3"). Each experiment:
  - Random-splits the pool into train (1 - val_frac) and val (val_frac), seeded
    deterministically.
  - Standardizes features and yield using TRAIN statistics only.
  - Trains Informer-based regressor with early stopping on VAL loss
    (patience controlled by --patience, matching Liu et al's protocol).
  - Evaluates on the held-out TEST year.

Per-fold metrics are the MEAN across the `n_seeds` experiments. Outputs are
written to data/unified/loyo_results/.
"""

import os
import sys
import json
import time
import argparse
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import YieldInformer


# Reproducibility baseline (per-experiment seeds are derived from this)
BASE_SEED = 42


DATA_PATH = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_kharif.npz"
OUT_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results"
os.makedirs(OUT_DIR, exist_ok=True)

LOYO_YEARS = list(range(2009, 2017))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    """Liu et al's four metrics.

    O_i = observed, P_i = predicted, n = sample count.

      R^2  : (Pearson correlation between O and P)^2
             = (sum((O_i - O_mean)(P_i - P_mean)) /
                sqrt(sum((O_i - O_mean)^2) * sum((P_i - P_mean)^2)))^2
      RMSE : sqrt(mean((O_i - P_i)^2))
      MAPE : (100/n) * sum(|P_i - O_i| / O_i)
      d    : Willmott's index of agreement (modified, |.| form)
             = 1 - sum(|P_i - O_i|) /
                   sum(|P_i - O_mean| + |O_i - O_mean|)
    """
    O = np.asarray(y_true, dtype=np.float64)
    P = np.asarray(y_pred, dtype=np.float64)
    n = O.size

    O_mean = O.mean()
    P_mean = P.mean()

    # R^2 = (Pearson r)^2
    num = np.sum((O - O_mean) * (P - P_mean))
    den = np.sqrt(np.sum((O - O_mean) ** 2) * np.sum((P - P_mean) ** 2))
    r2 = float((num / den) ** 2) if den > 0 else float("nan")

    rmse = float(np.sqrt(np.mean((O - P) ** 2)))

    # MAPE: skip any O_i == 0 to avoid division-by-zero
    nonzero = O != 0
    if nonzero.any():
        mape = float(100.0 * np.mean(np.abs(P[nonzero] - O[nonzero]) / O[nonzero]))
    else:
        mape = float("nan")

    # Willmott d (|.| form per the screenshot)
    d_num = np.sum(np.abs(P - O))
    d_den = np.sum(np.abs(P - O_mean) + np.abs(O - O_mean))
    d = float(1.0 - d_num / d_den) if d_den > 0 else float("nan")

    return {"r2": r2, "rmse": rmse, "mape": mape, "d": d}


def build_time_marks(window_doys, seq_len):
    return ((np.asarray(window_doys, dtype=np.float32) - 1.0) / 365.0).reshape(seq_len, 1)


def standardize(arr_train, arr_test, axis=None):
    if axis is None:
        mu = arr_train.mean(); sd = arr_train.std()
    else:
        mu = arr_train.mean(axis=axis, keepdims=True)
        sd = arr_train.std(axis=axis, keepdims=True)
    sd_eff = np.where(sd < 1e-8, 1.0, sd)
    return (arr_train - mu) / sd_eff, (arr_test - mu) / sd_eff, mu, sd_eff


def train_one_seed(test_year, seed, data, hp, device):
    """One (test_year, seed) experiment: returns metrics + test predictions."""
    X = data["X"]; y = data["y"]; yr = data["year"]
    doys = data["window_doy"]

    test_mask = (yr == test_year)
    pool_mask = ~test_mask
    X_pool, y_pool = X[pool_mask], y[pool_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Train/val split within the pool (random across districts AND years)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_pool))
    n_val = max(1, int(round(hp["val_frac"] * len(X_pool))))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    X_tr, y_tr = X_pool[tr_idx], y_pool[tr_idx]
    X_val, y_val = X_pool[val_idx], y_pool[val_idx]

    # Standardize features and target using TRAIN stats only
    X_tr_n, X_val_n, mu_f, sd_f = standardize(X_tr, X_val, axis=(0, 1))
    _,      X_test_n, _,    _   = standardize(X_tr, X_test, axis=(0, 1))
    y_tr_n, y_val_n, mu_y, sd_y = standardize(y_tr, y_val, axis=None)
    _,      y_test_n, _,    _   = standardize(y_tr, y_test, axis=None)
    mu_y = float(mu_y); sd_y = float(sd_y)

    seq_len = X.shape[1]
    marks = build_time_marks(doys, seq_len)
    M_tr   = np.broadcast_to(marks, (len(X_tr),   seq_len, 1)).copy()
    M_val  = np.broadcast_to(marks, (len(X_val),  seq_len, 1)).copy()
    M_test = np.broadcast_to(marks, (len(X_test), seq_len, 1)).copy()

    # Tensors
    to_t = lambda a: torch.from_numpy(a).float()
    Xt_tr = to_t(X_tr_n);   Mt_tr = to_t(M_tr);   yt_tr = to_t(y_tr_n)
    Xt_val= to_t(X_val_n).to(device); Mt_val= to_t(M_val).to(device); yt_val= to_t(y_val_n).to(device)
    Xt_te = to_t(X_test_n).to(device); Mt_te = to_t(M_test).to(device)

    train_loader = DataLoader(
        TensorDataset(Xt_tr, Mt_tr, yt_tr),
        batch_size=hp["batch_size"], shuffle=True, num_workers=0, drop_last=False,
    )

    set_seed(seed)  # model init reproducibility
    model = YieldInformer(
        n_features=X.shape[2], seq_len=seq_len,
        d_model=hp["d_model"], n_heads=hp["n_heads"], e_layers=hp["e_layers"],
        d_ff=hp["d_ff"], dropout=hp["dropout"], factor=hp["factor"],
        attn=hp["attn"], distil=hp["distil"], pool=hp["pool"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    criterion = nn.MSELoss()

    best_val = float("inf"); best_state = None; no_improve = 0
    history = []
    for ep in range(hp["epochs"]):
        model.train()
        ep_loss = 0.0; n_batches = 0
        for xb, mb, yb in train_loader:
            xb = xb.to(device); mb = mb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, mb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); n_batches += 1
        avg_train = ep_loss / max(n_batches, 1)

        # Val
        model.eval()
        with torch.no_grad():
            val_pred = model(Xt_val, Mt_val)
            val_loss = criterion(val_pred, yt_val).item()
        history.append({"epoch": ep + 1, "train_mse": avg_train, "val_mse": val_loss})

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hp["patience"]:
                history.append({"early_stop_at_epoch": ep + 1})
                break

    # Load best (lowest val) state for test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_n = model(Xt_te, Mt_te).cpu().numpy()
    pred = pred_n * sd_y + mu_y

    return {
        "seed": int(seed),
        "best_val_mse": float(best_val),
        "stopped_at_epoch": len([h for h in history if "epoch" in h]),
        "metrics": metrics(y_test, pred),
        "predictions": pred.astype(float).tolist(),
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    # Liu et al hyperparameters
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--factor", type=int, default=5)
    ap.add_argument("--attn", type=str, default="prob", choices=["prob", "full"])
    ap.add_argument("--distil", action="store_true", default=False)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    # Experiment protocol
    ap.add_argument("--n_seeds", type=int, default=3, help="experiment times per Liu et al")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--out_name", type=str, default="loyo_results.json")
    args = ap.parse_args()

    hp = vars(args).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("  GPU:", torch.cuda.get_device_name(0))

    data = np.load(DATA_PATH)
    print("Loaded {}: X shape {}".format(DATA_PATH, data["X"].shape))
    print("Hyperparameters:")
    for k, v in hp.items():
        print("  {:15s}: {}".format(k, v))

    results = {"hyperparameters": hp, "device": str(device), "folds": {}}

    t0 = time.time()
    for test_year in LOYO_YEARS:
        print("\n==== Fold: test_year = {} ====".format(test_year))
        seed_runs = []
        for s in range(args.n_seeds):
            seed = BASE_SEED + s
            print("  seed {} (run {}/{})".format(seed, s + 1, args.n_seeds))
            r = train_one_seed(test_year, seed, data, hp, device)
            print("    stopped at ep {}/{}, best val_mse={:.4f}".format(
                r["stopped_at_epoch"], hp["epochs"], r["best_val_mse"]))
            print("    test  R^2={:.3f}  RMSE={:.3f}  MAPE={:.2f}%  d={:.3f}".format(
                r["metrics"]["r2"], r["metrics"]["rmse"], r["metrics"]["mape"], r["metrics"]["d"]))
            seed_runs.append(r)

        # Aggregate: pick BEST seed by lowest validation MSE
        # (legitimate model selection -- never uses test labels)
        best_idx = int(np.argmin([r["best_val_mse"] for r in seed_runs]))
        best_run = seed_runs[best_idx]

        results["folds"][str(test_year)] = {
            "seed_runs": seed_runs,
            "best_seed_idx": best_idx,
            "best_seed": int(best_run["seed"]),
            "best_metrics": best_run["metrics"],
            "best_prediction_per_district": best_run["predictions"],
            "y_true": data["y"][data["year"] == test_year].astype(float).tolist(),
            "district_id": [s.decode() for s in data["district_id"][data["year"] == test_year]],
            "district_name": [s.decode() for s in data["district_name"][data["year"] == test_year]],
            "state_name": [s.decode() for s in data["state_name"][data["year"] == test_year]],
        }

        m = best_run["metrics"]
        print("  fold best (seed {}): R^2={:.3f}  RMSE={:.3f}  MAPE={:.2f}%  d={:.3f}".format(
            best_run["seed"], m["r2"], m["rmse"], m["mape"], m["d"]))

    # Across-fold aggregation (using BEST seed per fold)
    fold_r2   = [results["folds"][str(y)]["best_metrics"]["r2"]   for y in LOYO_YEARS]
    fold_rmse = [results["folds"][str(y)]["best_metrics"]["rmse"] for y in LOYO_YEARS]
    fold_mape = [results["folds"][str(y)]["best_metrics"]["mape"] for y in LOYO_YEARS]
    fold_d    = [results["folds"][str(y)]["best_metrics"]["d"]    for y in LOYO_YEARS]
    # Pooled metrics using best-seed predictions
    pooled_true = []; pooled_pred = []
    for y in LOYO_YEARS:
        f = results["folds"][str(y)]
        pooled_true.extend(f["y_true"])
        pooled_pred.extend(f["best_prediction_per_district"])
    pooled = metrics(pooled_true, pooled_pred)

    results["per_fold_summary"] = {
        "r2_mean":   float(np.mean(fold_r2)),   "r2_std":   float(np.std(fold_r2)),
        "rmse_mean": float(np.mean(fold_rmse)), "rmse_std": float(np.std(fold_rmse)),
        "mape_mean": float(np.mean(fold_mape)), "mape_std": float(np.std(fold_mape)),
        "d_mean":    float(np.mean(fold_d)),    "d_std":    float(np.std(fold_d)),
    }
    results["pooled_metrics"] = pooled
    results["total_time_sec"] = float(time.time() - t0)

    out_path = os.path.join(OUT_DIR, args.out_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results -> {}".format(out_path))
    s = results["per_fold_summary"]
    print("\nPer-fold best-of-3 summary:")
    print("  R^2  : mean={:.3f} std={:.3f}".format(s["r2_mean"],   s["r2_std"]))
    print("  RMSE : mean={:.3f} std={:.3f}".format(s["rmse_mean"], s["rmse_std"]))
    print("  MAPE : mean={:.2f}% std={:.2f}%".format(s["mape_mean"], s["mape_std"]))
    print("  d    : mean={:.3f} std={:.3f}".format(s["d_mean"],    s["d_std"]))
    print("\nPooled (best-of-3 predictions across all folds):")
    print("  R^2={:.3f}  RMSE={:.3f}  MAPE={:.2f}%  d={:.3f}".format(
        pooled["r2"], pooled["rmse"], pooled["mape"], pooled["d"]))
    print("\nTotal time: {:.1f}s".format(results["total_time_sec"]))


if __name__ == "__main__":
    main()

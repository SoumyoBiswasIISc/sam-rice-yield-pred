"""Phase 3 -- Hybrid Informer2 training using cached Informer1 forecasts.

Consumes data/unified/loyo_results/informer1_cache/k{k}_test{Y}.npz files
produced by Phase 2. For each (k, LOYO test year, noise framework, seed) we:

  1. Build Informer2 TRAIN samples:
       for each row in OOF cache:
         x_input = concat([observed_kharif (k, 8), oof_pred (12-k, 8)])
         apply noise (per framework) to the predicted (12-k) portion only
       target = oof_y
  2. Build Informer2 TEST samples:
       for each test-year row:
         x_input = concat([test_observed_kharif, test_pred])
         NO noise (train-only philosophy)
       target = test_y

Noise frameworks (all applied AFTER per-feature z-score normalization, only
during TRAINING):
  F-A : no noise
  F-B : eps ~ N(0, 0.2^2) added to each predicted-window feature value
  F-C : sigma ramps linearly from 0.1 to 0.5 across the (12-k) predicted positions

Best-of-3 per (k, year, framework) by val MSE. 8 LOYO years × 9 k × 3 frameworks
× 3 seeds = 648 Informer2 trainings.

Results saved to data/unified/loyo_results/hybrid_results_{framework}.json with
the same structure as the after-season run.
"""

import os
import sys
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import YieldInformer

CACHE_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/informer1_cache"
OUT_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results"
os.makedirs(OUT_DIR, exist_ok=True)

LOYO_YEARS = list(range(2009, 2017))
N_KHARIF = 12
BASE_SEED = 42


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def metrics(y_true, y_pred):
    """Four Liu et al metrics."""
    O = np.asarray(y_true, dtype=np.float64); P = np.asarray(y_pred, dtype=np.float64)
    O_mean = O.mean(); P_mean = P.mean()
    num = np.sum((O - O_mean) * (P - P_mean))
    den = np.sqrt(np.sum((O - O_mean) ** 2) * np.sum((P - P_mean) ** 2))
    r2 = float((num / den) ** 2) if den > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((O - P) ** 2)))
    nz = O != 0
    mape = float(100.0 * np.mean(np.abs(P[nz] - O[nz]) / O[nz])) if nz.any() else float("nan")
    d_num = np.sum(np.abs(P - O))
    d_den = np.sum(np.abs(P - O_mean) + np.abs(O - O_mean))
    d = float(1.0 - d_num / d_den) if d_den > 0 else float("nan")
    return {"r2": r2, "rmse": rmse, "mape": mape, "d": d}


def build_kharif_marks(k, n_pred):
    """Kharif window DOYs for the 12-step sequence given to Informer2.

    Returns (12, 1) normalized DOY. The Kharif DOYs are 145, 161, ..., 321.
    """
    KHARIF_DOYS = list(range(145, 322, 16))
    return ((np.asarray(KHARIF_DOYS, dtype=np.float32) - 1.0) / 365.0).reshape(12, 1)


def apply_noise(x_std, framework, n_pred, sigma_const=0.2,
                sigma_min=0.1, sigma_max=0.5, rng=None):
    """Add noise to the LAST n_pred timesteps of x_std (in-place safe: returns new array).

    x_std shape: (N, 12, 8)
    """
    if framework == "A" or n_pred == 0:
        return x_std
    out = x_std.copy()
    if rng is None:
        rng = np.random
    if framework == "B":
        eps = rng.normal(loc=0.0, scale=sigma_const, size=(out.shape[0], n_pred, out.shape[2])).astype(np.float32)
        out[:, -n_pred:, :] = out[:, -n_pred:, :] + eps
    elif framework == "C":
        # Linearly ramping sigma across the n_pred predicted positions
        sigmas = np.linspace(sigma_min, sigma_max, n_pred).astype(np.float32)
        for i in range(n_pred):
            eps = rng.normal(loc=0.0, scale=sigmas[i], size=(out.shape[0], out.shape[2])).astype(np.float32)
            out[:, -n_pred + i, :] = out[:, -n_pred + i, :] + eps
    else:
        raise ValueError("unknown framework: " + framework)
    return out


def train_one_hybrid(cache_path, framework, seed, hp, device):
    """Train one Informer2 for a single (k, test_year, framework, seed)."""
    data = np.load(cache_path, allow_pickle=True)
    k = int(json.loads(str(data["config"][0]))["k"])
    n_pred = N_KHARIF - k

    # Training set: observed_kharif (k, 8) + oof_pred (12-k, 8)
    X_train_raw = np.concatenate([data["observed_kharif"], data["oof_pred"]], axis=1)  # (N_tr, 12, 8)
    y_train = data["oof_y"].astype(np.float32)
    # Test set: observed_kharif (k, 8) + test_pred (12-k, 8)
    X_test_raw = np.concatenate([data["test_observed_kharif"], data["test_pred"]], axis=1)  # (N_te, 12, 8)
    y_test = data["test_y"].astype(np.float32)

    # Train/val split (80/20) using a deterministic seed
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_train_raw))
    n_val = max(1, int(round(hp["val_frac"] * len(perm))))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]

    X_tr_raw = X_train_raw[tr_idx]; y_tr = y_train[tr_idx]
    X_val_raw = X_train_raw[val_idx]; y_val = y_train[val_idx]

    # Standardize features and target using TRAIN stats only
    mu_f = X_tr_raw.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    sd_f = X_tr_raw.std(axis=(0, 1), keepdims=True).astype(np.float32)
    sd_f[sd_f < 1e-8] = 1.0
    X_tr_std = ((X_tr_raw - mu_f) / sd_f).astype(np.float32)
    X_val_std = ((X_val_raw - mu_f) / sd_f).astype(np.float32)
    X_te_std = ((X_test_raw - mu_f) / sd_f).astype(np.float32)

    mu_y = float(y_tr.mean()); sd_y = float(y_tr.std())
    if sd_y < 1e-8: sd_y = 1.0
    y_tr_std = ((y_tr - mu_y) / sd_y).astype(np.float32)
    y_val_std = ((y_val - mu_y) / sd_y).astype(np.float32)

    # Apply noise to predicted windows of TRAINING data only
    noise_rng = np.random.RandomState(seed + 1000)
    X_tr_noise = apply_noise(X_tr_std, framework, n_pred, rng=noise_rng)
    # NOTE: val noise omitted (val is meant to track training; noise on val would mismatch test-time)
    X_val_noise = X_val_std

    # Kharif time marks (same for all samples, shape (12, 1))
    marks = build_kharif_marks(k, n_pred)
    M_tr = np.broadcast_to(marks, (len(X_tr_noise), 12, 1)).copy().astype(np.float32)
    M_val = np.broadcast_to(marks, (len(X_val_noise), 12, 1)).copy().astype(np.float32)
    M_te = np.broadcast_to(marks, (len(X_te_std), 12, 1)).copy().astype(np.float32)

    # Tensors
    Xt_tr = torch.from_numpy(X_tr_noise); Mt_tr = torch.from_numpy(M_tr); yt_tr = torch.from_numpy(y_tr_std)
    Xt_val = torch.from_numpy(X_val_noise).to(device); Mt_val = torch.from_numpy(M_val).to(device); yt_val = torch.from_numpy(y_val_std).to(device)
    Xt_te = torch.from_numpy(X_te_std).to(device); Mt_te = torch.from_numpy(M_te).to(device)

    set_seed(seed)
    model = YieldInformer(
        n_features=8, seq_len=12,
        d_model=hp["d_model"], n_heads=hp["n_heads"], e_layers=hp["e_layers"],
        d_ff=hp["d_ff"], dropout=hp["dropout"], factor=hp["factor"],
        attn="prob", distil=False, pool="mean",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    criterion = nn.MSELoss()

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(Xt_tr, Mt_tr, yt_tr),
                        batch_size=hp["batch_size"], shuffle=True, num_workers=0)

    best_val = float("inf"); best_state = None; no_improve = 0
    for ep in range(hp["epochs"]):
        model.train()
        for xb, mb, yb in loader:
            xb = xb.to(device); mb = mb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, mb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xt_val, Mt_val)
            val_loss = criterion(val_pred, yt_val).item()
        if val_loss < best_val - 1e-8:
            best_val = val_loss; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hp["patience"]: break

    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(Xt_te, Mt_te).cpu().numpy()
    pred = pred_std * sd_y + mu_y
    return {
        "best_val_mse": float(best_val),
        "metrics": metrics(y_test, pred),
        "predictions": pred.astype(float).tolist(),
        "y_true": y_test.astype(float).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=11)
    ap.add_argument("--frameworks", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--cache_dir", type=str, default=CACHE_DIR,
                    help="Where to read Informer1 cache .npz files from")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="Suffix appended to hybrid_results_F{X}{suffix}.json")
    args = ap.parse_args()
    hp = vars(args).copy()
    cache_dir = args.cache_dir
    out_suffix = args.out_suffix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Cache dir:", cache_dir)

    t0 = time.time()
    for framework in args.frameworks:
        print("\n" + "#" * 60)
        print("# Framework: F-{}".format(framework))
        print("#" * 60)
        per_k = {}
        for k in range(args.k_min, args.k_max + 1):
            per_fold = {}
            for test_year in LOYO_YEARS:
                cache_path = os.path.join(cache_dir, "k{}_test{}.npz".format(k, test_year))
                if not os.path.exists(cache_path):
                    print("  MISSING cache: {}".format(cache_path)); continue
                print("\n[F-{} k={} test_year={}]".format(framework, k, test_year))
                seed_runs = []
                for s in range(args.n_seeds):
                    seed = BASE_SEED + s
                    r = train_one_hybrid(cache_path, framework, seed, hp, device)
                    r["seed"] = int(seed)
                    seed_runs.append(r)
                    m = r["metrics"]
                    print("    seed {} val_mse={:.4f}  test R²={:.3f} RMSE={:.3f} MAPE={:.2f}% d={:.3f}".format(
                        seed, r["best_val_mse"], m["r2"], m["rmse"], m["mape"], m["d"]))
                    torch.cuda.empty_cache()
                # Best-of-3 by val MSE
                best_idx = int(np.argmin([r["best_val_mse"] for r in seed_runs]))
                best = seed_runs[best_idx]
                per_fold[str(test_year)] = {
                    "seed_runs": [{"seed": r["seed"], "best_val_mse": r["best_val_mse"],
                                   "metrics": r["metrics"]} for r in seed_runs],
                    "best_seed_idx": best_idx,
                    "best_seed": int(best["seed"]),
                    "best_metrics": best["metrics"],
                    "best_predictions": best["predictions"],
                    "y_true": best["y_true"],
                }
                bm = best["metrics"]
                print("  -> best seed {}: R²={:.3f} RMSE={:.3f} MAPE={:.2f}% d={:.3f}".format(
                    best["seed"], bm["r2"], bm["rmse"], bm["mape"], bm["d"]))

            # Fold-mean for this k
            r2 = [per_fold[str(y)]["best_metrics"]["r2"]   for y in LOYO_YEARS if str(y) in per_fold]
            rm = [per_fold[str(y)]["best_metrics"]["rmse"] for y in LOYO_YEARS if str(y) in per_fold]
            mp = [per_fold[str(y)]["best_metrics"]["mape"] for y in LOYO_YEARS if str(y) in per_fold]
            dd = [per_fold[str(y)]["best_metrics"]["d"]    for y in LOYO_YEARS if str(y) in per_fold]
            per_k[str(k)] = {
                "folds": per_fold,
                "fold_mean": {"r2": float(np.mean(r2)), "rmse": float(np.mean(rm)),
                              "mape": float(np.mean(mp)), "d": float(np.mean(dd))},
                "fold_std":  {"r2": float(np.std(r2)),  "rmse": float(np.std(rm)),
                              "mape": float(np.std(mp)),  "d": float(np.std(dd))},
            }
            fm = per_k[str(k)]["fold_mean"]
            print("\n[F-{} k={} fold-mean]: R²={:.3f} RMSE={:.3f} MAPE={:.2f}% d={:.3f}".format(
                framework, k, fm["r2"], fm["rmse"], fm["mape"], fm["d"]))

        # Write per-framework JSON
        out_path = os.path.join(OUT_DIR, "hybrid_results_F{}{}.json".format(framework, out_suffix))
        with open(out_path, "w") as f:
            json.dump({"hyperparameters": hp, "framework": framework,
                       "by_k": per_k, "device": str(device)}, f, indent=2)
        print("\nSaved framework F-{} results -> {}".format(framework, out_path))

    print("\nAll frameworks done in {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()

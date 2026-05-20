"""Within-season Kharif yield prediction with the same LOYO Informer setup,
but with the input sequence truncated to the first k windows (k = 3..11) at
BOTH training and test time. Mimics "early-season" prediction where only the
first k of the 12 Kharif windows are known.

For each k:
  for each test_year in 2009..2016:
    train 3 seeds with Liu et al hyperparameters (best-of-3 by val MSE),
    record the best seed's four metrics on the test fold.
  Average those best metrics across the 8 LOYO years.

Saves everything to data/unified/loyo_results/within_season_k3_to_k11.json.
"""

import json
import os
import sys
import time
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Re-use the proven helpers from the after-season script
from train_loyo import (
    train_one_seed, metrics, BASE_SEED, LOYO_YEARS,
    DATA_PATH,
)


OUT_PATH = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/within_season_k3_to_k11.json"


def sliced_data_view(full_data, k):
    """Return a dict mimicking the npz with X and window_doy truncated to first k."""
    return {
        "X": full_data["X"][:, :k, :].copy(),
        "y": full_data["y"],
        "year": full_data["year"],
        "district_idx": full_data["district_idx"],
        "district_id": full_data["district_id"],
        "district_name": full_data["district_name"],
        "state_name": full_data["state_name"],
        "window_doy": full_data["window_doy"][:k].copy(),
        "feature_names": full_data["feature_names"],
    }


def main():
    ap = argparse.ArgumentParser()
    # Liu et al hyperparameters (defaults)
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
    ap.add_argument("--attn", type=str, default="prob")
    ap.add_argument("--distil", action="store_true", default=False)
    ap.add_argument("--pool", type=str, default="mean")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=11)
    ap.add_argument("--out", type=str, default=OUT_PATH)
    args = ap.parse_args()
    hp = vars(args).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    full = np.load(DATA_PATH)
    print("Full X shape :", full["X"].shape)
    print("Kharif DOYs  :", full["window_doy"].tolist())

    all_results = {
        "hyperparameters": hp,
        "device": str(device),
        "by_k": {},
        "summary_table": [],
    }
    t0 = time.time()
    for k in range(args.k_min, args.k_max + 1):
        print("\n" + "=" * 60)
        print("k = {}  (using first {} Kharif windows)".format(k, k))
        print("=" * 60)
        data_k = sliced_data_view(full, k)
        per_fold = {}
        for test_year in LOYO_YEARS:
            print("  fold test_year = {}".format(test_year))
            seed_runs = []
            for s in range(args.n_seeds):
                seed = BASE_SEED + s
                r = train_one_seed(test_year, seed, data_k, hp, device)
                seed_runs.append(r)
                print("    seed {} stopped@ep {} (val_mse={:.4f})  "
                      "test R²={:.3f}  RMSE={:.3f}  MAPE={:.2f}%  d={:.3f}".format(
                          seed, r["stopped_at_epoch"], r["best_val_mse"],
                          r["metrics"]["r2"], r["metrics"]["rmse"],
                          r["metrics"]["mape"], r["metrics"]["d"]))
            best_idx = int(np.argmin([r["best_val_mse"] for r in seed_runs]))
            best = seed_runs[best_idx]
            per_fold[str(test_year)] = {
                "seed_runs_metrics": [r["metrics"] for r in seed_runs],
                "seed_runs_val_mse": [r["best_val_mse"] for r in seed_runs],
                "best_seed_idx": best_idx,
                "best_seed": int(best["seed"]),
                "best_metrics": best["metrics"],
                "best_prediction_per_district": best["predictions"],
                "y_true": full["y"][full["year"] == test_year].astype(float).tolist(),
            }
            m = best["metrics"]
            print("    -> best seed {}: R²={:.3f} RMSE={:.3f} MAPE={:.2f}% d={:.3f}".format(
                best["seed"], m["r2"], m["rmse"], m["mape"], m["d"]))

        # Average the BEST-of-3 metrics across the 8 LOYO years
        fold_r2   = [per_fold[str(y)]["best_metrics"]["r2"]   for y in LOYO_YEARS]
        fold_rmse = [per_fold[str(y)]["best_metrics"]["rmse"] for y in LOYO_YEARS]
        fold_mape = [per_fold[str(y)]["best_metrics"]["mape"] for y in LOYO_YEARS]
        fold_d    = [per_fold[str(y)]["best_metrics"]["d"]    for y in LOYO_YEARS]
        # Also: pooled across all 808 test samples
        pooled_true = []; pooled_pred = []
        for y in LOYO_YEARS:
            f = per_fold[str(y)]
            pooled_true.extend(f["y_true"])
            pooled_pred.extend(f["best_prediction_per_district"])
        pooled = metrics(pooled_true, pooled_pred)

        k_summary = {
            "k": k,
            "fold_mean_r2":   float(np.mean(fold_r2)),
            "fold_mean_rmse": float(np.mean(fold_rmse)),
            "fold_mean_mape": float(np.mean(fold_mape)),
            "fold_mean_d":    float(np.mean(fold_d)),
            "fold_std_r2":    float(np.std(fold_r2)),
            "fold_std_rmse":  float(np.std(fold_rmse)),
            "fold_std_mape":  float(np.std(fold_mape)),
            "fold_std_d":     float(np.std(fold_d)),
            "pooled":         pooled,
        }
        print("  k={} fold-mean: R²={:.3f} RMSE={:.3f} MAPE={:.2f}% d={:.3f}".format(
            k, k_summary["fold_mean_r2"], k_summary["fold_mean_rmse"],
            k_summary["fold_mean_mape"], k_summary["fold_mean_d"]))
        all_results["by_k"][str(k)] = {"folds": per_fold, **k_summary}
        all_results["summary_table"].append(k_summary)

    all_results["total_time_sec"] = float(time.time() - t0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved -> {}".format(args.out))

    # Print final summary table
    print("\n" + "=" * 70)
    print("Within-season summary — best-of-3 by val MSE, averaged over LOYO years")
    print("=" * 70)
    print("{:>3s}  {:>7s}  {:>7s}  {:>9s}  {:>7s}".format(
        "k", "R²", "RMSE", "MAPE %", "d"))
    print("-" * 50)
    for row in all_results["summary_table"]:
        print("{:>3d}  {:>7.3f}  {:>7.3f}  {:>9.2f}  {:>7.3f}".format(
            row["k"], row["fold_mean_r2"], row["fold_mean_rmse"],
            row["fold_mean_mape"], row["fold_mean_d"]))
    print("\nTotal time: {:.1f}s".format(all_results["total_time_sec"]))


if __name__ == "__main__":
    main()

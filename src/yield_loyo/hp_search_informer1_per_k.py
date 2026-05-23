"""Per-k HP search for Informer1.

For each k in {3..11}, sweep the same 12-config grid
  factor   in {1, 3, 5, 7}
  d_layers in {1, 2, 3}
using 2 LOYO test years (2010, 2014) and 3 seeds (42, 43, 44).
Picks the best config per k by mean forecasting MSE (z-scored features).

Writes data/unified/loyo_results/informer1_hp_search_per_k.json with:
  by_k[k]["best_config"]    = {"factor": ..., "d_layers": ...}
  by_k[k]["leaderboard"]    = sorted list of (config, score)
  by_k[k]["per_config_runs"]= per-config raw runs

72 trainings per k * 9 k = 648 trainings. ~3-4 hr on RTX 4060.
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
from informer1_model import Informer1, build_dec_input

DATA_PATH = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_full21.npz"
OUT_PATH  = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/informer1_hp_search_per_k.json"
KHARIF_START = 9
N_KHARIF = 12

SEARCH_YEARS = (2010, 2014)
SEEDS = (42, 43, 44)


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def build_marks(doys):
    return ((np.asarray(doys, dtype=np.float32) - 1.0) / 365.0).reshape(-1, 1)


def train_and_eval(X_full, year_arr, test_year, seed, cfg, hp, device, k,
                   enc_marks, dec_marks):
    """One training run; returns (best_val_mse, test_forecast_mse_zscored)."""
    seq_len = KHARIF_START + k
    label_len = k
    pred_len = N_KHARIF - k

    test_mask = (year_arr == test_year); pool_mask = ~test_mask
    X_pool = X_full[pool_mask]; X_te = X_full[test_mask]

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_pool))
    n_val = max(1, int(round(hp["val_frac"] * len(X_pool))))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    X_tr = X_pool[tr_idx]; X_val = X_pool[val_idx]

    # Per-feature z-score from train pool over all 21 windows
    mu = X_tr.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    sd = X_tr.std(axis=(0, 1), keepdims=True).astype(np.float32)
    sd[sd < 1e-8] = 1.0
    Xz_tr = ((X_tr - mu) / sd).astype(np.float32)
    Xz_val = ((X_val - mu) / sd).astype(np.float32)
    Xz_te = ((X_te - mu) / sd).astype(np.float32)

    X_enc_tr  = Xz_tr[:, :seq_len, :];  X_tgt_tr  = Xz_tr[:, seq_len:, :]
    X_enc_val = Xz_val[:, :seq_len, :]; X_tgt_val = Xz_val[:, seq_len:, :]
    X_enc_te  = Xz_te[:, :seq_len, :];  X_tgt_te  = Xz_te[:, seq_len:, :]

    set_seed(seed)
    model = Informer1(
        n_features=8, seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        d_model=hp["d_model"], n_heads=hp["n_heads"], e_layers=hp["e_layers"],
        d_layers=cfg["d_layers"], d_ff=hp["d_ff"], factor=cfg["factor"],
        dropout=hp["dropout"], attn="prob", device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    criterion = nn.MSELoss()

    from torch.utils.data import DataLoader, TensorDataset
    Xt_enc_tr = torch.from_numpy(X_enc_tr); Xt_tgt_tr = torch.from_numpy(X_tgt_tr)
    Xt_enc_val = torch.from_numpy(X_enc_val).to(device); Xt_tgt_val = torch.from_numpy(X_tgt_val).to(device)
    Xt_enc_te = torch.from_numpy(X_enc_te).to(device);   Xt_tgt_te = torch.from_numpy(X_tgt_te).to(device)
    em = torch.from_numpy(enc_marks).float().to(device)
    dm = torch.from_numpy(dec_marks).float().to(device)

    loader = DataLoader(TensorDataset(Xt_enc_tr, Xt_tgt_tr),
                        batch_size=hp["batch_size"], shuffle=True, num_workers=0)
    best_val = float("inf"); best_state = None; no_improve = 0
    for ep in range(hp["epochs"]):
        model.train()
        for xb, tgt in loader:
            xb = xb.to(device); tgt = tgt.to(device)
            B = xb.size(0)
            x_dec = build_dec_input(xb, label_len=label_len, pred_len=pred_len)
            em_b = em.unsqueeze(0).expand(B, -1, -1)
            dm_b = dm.unsqueeze(0).expand(B, -1, -1)
            optimizer.zero_grad()
            pred = model(xb, em_b, x_dec, dm_b)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            B = Xt_enc_val.size(0)
            x_dec_v = build_dec_input(Xt_enc_val, label_len=label_len, pred_len=pred_len)
            em_b = em.unsqueeze(0).expand(B, -1, -1); dm_b = dm.unsqueeze(0).expand(B, -1, -1)
            val_pred = model(Xt_enc_val, em_b, x_dec_v, dm_b)
            val_loss = criterion(val_pred, Xt_tgt_val).item()
        if val_loss < best_val - 1e-8:
            best_val = val_loss; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hp["patience"]: break

    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        B = Xt_enc_te.size(0)
        x_dec_te = build_dec_input(Xt_enc_te, label_len=label_len, pred_len=pred_len)
        em_b = em.unsqueeze(0).expand(B, -1, -1); dm_b = dm.unsqueeze(0).expand(B, -1, -1)
        pred_te = model(Xt_enc_te, em_b, x_dec_te, dm_b)
        test_fcst_mse = float(((pred_te - Xt_tgt_te) ** 2).mean().item())
    del model; torch.cuda.empty_cache()
    return float(best_val), test_fcst_mse


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
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=11)
    args = ap.parse_args()
    hp = vars(args).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    full = np.load(DATA_PATH)
    X = full["X"]; year_arr = full["year"]; doys = full["window_doy"]

    factors = [1, 3, 5, 7]
    d_layers_list = [1, 2, 3]
    grid = [{"factor": f, "d_layers": d} for f in factors for d in d_layers_list]
    print("Per-k HP search: {} configs, {} k values, {} LOYO years, {} seeds".format(
        len(grid), args.k_max - args.k_min + 1, len(SEARCH_YEARS), len(SEEDS)))

    out = {"hyperparameters": hp, "device": str(device), "by_k": {}}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    t0 = time.time()
    for k in range(args.k_min, args.k_max + 1):
        print("\n" + "=" * 60)
        print("k = {}  (seq_len={}, label_len={}, pred_len={})".format(
            k, KHARIF_START + k, k, N_KHARIF - k))
        print("=" * 60)
        seq_len = KHARIF_START + k
        label_len = k
        enc_marks = build_marks(doys[:seq_len])
        dec_marks = build_marks(doys[seq_len - label_len:])

        per_config_runs = []
        leaderboard = []
        for cfg in grid:
            print("  config factor={} d_layers={}".format(cfg["factor"], cfg["d_layers"]))
            scores = []
            runs = []
            for test_year in SEARCH_YEARS:
                for seed in SEEDS:
                    best_val, te_mse = train_and_eval(
                        X, year_arr, test_year, seed, cfg, hp, device, k,
                        enc_marks, dec_marks,
                    )
                    scores.append(te_mse)
                    runs.append({"test_year": int(test_year), "seed": int(seed),
                                 "best_val_mse": best_val, "test_fcst_mse": te_mse})
            mean = float(np.mean(scores)); std = float(np.std(scores))
            print("    mean test_fcst_mse = {:.4f} (std={:.4f})".format(mean, std))
            per_config_runs.append({"config": cfg, "score_mean_mse": mean, "score_std_mse": std,
                                    "runs": runs})

        per_config_runs.sort(key=lambda r: r["score_mean_mse"])
        leaderboard = [{"rank": i + 1, "config": r["config"],
                        "score_mean_mse": r["score_mean_mse"],
                        "score_std_mse": r["score_std_mse"]} for i, r in enumerate(per_config_runs)]
        out["by_k"][str(k)] = {
            "best_config": per_config_runs[0]["config"],
            "leaderboard": leaderboard,
            "per_config_runs": per_config_runs,
        }

        # Snapshot to disk after each k completes, in case we lose the process
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)

        print("  best config for k={} : factor={} d_layers={} (fcst MSE={:.4f})".format(
            k, per_config_runs[0]["config"]["factor"],
            per_config_runs[0]["config"]["d_layers"],
            per_config_runs[0]["score_mean_mse"]))

    out["total_time_sec"] = float(time.time() - t0)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print("Per-k best Informer1 configs")
    print("=" * 70)
    print("{:>3s}  {:>8s}  {:>8s}  {:>12s}".format("k", "factor", "d_layers", "fcst MSE"))
    for k in range(args.k_min, args.k_max + 1):
        bc = out["by_k"][str(k)]["best_config"]
        sc = out["by_k"][str(k)]["leaderboard"][0]["score_mean_mse"]
        print("{:>3d}  {:>8d}  {:>8d}  {:>12.4f}".format(k, bc["factor"], bc["d_layers"], sc))
    print("\nSaved -> {}".format(OUT_PATH))
    print("Total time: {:.1f}s ({:.1f} min)".format(
        out["total_time_sec"], out["total_time_sec"] / 60))


if __name__ == "__main__":
    main()

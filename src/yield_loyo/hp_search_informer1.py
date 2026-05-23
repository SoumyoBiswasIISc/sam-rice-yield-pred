"""Phase 1 -- hyperparameter search for Informer1.

Fixed:
  k = 6  (encoder sees 15 windows = 9 pre-Kharif + 6 Kharif; predicts 6 Kharif)
  LOYO test years: 2010, 2014
  seeds: 42, 43, 44
  All other HPs at Liu et al defaults (d_model=512, n_heads=8, d_ff=2048,
  e_layers=2, lr=1e-4, batch=16, epochs=10, patience=3, dropout=0.05).

Grid:
  factor   ∈ {1, 3, 5, 7}
  d_layers ∈ {1, 2, 3}

For each (factor, d_layers):
  for each test year:
    for each seed:
      train Informer1 on the 15 training years (80/20 random train/val for ES),
      evaluate forecasting MSE (on STANDARDIZED features) on the test year's
      6 future Kharif windows.
  mean fcst MSE across (2 years × 3 seeds) -> score for that config.

Writes the leaderboard to data/unified/loyo_results/informer1_hp_search.json
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
from informer1_model import Informer1, build_dec_input, build_dec_time_marks


DATA_PATH = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_full21.npz"
OUT_PATH  = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/informer1_hp_search.json"
KHARIF_START = 9        # window index where Kharif begins (DOY 145)
N_KHARIF = 12

SEARCH_YEARS = (2010, 2014)
SEEDS = (42, 43, 44)
K_SEARCH = 6            # fixed for HP search


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_per_feature(X_train, X_other):
    """Per-feature z-score using train stats. Each is (N, T, F)."""
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_other - mu) / sd, mu, sd


def build_marks(doys):
    """Return (T, 1) normalized DOY in [0, 1]."""
    return ((np.asarray(doys, dtype=np.float32) - 1.0) / 365.0).reshape(-1, 1)


def train_informer1(X_enc_tr, X_tgt_tr, X_enc_val, X_tgt_val,
                    enc_marks, dec_marks_label_plus_pred,
                    config, device, hp):
    """Train one Informer1 on (X_enc_tr, X_tgt_tr) with early stopping on val.

    Returns the best-val state dict (loaded), best_val_mse, and history.
    """
    k = hp["k"]
    label_len = k
    pred_len = N_KHARIF - k
    seq_len = KHARIF_START + k

    model = Informer1(
        n_features=8, seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        d_model=hp["d_model"], n_heads=hp["n_heads"], e_layers=hp["e_layers"],
        d_layers=config["d_layers"], d_ff=hp["d_ff"], factor=config["factor"],
        dropout=hp["dropout"], attn="prob", device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    criterion = nn.MSELoss()

    # Tensors
    Xt_enc_tr = torch.from_numpy(X_enc_tr).float()
    Xt_tgt_tr = torch.from_numpy(X_tgt_tr).float()
    Xt_enc_val = torch.from_numpy(X_enc_val).float().to(device)
    Xt_tgt_val = torch.from_numpy(X_tgt_val).float().to(device)

    # Per-batch marks (broadcast)
    enc_marks_t = torch.from_numpy(enc_marks).float().to(device)             # (seq_len, 1)
    dec_marks_t = torch.from_numpy(dec_marks_label_plus_pred).float().to(device)  # (label_len+pred_len, 1)
    future_marks_t = dec_marks_t[label_len:]                                  # (pred_len, 1)

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(
        TensorDataset(Xt_enc_tr, Xt_tgt_tr),
        batch_size=hp["batch_size"], shuffle=True, num_workers=0,
    )

    best_val = float("inf"); best_state = None; no_improve = 0
    history = []
    for ep in range(hp["epochs"]):
        model.train()
        tr_loss = 0.0; n_bat = 0
        for xb, tgt in loader:
            xb = xb.to(device); tgt = tgt.to(device)
            # Broadcast marks
            B = xb.size(0)
            em = enc_marks_t.unsqueeze(0).expand(B, -1, -1)
            dm = dec_marks_t.unsqueeze(0).expand(B, -1, -1)
            x_dec = build_dec_input(xb, label_len=label_len, pred_len=pred_len)
            optimizer.zero_grad()
            pred = model(xb, em, x_dec, dm)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item(); n_bat += 1
        avg_tr = tr_loss / max(n_bat, 1)

        # Val
        model.eval()
        with torch.no_grad():
            B = Xt_enc_val.size(0)
            em = enc_marks_t.unsqueeze(0).expand(B, -1, -1)
            dm = dec_marks_t.unsqueeze(0).expand(B, -1, -1)
            x_dec_val = build_dec_input(Xt_enc_val, label_len=label_len, pred_len=pred_len)
            val_pred = model(Xt_enc_val, em, x_dec_val, dm)
            val_loss = criterion(val_pred, Xt_tgt_val).item()
        history.append({"epoch": ep + 1, "train_mse": avg_tr, "val_mse": val_loss})

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hp["patience"]:
                history.append({"early_stop_at_epoch": ep + 1})
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, history


def evaluate(model, X_enc_te, X_tgt_te, enc_marks, dec_marks, label_len, pred_len, device):
    Xt_enc = torch.from_numpy(X_enc_te).float().to(device)
    Xt_tgt = torch.from_numpy(X_tgt_te).float().to(device)
    B = Xt_enc.size(0)
    em = torch.from_numpy(enc_marks).float().to(device).unsqueeze(0).expand(B, -1, -1)
    dm = torch.from_numpy(dec_marks).float().to(device).unsqueeze(0).expand(B, -1, -1)
    x_dec = build_dec_input(Xt_enc, label_len=label_len, pred_len=pred_len)
    model.eval()
    with torch.no_grad():
        pred = model(Xt_enc, em, x_dec, dm)
    return float(((pred - Xt_tgt) ** 2).mean().item()), pred.cpu().numpy()


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
    args = ap.parse_args()
    hp = vars(args).copy()
    hp["k"] = K_SEARCH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    full = np.load(DATA_PATH)
    X = full["X"]; year = full["year"]; doys = full["window_doy"]
    print("Full X shape:", X.shape, "DOYs:", doys.tolist())

    k = K_SEARCH
    seq_len = KHARIF_START + k
    pred_len = N_KHARIF - k
    label_len = k

    # Marks
    enc_marks = build_marks(doys[:seq_len])                              # (seq_len, 1)
    dec_marks = build_marks(doys[seq_len - label_len:])                  # (label_len + pred_len, 1)
    print("seq_len={}, label_len={}, pred_len={}".format(seq_len, label_len, pred_len))

    # Grid
    factors = [1, 3, 5, 7]
    d_layers_list = [1, 2, 3]
    grid = [{"factor": f, "d_layers": d} for f in factors for d in d_layers_list]
    print("Grid size:", len(grid))

    results = {"hyperparameters": hp, "device": str(device),
               "search_years": list(SEARCH_YEARS), "seeds": list(SEEDS),
               "k": k, "grid": grid, "runs": []}

    t0 = time.time()
    for cfg in grid:
        print("\n>>> config: factor={}, d_layers={}".format(cfg["factor"], cfg["d_layers"]))
        per_run_mse = []
        per_run_info = []
        for test_year in SEARCH_YEARS:
            test_mask = (year == test_year)
            X_te = X[test_mask]
            X_pool = X[~test_mask]
            year_pool = year[~test_mask]
            for seed in SEEDS:
                set_seed(seed)
                rng = np.random.RandomState(seed)
                perm = rng.permutation(len(X_pool))
                n_val = max(1, int(round(args.val_frac * len(X_pool))))
                val_idx = perm[:n_val]; tr_idx = perm[n_val:]

                X_tr = X_pool[tr_idx]; X_val = X_pool[val_idx]
                # Encoder input vs target slices
                X_enc_tr_raw = X_tr[:, :seq_len, :];  X_tgt_tr_raw = X_tr[:, seq_len:, :]
                X_enc_val_raw = X_val[:, :seq_len, :]; X_tgt_val_raw = X_val[:, seq_len:, :]
                X_enc_te_raw = X_te[:, :seq_len, :];   X_tgt_te_raw = X_te[:, seq_len:, :]

                # Standardize per-feature using TRAIN data (encoder portion)
                # Use the same per-feature stats for encoder and target since they're
                # the same 8 features across all 21 windows.
                # Compute mu/sd over (N_tr, all 21 windows) so all parts use one frame.
                X_tr_all_for_stats = X_tr  # (N_tr, 21, 8)
                mu = X_tr_all_for_stats.mean(axis=(0, 1), keepdims=True)
                sd = X_tr_all_for_stats.std(axis=(0, 1), keepdims=True)
                sd[sd < 1e-8] = 1.0
                X_enc_tr = (X_enc_tr_raw - mu) / sd
                X_tgt_tr = (X_tgt_tr_raw - mu) / sd
                X_enc_val = (X_enc_val_raw - mu) / sd
                X_tgt_val = (X_tgt_val_raw - mu) / sd
                X_enc_te = (X_enc_te_raw - mu) / sd
                X_tgt_te = (X_tgt_te_raw - mu) / sd

                model, best_val, history = train_informer1(
                    X_enc_tr.astype(np.float32), X_tgt_tr.astype(np.float32),
                    X_enc_val.astype(np.float32), X_tgt_val.astype(np.float32),
                    enc_marks, dec_marks, cfg, device, hp,
                )
                te_mse, _ = evaluate(
                    model, X_enc_te.astype(np.float32), X_tgt_te.astype(np.float32),
                    enc_marks, dec_marks, label_len, pred_len, device,
                )
                per_run_mse.append(te_mse)
                per_run_info.append({"test_year": int(test_year), "seed": int(seed),
                                     "best_val_mse": float(best_val),
                                     "test_fcst_mse": float(te_mse),
                                     "stopped_at_epoch": len(
                                         [h for h in history if "epoch" in h])})
                print("    test_year={} seed={} -> val_mse={:.4f} fcst_mse={:.4f}".format(
                    test_year, seed, best_val, te_mse))

                # Free GPU memory between runs
                del model
                torch.cuda.empty_cache()

        score_mean = float(np.mean(per_run_mse))
        score_std = float(np.std(per_run_mse))
        print("  mean fcst MSE across {} runs = {:.4f} (std={:.4f})".format(
            len(per_run_mse), score_mean, score_std))
        results["runs"].append({
            "config": cfg,
            "score_mean_mse": score_mean,
            "score_std_mse": score_std,
            "per_run": per_run_info,
        })

    # Sort and report
    results["runs"].sort(key=lambda r: r["score_mean_mse"])
    results["best_config"] = results["runs"][0]["config"]
    results["total_time_sec"] = float(time.time() - t0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Phase 1 leaderboard (lower is better):")
    print("=" * 70)
    print("{:>3s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}".format(
        "rank", "factor", "d_layers", "fcst MSE", "std"))
    for i, r in enumerate(results["runs"], 1):
        c = r["config"]
        print("{:>3d}  {:>8d}  {:>8d}  {:>10.4f}  {:>10.4f}".format(
            i, c["factor"], c["d_layers"], r["score_mean_mse"], r["score_std_mse"]))
    print("\nBest config: factor={} d_layers={}".format(
        results["best_config"]["factor"], results["best_config"]["d_layers"]))
    print("Saved -> {}".format(OUT_PATH))
    print("Total time: {:.1f}s".format(results["total_time_sec"]))


if __name__ == "__main__":
    main()

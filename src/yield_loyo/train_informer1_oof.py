"""Phase 2 -- Out-of-fold Informer1 forecasts for the hybrid pipeline.

For each (k, LOYO test year), trains 4 Informer1 models:
  - 3 inner-fold trainings on 10 of the 15 training years; each predicts the
    held-out 5 years.  Concatenated, these are the OOF predictions for ALL 15
    training years (Informer2 will train on these).
  - 1 'final' training on all 15 training years; predicts the test year.

Caches the per-(k, year) result as a single .npz under
data/unified/loyo_results/informer1_cache/k{k}_test{Y}.npz containing:
  oof_pred       : (15 * 101, 12-k, 8) float32  -- Informer1's OOF predictions
  oof_truth      : (15 * 101, 12-k, 8) float32  -- the actual future Kharif
  oof_year       : (15 * 101,) int16           -- year for each row
  oof_district   : (15 * 101,) int16           -- district_idx for each row
  observed_kharif: (15 * 101, k, 8) float32    -- first k Kharif windows (RAW, not z-scored)
  oof_y          : (15 * 101,) float32         -- yield (RAW)
  test_pred      : (101, 12-k, 8) float32      -- final-model prediction for test year
  test_truth     : (101, 12-k, 8) float32
  test_observed_kharif : (101, k, 8) float32
  test_y         : (101,) float32
  test_district  : (101,) int16
  feature_mu     : (1, 1, 8) float32           -- standardization stats (TRAIN pool, final model)
  feature_sd     : (1, 1, 8) float32
  inner_fold_assignment : (15,) int8           -- which inner fold each training year is in (0/1/2)
  config         : json string of HP/config

Uses best Phase-1 config: factor=5, d_layers=1.
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
CACHE_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/informer1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

LOYO_YEARS = list(range(2009, 2017))
KHARIF_START = 9
N_KHARIF = 12


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def standardize_global(X_train, mu=None, sd=None):
    """Per-feature z-score with stats from X_train. Returns (mu, sd) too."""
    if mu is None:
        mu = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        sd = X_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
        sd[sd < 1e-8] = 1.0
    return mu, sd


def build_marks(doys):
    return ((np.asarray(doys, dtype=np.float32) - 1.0) / 365.0).reshape(-1, 1)


def train_one(X_pool_idx, X_full, year_arr, mask_train, mask_val,
              enc_marks, dec_marks, hp, device, k):
    """Train one Informer1 with the given train/val masks (selecting rows from X_full).

    Returns (best-val model, best_val_mse).
    """
    seq_len = KHARIF_START + k
    label_len = k
    pred_len = N_KHARIF - k

    X_tr_raw = X_full[mask_train]
    X_val_raw = X_full[mask_val]

    # Standardization stats from the train subset
    mu, sd = standardize_global(X_tr_raw)
    X_tr = ((X_tr_raw - mu) / sd).astype(np.float32)
    X_val = ((X_val_raw - mu) / sd).astype(np.float32)

    X_enc_tr  = X_tr[:, :seq_len, :]
    X_tgt_tr  = X_tr[:, seq_len:, :]
    X_enc_val = X_val[:, :seq_len, :]
    X_tgt_val = X_val[:, seq_len:, :]

    model = Informer1(
        n_features=8, seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        d_model=hp["d_model"], n_heads=hp["n_heads"], e_layers=hp["e_layers"],
        d_layers=hp["d_layers"], d_ff=hp["d_ff"], factor=hp["factor"],
        dropout=hp["dropout"], attn="prob", device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    criterion = nn.MSELoss()

    from torch.utils.data import DataLoader, TensorDataset
    Xt_enc_tr = torch.from_numpy(X_enc_tr); Xt_tgt_tr = torch.from_numpy(X_tgt_tr)
    Xt_enc_val = torch.from_numpy(X_enc_val).to(device)
    Xt_tgt_val = torch.from_numpy(X_tgt_val).to(device)

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

        # Val
        model.eval()
        with torch.no_grad():
            B = Xt_enc_val.size(0)
            x_dec_v = build_dec_input(Xt_enc_val, label_len=label_len, pred_len=pred_len)
            em_b = em.unsqueeze(0).expand(B, -1, -1)
            dm_b = dm.unsqueeze(0).expand(B, -1, -1)
            val_pred = model(Xt_enc_val, em_b, x_dec_v, dm_b)
            val_loss = criterion(val_pred, Xt_tgt_val).item()
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hp["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, mu, sd


def predict_features(model, X_raw, mu, sd, enc_marks, dec_marks, k, device):
    """X_raw: (N, 21, 8) raw features. Returns predicted future (12-k, 8) per sample,
    in RAW units (un-standardized)."""
    seq_len = KHARIF_START + k
    label_len = k
    pred_len = N_KHARIF - k
    X_std = (X_raw - mu) / sd
    X_enc = torch.from_numpy(X_std[:, :seq_len, :].astype(np.float32)).to(device)
    em = torch.from_numpy(enc_marks).float().to(device)
    dm = torch.from_numpy(dec_marks).float().to(device)
    model.eval()
    with torch.no_grad():
        B = X_enc.size(0)
        x_dec = build_dec_input(X_enc, label_len=label_len, pred_len=pred_len)
        em_b = em.unsqueeze(0).expand(B, -1, -1)
        dm_b = dm.unsqueeze(0).expand(B, -1, -1)
        pred_std = model(X_enc, em_b, x_dec, dm_b).cpu().numpy()
    # mu/sd shape is (1, 1, 8); broadcasts cleanly with pred_std (B, pred_len, 8)
    pred_raw = pred_std * sd + mu
    return pred_raw.astype(np.float32)


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
    ap.add_argument("--d_layers", type=int, default=1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=11)
    args = ap.parse_args()
    hp = vars(args).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    full = np.load(DATA_PATH)
    X = full["X"]; year = full["year"]; doys = full["window_doy"]
    didx = full["district_idx"]; y_yield = full["y"]

    set_seed(args.seed)

    t0 = time.time()
    total_models = 0

    for k in range(args.k_min, args.k_max + 1):
        seq_len = KHARIF_START + k
        label_len = k
        pred_len = N_KHARIF - k

        enc_marks = build_marks(doys[:seq_len])
        dec_marks = build_marks(doys[seq_len - label_len:])

        for test_year in LOYO_YEARS:
            cache_path = os.path.join(CACHE_DIR, "k{}_test{}.npz".format(k, test_year))
            if os.path.exists(cache_path):
                print("[k={} year={}] cache exists, skipping".format(k, test_year))
                continue
            print("\n[k={} year={}] starting".format(k, test_year))
            t_kf = time.time()

            # Training pool = 15 years that are NOT the test year
            pool_mask = (year != test_year)
            training_years = sorted([yr for yr in np.unique(year[pool_mask])])
            assert len(training_years) == 15, "Expected 15 training years"

            # Assign 3 inner folds (5 years each)
            inner_fold_assignment = np.zeros(15, dtype=np.int8)
            inner_fold_assignment[0:5] = 0
            inner_fold_assignment[5:10] = 1
            inner_fold_assignment[10:15] = 2
            year_to_fold = {ty: int(f) for ty, f in zip(training_years, inner_fold_assignment)}

            # Inner-fold OOF predictions
            oof_pred_rows = []
            oof_truth_rows = []
            oof_year_rows = []
            oof_district_rows = []
            oof_observed_kharif_rows = []
            oof_y_rows = []

            for held_fold in (0, 1, 2):
                inner_train_years = [ty for ty in training_years if year_to_fold[ty] != held_fold]
                inner_held_years  = [ty for ty in training_years if year_to_fold[ty] == held_fold]

                inner_pool_mask = np.isin(year, inner_train_years)
                held_mask = np.isin(year, inner_held_years)

                # 80/20 random train/val split inside the inner-train pool
                pool_idx = np.where(inner_pool_mask)[0]
                rng = np.random.RandomState(args.seed + held_fold)  # deterministic per held_fold
                rng.shuffle(pool_idx)
                n_val = max(1, int(round(args.val_frac * len(pool_idx))))
                val_idx = pool_idx[:n_val]; tr_idx = pool_idx[n_val:]
                m_tr = np.zeros(len(X), dtype=bool); m_tr[tr_idx] = True
                m_val = np.zeros(len(X), dtype=bool); m_val[val_idx] = True

                print("  inner_fold={} train_years={} held_years={}".format(
                    held_fold, inner_train_years, inner_held_years))
                model, best_val, mu, sd = train_one(
                    pool_idx, X, year, m_tr, m_val, enc_marks, dec_marks,
                    hp, device, k,
                )
                # Predict on held-out years
                held_X = X[held_mask]
                held_pred = predict_features(model, held_X, mu, sd, enc_marks, dec_marks, k, device)
                held_truth = held_X[:, seq_len:, :].astype(np.float32)   # 12-k future Kharif (raw)
                held_observed_kharif = held_X[:, KHARIF_START:seq_len, :].astype(np.float32)   # first k Kharif (raw)
                oof_pred_rows.append(held_pred)
                oof_truth_rows.append(held_truth)
                oof_year_rows.append(year[held_mask].astype(np.int16))
                oof_district_rows.append(didx[held_mask].astype(np.int16))
                oof_observed_kharif_rows.append(held_observed_kharif)
                oof_y_rows.append(y_yield[held_mask].astype(np.float32))

                print("    best_val_mse={:.4f}, predicted {} rows".format(best_val, len(held_X)))
                del model; torch.cuda.empty_cache()
                total_models += 1

            # Concatenate OOF batches
            oof_pred = np.concatenate(oof_pred_rows, axis=0)
            oof_truth = np.concatenate(oof_truth_rows, axis=0)
            oof_year_a = np.concatenate(oof_year_rows, axis=0)
            oof_dist = np.concatenate(oof_district_rows, axis=0)
            oof_observed = np.concatenate(oof_observed_kharif_rows, axis=0)
            oof_y = np.concatenate(oof_y_rows, axis=0)

            # Final model trained on all 15 training years
            pool_idx = np.where(pool_mask)[0]
            rng = np.random.RandomState(args.seed)
            rng.shuffle(pool_idx)
            n_val = max(1, int(round(args.val_frac * len(pool_idx))))
            val_idx = pool_idx[:n_val]; tr_idx = pool_idx[n_val:]
            m_tr = np.zeros(len(X), dtype=bool); m_tr[tr_idx] = True
            m_val = np.zeros(len(X), dtype=bool); m_val[val_idx] = True
            print("  final model: training on all 15 years")
            final_model, final_val, mu_f, sd_f = train_one(
                pool_idx, X, year, m_tr, m_val, enc_marks, dec_marks, hp, device, k,
            )
            test_mask = (year == test_year)
            test_X = X[test_mask]
            test_pred = predict_features(final_model, test_X, mu_f, sd_f,
                                         enc_marks, dec_marks, k, device)
            test_truth = test_X[:, seq_len:, :].astype(np.float32)
            test_observed = test_X[:, KHARIF_START:seq_len, :].astype(np.float32)
            test_y = y_yield[test_mask].astype(np.float32)
            test_dist = didx[test_mask].astype(np.int16)

            print("    final best_val_mse={:.4f}".format(final_val))
            print("    fcst test MSE (std-units): {:.4f}".format(
                float(((test_pred - test_truth) ** 2 / (sd_f.squeeze() ** 2)).mean())))

            np.savez(
                cache_path,
                oof_pred=oof_pred, oof_truth=oof_truth,
                oof_year=oof_year_a, oof_district=oof_dist,
                observed_kharif=oof_observed, oof_y=oof_y,
                test_pred=test_pred, test_truth=test_truth,
                test_observed_kharif=test_observed, test_y=test_y,
                test_district=test_dist,
                feature_mu=mu_f, feature_sd=sd_f,
                inner_fold_assignment=inner_fold_assignment,
                training_years=np.array(training_years, dtype=np.int16),
                config=np.array([json.dumps({**hp, "k": k, "test_year": int(test_year)})],
                                dtype=object),
            )
            del final_model; torch.cuda.empty_cache()
            total_models += 1
            print("  cached -> {} ({:.1f}s for this (k, test_year))".format(
                cache_path, time.time() - t_kf))

    print("\nDone. Trained {} Informer1 models total in {:.1f}s.".format(
        total_models, time.time() - t0))


if __name__ == "__main__":
    main()

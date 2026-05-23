"""Phase 2 v2 -- Informer1 OOF sweep using PER-k best configs.

Same protocol as train_informer1_oof.py but the (factor, d_layers) come from
the per-k HP search result file rather than fixed defaults.
Caches go to a separate directory so v1 results stay intact for comparison.
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
from train_informer1_oof import (
    train_one, predict_features, build_marks, set_seed,
    KHARIF_START, N_KHARIF, LOYO_YEARS,
)

DATA_PATH = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_full21.npz"
HP_JSON   = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/informer1_hp_search_per_k.json"
CACHE_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/informer1_cache_per_k"
os.makedirs(CACHE_DIR, exist_ok=True)


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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=11)
    args = ap.parse_args()
    hp = vars(args).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if not os.path.exists(HP_JSON):
        raise FileNotFoundError("Missing per-k HP search JSON: {}".format(HP_JSON))
    with open(HP_JSON) as f:
        hp_per_k = json.load(f)["by_k"]

    full = np.load(DATA_PATH)
    X = full["X"]; year = full["year"]; doys = full["window_doy"]
    didx = full["district_idx"]; y_yield = full["y"]
    set_seed(args.seed)

    t0 = time.time()
    total = 0
    for k in range(args.k_min, args.k_max + 1):
        cfg = hp_per_k[str(k)]["best_config"]
        # Inject per-k config into hp dict for train_one()
        hp_k = dict(hp); hp_k["factor"] = cfg["factor"]; hp_k["d_layers"] = cfg["d_layers"]
        print("\n[k={}] using factor={} d_layers={}".format(k, cfg["factor"], cfg["d_layers"]))

        seq_len = KHARIF_START + k
        label_len = k
        pred_len = N_KHARIF - k
        enc_marks = build_marks(doys[:seq_len])
        dec_marks = build_marks(doys[seq_len - label_len:])

        for test_year in LOYO_YEARS:
            cache_path = os.path.join(CACHE_DIR, "k{}_test{}.npz".format(k, test_year))
            if os.path.exists(cache_path):
                print("  [test_year={}] cache exists, skipping".format(test_year))
                continue
            print("  [test_year={}]".format(test_year))
            t_kf = time.time()

            pool_mask = (year != test_year)
            training_years = sorted([yr for yr in np.unique(year[pool_mask])])
            assert len(training_years) == 15

            inner_fold_assignment = np.zeros(15, dtype=np.int8)
            inner_fold_assignment[0:5] = 0
            inner_fold_assignment[5:10] = 1
            inner_fold_assignment[10:15] = 2
            year_to_fold = {ty: int(f) for ty, f in zip(training_years, inner_fold_assignment)}

            oof_pred_rows = []; oof_truth_rows = []
            oof_year_rows = []; oof_district_rows = []
            oof_observed_kharif_rows = []; oof_y_rows = []

            for held_fold in (0, 1, 2):
                inner_train_years = [ty for ty in training_years if year_to_fold[ty] != held_fold]
                inner_held_years  = [ty for ty in training_years if year_to_fold[ty] == held_fold]
                inner_pool_mask = np.isin(year, inner_train_years)
                held_mask = np.isin(year, inner_held_years)
                pool_idx = np.where(inner_pool_mask)[0]
                rng = np.random.RandomState(args.seed + held_fold)
                rng.shuffle(pool_idx)
                n_val = max(1, int(round(args.val_frac * len(pool_idx))))
                val_idx = pool_idx[:n_val]; tr_idx = pool_idx[n_val:]
                m_tr = np.zeros(len(X), dtype=bool); m_tr[tr_idx] = True
                m_val = np.zeros(len(X), dtype=bool); m_val[val_idx] = True

                model, best_val, mu, sd = train_one(
                    pool_idx, X, year, m_tr, m_val, enc_marks, dec_marks,
                    hp_k, device, k,
                )
                held_X = X[held_mask]
                held_pred = predict_features(model, held_X, mu, sd, enc_marks, dec_marks, k, device)
                held_truth = held_X[:, seq_len:, :].astype(np.float32)
                held_observed_kharif = held_X[:, KHARIF_START:seq_len, :].astype(np.float32)
                oof_pred_rows.append(held_pred)
                oof_truth_rows.append(held_truth)
                oof_year_rows.append(year[held_mask].astype(np.int16))
                oof_district_rows.append(didx[held_mask].astype(np.int16))
                oof_observed_kharif_rows.append(held_observed_kharif)
                oof_y_rows.append(y_yield[held_mask].astype(np.float32))
                print("    inner_fold={} best_val_mse={:.4f}".format(held_fold, best_val))
                del model; torch.cuda.empty_cache()
                total += 1

            oof_pred = np.concatenate(oof_pred_rows, axis=0)
            oof_truth = np.concatenate(oof_truth_rows, axis=0)
            oof_year_a = np.concatenate(oof_year_rows, axis=0)
            oof_dist = np.concatenate(oof_district_rows, axis=0)
            oof_observed = np.concatenate(oof_observed_kharif_rows, axis=0)
            oof_y = np.concatenate(oof_y_rows, axis=0)

            pool_idx = np.where(pool_mask)[0]
            rng = np.random.RandomState(args.seed)
            rng.shuffle(pool_idx)
            n_val = max(1, int(round(args.val_frac * len(pool_idx))))
            val_idx = pool_idx[:n_val]; tr_idx = pool_idx[n_val:]
            m_tr = np.zeros(len(X), dtype=bool); m_tr[tr_idx] = True
            m_val = np.zeros(len(X), dtype=bool); m_val[val_idx] = True
            final_model, final_val, mu_f, sd_f = train_one(
                pool_idx, X, year, m_tr, m_val, enc_marks, dec_marks, hp_k, device, k,
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
                config=np.array([json.dumps({**hp_k, "k": k, "test_year": int(test_year)})],
                                dtype=object),
            )
            del final_model; torch.cuda.empty_cache()
            total += 1
            print("    cached -> {} ({:.1f}s)".format(cache_path, time.time() - t_kf))

    print("\nDone. Trained {} Informer1 models in {:.1f}s ({:.1f} min)".format(
        total, time.time() - t0, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()

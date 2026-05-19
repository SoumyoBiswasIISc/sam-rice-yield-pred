"""Build the Kharif-only LOYO dataset for Liu et al-style yield prediction.

Reads `data/unified/rice_yield_features_long.csv` (long format with 21 windows
per district-year covering Jan 1 - Dec 2) and produces one .npz file holding:

  X            : (N, 12, 8) float32  — Kharif feature sequence (DOY 145..321)
  y            : (N,)        float32 — yield (tonne/ha)
  area         : (N,)        float32 — sown area (hectare)
  prod         : (N,)        float32 — production (tonnes)
  year         : (N,)        int16   — harvest_year (2001..2016)
  district_id  : (N,)        |S20    — IND.xx.yy_1 (master CSV id)
  district_idx : (N,)        int16   — 0..100 master-order index
  state_name   : (N,)        |S20
  district_name: (N,)        |S40
  window_doy   : (12,)       int16   — [145, 161, ..., 321]
  feature_names: (8,)        |S8     — ['ndvi','evi','nirv','sif','tmin','tmax','srad','pr']

NaNs in features are handled by per-(district,year) linear interpolation along
the time axis; any remaining NaNs (e.g. fully-NaN sequences) are filled with
the per-feature train-time mean LATER (during training-split normalization),
not here — so the npz preserves any NaN that interpolation couldn't fix and
the split logic can mark/handle them.
"""

import os
import numpy as np
import pandas as pd


CSV_PATH = "/media/sam/writable/Sam Rice Yield Pred/data/unified/rice_yield_features_long.csv"
OUT_PATH = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_kharif.npz"
MASTER_CSV = "/media/sam/writable/Sam Rice Yield Pred/district_index_map_IGP_LOYO_101_with_names.csv"

# Liu et al Kharif window: May 25 (DOY 145) to Dec 2 (DOY 336 = window starting DOY 321 + 16)
# DOY 145, 161, 177, 193, 209, 225, 241, 257, 273, 289, 305, 321  -> 12 windows
KHARIF_DOYS = list(range(145, 322, 16))
assert len(KHARIF_DOYS) == 12

FEATURES = ("ndvi", "evi", "nirv", "sif", "tmin", "tmax", "srad", "pr")


def main():
    print("Reading long-format dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print("  shape:", df.shape)

    # Filter to Kharif windows
    df = df[df["window_doy"].isin(KHARIF_DOYS)].copy()
    print("  after Kharif filter:", df.shape)

    # Establish district_index using the master CSV ordering (matches build done in Cell 28)
    master = pd.read_csv(MASTER_CSV).sort_values("district_number").reset_index(drop=True)
    master["district_index"] = np.arange(len(master), dtype=np.int16)
    id_to_idx = dict(zip(master["district_id"], master["district_index"]))
    df["district_idx"] = df["district_id"].map(id_to_idx).astype(np.int16)

    # Sort so each (district, year) block is contiguous in window_doy order
    df = df.sort_values(["district_idx", "harvest_year", "window_doy"]).reset_index(drop=True)

    # Group by (district, year) and build the (12, 8) feature matrix
    n_districts = master.shape[0]
    years = sorted(df["harvest_year"].unique())
    n_years = len(years)
    print("Districts: {}, Years: {} ({}..{})".format(n_districts, n_years, years[0], years[-1]))

    N = n_districts * n_years
    X = np.full((N, len(KHARIF_DOYS), len(FEATURES)), np.nan, dtype=np.float32)
    y = np.full((N,), np.nan, dtype=np.float32)
    area = np.full((N,), np.nan, dtype=np.float32)
    prod = np.full((N,), np.nan, dtype=np.float32)
    year_arr = np.full((N,), 0, dtype=np.int16)
    didx_arr = np.full((N,), -1, dtype=np.int16)
    dist_id_arr = np.full((N,), b"", dtype="|S20")
    state_arr = np.full((N,), b"", dtype="|S20")
    dist_name_arr = np.full((N,), b"", dtype="|S40")

    doy_to_col = {doy: i for i, doy in enumerate(KHARIF_DOYS)}

    grouped = df.groupby(["district_idx", "harvest_year"])
    sample_idx = 0
    for (didx, yr), g in grouped:
        # Should have 12 rows (one per Kharif window)
        if len(g) != len(KHARIF_DOYS):
            print("  WARN: (district_idx={}, year={}) has {} rows (expected {})".format(
                didx, yr, len(g), len(KHARIF_DOYS)))
        meta = g.iloc[0]
        year_arr[sample_idx] = int(yr)
        didx_arr[sample_idx] = int(didx)
        dist_id_arr[sample_idx] = str(meta["district_id"]).encode()
        state_arr[sample_idx] = str(meta["state_name"]).encode()
        dist_name_arr[sample_idx] = str(meta["district_name"]).encode()
        y[sample_idx] = float(meta["yield_tonne_per_hectare"])
        area[sample_idx] = float(meta["area_hectare"])
        prod[sample_idx] = float(meta["production_tonnes"])

        for _, row in g.iterrows():
            col = doy_to_col[int(row["window_doy"])]
            for f, feat in enumerate(FEATURES):
                X[sample_idx, col, f] = float(row[feat])
        sample_idx += 1

    # Trim if some (district, year) combinations were missing
    if sample_idx != N:
        print("  WARN: {} sample slots, only {} filled".format(N, sample_idx))
        X = X[:sample_idx]
        y = y[:sample_idx]
        area = area[:sample_idx]
        prod = prod[:sample_idx]
        year_arr = year_arr[:sample_idx]
        didx_arr = didx_arr[:sample_idx]
        dist_id_arr = dist_id_arr[:sample_idx]
        state_arr = state_arr[:sample_idx]
        dist_name_arr = dist_name_arr[:sample_idx]

    print("Stack shape :", X.shape)
    print("Yield range :", "{:.2f} .. {:.2f} (mean={:.2f})".format(
        np.nanmin(y), np.nanmax(y), np.nanmean(y)))

    # ------------ NaN handling: linear interpolation along time per (sample, feature) ------------
    print("\nNaN cells BEFORE interpolation: {} / {}".format(
        int(np.isnan(X).sum()), X.size))
    # Per sample, per feature: 1-D linear interp over the 12 time points
    for i in range(X.shape[0]):
        for f in range(X.shape[2]):
            col = X[i, :, f]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue
            if nan_mask.all():
                continue  # leave fully NaN; will be filled at training time
            valid = ~nan_mask
            xs = np.arange(len(col))
            X[i, nan_mask, f] = np.interp(xs[nan_mask], xs[valid], col[valid])
    print("NaN cells AFTER  interpolation: {} / {}".format(
        int(np.isnan(X).sum()), X.size))

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        X=X, y=y, area=area, prod=prod,
        year=year_arr, district_idx=didx_arr,
        district_id=dist_id_arr, state_name=state_arr, district_name=dist_name_arr,
        window_doy=np.array(KHARIF_DOYS, dtype=np.int16),
        feature_names=np.array(FEATURES, dtype="|S8"),
    )
    print("\nWrote {} ({:.2f} MB)".format(OUT_PATH, os.path.getsize(OUT_PATH) / (1024 * 1024)))


if __name__ == "__main__":
    main()

"""Build the full Jan 1 - Dec 2 LOYO dataset (21 16-day windows).

This is the same content as rice_yield_features_long.csv, just reshaped to a
fixed numpy array with shape (1616, 21, 8). Required by Informer1 since it
sees 9 pre-Kharif windows plus k observed Kharif windows.

Produces `data/unified/loyo_full21.npz` with:
  X            : (N, 21, 8) float32
  y            : (N,) float32 yield (tonne/ha)
  area, prod   : (N,) float32
  year         : (N,) int16
  district_idx : (N,) int16
  district_id  : (N,) |S20
  state_name   : (N,) |S20
  district_name: (N,) |S40
  window_doy   : (21,) int16  -- [1, 17, ..., 321]
  kharif_start : int -- index where Kharif begins (9, i.e. DOY 145)
  feature_names: (8,) |S8

NaN cells in features are linear-interpolated along time per-sample-per-feature,
identical to build_dataset.py.
"""

import os
import numpy as np
import pandas as pd


CSV_PATH    = "/media/sam/writable/Sam Rice Yield Pred/data/unified/rice_yield_features_long.csv"
OUT_PATH    = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_full21.npz"
MASTER_CSV  = "/media/sam/writable/Sam Rice Yield Pred/district_index_map_IGP_LOYO_101_with_names.csv"

ALL_DOYS = list(range(1, 322, 16))    # [1, 17, ..., 321]  -> 21 windows
assert len(ALL_DOYS) == 21
KHARIF_START_DOY = 145
KHARIF_START_IDX = ALL_DOYS.index(KHARIF_START_DOY)   # 9 -> windows 9..20 are Kharif (12 of them)

FEATURES = ("ndvi", "evi", "nirv", "sif", "tmin", "tmax", "srad", "pr")


def main():
    print("Reading long-format dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print("  shape:", df.shape)

    master = pd.read_csv(MASTER_CSV).sort_values("district_number").reset_index(drop=True)
    master["district_index"] = np.arange(len(master), dtype=np.int16)
    id_to_idx = dict(zip(master["district_id"], master["district_index"]))
    df["district_idx"] = df["district_id"].map(id_to_idx).astype(np.int16)

    df = df.sort_values(["district_idx", "harvest_year", "window_doy"]).reset_index(drop=True)
    n_districts = master.shape[0]
    years = sorted(df["harvest_year"].unique())
    n_years = len(years)
    N = n_districts * n_years
    T = len(ALL_DOYS)
    F = len(FEATURES)
    print("Districts: {}, Years: {} ({}..{}), windows/sample: {}".format(
        n_districts, n_years, years[0], years[-1], T))

    X = np.full((N, T, F), np.nan, dtype=np.float32)
    y = np.full((N,), np.nan, dtype=np.float32)
    area = np.full((N,), np.nan, dtype=np.float32)
    prod = np.full((N,), np.nan, dtype=np.float32)
    year_arr = np.zeros((N,), dtype=np.int16)
    didx_arr = np.full((N,), -1, dtype=np.int16)
    dist_id_arr = np.full((N,), b"", dtype="|S20")
    state_arr = np.full((N,), b"", dtype="|S20")
    dist_name_arr = np.full((N,), b"", dtype="|S40")
    doy_to_col = {doy: i for i, doy in enumerate(ALL_DOYS)}

    idx = 0
    for (didx, yr), g in df.groupby(["district_idx", "harvest_year"]):
        meta = g.iloc[0]
        year_arr[idx] = int(yr); didx_arr[idx] = int(didx)
        dist_id_arr[idx] = str(meta["district_id"]).encode()
        state_arr[idx] = str(meta["state_name"]).encode()
        dist_name_arr[idx] = str(meta["district_name"]).encode()
        y[idx] = float(meta["yield_tonne_per_hectare"])
        area[idx] = float(meta["area_hectare"])
        prod[idx] = float(meta["production_tonnes"])
        for _, row in g.iterrows():
            col = doy_to_col[int(row["window_doy"])]
            for f, feat in enumerate(FEATURES):
                X[idx, col, f] = float(row[feat])
        idx += 1
    if idx != N:
        print("  WARN: filled {} of {} expected slots".format(idx, N))
        X = X[:idx]; y = y[:idx]; area = area[:idx]; prod = prod[:idx]
        year_arr = year_arr[:idx]; didx_arr = didx_arr[:idx]
        dist_id_arr = dist_id_arr[:idx]; state_arr = state_arr[:idx]
        dist_name_arr = dist_name_arr[:idx]

    print("Stack shape:", X.shape)
    print("NaN before interp:", int(np.isnan(X).sum()))
    for i in range(X.shape[0]):
        for f in range(X.shape[2]):
            col = X[i, :, f]
            m = np.isnan(col)
            if not m.any(): continue
            if m.all(): continue
            valid = ~m
            xs = np.arange(len(col))
            X[i, m, f] = np.interp(xs[m], xs[valid], col[valid])
    print("NaN after  interp:", int(np.isnan(X).sum()))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        X=X, y=y, area=area, prod=prod,
        year=year_arr, district_idx=didx_arr,
        district_id=dist_id_arr, state_name=state_arr, district_name=dist_name_arr,
        window_doy=np.array(ALL_DOYS, dtype=np.int16),
        kharif_start=np.array([KHARIF_START_IDX], dtype=np.int16),
        feature_names=np.array(FEATURES, dtype="|S8"),
    )
    print("\nWrote {} ({:.2f} MB)".format(OUT_PATH, os.path.getsize(OUT_PATH) / (1024 * 1024)))
    print("Kharif starts at window index {} (DOY {})".format(KHARIF_START_IDX, KHARIF_START_DOY))


if __name__ == "__main__":
    main()

"""Subprocess worker: read one MOD13A2 16-day window (4 tiles), build a
4800x4800 mosaic at 500 m (upsampled from 1 km), apply QC + scale + NIRV,
and save to disk as a .npz file. Designed to be invoked as:

    python mod13a2_worker.py <year> <doy> <output_npz> [qc_max=1]

This keeps each pyhdf SD/SDS open/close cycle isolated to a short-lived
process so HDF4's internal file-slot table can't accumulate stale state
across many windows in a long-running parent (Jupyter) process.
"""

import sys
import glob
import os
import numpy as np
from pyhdf.SD import SD, SDC


DATA_ROOT = "/media/sam/writable/Sam Rice Yield Pred/data"
MOSAIC_SHAPE = (4800, 4800)
# (row_start, col_start) at 500 m for each tile
TILE_POS = {
    "h24v05": (0, 0),
    "h25v05": (0, 2400),
    "h24v06": (2400, 0),
    "h25v06": (2400, 2400),
}


def _find_hdf(tile, year, doy):
    pattern = "*.A{:04d}{:03d}.{}.061.*.hdf".format(year, doy, tile)
    matches = glob.glob(os.path.join(DATA_ROOT, "MOD13A2", tile, pattern))
    return matches[0] if matches else None


def _upsample_2x(arr):
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


def main():
    year = int(sys.argv[1])
    doy = int(sys.argv[2])
    out_path = sys.argv[3]
    qc_max = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    ndvi_mos = np.full(MOSAIC_SHAPE, np.nan, dtype=np.float32)
    evi_mos = np.full(MOSAIC_SHAPE, np.nan, dtype=np.float32)
    nirv_mos = np.full(MOSAIC_SHAPE, np.nan, dtype=np.float32)

    for tile, (r0, c0) in TILE_POS.items():
        hdf_path = _find_hdf(tile, year, doy)
        if hdf_path is None:
            continue
        hdf = SD(hdf_path, SDC.READ)
        try:
            ndvi_i = hdf.select("1 km 16 days NDVI").get()
            evi_i = hdf.select("1 km 16 days EVI").get()
            nir_i = hdf.select("1 km 16 days NIR reflectance").get()
            pr = hdf.select("1 km 16 days pixel reliability").get()
        finally:
            hdf.end()

        ndvi = ndvi_i.astype(np.float32)
        ndvi[ndvi_i == -3000] = np.nan
        ndvi /= 10000.0
        evi = evi_i.astype(np.float32)
        evi[evi_i == -3000] = np.nan
        evi /= 10000.0
        nir = nir_i.astype(np.float32)
        nir[nir_i == -1000] = np.nan
        nir /= 10000.0

        bad = (pr > qc_max) | (pr < 0)
        ndvi[bad] = np.nan
        evi[bad] = np.nan
        nir[bad] = np.nan
        # NIRV at 1 km native, BEFORE upsampling
        nirv = nir * ndvi

        ndvi_mos[r0 : r0 + 2400, c0 : c0 + 2400] = _upsample_2x(ndvi)
        evi_mos[r0 : r0 + 2400, c0 : c0 + 2400] = _upsample_2x(evi)
        nirv_mos[r0 : r0 + 2400, c0 : c0 + 2400] = _upsample_2x(nirv)

    np.savez(out_path, ndvi=ndvi_mos, evi=evi_mos, nirv=nirv_mos)


if __name__ == "__main__":
    main()

"""Subprocess worker: read MCD12Q1 LC_Type1 for one year, build the 4800x4800
cropland mask (IGBP classes 12 OR 14), save as a .npz file. Invoked as:

    python mcd12q1_worker.py <year> <output_npz>
"""

import sys
import glob
import os
import numpy as np
from pyhdf.SD import SD, SDC


DATA_ROOT = "/media/sam/writable/Sam Rice Yield Pred/data"
MOSAIC_SHAPE = (4800, 4800)
TILE_POS = {
    "h24v05": (0, 0),
    "h25v05": (0, 2400),
    "h24v06": (2400, 0),
    "h25v06": (2400, 2400),
}
CROPLAND_CLASSES = (12, 14)


def main():
    year = int(sys.argv[1])
    out_path = sys.argv[2]
    mask = np.zeros(MOSAIC_SHAPE, dtype=np.uint8)  # uint8 instead of bool to be safe
    for tile, (r0, c0) in TILE_POS.items():
        tile_dir = os.path.join(DATA_ROOT, "MCD12Q1", tile)
        matches = glob.glob(
            os.path.join(tile_dir, "*.A{:04d}001.{}.061.*.hdf".format(year, tile))
        )
        if not matches:
            raise FileNotFoundError("MCD12Q1 missing for {} {}".format(tile, year))
        hdf = SD(matches[0], SDC.READ)
        try:
            lc = hdf.select("LC_Type1").get()
        finally:
            hdf.end()
        is_crop = np.isin(lc, CROPLAND_CLASSES).astype(np.uint8)
        mask[r0 : r0 + 2400, c0 : c0 + 2400] = is_crop
    np.savez(out_path, cropland=mask)


if __name__ == "__main__":
    main()

"""
Examine CSIF metadata files from OSF (https://osf.io/8xqy6/).

This script parses the three DataCite JSON metadata files representing the
three CSIF product variants available on OSF, and summarizes their
characteristics to inform our download decision.

Usage:
    python src/examine_csif_metadata.py
"""

import json
import os
import re
from datetime import datetime, timedelta

# Project root
PROJECT_ROOT = "/media/sam/writable/Sam Rice Yield Pred"

# Metadata files we have on disk (one sample from each CSIF variant)
METADATA_FILES = [
    "7yag9-datacite.json",
    "b3rym-datacite.json",
    "pk2u4-datacite.json",
]


def parse_csif_filename(fname):
    """
    Parse a CSIF NetCDF filename into its components.
    Pattern: OCO2.SIF.{sky}.{temporal}.YYYYDDD[.vN].nc
    Example: OCO2.SIF.clear.inst.2000049.v2.nc
    """
    pattern = (
        r"^OCO2\.SIF\."
        r"(?P<sky>clear|all)\."
        r"(?P<temporal>inst|daily)\."
        r"(?P<year>\d{4})(?P<doy>\d{3})"
        r"(?:\.v(?P<version>\d+))?"
        r"\.nc$"
    )
    m = re.match(pattern, fname)
    if not m:
        return None
    parts = m.groupdict()
    year = int(parts["year"])
    doy = int(parts["doy"])
    parts["date"] = datetime(year, 1, 1) + timedelta(days=doy - 1)
    parts["version"] = parts["version"] if parts["version"] else "1"
    return parts


def load_metadata(path):
    """Load a DataCite JSON file and extract the file title."""
    with open(path, "r") as f:
        data = json.load(f)
    title = data["titles"][0]["title"]
    created = next(
        (d["date"] for d in data.get("dates", []) if d["dateType"] == "Created"),
        None,
    )
    osf_url = data["identifier"]["identifier"]
    return {
        "title": title,
        "created": created,
        "osf_url": osf_url,
    }


def main():
    print("=" * 70)
    print("CSIF Metadata Examination")
    print("=" * 70)

    variants = []
    for meta_file in METADATA_FILES:
        path = os.path.join(PROJECT_ROOT, meta_file)
        if not os.path.exists(path):
            print("MISSING: {}".format(path))
            continue
        meta = load_metadata(path)
        parsed = parse_csif_filename(meta["title"])
        variants.append({"meta": meta, "parsed": parsed, "source": meta_file})

    print("\nVariants discovered:\n")
    for v in variants:
        m, p, src = v["meta"], v["parsed"], v["source"]
        print("-" * 70)
        print("Source metadata file : {}".format(src))
        print("OSF page             : {}".format(m["osf_url"]))
        print("Sample filename      : {}".format(m["title"]))
        print("Sky condition        : {}".format(p["sky"]))
        print("Temporal type        : {}".format(p["temporal"]))
        print("Sample date          : {}".format(p["date"].strftime("%Y-%m-%d")))
        print("Version              : v{}".format(p["version"]))
        print("Published            : {}".format(m["created"]))

    # Show 4-day pattern check
    print("\n" + "=" * 70)
    print("4-day composite verification")
    print("=" * 70)
    print("CSIF files use Julian day suffix. Expected 4-day pattern: 001, 005, 009...")
    for v in variants:
        doy = int(v["parsed"]["doy"])
        is_4day = (doy - 1) % 4 == 0
        print("  {}: DOY={} -> 4-day grid? {}".format(
            v["meta"]["title"], doy, "YES" if is_4day else "NO"
        ))

    # Recommendation
    print("\n" + "=" * 70)
    print("Recommendation")
    print("=" * 70)
    print("Two product types exist:")
    print("  1. clear.inst (v2)  -- clear-sky only, instantaneous, has gaps")
    print("  2. all.daily        -- gap-filled, daily-averaged, smoother")
    print("")
    print("For crop yield prediction over 16 years in monsoon-affected IGP")
    print("(many cloudy days), the all-sky gap-filled product is typically")
    print("preferred for time-series consistency.")
    print("")
    print("Liu et al. (2022) did NOT specify which variant they used.")
    print("Awaiting user decision before proceeding.")


if __name__ == "__main__":
    main()

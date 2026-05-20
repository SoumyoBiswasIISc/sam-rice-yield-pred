"""Line-plot comparison with all four metrics rendered on a unified 0-100 %
y-axis, so the *relative* gap between our replication and Liu et al's
published numbers is visible at one glance.

Transformations:
  R²       -> R² * 100                                 (correlation squared, %)
  RMSE     -> 100 * RMSE / mean(observed yield)        (Normalized RMSE, % of mean)
  MAPE     -> already %                                (unchanged)
  d        -> d * 100                                  (Willmott's d, %)

Reads:
  data/unified/loyo_results/loyo_results_liu_metrics_bestof3.json (ours)
  DEBUG_liu_all.csv (Liu et al)

Writes 5 PNGs to data/unified/loyo_results/figures/:
  lineplot_pct_R2.png
  lineplot_pct_RMSE.png
  lineplot_pct_MAPE.png
  lineplot_pct_d.png
  lineplot_pct_all_metrics.png  (2x2)
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUR_JSON = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/loyo_results_liu_metrics_bestof3.json"
LIU_CSV = "/media/sam/writable/Sam Rice Yield Pred/DEBUG_liu_all.csv"
DATASET_NPZ = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_kharif.npz"
OUT_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_ours():
    with open(OUR_JSON) as f:
        res = json.load(f)
    years = sorted(int(y) for y in res["folds"])
    rows = []
    for y in years:
        m = res["folds"][str(y)]["best_metrics"]
        rows.append({"Year": y, "R2": m["r2"], "RMSE": m["rmse"],
                     "MAPE": m["mape"], "d": m["d"]})
    return pd.DataFrame(rows).set_index("Year")


def load_liu():
    df = pd.read_csv(LIU_CSV).dropna(how="all")
    df["Year"] = df["Year"].astype(int)
    return df.set_index("Year")


def line_plot(ax, years, liu_vals, our_vals, title, ylabel):
    ax.plot(years, liu_vals, marker="o", markersize=8, linewidth=2,
            color="#4C72B0", label="Liu et al (published)")
    ax.plot(years, our_vals, marker="s", markersize=8, linewidth=2,
            color="#DD8452", label="Ours (replication)")
    ax.axhline(np.mean(liu_vals), color="#4C72B0", linestyle="--", alpha=0.4,
               label="Liu mean = {:.1f}%".format(np.mean(liu_vals)))
    ax.axhline(np.mean(our_vals), color="#DD8452", linestyle="--", alpha=0.4,
               label="Ours mean = {:.1f}%".format(np.mean(our_vals)))
    ax.set_xticks(years)
    ax.set_xlabel("Test year (LOYO held-out)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", ncol=2, fontsize=9)
    # Annotate per-point
    for yr, v in zip(years, liu_vals):
        ax.annotate("{:.1f}".format(v), xy=(yr, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#4C72B0")
    for yr, v in zip(years, our_vals):
        ax.annotate("{:.1f}".format(v), xy=(yr, v), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#DD8452")


def main():
    ours = load_ours()
    liu = load_liu()
    years = sorted(set(ours.index) & set(liu.index))
    print("Years in common:", years)

    # Compute mean observed yield from the dataset, restricted to those years
    data = np.load(DATASET_NPZ)
    yr_arr = data["year"]; y_arr = data["y"]
    mean_yield = float(np.mean([y_arr[yr_arr == y].mean() for y in years]))
    print("Mean observed yield across LOYO years: {:.3f} tonne/ha".format(mean_yield))

    # Build percentage-scale dataframes
    pct = {}
    pct["R2"]   = (liu["R2"]   * 100.0).rename("Liu"), (ours["R2"]   * 100.0).rename("Ours")
    pct["RMSE"] = (liu["RMSE"] * 100.0 / mean_yield).rename("Liu"), (ours["RMSE"] * 100.0 / mean_yield).rename("Ours")
    pct["MAPE"] = (liu["MAPE"]).rename("Liu"),         (ours["MAPE"]).rename("Ours")
    pct["d"]    = (liu["d"]    * 100.0).rename("Liu"), (ours["d"]    * 100.0).rename("Ours")

    metric_meta = [
        ("R2",   "R² × 100",
         "R² (Pearson² × 100) [%]",
         "LOYO R² across test years (×100) — Liu et al vs Ours"),
        ("RMSE", "Normalized RMSE",
         "Normalized RMSE = 100 · RMSE / mean(y_obs)  [%]",
         "LOYO Normalized RMSE across test years — Liu et al vs Ours"),
        ("MAPE", "MAPE",
         "MAPE [%]",
         "LOYO MAPE across test years — Liu et al vs Ours"),
        ("d",    "Willmott's d × 100",
         "Willmott's index of agreement (× 100) [%]",
         "LOYO Willmott's d across test years (×100) — Liu et al vs Ours"),
    ]

    for col, short, ylabel, title in metric_meta:
        liu_vals = pct[col][0].loc[years].values
        our_vals = pct[col][1].loc[years].values
        fig, ax = plt.subplots(figsize=(11, 6))
        line_plot(ax, years, liu_vals, our_vals, title=title, ylabel=ylabel)
        out = os.path.join(OUT_DIR, "lineplot_pct_{}.png".format(col))
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("Wrote", out)

    # 2x2 combined
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for (col, short, ylabel, title), ax in zip(metric_meta, axes.flat):
        liu_vals = pct[col][0].loc[years].values
        our_vals = pct[col][1].loc[years].values
        line_plot(ax, years, liu_vals, our_vals, title=short, ylabel=ylabel)
    fig.suptitle(
        "Rice yield prediction (Kharif Informer, LOYO) — Liu et al vs Ours\n"
        "All four metrics on a unified 0–100% scale  (RMSE shown as % of mean observed yield = {:.2f} t/ha)".format(mean_yield),
        fontsize=13, y=0.99,
    )
    out = os.path.join(OUT_DIR, "lineplot_pct_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()

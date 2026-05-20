"""Line-plot comparison of our LOYO Informer metrics vs Liu et al's published
metrics, with LOYO test year on the x-axis and the metric on the y-axis.

Reads:
  - data/unified/loyo_results/loyo_results_liu_metrics_bestof3.json (ours)
  - DEBUG_liu_all.csv (Liu et al)

Writes 5 PNGs to data/unified/loyo_results/figures/:
  - lineplot_R2.png
  - lineplot_RMSE.png
  - lineplot_MAPE.png
  - lineplot_d.png
  - lineplot_all_metrics.png  (2x2 panel)
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


def line_plot(ax, years, liu_vals, our_vals, title, ylabel, fmt="{:.3f}"):
    ax.plot(years, liu_vals, marker="o", markersize=8, linewidth=2,
            color="#4C72B0", label="Liu et al (published)")
    ax.plot(years, our_vals, marker="s", markersize=8, linewidth=2,
            color="#DD8452", label="Ours (replication)")
    # Mean horizontal reference lines (light)
    ax.axhline(np.mean(liu_vals), color="#4C72B0", linestyle="--", alpha=0.4,
               label="Liu mean = {}".format(fmt.format(np.mean(liu_vals))))
    ax.axhline(np.mean(our_vals), color="#DD8452", linestyle="--", alpha=0.4,
               label="Ours mean = {}".format(fmt.format(np.mean(our_vals))))
    ax.set_xticks(years)
    ax.set_xlabel("Test year (LOYO held-out)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    # annotate per-point
    for yr, v in zip(years, liu_vals):
        ax.annotate(fmt.format(v), xy=(yr, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#4C72B0")
    for yr, v in zip(years, our_vals):
        ax.annotate(fmt.format(v), xy=(yr, v), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#DD8452")


def main():
    ours = load_ours()
    liu = load_liu()
    years = sorted(set(ours.index) & set(liu.index))
    print("Years in common:", years)

    metric_meta = [
        ("R2",   "R²",            "R² (Pearson² between observed & predicted)", "{:.3f}"),
        ("RMSE", "RMSE",          "RMSE (tonne/ha)",                              "{:.3f}"),
        ("MAPE", "MAPE",          "MAPE (%)",                                     "{:.1f}"),
        ("d",    "Willmott's d",  "Willmott's index of agreement (d)",            "{:.3f}"),
    ]

    # Individual plots
    for col, short, ylabel, fmt in metric_meta:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        line_plot(
            ax, years,
            liu.loc[years, col].values, ours.loc[years, col].values,
            title="LOYO {} across test years — Liu et al vs Ours".format(short),
            ylabel=ylabel, fmt=fmt,
        )
        out = os.path.join(OUT_DIR, "lineplot_{}.png".format(col))
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("Wrote", out)

    # 2x2 combined
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for (col, short, ylabel, fmt), ax in zip(metric_meta, axes.flat):
        line_plot(
            ax, years,
            liu.loc[years, col].values, ours.loc[years, col].values,
            title="{}".format(short), ylabel=ylabel, fmt=fmt,
        )
    fig.suptitle("Rice yield prediction (Kharif Informer, LOYO) — Liu et al vs Ours",
                 fontsize=14, y=0.99)
    out = os.path.join(OUT_DIR, "lineplot_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()

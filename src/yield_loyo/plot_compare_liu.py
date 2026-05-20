"""Compare our LOYO Informer metrics vs Liu et al's published metrics.

Reads:
  - data/unified/loyo_results/loyo_results_liu_metrics_bestof3.json (ours)
  - DEBUG_liu_all.csv (Liu et al's published numbers)

Writes 5 PNGs to data/unified/loyo_results/figures/:
  - comparison_R2.png
  - comparison_RMSE.png
  - comparison_MAPE.png
  - comparison_d.png
  - comparison_all_metrics.png  (2x2 panel of all four)
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


def grouped_bar(ax, years, liu_vals, our_vals, title, ylabel, fmt="{:.3f}"):
    x = np.arange(len(years))
    width = 0.38
    bars_liu = ax.bar(x - width / 2, liu_vals, width, label="Liu et al (published)",
                       color="#4C72B0", edgecolor="black", linewidth=0.5)
    bars_our = ax.bar(x + width / 2, our_vals, width, label="Ours (replication)",
                       color="#DD8452", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Test year (LOYO)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    # Annotate bars with values
    for bar, v in list(zip(bars_liu, liu_vals)) + list(zip(bars_our, our_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt.format(v), ha="center", va="bottom", fontsize=8)


def main():
    ours = load_ours()
    liu = load_liu()
    years = sorted(set(ours.index) & set(liu.index))
    print("Years in common:", years)

    # Per-year mean + values
    print("\nLiu et al means:", liu.mean().round(3).to_dict())
    print("Our  means     :", ours.mean().round(3).to_dict())

    metric_meta = [
        ("R2",   "R²",                 "R² (Pearson² between observed & predicted)", "{:.3f}"),
        ("RMSE", "RMSE",               "RMSE (tonne/ha)",                              "{:.3f}"),
        ("MAPE", "MAPE",               "MAPE (%)",                                     "{:.1f}"),
        ("d",    "Willmott's d",       "Willmott's index of agreement (d)",            "{:.3f}"),
    ]

    # Individual plots
    for col, short, ylabel, fmt in metric_meta:
        fig, ax = plt.subplots(figsize=(11, 5))
        grouped_bar(
            ax, years,
            liu.loc[years, col].values, ours.loc[years, col].values,
            title="LOYO {} — Liu et al vs Ours".format(short),
            ylabel=ylabel, fmt=fmt,
        )
        out = os.path.join(OUT_DIR, "comparison_{}.png".format(col))
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("Wrote", out)

    # 2x2 combined panel
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for (col, short, ylabel, fmt), ax in zip(metric_meta, axes.flat):
        grouped_bar(
            ax, years,
            liu.loc[years, col].values, ours.loc[years, col].values,
            title="{}".format(short), ylabel=ylabel, fmt=fmt,
        )
    fig.suptitle("Rice yield prediction (Kharif Informer, LOYO) — Liu et al vs Ours",
                 fontsize=14, y=0.99)
    out = os.path.join(OUT_DIR, "comparison_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()

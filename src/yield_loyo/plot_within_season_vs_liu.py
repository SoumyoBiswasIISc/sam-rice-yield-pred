"""Compare our within-season Informer metrics vs Liu et al's published ones.

X axis : k (number of known Kharif windows, 3..12)
Y axis : R² / RMSE / MAPE / d  (native units AND % rescaled to 0-100)

Reads:
  - data/unified/loyo_results/within_season_k3_to_k11.json (our k=3..11)
  - data/unified/loyo_results/loyo_results_liu_metrics_bestof3.json (our k=12)
  - liu_withinseason_extracted.csv (Liu et al, k=3..12)

Writes to data/unified/loyo_results/figures/:
  - lineplot_within_R2.png
  - lineplot_within_RMSE.png
  - lineplot_within_MAPE.png
  - lineplot_within_d.png
  - lineplot_within_all_metrics.png            (combined 2x2)
  - lineplot_within_pct_R2.png
  - lineplot_within_pct_RMSE.png
  - lineplot_within_pct_MAPE.png
  - lineplot_within_pct_d.png
  - lineplot_within_pct_all_metrics.png        (combined 2x2, unified 0-100% axis)
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WITHIN_JSON = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/within_season_k3_to_k11.json"
FULL_JSON   = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/loyo_results_liu_metrics_bestof3.json"
LIU_CSV     = "/media/sam/writable/Sam Rice Yield Pred/liu_withinseason_extracted.csv"
DATASET_NPZ = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_kharif.npz"
OUT_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_ours():
    """Return DataFrame indexed by k with columns R2, RMSE, MAPE, d (k=3..12)."""
    with open(WITHIN_JSON) as f:
        wr = json.load(f)
    rows = []
    for row in wr["summary_table"]:
        rows.append({
            "k": row["k"],
            "R2": row["fold_mean_r2"],
            "RMSE": row["fold_mean_rmse"],
            "MAPE": row["fold_mean_mape"],
            "d": row["fold_mean_d"],
        })
    # Add k=12 row from the after-season run
    with open(FULL_JSON) as f:
        fr = json.load(f)
    s = fr["per_fold_summary"]
    rows.append({
        "k": 12,
        "R2": s["r2_mean"],
        "RMSE": s["rmse_mean"],
        "MAPE": s["mape_mean"],
        "d": s["d_mean"],
    })
    return pd.DataFrame(rows).set_index("k").sort_index()


def load_liu():
    df = pd.read_csv(LIU_CSV).dropna(how="all")
    df["k"] = df["k"].astype(int)
    df = df.rename(columns={"Willmott_d": "d"})
    return df.set_index("k").sort_index()


def line_plot(ax, ks, liu_vals, our_vals, title, ylabel, fmt="{:.3f}",
              y_lim=None, y_ticks=None):
    ax.plot(ks, liu_vals, marker="o", markersize=8, linewidth=2,
            color="#4C72B0", label="Liu et al (published)")
    ax.plot(ks, our_vals, marker="s", markersize=8, linewidth=2,
            color="#DD8452", label="Ours (replication)")
    ax.axhline(np.mean(liu_vals), color="#4C72B0", linestyle="--", alpha=0.4,
               label="Liu mean = {}".format(fmt.format(np.mean(liu_vals))))
    ax.axhline(np.mean(our_vals), color="#DD8452", linestyle="--", alpha=0.4,
               label="Ours mean = {}".format(fmt.format(np.mean(our_vals))))
    ax.set_xticks(ks)
    ax.set_xlabel("k  (number of known Kharif 16-day windows)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_lim is not None: ax.set_ylim(*y_lim)
    if y_ticks is not None: ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2 if y_lim else 1)
    for k, v in zip(ks, liu_vals):
        ax.annotate(fmt.format(v), xy=(k, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#4C72B0")
    for k, v in zip(ks, our_vals):
        ax.annotate(fmt.format(v), xy=(k, v), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#DD8452")


def main():
    ours = load_ours()
    liu = load_liu()
    ks = sorted(set(ours.index) & set(liu.index))
    print("k values in common:", ks)
    print("\nOur means      :", ours.loc[ks].mean().round(3).to_dict())
    print("Liu et al means:", liu.loc[ks].mean().round(3).to_dict())

    # Native units
    metric_meta_nat = [
        ("R2",   "R²",            "R² (Pearson² × scalar)",                            "{:.3f}"),
        ("RMSE", "RMSE",          "RMSE (tonne/ha)",                                  "{:.3f}"),
        ("MAPE", "MAPE",          "MAPE (%)",                                         "{:.1f}"),
        ("d",    "Willmott's d",  "Willmott's index of agreement (d)",                "{:.3f}"),
    ]
    for col, short, ylabel, fmt in metric_meta_nat:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        line_plot(
            ax, ks,
            liu.loc[ks, col].values, ours.loc[ks, col].values,
            title="Within-season {} vs k — Liu et al vs Ours".format(short),
            ylabel=ylabel, fmt=fmt,
        )
        out = os.path.join(OUT_DIR, "lineplot_within_{}.png".format(col))
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("Wrote", out)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for (col, short, ylabel, fmt), ax in zip(metric_meta_nat, axes.flat):
        line_plot(
            ax, ks,
            liu.loc[ks, col].values, ours.loc[ks, col].values,
            title=short, ylabel=ylabel, fmt=fmt,
        )
    fig.suptitle("Within-season rice yield prediction — Liu et al vs Ours",
                 fontsize=14, y=0.99)
    out = os.path.join(OUT_DIR, "lineplot_within_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)

    # Percentage-scale (0..100 % y-axis)
    data = np.load(DATASET_NPZ)
    mean_yield = float(data["y"].mean())
    print("\nMean observed yield: {:.3f} t/ha (used to normalize RMSE)".format(mean_yield))

    pct = {}
    pct["R2"]   = liu["R2"]   * 100.0, ours["R2"]   * 100.0
    pct["RMSE"] = liu["RMSE"] * 100.0 / mean_yield, ours["RMSE"] * 100.0 / mean_yield
    pct["MAPE"] = liu["MAPE"], ours["MAPE"]
    pct["d"]    = liu["d"]    * 100.0, ours["d"]    * 100.0

    metric_meta_pct = [
        ("R2",   "R² × 100",            "R² (× 100) [%]",                             "{:.1f}"),
        ("RMSE", "Normalized RMSE",     "Normalized RMSE = 100 · RMSE / mean(y_obs) [%]", "{:.1f}"),
        ("MAPE", "MAPE",                "MAPE [%]",                                   "{:.1f}"),
        ("d",    "Willmott's d × 100",  "Willmott's d (× 100) [%]",                   "{:.1f}"),
    ]
    yticks = np.arange(0, 101, 10)
    for col, short, ylabel, fmt in metric_meta_pct:
        liu_vals = pct[col][0].loc[ks].values
        our_vals = pct[col][1].loc[ks].values
        fig, ax = plt.subplots(figsize=(11, 6))
        line_plot(
            ax, ks, liu_vals, our_vals,
            title="Within-season {} vs k — Liu et al vs Ours".format(short),
            ylabel=ylabel, fmt=fmt, y_lim=(0, 100), y_ticks=yticks,
        )
        out = os.path.join(OUT_DIR, "lineplot_within_pct_{}.png".format(col))
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("Wrote", out)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for (col, short, ylabel, fmt), ax in zip(metric_meta_pct, axes.flat):
        liu_vals = pct[col][0].loc[ks].values
        our_vals = pct[col][1].loc[ks].values
        line_plot(
            ax, ks, liu_vals, our_vals,
            title=short, ylabel=ylabel, fmt=fmt,
            y_lim=(0, 100), y_ticks=yticks,
        )
    fig.suptitle(
        "Within-season rice yield prediction — Liu et al vs Ours\n"
        "All four metrics on unified 0–100% scale (RMSE shown as % of mean observed yield = {:.2f} t/ha)".format(mean_yield),
        fontsize=13, y=0.99,
    )
    out = os.path.join(OUT_DIR, "lineplot_within_pct_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()

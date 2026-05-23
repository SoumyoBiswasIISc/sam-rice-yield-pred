"""5-line within-season comparison plot:
  - Liu et al (published)
  - Ours replicated (plain Informer, within-season)
  - Hybrid F-A (no noise)
  - Hybrid F-B (constant noise, train-only)
  - Hybrid F-C (increasing noise, train-only)

X axis : k (3..11)
Y axis : metric (R², RMSE, MAPE, d) — both native units AND % rescaled to 0-100

Outputs written to data/unified/loyo_results/figures/:
  lineplot_hybrid_R2.png / lineplot_hybrid_pct_R2.png
  lineplot_hybrid_RMSE.png / lineplot_hybrid_pct_RMSE.png
  lineplot_hybrid_MAPE.png / lineplot_hybrid_pct_MAPE.png
  lineplot_hybrid_d.png / lineplot_hybrid_pct_d.png
  lineplot_hybrid_all_metrics.png / lineplot_hybrid_pct_all_metrics.png
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results/figures"
RES_DIR = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_results"
LIU_CSV = "/media/sam/writable/Sam Rice Yield Pred/liu_withinseason_extracted.csv"
WITHIN_JSON = os.path.join(RES_DIR, "within_season_k3_to_k11.json")
DATASET_NPZ = "/media/sam/writable/Sam Rice Yield Pred/data/unified/loyo_kharif.npz"
os.makedirs(OUT_DIR, exist_ok=True)


def load_liu():
    df = pd.read_csv(LIU_CSV).dropna(how="all")
    df["k"] = df["k"].astype(int)
    df = df.rename(columns={"Willmott_d": "d"})
    return df.set_index("k").sort_index()


def load_ours_within():
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
    return pd.DataFrame(rows).set_index("k").sort_index()


def load_hybrid(framework):
    with open(os.path.join(RES_DIR, "hybrid_results_F{}.json".format(framework))) as f:
        r = json.load(f)
    rows = []
    for k_str, blk in r["by_k"].items():
        fm = blk["fold_mean"]
        rows.append({"k": int(k_str), "R2": fm["r2"], "RMSE": fm["rmse"],
                     "MAPE": fm["mape"], "d": fm["d"]})
    return pd.DataFrame(rows).set_index("k").sort_index()


SERIES_STYLES = {
    "Liu et al (published)":   ("#4C72B0", "o", "-"),
    "Ours (replicated Liu)":   ("#DD8452", "s", "-"),
    "Hybrid F-A (no noise)":   ("#55A868", "D", "-"),
    "Hybrid F-B (const noise)":("#C44E52", "^", "--"),
    "Hybrid F-C (incr. noise)":("#8172B2", "v", "--"),
}


def line_plot(ax, ks, series, title, ylabel, fmt="{:.3f}", y_lim=None, y_ticks=None):
    """series: list of (label, values_array) in the order to plot."""
    for label, vals in series:
        color, marker, ls = SERIES_STYLES[label]
        ax.plot(ks, vals, marker=marker, markersize=8, linewidth=2,
                color=color, linestyle=ls, label=label)
    ax.set_xticks(ks)
    ax.set_xlabel("k  (number of known Kharif 16-day windows)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_lim is not None: ax.set_ylim(*y_lim)
    if y_ticks is not None: ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def main():
    liu = load_liu()
    ours = load_ours_within()
    fa = load_hybrid("A"); fb = load_hybrid("B"); fc = load_hybrid("C")

    ks = sorted(set(liu.index) & set(ours.index) & set(fa.index) & set(fb.index) & set(fc.index))
    print("k values in common:", ks)

    # Print means
    print("\nMeans across k = {}-{}:".format(ks[0], ks[-1]))
    for name, df in [("Liu", liu), ("Ours", ours), ("F-A", fa), ("F-B", fb), ("F-C", fc)]:
        print("  {:>6s}: {}".format(name, df.loc[ks].mean().round(3).to_dict()))

    # ===== Native units =====
    metric_meta_nat = [
        ("R2",   "R²",            "R² (Pearson² × scalar)",                            "{:.3f}"),
        ("RMSE", "RMSE",          "RMSE (tonne/ha)",                                  "{:.3f}"),
        ("MAPE", "MAPE",          "MAPE (%)",                                         "{:.1f}"),
        ("d",    "Willmott's d",  "Willmott's index of agreement (d)",                "{:.3f}"),
    ]
    for col, short, ylabel, fmt in metric_meta_nat:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        series = [
            ("Liu et al (published)",   liu.loc[ks, col].values),
            ("Ours (replicated Liu)",   ours.loc[ks, col].values),
            ("Hybrid F-A (no noise)",   fa.loc[ks, col].values),
            ("Hybrid F-B (const noise)", fb.loc[ks, col].values),
            ("Hybrid F-C (incr. noise)", fc.loc[ks, col].values),
        ]
        line_plot(ax, ks, series,
                  title="Within-season {} vs k — 5-way comparison".format(short),
                  ylabel=ylabel, fmt=fmt)
        out = os.path.join(OUT_DIR, "lineplot_hybrid_{}.png".format(col))
        fig.tight_layout()
        fig.savefig(out, dpi=140); plt.close(fig)
        print("Wrote", out)

    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    for (col, short, ylabel, fmt), ax in zip(metric_meta_nat, axes.flat):
        series = [
            ("Liu et al (published)",   liu.loc[ks, col].values),
            ("Ours (replicated Liu)",   ours.loc[ks, col].values),
            ("Hybrid F-A (no noise)",   fa.loc[ks, col].values),
            ("Hybrid F-B (const noise)", fb.loc[ks, col].values),
            ("Hybrid F-C (incr. noise)", fc.loc[ks, col].values),
        ]
        line_plot(ax, ks, series, title=short, ylabel=ylabel, fmt=fmt)
    fig.suptitle("Within-season rice yield prediction — Liu vs Ours-plain vs Hybrid (F-A/B/C)",
                 fontsize=14, y=0.995)
    out = os.path.join(OUT_DIR, "lineplot_hybrid_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("Wrote", out)

    # ===== 0–100 % scaled =====
    data = np.load(DATASET_NPZ)
    mean_yield = float(data["y"].mean())
    print("\nMean observed yield (for normalized RMSE): {:.3f} t/ha".format(mean_yield))
    def transform(df):
        return pd.DataFrame({
            "R2":   df["R2"] * 100.0,
            "RMSE": df["RMSE"] * 100.0 / mean_yield,
            "MAPE": df["MAPE"],
            "d":    df["d"] * 100.0,
        })
    pct_liu = transform(liu); pct_ours = transform(ours)
    pct_fa = transform(fa); pct_fb = transform(fb); pct_fc = transform(fc)

    metric_meta_pct = [
        ("R2",   "R² × 100",            "R² (× 100) [%]",                              "{:.1f}"),
        ("RMSE", "Normalized RMSE",     "Normalized RMSE = 100 · RMSE / mean(y_obs) [%]", "{:.1f}"),
        ("MAPE", "MAPE",                "MAPE [%]",                                    "{:.1f}"),
        ("d",    "Willmott's d × 100",  "Willmott's d (× 100) [%]",                    "{:.1f}"),
    ]
    yticks = np.arange(0, 101, 10)
    for col, short, ylabel, fmt in metric_meta_pct:
        fig, ax = plt.subplots(figsize=(12, 7))
        series = [
            ("Liu et al (published)",   pct_liu.loc[ks, col].values),
            ("Ours (replicated Liu)",   pct_ours.loc[ks, col].values),
            ("Hybrid F-A (no noise)",   pct_fa.loc[ks, col].values),
            ("Hybrid F-B (const noise)", pct_fb.loc[ks, col].values),
            ("Hybrid F-C (incr. noise)", pct_fc.loc[ks, col].values),
        ]
        line_plot(ax, ks, series,
                  title="Within-season {} vs k — 5-way comparison".format(short),
                  ylabel=ylabel, fmt=fmt, y_lim=(0, 100), y_ticks=yticks)
        out = os.path.join(OUT_DIR, "lineplot_hybrid_pct_{}.png".format(col))
        fig.tight_layout()
        fig.savefig(out, dpi=140); plt.close(fig)
        print("Wrote", out)

    fig, axes = plt.subplots(2, 2, figsize=(17, 11.5))
    for (col, short, ylabel, fmt), ax in zip(metric_meta_pct, axes.flat):
        series = [
            ("Liu et al (published)",   pct_liu.loc[ks, col].values),
            ("Ours (replicated Liu)",   pct_ours.loc[ks, col].values),
            ("Hybrid F-A (no noise)",   pct_fa.loc[ks, col].values),
            ("Hybrid F-B (const noise)", pct_fb.loc[ks, col].values),
            ("Hybrid F-C (incr. noise)", pct_fc.loc[ks, col].values),
        ]
        line_plot(ax, ks, series, title=short, ylabel=ylabel, fmt=fmt,
                  y_lim=(0, 100), y_ticks=yticks)
    fig.suptitle(
        "Within-season rice yield prediction — 5-way comparison\n"
        "All four metrics on unified 0–100% scale (RMSE shown as % of mean observed yield = {:.2f} t/ha)".format(mean_yield),
        fontsize=13, y=0.995,
    )
    out = os.path.join(OUT_DIR, "lineplot_hybrid_pct_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()

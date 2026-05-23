"""Compare hybrid v1 (single global HP) vs v2 (per-k HP) alongside baselines.

Produces both native-units and 0–100% rescaled plots, saved under
data/unified/loyo_results/figures/.
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
        rows.append({"k": row["k"], "R2": row["fold_mean_r2"], "RMSE": row["fold_mean_rmse"],
                     "MAPE": row["fold_mean_mape"], "d": row["fold_mean_d"]})
    return pd.DataFrame(rows).set_index("k").sort_index()


def load_hybrid(framework, suffix=""):
    with open(os.path.join(RES_DIR, "hybrid_results_F{}{}.json".format(framework, suffix))) as f:
        r = json.load(f)
    rows = []
    for k_str, blk in r["by_k"].items():
        fm = blk["fold_mean"]
        rows.append({"k": int(k_str), "R2": fm["r2"], "RMSE": fm["rmse"],
                     "MAPE": fm["mape"], "d": fm["d"]})
    return pd.DataFrame(rows).set_index("k").sort_index()


SERIES_STYLES = {
    "Liu et al (published)":     ("#4C72B0", "o", "-", 2.0),
    "Ours (replicated Liu)":     ("#DD8452", "s", "-", 2.0),
    "Hybrid F-A v1 (k=6 HP)":     ("#55A868", "D", "-", 1.6),
    "Hybrid F-B v1 (k=6 HP)":     ("#C44E52", "^", "-", 1.6),
    "Hybrid F-C v1 (k=6 HP)":     ("#8172B2", "v", "-", 1.6),
    "Hybrid F-A v2 (per-k HP)":   ("#55A868", "D", "--", 1.6),
    "Hybrid F-B v2 (per-k HP)":   ("#C44E52", "^", "--", 1.6),
    "Hybrid F-C v2 (per-k HP)":   ("#8172B2", "v", "--", 1.6),
}


def line_plot(ax, ks, series, title, ylabel, y_lim=None, y_ticks=None):
    for label, vals in series:
        c, m, ls, lw = SERIES_STYLES[label]
        ax.plot(ks, vals, marker=m, markersize=6, linewidth=lw,
                color=c, linestyle=ls, label=label, alpha=0.95)
    ax.set_xticks(ks)
    ax.set_xlabel("k  (number of known Kharif 16-day windows)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_lim is not None: ax.set_ylim(*y_lim)
    if y_ticks is not None: ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)


def main():
    liu = load_liu()
    ours = load_ours_within()
    fa1 = load_hybrid("A"); fb1 = load_hybrid("B"); fc1 = load_hybrid("C")
    fa2 = load_hybrid("A", "_perk"); fb2 = load_hybrid("B", "_perk"); fc2 = load_hybrid("C", "_perk")
    ks = sorted(set(liu.index) & set(ours.index) & set(fa1.index) & set(fa2.index))
    print("Common k:", ks)
    for nm, df in [("Liu", liu), ("Ours-plain", ours),
                   ("F-A v1", fa1), ("F-B v1", fb1), ("F-C v1", fc1),
                   ("F-A v2", fa2), ("F-B v2", fb2), ("F-C v2", fc2)]:
        means = df.loc[ks].mean().to_dict()
        print("  {:<14s}  R²={:.3f}  RMSE={:.3f}  MAPE={:.2f}  d={:.3f}".format(
            nm, means["R2"], means["RMSE"], means["MAPE"], means["d"]))

    metric_meta_nat = [
        ("R2",   "R²",            "R² (Pearson² × scalar)",                            "{:.3f}"),
        ("RMSE", "RMSE",          "RMSE (tonne/ha)",                                  "{:.3f}"),
        ("MAPE", "MAPE",          "MAPE (%)",                                         "{:.1f}"),
        ("d",    "Willmott's d",  "Willmott's index of agreement (d)",                "{:.3f}"),
    ]

    def build_series(col):
        return [
            ("Liu et al (published)",   liu.loc[ks, col].values),
            ("Ours (replicated Liu)",   ours.loc[ks, col].values),
            ("Hybrid F-A v1 (k=6 HP)",   fa1.loc[ks, col].values),
            ("Hybrid F-B v1 (k=6 HP)",   fb1.loc[ks, col].values),
            ("Hybrid F-C v1 (k=6 HP)",   fc1.loc[ks, col].values),
            ("Hybrid F-A v2 (per-k HP)", fa2.loc[ks, col].values),
            ("Hybrid F-B v2 (per-k HP)", fb2.loc[ks, col].values),
            ("Hybrid F-C v2 (per-k HP)", fc2.loc[ks, col].values),
        ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for (col, short, ylabel, _), ax in zip(metric_meta_nat, axes.flat):
        line_plot(ax, ks, build_series(col), title=short, ylabel=ylabel)
    fig.suptitle("Hybrid v1 (k=6 global HP) vs v2 (per-k HP) vs baselines",
                 fontsize=14, y=0.995)
    out = os.path.join(OUT_DIR, "lineplot_hybrid_v1_vs_v2_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("Wrote", out)

    # Percentage version
    data = np.load(DATASET_NPZ)
    mean_yield = float(data["y"].mean())
    def trans(df):
        return pd.DataFrame({
            "R2": df["R2"] * 100.0, "RMSE": df["RMSE"] * 100.0 / mean_yield,
            "MAPE": df["MAPE"], "d": df["d"] * 100.0,
        })
    p = {nm: trans(df) for nm, df in [
        ("liu", liu), ("ours", ours),
        ("fa1", fa1), ("fb1", fb1), ("fc1", fc1),
        ("fa2", fa2), ("fb2", fb2), ("fc2", fc2),
    ]}
    metric_meta_pct = [
        ("R2",   "R² × 100",            "R² (× 100) [%]"),
        ("RMSE", "Normalized RMSE",     "100 · RMSE / mean(y_obs) [%]"),
        ("MAPE", "MAPE",                "MAPE [%]"),
        ("d",    "Willmott's d × 100",  "Willmott's d (× 100) [%]"),
    ]
    yticks = np.arange(0, 101, 10)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for (col, short, ylabel), ax in zip(metric_meta_pct, axes.flat):
        series = [
            ("Liu et al (published)",   p["liu"].loc[ks, col].values),
            ("Ours (replicated Liu)",   p["ours"].loc[ks, col].values),
            ("Hybrid F-A v1 (k=6 HP)",   p["fa1"].loc[ks, col].values),
            ("Hybrid F-B v1 (k=6 HP)",   p["fb1"].loc[ks, col].values),
            ("Hybrid F-C v1 (k=6 HP)",   p["fc1"].loc[ks, col].values),
            ("Hybrid F-A v2 (per-k HP)", p["fa2"].loc[ks, col].values),
            ("Hybrid F-B v2 (per-k HP)", p["fb2"].loc[ks, col].values),
            ("Hybrid F-C v2 (per-k HP)", p["fc2"].loc[ks, col].values),
        ]
        line_plot(ax, ks, series, title=short, ylabel=ylabel,
                  y_lim=(0, 100), y_ticks=yticks)
    fig.suptitle(
        "Hybrid v1 vs v2 vs baselines — unified 0–100% scale\n"
        "(RMSE shown as % of mean observed yield = {:.2f} t/ha)".format(mean_yield),
        fontsize=13, y=0.995,
    )
    out = os.path.join(OUT_DIR, "lineplot_hybrid_v1_vs_v2_pct_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()

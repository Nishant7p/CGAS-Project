"""F15: D3 steering curves from steering/run_bon.py (and run_revise.py when present).

F15  Does best-of-N reranking with the fidelity scorer make LLM recipes more faithful, and at what cost in diversity
     and novelty? Held-out metrics only (never the scorer itself), mean over regions with 95% bootstrap CIs.
F15b Which scorer component matters? (ablation at the largest N)

Usage (repo root): .venv/bin/python analysis/figures/d3_figures.py [--tag main]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STEER = REPO / "data" / "interim" / "steering"
PANELS = {
    "indep_classifier_acc": "Independent classifier:\nshare assigned to the claimed cuisine",
    "prevalence_jsd": "Prevalence JSD vs real\n(lower = closer)",
    "sai": "Stereotype Amplification Index\n(1 = as in real recipes)",
    "diversity_ratio": "Diversity ratio\n(unique ingredients vs real)",
    "novelty_mean": "Novelty: mean max-Jaccard\nto CulinaryDB (lower = more novel)",
}


def ci(v: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    style.setup()
    df = pd.read_parquet(STEER / f"bon_{args.tag}.parquet")
    comp = df[df["method"] == "composite"]
    models = [m for m in style.LABELS if m in set(comp["model"])]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.3 * len(PANELS), 3.6))
    for ax, (metric, title) in zip(axes, PANELS.items(), strict=True):
        for k, m in enumerate(models):
            g = comp[comp["model"] == m]
            ns = sorted(g["N"].unique())
            means, los, his = [], [], []
            for n in ns:
                v = g.loc[g["N"] == n, metric].dropna().to_numpy()
                means.append(v.mean())
                lo, hi = ci(v)
                los.append(lo)
                his.append(hi)
            x = np.array(ns, dtype=float) * (1 + 0.04 * (k - (len(models) - 1) / 2))
            ax.errorbar(x, means, yerr=[np.subtract(means, los), np.subtract(his, means)], fmt=style.MARKERS[m] + "-",
                        color=style.color(m), label=style.label(m), ms=4, lw=1.2, capsize=2)  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlabel("N (candidates per pick)")
        ax.set_title(title, fontsize=9)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=len(l_), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.12))
    n_sel = int(df["n_selected"].median())
    fig.suptitle(f"F15. Does best-of-N reranking with the fidelity scorer make recipes more faithful? "
                 f"({n_sel} recipes per region and N; mean over regions, 95% bootstrap CI)", y=1.04)  # fmt: skip
    style.save(fig, "F15_best_of_n")

    nmax = df["N"].max()
    abl = df[df["N"] == nmax]
    methods = ["composite", "without_classifier", "without_cooccurrence", "without_ns_consistency"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric in zip(axes, ["indep_classifier_acc", "prevalence_jsd"], strict=True):
        for k, m in enumerate(models):
            vals = [abl[(abl["model"] == m) & (abl["method"] == meth)][metric].mean() for meth in methods]
            ax.bar(np.arange(len(methods)) + (k - (len(models) - 1) / 2) * 0.25, vals, width=0.25,
                   color=style.color(m), label=style.label(m))  # fmt: skip
        ax.set_xticks(range(len(methods)), ["all 3", "− classifier", "− co-occurrence", "− Ns consistency"],
                      rotation=20, fontsize=8)  # fmt: skip
        ax.set_title(PANELS[metric].replace("\n", " "), fontsize=9)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=len(l_), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(f"F15b. Which scorer component drives the gain? (ablation at N = {nmax})", y=1.03)
    style.save(fig, "F15b_ablation")
    rev_path = STEER / f"revise_{args.tag}.parquet"
    if rev_path.exists():
        rev = pd.read_parquet(rev_path)
        metrics = {
            "indep_classifier_acc": "Independent classifier accuracy",
            "fingerprint_spearman": "Fingerprint ρ vs real",
            "prevalence_jsd": "Prevalence JSD (lower = closer)",
            "identity_jaccard": "Ingredients kept from the original\n(Jaccard)",
        }
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.4))
        for ax, (metric, title) in zip(axes, metrics.items(), strict=True):
            for m in [m for m in style.LABELS if m in set(rev["model"])]:
                g = rev[rev["model"] == m].groupby("round")[metric].mean().dropna()
                ax.plot(g.index, g.values, style.MARKERS[m] + "-", color=style.color(m), label=style.label(m))
            ax.set_xticks(sorted(rev["round"].unique()))
            ax.set_xlabel("revision round (0 = original)")
            ax.set_title(title, fontsize=9)
        h_, l_ = axes[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="lower center", ncol=len(l_), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle(
            "F15c. Does critique-and-revise with fingerprint feedback make recipes more faithful while keeping "
            "the dish? (mean over 22 regions)",
            y=1.04,
        )
        style.save(fig, "F15c_critique_revise")
    print("wrote F15_best_of_n, F15b_ablation (+ F15c when revisions exist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

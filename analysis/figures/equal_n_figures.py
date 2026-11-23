"""F2c and F2d: food pairing at equal sample size (analysis/audit/equal_n_pairing.py).

F2c  Does each cuisine's food-pairing sign survive when every cuisine has the same number of recipes (290)?
F2d  Why does the food-pairing effect depend on the number of recipes? ΔNs against the sample's own ingredient pool
     (the published null) vs against the full cuisine's pool, as the sample grows.

Usage (repo root): .venv/bin/python analysis/figures/equal_n_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402
from audit_io import CONTRASTING, REAL, REPO  # noqa: E402

UNIFORM, CONTRAST = "#0072B2", "#D55E00"


def full_size() -> pd.DataFrame:
    z = pd.read_csv(REPO / "data" / "interim" / "replication" / "results_v1_keep_empty.csv")
    return z[z["null_model"] == "random"].set_index("region")[["d", "delta_ns", "n_recipes"]]


def f2c(matched: pd.DataFrame, full: pd.DataFrame) -> None:
    n = int(matched["N"].iloc[0])
    g = matched.groupby("region")["d"]
    med, lo, hi = g.median(), g.quantile(0.025), g.quantile(0.975)
    share = g.apply(lambda s: (s > 0).mean())
    order = full["d"].sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    for i, r in enumerate(order):
        col = CONTRAST if r in CONTRASTING else UNIFORM
        ax.plot([lo[r], hi[r]], [i, i], color=col, lw=2, alpha=0.5)
        ax.scatter(med[r], i, color=col, s=36, zorder=3)
        ax.scatter(full.loc[r, "d"], i, marker="D", facecolors="none", edgecolors="#333", s=30, zorder=4)
        ax.text(1.01, i, f"{share[r]:.0%} > 0   (N = {int(full.loc[r, 'n_recipes']):,})", va="center", fontsize=7,
                color="#444", transform=ax.get_yaxis_transform())  # fmt: skip
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(np.arange(len(order)), order)
    ax.set_xlabel(f"Food-pairing effect size d against the random null (dot: median of {matched['rep'].nunique()} "
                  f"random {n}-recipe samples, line: 95% range; ◇: full cuisine)")  # fmt: skip
    ax.set_title(f"F2c. Does each cuisine's food-pairing sign survive at equal size ({n} recipes)?\n"
                 "(vermillion: published contrasting; blue: published uniform)")  # fmt: skip
    style.save(fig, "F2c_matched_size_signs")


def f2d(eq: pd.DataFrame, full: pd.DataFrame, regions: list[str]) -> None:
    fig, axes = plt.subplots(1, len(regions), figsize=(3.4 * len(regions), 3.6), sharey=False)
    for ax, r in zip(axes, regions, strict=True):
        g = eq[eq["region"] == r].groupby("N")
        own, fixed = g["delta_ns"].median(), g["delta_ns_fullpool"].median()
        ax.plot(own.index, own.values, "o-", color=CONTRAST, ms=3, label="own-pool null (published)")
        ax.plot(fixed.index, fixed.values, "s-", color=UNIFORM, ms=3, label="full-cuisine pool")
        ax.axhline(full.loc[r, "delta_ns"], color="#555", lw=0.8, ls="--", label="full cuisine")
        ax.axhline(0, color="#bbb", lw=0.6)
        ax.set_title(f"{r} (N = {int(full.loc[r, 'n_recipes']):,})")
        ax.set_xlabel("recipes in the sample")
    axes[0].set_ylabel("ΔNs (shared compounds per pair)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("F2d. Why does food pairing depend on the number of recipes? Small samples lose rare ingredients, "
                 "so the own-pool null's mean rises", y=1.06)  # fmt: skip
    style.save(fig, "F2d_pool_effect")


def main() -> int:
    style.setup()
    full = full_size()
    eq_path, m_path = REAL / "equal_n_pairing.parquet", REAL / "equal_n_pairing_matched290.parquet"
    if m_path.exists():
        f2c(pd.read_parquet(m_path), full)
    if eq_path.exists():
        f2d(pd.read_parquet(eq_path), full, ["USA", "France", "Canada", "Korea", "Japan"])
    print("wrote", sorted(p.name for p in style.FIGURES.glob("F2[cd]*")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

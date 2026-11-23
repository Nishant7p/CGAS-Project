"""F2 and F2b: the replication gate (Phase 2) as figures, from the saved replication results.

F2   Does our pipeline reproduce the published Z of every region and null? (ours vs published, 4 nulls)
F2b  Do our food-pairing values look like Singh & Bagler's Fig. 4? (Fig. 4-style dot plot, ours filled, published open)

Inputs: data/interim/replication/results_<variant>.csv (analysis/replication/replicate_singh_bagler.py) and
data/reference/singh_bagler_2018_targets.json (values read from the published figures).
Usage (repo root): .venv/bin/python analysis/figures/f2_replication.py [--variant v1_keep_empty]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats as st

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TARGETS = json.loads((REPO / "data" / "reference" / "singh_bagler_2018_targets.json").read_text())
NULLS = {"random": "Random", "frequency": "Ingredient frequency", "category": "Ingredient category",
         "frequency_category": "Frequency + category"}  # fmt: skip
UNIFORM, CONTRASTING = "#0072B2", "#D55E00"  # Okabe-Ito blue and vermillion


def f2(df: pd.DataFrame, variant: str) -> None:
    neg = set(TARGETS["expected_sign"]["contrasting_negative"])
    nulls = [n for n in NULLS if n in TARGETS["z_vs_null"]]
    fig, axes = plt.subplots(1, len(nulls), figsize=(4.5 * len(nulls), 4.6))
    for ax, null in zip(np.atleast_1d(axes), nulls, strict=True):
        pub = TARGETS["z_vs_null"][null]
        sub = df[df["null_model"] == null].set_index("code")
        codes = list(pub)
        x, y = np.array([pub[c] for c in codes]), sub.loc[codes, "z_sb"].to_numpy()
        lim = [min(x.min(), y.min()) * 1.1 - 5, max(x.max(), y.max()) * 1.1 + 5]
        ax.plot(lim, lim, color="#999", lw=1, zorder=0)
        ax.axhline(0, color="#ddd", lw=0.8, zorder=0)
        ax.axvline(0, color="#ddd", lw=0.8, zorder=0)
        ax.scatter(x, y, c=[CONTRASTING if c in neg else UNIFORM for c in codes], s=28, zorder=3)
        for c, xi, yi in zip(codes, x, y, strict=True):
            ax.annotate(c, (xi, yi), fontsize=7, xytext=(3, 2), textcoords="offset points")
        rho = st.spearmanr(x, y).statistic
        agree = int((np.sign(x) == np.sign(y)).sum())
        ax.set_title(f"{NULLS[null]} null\nSpearman ρ = {rho:.2f}, signs {agree}/{len(codes)}")
        ax.set_xlabel("Published Z (Singh & Bagler 2018, Fig. 4)")
        ax.set_ylabel("Our Z (N_rand = 100,000)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    fig.suptitle(f"F2. Does our pipeline reproduce the published food-pairing Z? ({variant}; "
                 "blue: published uniform, vermillion: published contrasting)")  # fmt: skip
    style.save(fig, f"F2_replication_{variant}")


def f2b(df: pd.DataFrame, variant: str) -> None:
    neg = set(TARGETS["expected_sign"]["contrasting_negative"])
    pub = TARGETS["z_vs_null"]["random"]
    codes = sorted(pub, key=lambda c: pub[c])
    nulls = [n for n in NULLS if n in TARGETS["z_vs_null"]]
    markers = {"random": "o", "frequency": "s", "category": "^", "frequency_category": "D"}
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    y = np.arange(len(codes))
    for k, null in enumerate(nulls):
        sub = df[df["null_model"] == null].set_index("code")
        off = (k - (len(nulls) - 1) / 2) * 0.16
        ours = sub.loc[codes, "z_sb"].to_numpy()
        theirs = np.array([TARGETS["z_vs_null"][null].get(c, np.nan) for c in codes])
        col = plt.cm.cividis(k / max(1, len(nulls) - 1))
        ax.scatter(ours, y + off, marker=markers[null], color=col, s=26, label=f"{NULLS[null]} (ours)")
        ax.scatter(theirs, y + off, marker=markers[null], facecolors="none", edgecolors=col, s=26)
    for i, c in enumerate(codes):
        if c in neg:
            ax.axhspan(i - 0.5, i + 0.5, color="#f1f1f1", zorder=0)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(y, codes)
    ax.set_xlabel("Z (filled: ours; open: published)")
    ax.set_title("F2b. Do our results look like Singh & Bagler's Fig. 4?\n(grey bands: the 6 contrasting cuisines)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    style.save(fig, f"F2b_fig4_style_{variant}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1_keep_empty")
    args = ap.parse_args()
    style.setup()
    df = pd.read_csv(REPO / "data" / "interim" / "replication" / f"results_{args.variant}.csv")
    f2(df, args.variant)
    f2b(df, args.variant)
    print("wrote", sorted(p.name for p in style.FIGURES.glob("F2*")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

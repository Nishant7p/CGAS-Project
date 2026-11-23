"""D2 figures from the scorecards (plan §11): F3, F4, F5, F8, F12, F13, F14.

F3   Do LLM cuisines reproduce each region's food-pairing sign and strength? (Fig. 4-style dot plot of Z)
F4   How large is each model's pairing effect, region by region? (d with 95% bootstrap CIs vs the real d)
F5   Where do models make pre-registered sign errors? (model x region matrix)
F8   Do LLMs over-use each cuisine's signature ingredients? (SAI vs the genuine-sample 95% band)
F12  How does each model score on every audit metric? (scorecard heatmap, medians over regions)
F13  Do LLMs beat a simple copy-mutate model of cuisine evolution? (metric distributions at equal N)
F14  Does the home-cook persona change fidelity? (paired persona minus plain, per model)

Inputs: data/interim/audit/<tag>/scorecard.parquet, data/interim/audit/real/subsamples_n160.parquet,
data/interim/audit/copy_mutate/scorecard.parquet. Usage (repo root):
    .venv/bin/python analysis/figures/d2_figures.py [--tag full]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless, before pyplot is imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402
from audit_io import AUDIT, CONTRASTING, LLM_N_PER_REGION, REAL  # noqa: E402
from style import color, label, save  # noqa: E402

N_RAND = 100_000


def load(tag: str):
    llm = pd.read_parquet(AUDIT / tag / "scorecard.parquet")
    real = pd.read_parquet(REAL / f"subsamples_n{LLM_N_PER_REGION}.parquet")
    cm_path = AUDIT / "copy_mutate" / "scorecard.parquet"
    cm = pd.read_parquet(cm_path) if cm_path.exists() else None
    return llm, real, cm


def region_order(pooled: pd.DataFrame) -> list[str]:
    return pooled.groupby("region")["z_real"].first().sort_values().index.tolist()


def f3_pairing(pooled: pd.DataFrame, models: list[str]) -> None:
    order = region_order(pooled)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    y = np.arange(len(order))
    real = pooled.groupby("region")["z_real"].first().reindex(order)
    ax.scatter(real, y, marker=style.MARKERS["real"], s=46, color=color("real"), label=label("real"), zorder=4)
    for k, m in enumerate(models):
        g = pooled[pooled["model"] == m].set_index("region").reindex(order)
        ax.scatter(g["z_sb"], y + (k - len(models) / 2) * 0.08, marker=style.MARKERS.get(m, "o"), s=26,
                   color=color(m), label=label(m), alpha=0.9, zorder=3)  # fmt: skip
    ax.axvline(0, color="#555", lw=0.8)
    for i, r in enumerate(order):
        if r in CONTRASTING:
            ax.axhspan(i - 0.5, i + 0.5, color="#eeeeee", zorder=0)
    ax.set_yticks(y, order)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_xlabel("Z against the random null cuisine (N_rand = 100,000; symlog scale)")
    ax.set_title(
        "F3. Do LLM cuisines reproduce each region's food-pairing sign?\n(grey bands: the 6 contrasting regions)"
    )
    ax.legend(loc="lower right", frameon=False)
    save(fig, "F3_pairing_dotplot")


def f4_effect_sizes(pooled: pd.DataFrame, models: list[str]) -> None:
    order = region_order(pooled)
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models) + 1.5, 6.8), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models, strict=True):
        g = pooled[pooled["model"] == m].set_index("region").reindex(order)
        y = np.arange(len(order))
        ax.errorbar(g["d"], y, xerr=[g["d"] - g["d_lo"], g["d_hi"] - g["d"]], fmt=style.MARKERS.get(m, "o"),
                    color=color(m), ms=4, lw=1, capsize=0)  # fmt: skip
        ax.scatter(g["z_real"] / np.sqrt(N_RAND), y, marker="|", s=80, color=color("real"), zorder=4)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_title(label(m))
        ax.set_xlabel("effect size d = ΔNs / SD(null)")
    axes[0][0].set_yticks(np.arange(len(order)), order)
    fig.suptitle("F4. How large is each model's food-pairing effect? (95% bootstrap CI; black tick: real cuisine)")
    save(fig, "F4_effect_sizes")


def f5_sign_matrix(pooled: pd.DataFrame, models: list[str]) -> None:
    order = region_order(pooled)
    code = np.zeros((len(models), len(order)))
    for i, m in enumerate(models):
        g = pooled[pooled["model"] == m].set_index("region").reindex(order)
        sig = (g["delta_ns_lo"] > 0) | (g["delta_ns_hi"] < 0)
        covered = g["z_sb"].notna().to_numpy()
        err = g["sign_error"].fillna(False).astype(bool).to_numpy()
        ok = (sig & g["sign_match"].fillna(False).astype(bool)).to_numpy()
        code[i] = np.where(~covered, np.nan, np.where(err, -1, np.where(ok, 1, 0)))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(models) + 2.2))
    cmap = plt.matplotlib.colors.ListedColormap(["#D55E00", "#eeeeee", "#009E73"])
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(code), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i, j in zip(*np.where(np.isnan(code)), strict=True):
        ax.text(j, i, "·", ha="center", va="center", color="#999999", fontsize=9)
    ax.set_yticks(range(len(models)), [label(m) for m in models])
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right")
    ax.set_title("F5. Where do models get the food-pairing sign wrong? "
                 "(orange: CI-significant sign error; green: significant and correct; grey: not significant; · white: region not covered yet)")  # fmt: skip
    save(fig, "F5_sign_matrix")


def f8_sai(pooled: pd.DataFrame, real: pd.DataFrame, models: list[str]) -> None:
    order = region_order(pooled)
    band = real.groupby("region")["sai"].quantile([0.025, 0.975]).unstack().reindex(order)
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    y = np.arange(len(order))
    ax.fill_betweenx(y, band[0.025], band[0.975], color="#dddddd", step="mid", label="genuine 160-recipe samples (95%)")
    for m in models:
        g = pooled[pooled["model"] == m].set_index("region").reindex(order)
        ax.scatter(g["sai"], y, marker=style.MARKERS.get(m, "o"), color=color(m), s=26, label=label(m))
    ax.axvline(1, color="#555", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, order)
    ax.set_xlabel("Stereotype Amplification Index (SAI; log scale; 1 = as in real recipes)")
    ax.set_title("F8. Do LLMs over-use each cuisine's signature ingredients?")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "F8_sai")


METRICS = {  # column -> (label, higher is better)
    "sign_match": ("Sign agreement", True),
    "rel_abs_dz": ("|ΔZ| / |Z_real|", False),
    "fingerprint_spearman": ("Fingerprint ρ", True),
    "prevalence_jsd": ("Prevalence JSD", False),
    "category_jsd": ("Category JSD", False),
    "sai": ("SAI", None),
    "freq_rank_mae": ("Frequency-rank MAE", False),
    "size_ks": ("Size KS", False),
    "diversity_ratio": ("Diversity ratio", True),
    "near_copy_rate": ("Near-copy rate", False),
}


def f12_scorecard(pooled: pd.DataFrame, real: pd.DataFrame, cm: pd.DataFrame | None, models: list[str]) -> None:
    rows = {
        f"{label(m)} ({(pooled['model'] == m).sum()} regions)": pooled[pooled["model"] == m][list(METRICS)].median()
        for m in models
    }
    rows["Real @160 (genuine samples)"] = real[list(METRICS)].median()
    if cm is not None:
        base = cm[(cm["source"] == "CM-R/ranked") & (cm["scope"] == f"n={LLM_N_PER_REGION}")]
        rows[label("copy-mutate")] = base[list(METRICS)].median()
    tab = pd.DataFrame(rows).T
    tab["sign_match"] = [
        pooled[pooled["model"] == m]["sign_match"].mean() for m in models
    ] + [real["sign_match"].mean()] + ([base["sign_match"].mean()] if cm is not None else [])  # fmt: skip
    norm = tab.copy()
    for c, (_, better) in METRICS.items():
        v = tab[c].astype(float)
        if better is None:  # SAI: distance from 1 on the log scale
            v = -np.abs(np.log(v))
        elif not better:
            v = -v
        rng = v.max() - v.min()
        norm[c] = (v - v.min()) / rng if rng > 0 else 0.5
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(tab) + 2))
    ax.imshow(norm.to_numpy(dtype=float), cmap="cividis", vmin=0, vmax=1, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, f"{tab.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm.iat[i, j] < 0.5 else "black")  # fmt: skip
    ax.set_xticks(range(len(METRICS)), [v[0] for v in METRICS.values()], rotation=35, ha="right")
    ax.set_yticks(range(len(tab)), tab.index)
    ax.set_title("F12. How faithful is each source to the real cuisines? (median over the regions each source covers; "
                 "colour: better → brighter, within each column)")  # fmt: skip
    save(fig, "F12_scorecard")


def f13_copy_mutate(pooled: pd.DataFrame, real: pd.DataFrame, cm: pd.DataFrame, models: list[str]) -> None:
    base = cm[(cm["source"] == "CM-R/ranked") & (cm["scope"] == f"n={LLM_N_PER_REGION}")]
    panels = ["prevalence_jsd", "fingerprint_spearman", "freq_rank_mae", "diversity_ratio"]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.2))
    for ax, metric in zip(axes, panels, strict=True):
        data = [real.groupby("region")[metric].median()]
        names = ["real@160"]
        cols = [color("real")]
        for m in models:
            data.append(pooled[pooled["model"] == m].set_index("region")[metric])
            names.append(label(m))
            cols.append(color(m))
        data.append(base.groupby("region")[metric].mean())
        names.append("copy-mutate")
        cols.append(color("copy-mutate"))
        bp = ax.boxplot([d.dropna() for d in data], patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], cols, strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(names) + 1), names, rotation=60, ha="right", fontsize=8)
        ax.set_title(METRICS[metric][0])
    fig.suptitle("F13. Do LLMs capture cuisines better than a copy-mutate evolution model? (one point per region)")
    save(fig, "F13_copy_mutate")


def f14_persona(llm: pd.DataFrame, models: list[str]) -> None:
    metrics = ["prevalence_jsd", "fingerprint_spearman", "sai", "diversity_ratio"]
    cond = llm[llm["scope"].isin(["plain", "persona"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.8))
    for ax, metric in zip(axes, metrics, strict=True):
        for k, m in enumerate(models):
            w = cond[cond["model"] == m].pivot_table(index="region", columns="scope", values=metric)
            if {"plain", "persona"} <= set(w.columns):
                d = (w["persona"] - w["plain"]).dropna()
                ax.scatter(np.full(len(d), k) + np.linspace(-0.15, 0.15, len(d)), d, color=color(m), s=12)
                ax.hlines(d.median(), k - 0.3, k + 0.3, color="black", lw=1.5)
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(range(len(models)), [label(m) for m in models], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{METRICS[metric][0]}: persona − plain")
    fig.suptitle("F14. Does the home-cook persona change fidelity? (one point per region; bar: median)")
    save(fig, "F14_persona_vs_plain")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    style.setup()
    llm, real, cm = load(args.tag)
    pooled = llm[llm["scope"] == "pooled"]
    models = [m for m in style.LABELS if m in set(pooled["model"])]
    f3_pairing(pooled, models)
    f4_effect_sizes(pooled, models)
    f5_sign_matrix(pooled, models)
    f8_sai(pooled, real, models)
    f12_scorecard(pooled, real, cm, models)
    if cm is not None:
        f13_copy_mutate(pooled, real, cm, models)
    f14_persona(llm, models)
    print("figures:", sorted(p.name for p in style.FIGURES.glob("F*.pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

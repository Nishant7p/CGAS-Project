"""The plan's remaining figures (plan §11), from saved outputs and the aliased LLM lines.

F4b  What does the food-pairing statistic measure? (recipe-level Ns, real vs random null, plus LLM recipes)
F1   How does the project go from real cuisines to audited and steered LLM recipes? (pipeline; numbers from files)
F6   Do LLM cuisines keep each cuisine's culinary fingerprint? (ΔNs_cat by category for the showcase regions, and the
     similarity to the real fingerprint across regions against genuine 160-recipe samples)
F7   Do LLMs build food pairing from the same ingredients? (χ% drivers for Korea; top-5 overlap across regions)
F9   Do LLM cuisines use ingredients as diversely as real cuisines? (rank-frequency for Italy; diversity ratio against
     genuine samples and copy-mutate)
F10  Are LLM recipes the right size? (pooled distribution and per-region means)
F11  Do LLM cuisines combine ingredients like the real cuisines? (Jaccard@100 of NPMI networks against the genuine
     range; backbone size)
F11b Which of the real cuisine's most distinctive pairings do LLMs reproduce? (Korea and Italy)

Showcase regions, fixed by rule: Korea (Singh & Bagler's showcase cuisine) and Italy (the largest European cuisine in
CulinaryDB). LLM cuisines are the pooled (plain + persona) full-mode cuisines of analysis/audit/llm_audit.py.
Usage (repo root): .venv/bin/python analysis/figures/d2_more_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "audit"))
import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402
from audit_io import AUDIT, LLM_N_PER_REGION, REAL, REPO, load_reference  # noqa: E402
from llm_audit import LINES, RECIPES_PER_CALL, cuisines_from_lines  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from networks import cooccurrence, npmi_top  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from style import color, label, save  # noqa: E402

from flavoraudit.contribution import chi_percent  # noqa: E402
from flavoraudit.fingerprint import fingerprint  # noqa: E402
from flavoraudit.nulls import generate  # noqa: E402
from flavoraudit.pairing import ns_from_groups, ns_values  # noqa: E402

SHOWCASE = ("Korea", "Italy")
GENUINE = "#8C8C8C"
TOP_PAIRS = 12
N_NULL_SHOW = 20_000


def takeaway(fig_id: str, text: str) -> None:
    print(f"{fig_id}: {text}")


# ---------------------------------------------------------------------------------------------------------------- F4b
def f4b_ns_distributions(ref, llm: dict, models: list[str]) -> None:
    """Recipe-level Ns: what the food-pairing statistic summarises (real vs its random null, plus the LLM recipes)."""
    fig, axes = plt.subplots(1, len(SHOWCASE), figsize=(13.5, 4.8))
    notes = []
    for ax, region in zip(axes, SHOWCASE, strict=True):
        real = ref.real[region].pairing_recipes()
        ns_real = ns_values(real, ref.S)
        ns_null = ns_from_groups(generate(real, "random", n=N_NULL_SHOW, seed=0), ref.S)
        hi = float(np.quantile(np.concatenate([ns_real, ns_null]), 0.99))
        bins = np.linspace(0, hi, 40)
        grid = np.linspace(0, hi, 300)
        ax.hist(
            np.clip(ns_null, 0, hi), bins=bins, density=True, color="#DADADA", label="random null (same pool and sizes)"
        )
        ax.plot(grid, gaussian_kde(ns_real)(grid), color="black", lw=2.6, label=f"real {region} recipes")
        for m in models:
            if (m, region) in llm:
                v = ns_values(llm[(m, region)].pairing_recipes(), ref.S)
                ax.plot(grid, gaussian_kde(v)(grid), color=color(m), lw=1.5, label=f"{label(m)} recipes")
        ax.axvline(ns_null.mean(), color="#666", ls="--", lw=1.2, label="mean, random null")
        ax.axvline(ns_real.mean(), color="black", ls="--", lw=1.2, label="mean, real recipes")
        dns = ns_real.mean() - ns_null.mean()
        word = "more" if dns > 0 else "fewer"
        ax.set_title(f"{region}: real recipes share {word} compounds than chance\n"
                     f"(real mean {ns_real.mean():.1f} vs null {ns_null.mean():.1f}; ΔNs = {dns:+.1f})",
                     fontsize=10)  # fmt: skip
        ax.set_xlabel("Ns of a recipe (mean flavor compounds shared per ingredient pair)")
        ax.set_ylabel("Density")
        notes.append(f"{region} ΔNs {dns:+.2f} (real mean {ns_real.mean():.1f}, null {ns_null.mean():.1f})")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("F4b. What does the food-pairing statistic measure? Recipe-level Ns against a random null",
                 x=0.06, ha="left", fontsize=12, y=1.04)  # fmt: skip
    save(fig, "F4b_ns_distributions")
    takeaway("F4b", "; ".join(notes))


# ----------------------------------------------------------------------------------------------------------------- F1
def f1_pipeline(summary: dict, n_real: int, alias_acc: float) -> None:
    d, rep = summary["dataset"], summary["replication"]["by_null"]["random"]
    width = 14.4
    fig, ax = plt.subplots(figsize=(width, 4.7))
    ax.set_xlim(0, width)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    steps = [
        ("Real cuisines", f"CulinaryDB\n{n_real:,} recipes\n22 regions analysed\n+ FlavorDB molecules"),
        (
            "Replication gate",
            f"Singh & Bagler (2018)\nZ vs 4 null models\nsigns {rep['sign_agreement']}/{rep['n']}\n"
            f"Spearman ρ = {rep['spearman']:.2f}",
        ),  # fmt: skip
        (
            "LLM generation",
            f"{d['n_models']} LLMs\n22 regions\nplain + persona prompts\n{d['total_calls']:,} logged calls",
        ),
        (
            "Parse and alias",
            f"ingredient lines\n→ CulinaryDB entities\naccuracy {alias_acc:.1%}\n(held-out test lines)",
        ),
        ("Audit", "Ns and Z vs 4 nulls\nequal-N reference\nχ% drivers, fingerprints\nSAI, networks, baseline"),
        ("Steering", "fidelity scorer\nbest-of-N reranking\ncritique-and-revise\nheld-out evaluation"),
    ]
    n, gap, y0, h = len(steps), 0.3, 2.25, 2.0
    w = (width - 0.24 - (n - 1) * gap) / n
    for i, (title, body) in enumerate(steps):
        x = 0.12 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#F4F6F8", ec="#34495E", lw=1.2))  # fmt: skip
        ax.text(x + w / 2, y0 + h - 0.27, title, ha="center", va="center", fontsize=11, weight="bold", color="#1B2631")
        ax.text(x + w / 2, y0 + 0.78, body, ha="center", va="center", fontsize=8.8, color="#2C3E50", linespacing=1.4)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, y0 + h / 2), xytext=(x + w + 0.03, y0 + h / 2),
                        arrowprops={"arrowstyle": "-|>", "color": "#34495E", "lw": 1.4})  # fmt: skip
    deliverables = [
        ("D1  LLMCuisine-22", f"{d['total_recipes']:,} LLM recipes\nwith logs and datasheet", "#0072B2"),
        ("D2  Audit", "food pairing, fingerprints,\nnetworks, scorecard", "#009E73"),
        ("D3  Steering", "fidelity scorer,\nbest-of-N, revise", "#D55E00"),
        ("D4  FlavorAudit", "pip-installable\nPython library", "#CC79A7"),
        ("D5  FlavorPrint", "web app:\naudit any recipe", "#E69F00"),
    ]
    dgap = 0.3
    dw = (width - 0.24 - 4 * dgap) / 5
    for i, (title, body, c) in enumerate(deliverables):
        x = 0.12 + i * (dw + dgap)
        ax.add_patch(FancyBboxPatch((x, 0.3), dw, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="white", ec=c, lw=2.0))  # fmt: skip
        ax.text(x + dw / 2, 1.27, title, ha="center", va="center", fontsize=10.5, weight="bold", color=c)
        ax.text(x + dw / 2, 0.72, body, ha="center", va="center", fontsize=9, color="#2C3E50", linespacing=1.35)
    ax.text(0.12, 1.88, "Deliverables", fontsize=10, weight="bold", color="#1B2631")
    ax.set_title("F1. How does the project go from real cuisines to audited and steered LLM recipes?", loc="left")
    save(fig, "F1_pipeline")
    takeaway(
        "F1", f"{d['n_models']} LLMs, {d['total_recipes']:,} recipes, replication signs {rep['sign_agreement']}/22"
    )


# ----------------------------------------------------------------------------------------------------------------- F6
def f6_fingerprints(ref, llm: dict, models: list[str], pooled: pd.DataFrame, subs: pd.DataFrame) -> None:
    real_fp = pd.read_parquet(REAL / "fingerprints.parquet")
    fig = plt.figure(figsize=(14, 6.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], height_ratios=[1, 0.035], wspace=0.55, hspace=0.42)
    mats = {}
    for region in SHOWCASE:
        rf = real_fp[real_fp["region"] == region].sort_values("delta_ns_cat", ascending=False)
        cats = rf["category"].tolist()
        cols = ["real"] + [m for m in models if (m, region) in llm]
        M = np.full((len(cats), len(cols)), np.nan)
        M[:, 0] = rf["delta_ns_cat"].to_numpy()
        for j, m in enumerate(cols[1:], start=1):
            f = fingerprint(llm[(m, region)], ref.S, n_null=5000, seed=2, min_recipes=10)
            if not len(f):
                f = pd.Series(dtype=float)
            else:
                f = (f if "category" in f.columns else f.reset_index()).set_index("category")["delta_ns_cat"]
            M[:, j] = [f.get(c, np.nan) for c in cats]
        mats[region] = (cats, cols, M)
    vmax = float(np.nanquantile(np.abs(np.concatenate([m[2].ravel() for m in mats.values()])), 0.95))
    im = None
    for k, region in enumerate(SHOWCASE):
        cats, cols, M = mats[region]
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(np.ma.masked_invalid(M), cmap="PuOr_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_facecolor("#E5E5E5")
        for (i, j), v in np.ndenumerate(M):
            txt = "–" if np.isnan(v) else f"{v:+.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if not np.isnan(v) and abs(v) > 0.6 * vmax else "#222")  # fmt: skip
        ax.set_xticks(range(len(cols)), [label(c).split(" (")[0] for c in cols], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(cats)), cats, fontsize=8)
        ax.set_title(f"({'ab'[k]}) {region}: ΔNs_cat by category", fontsize=10)
        ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, :2]), orientation="horizontal")
    cb.set_label("ΔNs_cat: mean shared compounds per pair, real minus category-randomized (–: category too rare)",
                 fontsize=8.5)  # fmt: skip
    ax = fig.add_subplot(gs[:, 2])
    gen = subs.groupby("region")["fingerprint_spearman"].median().dropna()
    rows = [("genuine 160-recipe\nsamples", gen.to_numpy(), GENUINE)]
    rows += [(label(m), pooled.loc[pooled["model"] == m, "fingerprint_spearman"].dropna().to_numpy(), color(m))
             for m in models]  # fmt: skip
    rng = np.random.default_rng(0)
    for i, (_name, vals, c) in enumerate(rows):
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=16, color=c, alpha=0.8, lw=0)
        if len(vals):
            ax.plot([np.median(vals)] * 2, [i - 0.32, i + 0.32], color="black", lw=2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("Spearman ρ with the real fingerprint (one dot per region; bar: median)")
    ax.set_title("(c) How similar is each fingerprint to the real one?", fontsize=10)
    fig.suptitle("F6. Do LLM cuisines keep each cuisine's culinary fingerprint?", x=0.06, ha="left", fontsize=12)
    save(fig, "F6_fingerprints")
    meds = ", ".join(
        f"{label(m)} {np.median(v):.2f}" for m, (_n, v, _c) in zip(models, rows[1:], strict=True) if len(v)
    )
    takeaway("F6", f"median fingerprint ρ: genuine samples {gen.median():.2f}; {meds}")


# ----------------------------------------------------------------------------------------------------------------- F7
def f7_drivers(ref, llm: dict, models: list[str], names: dict[int, str]) -> None:
    region, top = "Korea", 6
    real = chi_percent(ref.real[region], ref.S, top_k=30, n_sd=20_000, seed=0).head(top)
    real_rows = set(real["row"].astype(int))
    sources = ["real"] + [m for m in models if (m, region) in llm]
    tables = {"real": real}
    for m in sources[1:]:
        tables[m] = chi_percent(llm[(m, region)], ref.S, top_k=30, n_sd=2000, seed=0).head(top)
    drv = pd.read_parquet(AUDIT / "full" / "drivers.parquet")
    gen = pd.read_parquet(REAL / "drivers_subsamples_n160.parquet").groupby("region")["jaccard5"].median()
    fig = plt.figure(figsize=(16, 4.9))
    gs = fig.add_gridspec(1, len(sources) + 1, width_ratios=[1] * len(sources) + [1.35], wspace=0.95)
    for k, s in enumerate(sources):
        ax = fig.add_subplot(gs[0, k])
        t = tables[s]
        rows = t["row"].astype(int).tolist()
        cols = [color(s) if (s == "real" or r in real_rows) else "#CFCFCF" for r in rows]
        if s == "real":
            cols = ["#333333"] * len(rows)
        ax.barh(range(len(rows)), t["chi_percent"], color=cols)
        ax.set_yticks(range(len(rows)), [names.get(r, str(r)) for r in rows], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("χ% (change in mean Ns\nwhen removed)", fontsize=8)
        ax.set_title(label(s).split(" (")[0], fontsize=9.5, color=color(s) if s != "real" else "black")
    ax = fig.add_subplot(gs[0, -1])
    rows_ = [("genuine 160-recipe\nsamples", gen.to_numpy(), GENUINE)]
    rows_ += [(label(m), drv.loc[drv["model"] == m, "jaccard5"].to_numpy(), color(m)) for m in models]
    rng = np.random.default_rng(0)
    for i, (_name, vals, c) in enumerate(rows_):
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=16, color=c, alpha=0.8, lw=0)
        if len(vals):
            ax.plot([np.median(vals)] * 2, [i - 0.32, i + 0.32], color="black", lw=2)
    ax.set_yticks(range(len(rows_)), [r[0] for r in rows_], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Jaccard of top-5 drivers with the real\ncuisine (one dot per region; bar: median)", fontsize=8)
    ax.set_title("All regions", fontsize=9.5)
    fig.suptitle(
        f"F7. Do LLMs build {region}'s food pairing from the same ingredients? (top-{top} χ% contributors; "
        "coloured: also a real top contributor, grey: not)",
        x=0.06, ha="left", fontsize=11.5, y=1.02,
    )  # fmt: skip
    save(fig, "F7_drivers")
    firsts = "; ".join(f"{label(s).split(' (')[0]}: {names.get(int(tables[s]['row'].iloc[0]), '?')}" for s in sources)
    takeaway("F7", f"{region} top contributor — {firsts}")


# ----------------------------------------------------------------------------------------------------------------- F9
def f9_diversity(ref, llm: dict, models: list[str], pooled: pd.DataFrame, subs: pd.DataFrame, cm: pd.DataFrame) -> None:
    region, ranks = "Italy", 60
    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])

    def curve(cu) -> np.ndarray:
        counts = np.bincount(np.concatenate(cu.recipes), minlength=ref.S.shape[0])
        return np.sort(counts[counts > 0])[::-1] / len(cu.recipes)

    real = curve(ref.real[region])
    ax.plot(np.arange(1, min(ranks, len(real)) + 1), real[:ranks], color="black", lw=2.2, label="Real (full cuisine)")
    for m in models:
        if (m, region) in llm:
            c = curve(llm[(m, region)])
            ax.plot(np.arange(1, min(ranks, len(c)) + 1), c[:ranks], color=color(m), lw=1.6, label=label(m))
    ax.set_yscale("log")
    ax.set_xlabel("Ingredient rank in the cuisine")
    ax.set_ylabel("Share of recipes containing the ingredient")
    ax.set_title(f"(a) {region}: rank-frequency of ingredients", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    gen = subs.groupby("region")["diversity_ratio"].median()
    cmr = cm[(cm["source"] == "CM-R/ranked") & (cm["scope"] == "n=160")].groupby("region")["diversity_ratio"].median()
    rows = [("genuine 160-recipe\nsamples", gen.to_numpy(), GENUINE)]
    rows += [
        (label(m), pooled.loc[pooled["model"] == m, "diversity_ratio"].dropna().to_numpy(), color(m)) for m in models
    ]
    rows += [(label("copy-mutate"), cmr.to_numpy(), color("copy-mutate"))]
    rng = np.random.default_rng(0)
    for i, (_name, vals, c) in enumerate(rows):
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=16, color=c, alpha=0.8, lw=0)
        if len(vals):
            ax.plot([np.median(vals)] * 2, [i - 0.32, i + 0.32], color="black", lw=2)
    ax.axvline(1, color="#bbb", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Diversity ratio: unique ingredients ÷ genuine\nsamples of the same size (1 = as diverse)")
    ax.set_title("(b) Ingredient diversity, all regions (bar: median)", fontsize=10)
    fig.suptitle("F9. Do LLM cuisines use ingredients as diversely as real cuisines?", x=0.06, ha="left", fontsize=12)
    save(fig, "F9_diversity")
    meds = ", ".join(f"{r[0]} {np.median(r[2 - 1]):.2f}" for r in rows[1:] if len(r[1]))
    takeaway("F9", f"median diversity ratio: {meds}")


# ---------------------------------------------------------------------------------------------------------------- F10
def f10_sizes(ref, llm: dict, models: list[str]) -> None:
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    bins = np.arange(0.5, 31.5, 1)
    real_sizes = np.concatenate([[len(r) for r in c.recipes] for c in ref.real.values()])
    ax.hist(np.clip(real_sizes, 1, 30), bins=bins, density=True, histtype="stepfilled", color="#DDDDDD",
            ec="#555555", lw=1.2, label=f"Real (median {np.median(real_sizes):.0f})")  # fmt: skip
    meds = {}
    for m in models:
        sz = np.concatenate([[len(r) for r in cu.recipes] for (mm, _r), cu in llm.items() if mm == m])
        meds[m] = float(np.median(sz))
        ax.hist(np.clip(sz, 1, 30), bins=bins, density=True, histtype="step", color=color(m), lw=1.8,
                label=f"{label(m)} (median {meds[m]:.0f})")  # fmt: skip
    ax.set_xlabel("Ingredients per recipe (distinct CulinaryDB entities; ≥30 pooled)")
    ax.set_ylabel("Share of recipes")
    ax.set_title("(a) Recipe size, all regions pooled", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    real_mean = {r: np.mean([len(x) for x in c.recipes]) for r, c in ref.real.items()}
    lo, hi = 4, 16
    for m in models:
        pts = [(real_mean[r], np.mean([len(x) for x in cu.recipes])) for (mm, r), cu in llm.items() if mm == m]
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.scatter(xs, ys, color=color(m), marker=style.MARKERS.get(m, "o"), s=26, label=label(m), alpha=0.85)
            lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
    ax.plot([lo - 0.5, hi + 0.5], [lo - 0.5, hi + 0.5], color="#999", lw=1, ls="--")
    ax.set_xlabel("Real mean recipe size (full cuisine)")
    ax.set_ylabel("LLM mean recipe size")
    ax.set_title("(b) Per region (dashed: equal size)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("F10. Are LLM recipes the right size?", x=0.06, ha="left", fontsize=12)
    save(fig, "F10_sizes")
    takeaway(
        "F10",
        f"median size: real {np.median(real_sizes):.0f}; " + ", ".join(f"{label(m)} {v:.0f}" for m, v in meds.items()),
    )


# ---------------------------------------------------------------------------------------------------------------- F11
def f11_networks(models: list[str]) -> None:
    net = pd.read_parquet(AUDIT / "full" / "networks.parquet")
    prim = net[net["metric"] == "npmi_top100"]
    gen = prim[prim["source"] == "real@160"].set_index("region").sort_values("value")
    regions = gen.index.tolist()
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(regions))
    ax.hlines(y, gen["ref_p05"], gen["ref_p95"], color="#CFCFCF", lw=7, label="genuine 160-recipe samples (5–95%)")
    ax.scatter(gen["value"], y, color=GENUINE, s=22, zorder=3, label="genuine median")
    for k, m in enumerate(models):
        sub = prim[prim["source"] == m].set_index("region")
        yy = [regions.index(r) + (k - (len(models) - 1) / 2) * 0.12 for r in sub.index]
        ax.scatter(sub["value"], yy, color=color(m), marker=style.MARKERS.get(m, "o"), s=26, zorder=4, label=label(m))
    ax.set_yticks(y, regions, fontsize=8.5)
    ax.set_xlabel("Jaccard similarity of the top-100 NPMI ingredient pairs with the real cuisine")
    ax.set_title("(a) Distinctive pairings, by region", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax = fig.add_subplot(gs[0, 1])
    be = net[net["metric"] == "backbone_edges"]
    rows = [("genuine 160-recipe\nsamples", be.loc[be["source"] == "real@160", "value"].to_numpy(), GENUINE)]
    rows += [(label(m), be.loc[be["source"] == m, "value"].to_numpy(), color(m)) for m in models]
    rng = np.random.default_rng(0)
    for i, (_name, vals, c) in enumerate(rows):
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=16, color=c, alpha=0.8, lw=0)
        if len(vals):
            ax.plot([np.median(vals)] * 2, [i - 0.32, i + 0.32], color="black", lw=2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Edges in the disparity-filter backbone\n(α = 0.05; one dot per region; bar: median)")
    ax.set_title("(b) How many strong pairings does each network keep?", fontsize=10)
    fig.suptitle(
        "F11. Do LLM cuisines combine ingredients the way the real cuisines do?", x=0.06, ha="left", fontsize=12
    )
    save(fig, "F11_networks")
    meds = ", ".join(f"{label(m)} {prim.loc[prim['source'] == m, 'value'].median():.2f}" for m in models)
    takeaway("F11", f"median Jaccard@100: genuine {gen['value'].median():.2f}; {meds}")


def f11b_showcase_pairs(ref, llm: dict, models: list[str], names: dict[int, str]) -> None:
    n_cols = ref.S.shape[0]
    fig, axes = plt.subplots(1, len(SHOWCASE), figsize=(14, 5.6))
    summary = []
    for ax, region in zip(axes, SHOWCASE, strict=True):
        C, f, n = cooccurrence(ref.real[region].recipes, n_cols)
        Cr = C.tocsr()
        pairs = sorted(npmi_top(C, f, n, 10_000), key=lambda p: -npmi_of(Cr, f, n, *p))[:TOP_PAIRS]
        cols = ["real"] + [m for m in models if (m, region) in llm]
        M = np.full((len(pairs), len(cols)), np.nan)
        M[:, 0] = [npmi_of(Cr, f, n, i, j) for i, j in pairs]
        for c_idx, m in enumerate(cols[1:], start=1):
            Cm, fm, nm = cooccurrence(llm[(m, region)].recipes, n_cols)
            Cmr = Cm.tocsr()
            for p_idx, (i, j) in enumerate(pairs):
                if Cmr[i, j] >= max(3, 0.02 * nm):
                    M[p_idx, c_idx] = npmi_of(Cmr, fm, nm, i, j)
        ax.imshow(np.ma.masked_invalid(M), cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_facecolor("#EEEEEE")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, "absent" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#888" if np.isnan(v) else ("white" if v > 0.6 else "#222"))  # fmt: skip
        ax.set_xticks(range(len(cols)), [label(c).split(" (")[0] for c in cols], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(pairs)), [f"{names.get(i, i)} + {names.get(j, j)}" for i, j in pairs], fontsize=8)
        ax.set_title(f"{region}: the real cuisine's {TOP_PAIRS} most distinctive pairs (NPMI)", fontsize=10)
        ax.spines[:].set_visible(False)
        for c_idx, m in enumerate(cols[1:], start=1):
            summary.append(f"{region}/{label(m).split(' (')[0]} {int(np.isfinite(M[:, c_idx]).sum())}/{len(pairs)}")
    fig.suptitle(
        "F11b. Which of the real cuisine's most distinctive pairings do LLMs reproduce? (cell: NPMI in that source; "
        "absent: pair in < max(3, 2%) of recipes)",
        x=0.02, ha="left", fontsize=11,
    )  # fmt: skip
    fig.tight_layout()
    save(fig, "F11b_showcase_pairs")
    takeaway("F11b", "reproduced pairs: " + "; ".join(summary))


def npmi_of(Cr, f: np.ndarray, n: int, i: int, j: int) -> float:
    c = Cr[i, j]
    if c <= 0:
        return float("nan")
    pij, pi, pj = c / n, f[i] / n, f[j] / n
    return float(np.log(pij / (pi * pj)) / -np.log(pij)) if pij < 1 else 1.0


def main() -> int:
    style.setup()
    summary = json.loads((REPO / "results" / "summary.json").read_text())
    ref, fp, idx, cdb = load_reference()
    inv = {r: e for e, r in idx.items()}
    names = {r: str(fp.names.get(inv.get(r), r)) for r in range(ref.S.shape[0])}
    names = {r: (v[:1].upper() + v[1:]) for r, v in names.items()}
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs" / "regions.yaml").read_text()).items()}
    lines = pd.read_parquet(LINES)
    lines = lines[(lines["mode"] == "full") & (lines["recipe_idx"] < RECIPES_PER_CALL)]
    cats = next(iter(ref.real.values())).categories
    llm = {(mk, labels[code]): cuisines_from_lines(g, idx, cats, f"{mk}:{code}")
           for (mk, code), g in lines.groupby(["model_key", "region_code"], sort=True)}  # fmt: skip
    models = [m for m in style.LABELS if m in {k[0] for k in llm}]
    sc = pd.read_parquet(AUDIT / "full" / "scorecard.parquet")
    pooled = sc[sc["scope"] == "pooled"]
    subs = pd.read_parquet(REAL / f"subsamples_n{LLM_N_PER_REGION}.parquet")
    cm = pd.read_parquet(AUDIT / "copy_mutate" / "scorecard.parquet")
    alias = json.loads((REPO / "data" / "interim" / "aliasing" / "metrics.json").read_text())["test"]["accuracy_lines"]
    f1_pipeline(summary, len(cdb.recipes), alias)  # CulinaryDB's full size (45,772); 22 regions are analysed
    f4b_ns_distributions(ref, llm, models)
    f6_fingerprints(ref, llm, models, pooled, subs)
    f7_drivers(ref, llm, models, names)
    f9_diversity(ref, llm, models, pooled, subs, cm)
    f10_sizes(ref, llm, models)
    f11_networks(models)
    f11b_showcase_pairs(ref, llm, models, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

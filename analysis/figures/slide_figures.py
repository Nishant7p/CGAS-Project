"""Slide versions of the key results: few elements, large text (readable from the back of a room), no overlapping
labels, background matching the deck. Values come from results/summary.json and the same audit functions as the
report figures; nothing is typed by hand.

S_pairing      What does food pairing look like? (Korea vs Italy: real recipes against the random null)
S_replication  Do we reproduce Singh & Bagler (2018)? (our Z vs the published Z, random null, 22 regions)
S_glance       Where do AI cuisines match real ones? (4 measures: AI models vs genuine 160-recipe samples)
S_fingerprint  Korea's category fingerprint: real vs AI
S_pairs        Korea's most distinctive ingredient pairs: which AI models use them?
S_steering     Does fingerprint feedback make AI recipes more recognisable?

Usage (repo root): .venv/bin/python analysis/figures/slide_figures.py   -> figures/slides/*.png
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
from audit_io import REAL, REPO, load_reference  # noqa: E402
from d2_more_figures import npmi_of  # noqa: E402
from llm_audit import LINES, RECIPES_PER_CALL, cuisines_from_lines  # noqa: E402
from networks import cooccurrence, npmi_top  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from flavoraudit.fingerprint import fingerprint  # noqa: E402
from flavoraudit.nulls import generate  # noqa: E402
from flavoraudit.pairing import ns_from_groups, ns_values  # noqa: E402

OUT = REPO / "figures" / "slides"
BG = "#FAFAF7"
INK = "#1B2631"
GENUINE = "#7F8C8D"
S = json.loads((REPO / "results" / "summary.json").read_text())
ORDER = ["openai-gpt-6-luna", "gemini-3.5-flash-lite", "qwen3.5-9b", "llama-3.1-8b"]
SHORT = {"openai-gpt-6-luna": "GPT-6-Luna", "gemini-3.5-flash-lite": "Gemini 3.5 FL", "qwen3.5-9b": "Qwen3.5-9B",
         "llama-3.1-8b": "Llama-3.1-8B"}  # fmt: skip


def setup() -> None:
    style.setup()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 13, "figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG, "axes.edgecolor": "#555", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold"})  # fmt: skip


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def s_pairing(ref) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, region in zip(axes, ("Korea", "Italy"), strict=True):
        real = ref.real[region].pairing_recipes()
        ns_real = ns_values(real, ref.S)
        ns_null = ns_from_groups(generate(real, "random", n=20_000, seed=0), ref.S)
        hi = float(np.quantile(np.concatenate([ns_real, ns_null]), 0.99))
        grid = np.linspace(0, hi, 300)
        ax.hist(np.clip(ns_null, 0, hi), bins=np.linspace(0, hi, 36), density=True, color="#D5D8DC",
                label="random recipes (same pantry)")  # fmt: skip
        ax.plot(grid, gaussian_kde(ns_real)(grid), color=INK, lw=3.2, label="real recipes")
        ax.axvline(ns_null.mean(), color="#7B7D7D", ls="--", lw=2)
        ax.axvline(ns_real.mean(), color="#D55E00", ls="--", lw=2.4)
        d = ns_real.mean() - ns_null.mean()
        ax.set_title(f"{region}: {'more' if d > 0 else 'fewer'} than chance", pad=10)
        ax.annotate(f"ΔNs = {d:+.1f}", xy=(0.97, 0.9), xycoords="axes fraction", ha="right", fontsize=18,
                    color="#D55E00", weight="bold")  # fmt: skip
        ax.set_xlabel("Ns: flavor compounds shared per ingredient pair")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    h, labels_ = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="#D55E00", ls="--", lw=2.4), plt.Line2D([], [], color="#7B7D7D", ls="--", lw=2)]
    labels_ += ["real mean", "random mean"]
    fig.legend(h, labels_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    save(fig, "S_pairing")


def s_replication() -> None:
    ours = pd.read_csv(REPO / "data/interim/replication/results_v1_keep_empty.csv")
    ours = ours[ours["null_model"] == "random"].set_index("code")["z_sb"]
    pub = json.loads((REPO / "data/reference/singh_bagler_2018_targets.json").read_text())["z_vs_null"]["random"]
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs/regions.yaml").read_text()).items()}
    codes = sorted(pub, key=lambda c: pub[c])
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    y = np.arange(len(codes))
    for i, c in enumerate(codes):
        if pub[c] < 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#FDEBD0", zorder=0)
    ax.scatter([pub[c] for c in codes], y, s=90, facecolors="none", edgecolors=INK, lw=1.8, label="published (2018)",
               zorder=3)  # fmt: skip
    ax.scatter([ours[c] for c in codes], y, s=55, color="#0E7C86", label="our pipeline", zorder=4)
    ax.axvline(0, color="#555", lw=1)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yticks(y, [labels[c] for c in codes], fontsize=11.5)
    ax.set_xlabel("Food-pairing Z against the random null (symlog)")
    handles, labels_ = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#FDEBD0"))
    labels_.append("contrasting cuisines")
    ax.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=12)
    save(fig, "S_replication")


def s_glance() -> None:
    aud, gen_sign = S["llm_audit"], float(pd.read_parquet(REAL / "subsamples_n160.parquet")["sign_match"].mean())
    net = S["networks"]["npmi_top100"]
    gen = S["equal_n"]["n160_reference_medians"]
    panels = [
        ("Pairing direction\nmatches the real cuisine", gen_sign, {m: aud[m]["sign_agreement"] for m in ORDER}, "pct"),
        ("Category fingerprint\nsimilarity (ρ)", gen["fingerprint_spearman"],
         {m: aud[m]["median_fingerprint_spearman"] for m in ORDER}, "num"),
        ("Signature ingredient\npairs shared", net["real@160"]["median"], {m: net[m]["median"] for m in ORDER}, "num"),
        ("Ingredient variety\n(real = 1.00)", 1.0, {m: aud[m]["median_diversity_ratio"] for m in ORDER}, "num"),
    ]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.2), sharey=True)
    names = ["Real (160 recipes)"] + [SHORT[m] for m in ORDER]
    cols = [GENUINE] + [style.color(m) for m in ORDER]
    for ax, (title, g, vals, kind) in zip(axes, panels, strict=True):
        v = [g] + [vals[m] for m in ORDER]
        y = np.arange(len(v))[::-1]
        ax.barh(y, v, color=cols, height=0.68)
        for yy, vv in zip(y, v, strict=True):
            ax.text(vv + max(v) * 0.03, yy, f"{100 * vv:.0f}%" if kind == "pct" else f"{vv:.2f}", va="center",
                    fontsize=14, weight="bold", color=INK)  # fmt: skip
        ax.set_xlim(0, max(v) * 1.38)
        ax.set_xticks([])
        ax.set_title(title, fontsize=15.5, pad=10)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(np.arange(len(names))[::-1], names, fontsize=14)
    save(fig, "S_glance")


def s_fingerprint(ref, llm) -> None:
    region = "Korea"
    rf = pd.read_parquet(REAL / "fingerprints.parquet")
    rf = rf[rf["region"] == region]
    rf = rf.reindex(rf["delta_ns_cat"].abs().sort_values(ascending=False).index).head(10)
    rf = rf.sort_values("delta_ns_cat", ascending=False)
    cats = rf["category"].tolist()
    cols = ["real"] + [m for m in ORDER if (m, region) in llm]
    M = np.full((len(cats), len(cols)), np.nan)
    M[:, 0] = rf["delta_ns_cat"].to_numpy()
    for j, m in enumerate(cols[1:], start=1):
        f = fingerprint(llm[(m, region)], ref.S, n_null=5000, seed=2, min_recipes=10)
        f = (f if "category" in f.columns else f.reset_index()).set_index("category")["delta_ns_cat"] if len(f) else {}
        M[:, j] = [f.get(c, np.nan) for c in cats]
    vmax = float(np.nanquantile(np.abs(M), 0.95))
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.imshow(np.ma.masked_invalid(M), cmap="PuOr_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_facecolor("#E5E7E9")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, "–" if np.isnan(v) else f"{v:+.1f}", ha="center", va="center", fontsize=13,
                color="white" if not np.isnan(v) and abs(v) > 0.55 * vmax else INK)  # fmt: skip
    ax.set_xticks(range(len(cols)), ["Real" if c == "real" else SHORT[c] for c in cols], fontsize=13.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(cats)), cats, fontsize=13.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("ΔNs_cat: change in flavor sharing caused by each category\n(orange = increases, purple = "
                  "decreases; – = category too rare)", fontsize=12.5)  # fmt: skip
    save(fig, "S_fingerprint")


def s_pairs(ref, llm, names) -> None:
    region, n_cols = "Korea", ref.S.shape[0]
    C, f, n = cooccurrence(ref.real[region].recipes, n_cols)
    Cr = C.tocsr()
    pairs = sorted(npmi_top(C, f, n, 10_000), key=lambda p: -npmi_of(Cr, f, n, *p))[:10]
    cols = ["real"] + [m for m in ORDER if (m, region) in llm]
    M = np.full((len(pairs), len(cols)), np.nan)
    M[:, 0] = [npmi_of(Cr, f, n, i, j) for i, j in pairs]
    for c_idx, m in enumerate(cols[1:], start=1):
        Cm, fm, nm = cooccurrence(llm[(m, region)].recipes, n_cols)
        Cmr = Cm.tocsr()
        for p_idx, (i, j) in enumerate(pairs):
            if Cmr[i, j] >= max(3, 0.02 * nm):
                M[p_idx, c_idx] = npmi_of(Cmr, fm, nm, i, j)
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.imshow(np.ma.masked_invalid(M), cmap="Greens", vmin=-0.1, vmax=1.0, aspect="auto")
    ax.set_facecolor("#EEEEEE")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, "not used" if np.isnan(v) else ("✓" if j else f"{v:.2f}"), ha="center", va="center",
                fontsize=13 if not np.isnan(v) else 11.5,
                color="#9A9A9A" if np.isnan(v) else (INK if j == 0 or v < 0.6 else "white"),
                weight="bold" if j and not np.isnan(v) else "normal")  # fmt: skip
    ax.set_xticks(range(len(cols)), ["Real (NPMI)" if c == "real" else SHORT[c] for c in cols], fontsize=13.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(pairs)), [f"{names[i]} + {names[j]}" for i, j in pairs], fontsize=13.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    save(fig, "S_pairs")
    print("S_pairs reproduced:", {SHORT[m]: int(np.isfinite(M[:, k + 1]).sum()) for k, m in enumerate(cols[1:])})


def s_steering() -> None:
    rev = pd.DataFrame(S["steering"]["revise_main"])
    bon = pd.DataFrame(S["steering"]["bon_main"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.28})
    ax = axes[0]
    rev_models = [m for m in ORDER if m in set(rev["model"])]
    for k, m in enumerate(rev_models):
        g = rev[rev["model"] == m].sort_values("round")
        ax.plot(g["round"], 100 * g["indep_classifier_acc"], "o-", color=style.color(m), lw=3, ms=10, label=SHORT[m])
        below = k % 2 == 0  # alternate label sides so close values never collide
        for x, yv in zip(g["round"], 100 * g["indep_classifier_acc"], strict=True):
            ax.annotate(f"{yv:.0f}%", (x, yv), textcoords="offset points", xytext=(0, -24 if below else 13),
                        ha="center", fontsize=13.5, weight="bold", color=style.color(m))  # fmt: skip
    ax.set_xticks([0, 1, 2], ["original", "round 1", "round 2"])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylabel("Recognised as the claimed cuisine (%)")
    ax.set_ylim(36, 66)
    ax.set_title("Feedback rounds (critique-and-revise)", pad=12)
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    models = [m for m in ORDER if m in set(bon["model"])]
    x = np.arange(len(models))
    for k, (nn, alpha) in enumerate(((1, 0.45), (8, 1.0))):
        vals = [100 * bon[(bon["model"] == m) & (bon["N"] == nn)]["indep_classifier_acc"].mean() for m in models]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=[style.color(m) for m in models], alpha=alpha,
                      label="one try (N = 1)" if nn == 1 else "best of 8 (N = 8)")  # fmt: skip
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", fontsize=13, weight="bold")
    ax.set_xticks(x, [SHORT[m] for m in models])
    ax.set_ylim(0, 72)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Best-of-N selection", pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, "S_steering")


def main() -> int:
    setup()
    ref, fp, idx, _ = load_reference()
    inv = {r: e for e, r in idx.items()}
    names = {r: str(fp.names.get(inv.get(r), r)) for r in range(ref.S.shape[0])}
    names = {r: v[:1].upper() + v[1:] for r, v in names.items()}
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs/regions.yaml").read_text()).items()}
    lines = pd.read_parquet(LINES)
    lines = lines[(lines["mode"] == "full") & (lines["recipe_idx"] < RECIPES_PER_CALL)]
    cats = next(iter(ref.real.values())).categories
    llm = {(mk, labels[code]): cuisines_from_lines(g, idx, cats, f"{mk}:{code}")
           for (mk, code), g in lines.groupby(["model_key", "region_code"], sort=True)}  # fmt: skip
    s_pairing(ref)
    s_replication()
    s_glance()
    s_fingerprint(ref, llm)
    s_pairs(ref, llm, names)
    s_steering()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

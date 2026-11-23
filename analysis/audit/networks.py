"""D2 co-occurrence networks (plan §6.5): do LLM cuisines combine the same ingredient pairs as the real cuisines?

Pre-registered comparison (plan §6.5): PMI networks per cuisine; the top-k edges (Jaccard@100) and the overlap of the
disparity-filter backbones. Each recipe set gives three edge sets:

  npmi_top100  (primary) the 100 pairs with the highest normalized PMI (Bouma 2009),
               npmi = log(p_ij / (p_i p_j)) / -log p_ij, among pairs found in >= max(3, 2% of recipes) recipes;
  npmi_top50   (sensitivity) the same with k = 50;
  backbone     the disparity-filter backbone (Serrano, Boguna & Vespignani 2009, PNAS 106:6483) of the co-occurrence
               network weighted by co-occurrence counts: edge ij is kept if (1 - w_ij/s_i)^(k_i - 1) < 0.05 for either
               endpoint (s = strength, k = degree).

Each LLM cuisine's edge set is compared with the full real cuisine's by Jaccard similarity. Because the full cuisine's
backbone is far larger than a 160-recipe one, the backbone is also compared by overlap, |A ∩ B| / |A| (the share of the
cuisine's own backbone edges that the real backbone has), and its size |A| is reported. Reference (sampling noise at
the LLM's size, held out): N_DRAWS random samples of the real cuisine with the LLM cuisine's own recipe count n, each
compared with the edge set of the remaining recipes, so no recipe is on both sides. Each LLM value is placed in that
distribution by its empirical percentile p_low = (1 + #draws <= value) / (1 + N_DRAWS) (one-sided: is the LLM network
less real-like than a genuine sample of the same size?), with BH-FDR across regions within each model and metric.

Usage (repo root): .venv/bin/python analysis/audit/networks.py
Output: data/interim/audit/full/networks.parquet (one row per source x region x metric)
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_io import AUDIT, LLM_N_PER_REGION, REPO, load_reference  # noqa: E402
from llm_audit import LINES, RECIPES_PER_CALL, cuisines_from_lines  # noqa: E402

N_DRAWS = 100
ALPHA = 0.05
METRICS = ("npmi_top100", "npmi_top50", "backbone", "backbone_overlap", "backbone_edges")
PRIMARY = "npmi_top100"


def cooccurrence(recipes: list[np.ndarray], n_cols: int) -> tuple[sparse.coo_matrix, np.ndarray, int]:
    """Ingredient co-occurrence counts (symmetric, diagonal = ingredient frequency), frequencies and recipe count."""
    n = len(recipes)
    rows = np.concatenate([np.full(len(r), i) for i, r in enumerate(recipes)])
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, np.concatenate(recipes))), shape=(n, n_cols))
    return (X.T @ X).tocoo(), np.asarray(X.sum(axis=0)).ravel(), n


def npmi_top(C: sparse.coo_matrix, f: np.ndarray, n: int, k: int) -> set[tuple[int, int]]:
    """The k pairs (i < j) with the highest NPMI among pairs in >= max(3, 2% of) recipes."""
    mask = (C.row < C.col) & (C.data >= max(3, 0.02 * n))
    i, j, c = C.row[mask], C.col[mask], C.data[mask]
    if len(c) == 0:
        return set()
    pij, pi, pj = c / n, f[i] / n, f[j] / n
    npmi = np.log(pij / (pi * pj)) / -np.log(pij)
    order = np.argsort(-npmi, kind="stable")[:k]
    return {(int(i[t]), int(j[t])) for t in order}


def backbone(C: sparse.coo_matrix, n_cols: int, alpha: float = ALPHA) -> set[tuple[int, int]]:
    """Disparity-filter backbone (Serrano et al. 2009): edges significant at alpha for at least one endpoint."""
    off = C.row != C.col
    r, c, w = C.row[off], C.col[off], C.data[off]  # both directions of every edge are present
    if len(w) == 0:
        return set()
    s = np.bincount(r, weights=w, minlength=n_cols)
    k = np.bincount(r, minlength=n_cols)
    a = (1.0 - w / s[r]) ** (k[r] - 1)
    sig = (a < alpha) & (k[r] > 1)
    return {(int(min(x, y)), int(max(x, y))) for x, y in zip(r[sig], c[sig], strict=True)}


def edge_sets(recipes: list[np.ndarray], n_cols: int) -> dict[str, set[tuple[int, int]]]:
    C, f, n = cooccurrence(recipes, n_cols)
    return {
        "npmi_top100": npmi_top(C, f, n, 100),
        "npmi_top50": npmi_top(C, f, n, 50),
        "backbone": backbone(C, n_cols),
    }


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


def compare(mine: dict[str, set], real: dict[str, set]) -> dict[str, float]:
    """All metrics of one recipe set's networks against the real ones.

    backbone_overlap = |A ∩ B| / |A|: the share of this set's backbone edges that the real backbone also has (robust to
    the real backbone being larger); backbone_edges = |A| (LLM networks can be sparser than genuine ones).
    """
    a, b = mine["backbone"], real["backbone"]
    return {
        "npmi_top100": jaccard(mine["npmi_top100"], real["npmi_top100"]),
        "npmi_top50": jaccard(mine["npmi_top50"], real["npmi_top50"]),
        "backbone": jaccard(a, b),
        "backbone_overlap": len(a & b) / len(a) if a else float("nan"),
        "backbone_edges": float(len(a)),
    }


def held_out_reference(
    recipes: list[np.ndarray], n: int, n_cols: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Per metric: N_DRAWS random n-recipe samples compared with the networks of the recipes not drawn."""
    vals = {m: np.empty(N_DRAWS) for m in METRICS}
    for d in range(N_DRAWS):
        pick = np.zeros(len(recipes), dtype=bool)
        pick[rng.choice(len(recipes), size=n, replace=False)] = True
        a = edge_sets([r for r, p in zip(recipes, pick, strict=True) if p], n_cols)
        b = edge_sets([r for r, p in zip(recipes, pick, strict=True) if not p], n_cols)
        for m, v in compare(a, b).items():
            vals[m][d] = v
    return vals


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def main() -> int:
    ref, _fp, idx, _ = load_reference()
    n_cols = ref.S.shape[0]
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs" / "regions.yaml").read_text()).items()}
    real_edges = {r: edge_sets(c.recipes, n_cols) for r, c in ref.real.items()}
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def draws_for(region: str, n: int) -> dict[str, np.ndarray]:
        """The held-out reference at exactly n recipes (each LLM cuisine is placed among samples of its own size)."""
        if (region, n) not in cache:
            rng = np.random.default_rng([2, zlib.crc32(region.encode()), n])
            cache[(region, n)] = held_out_reference(ref.real[region].recipes, n, n_cols, rng)
        return cache[(region, n)]

    rows = []
    for region, cu in ref.real.items():
        draws = draws_for(region, LLM_N_PER_REGION)
        for m in METRICS:
            rows.append(
                {
                    "source": "real@160",
                    "region": region,
                    "metric": m,
                    "n_recipes": LLM_N_PER_REGION,
                    "n_real": len(cu.recipes),
                    "real_backbone_edges": len(real_edges[region]["backbone"]),
                    "value": float(np.nanmedian(draws[m])),
                    "ref_p05": float(np.nanquantile(draws[m], 0.05)),
                    "ref_p95": float(np.nanquantile(draws[m], 0.95)),
                }
            )
    lines = pd.read_parquet(LINES)
    lines = lines[(lines["mode"] == "full") & (lines["recipe_idx"] < RECIPES_PER_CALL)]
    cats = next(iter(ref.real.values())).categories
    for (mk, code), g in lines.groupby(["model_key", "region_code"], sort=True):
        region = labels[code]
        cu = cuisines_from_lines(g, idx, cats, f"{mk}:{code}")
        values = compare(edge_sets(cu.recipes, n_cols), real_edges[region])
        draws = draws_for(region, len(cu.recipes))
        for m in METRICS:
            value = values[m]
            rows.append(
                {
                    "source": mk,
                    "region": region,
                    "metric": m,
                    "n_recipes": len(cu.recipes),
                    "n_real": len(ref.real[region].recipes),
                    "real_backbone_edges": len(real_edges[region]["backbone"]),
                    "value": value,
                    "ref_median": float(np.nanmedian(draws[m])),
                    "ref_p05": float(np.nanquantile(draws[m], 0.05)),
                    "ref_p95": float(np.nanquantile(draws[m], 0.95)),
                    "p_low": float((1 + (draws[m] <= value).sum()) / (1 + N_DRAWS)),
                }
            )
    df = pd.DataFrame(rows)
    llm = df["source"] != "real@160"
    df.loc[llm, "q_low"] = df[llm].groupby(["source", "metric"])["p_low"].transform(lambda p: bh(p.to_numpy()))
    path = AUDIT / "full" / "networks.parquet"
    df.to_parquet(path, index=False)
    summary = df.groupby(["metric", "source"]).agg(
        median=("value", "median"),
        regions=("region", "count"),
        n_min=("n_recipes", "min"),
        below_ref_fdr05=("q_low", lambda q: int((q < 0.05).sum())),
    )
    print(summary.round(3).to_string())
    print(f"-> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

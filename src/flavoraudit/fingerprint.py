"""Category ('culinary') fingerprints: ΔNs_cat of Jain, Rakhi & Bagler (2015).

For one category c, a null cuisine is built by keeping every real recipe's other ingredients and re-drawing only
its c-ingredients from the cuisine's own c pool (distinct within the recipe). Comparing real Ns with this
category-randomized null gives ΔNs_cat and Z_cat. "The pattern of category contributions presents itself as a
'culinary fingerprint' of the cuisine" (Jain et al. 2015). Fingerprints of two cuisines (e.g. an LLM cuisine
and the real one) are compared by Spearman and cosine similarity over shared categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .nulls import _draw
from .pairing import Cuisine, group_by_size, ns_from_groups, ns_values

UNKNOWN = "Unknown"


def _recipe_categories(cuisine: Cuisine) -> list[np.ndarray]:
    return [np.array([cuisine.categories.get(int(i), UNKNOWN) for i in r], dtype=object) for r in cuisine.recipes]


def _randomize(
    recipes: list[np.ndarray],
    is_c: list[np.ndarray],
    templates: np.ndarray,
    cat_pool: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """n null recipes: a random template keeps its non-c ingredients; its k c-ingredients are re-drawn from cat_pool."""
    k_of = np.array([int(is_c[t].sum()) for t in templates])
    pick = rng.integers(0, len(templates), size=n)
    out: list[np.ndarray] = []
    for k in np.unique(k_of[pick]):
        rows = templates[pick[k_of[pick] == k]]
        new = _draw(cat_pool, min(int(k), len(cat_pool)), len(rows), rng, None)
        out.extend(np.concatenate([recipes[t][~is_c[t]], nw]) for t, nw in zip(rows, new, strict=True))
    return out


def category_randomized(cuisine: Cuisine, category: str, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """The category-randomized null for one category (empty if the category has < 2 pool items or no recipes)."""
    real = cuisine.pairing_recipes()
    cats = _recipe_categories(real)
    is_c = [c == category for c in cats]
    templates = np.array([t for t, m in enumerate(is_c) if m.any()], dtype=np.int64)
    pool, _ = real.pool()
    cat_pool = np.array([r for r in pool if real.categories.get(int(r), UNKNOWN) == category], dtype=np.int64)
    if len(cat_pool) < 2 or len(templates) == 0:
        return []
    return _randomize(real.recipes, is_c, templates, cat_pool, n, rng)


def fingerprint(
    cuisine: Cuisine, S: np.ndarray, n_null: int = 20_000, seed: int = 0, n_ref: int = 100_000, min_recipes: int = 20
) -> pd.DataFrame:
    """Per-category ΔNs_cat and Z_cat (Z_SB scaled to n_ref) against the category-randomized null.

    Only recipes containing the category enter the comparison (the real mean is over those same recipes).
    Categories present in fewer than min_recipes recipes are skipped.
    """
    rng = np.random.default_rng(seed)
    real = cuisine.pairing_recipes()
    cats_of = _recipe_categories(real)
    pool, _ = real.pool()
    pool_cat = np.array([real.categories.get(int(r), UNKNOWN) for r in pool], dtype=object)
    rows = []
    for c in sorted(set(pool_cat)):
        is_c = [cc == c for cc in cats_of]
        templates = np.array([t for t, m in enumerate(is_c) if m.any()], dtype=np.int64)
        cat_pool = pool[pool_cat == c]
        if len(templates) < min_recipes or len(cat_pool) < 2:
            continue
        null = _randomize(real.recipes, is_c, templates, cat_pool, n_null, rng)
        ns_real = ns_values(Cuisine(real.name, [real.recipes[t] for t in templates], real.categories), S)
        ns_null = ns_from_groups(group_by_size(null), S)
        delta = float(ns_real.mean() - ns_null.mean())
        sd = float(ns_null.std(ddof=1))
        rows.append(
            {
                "category": c,
                "n_recipes": len(templates),
                "delta_ns_cat": delta,
                "z_cat": float(np.sqrt(n_ref) * delta / sd) if sd > 0 else float("nan"),
            }
        )
    if not rows:  # e.g. a small cuisine where no category reaches min_recipes
        return pd.DataFrame(columns=["n_recipes", "delta_ns_cat", "z_cat"]).rename_axis("category")
    return pd.DataFrame(rows).set_index("category")


def fingerprint_similarity(a: pd.DataFrame, b: pd.DataFrame, column: str = "z_cat") -> dict:
    common = a.index.intersection(b.index)
    x, y = a.loc[common, column].to_numpy(dtype=float), b.loc[common, column].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)  # drop categories whose null had zero variance
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return {"n_categories": int(len(x)), "spearman": float("nan"), "cosine": float("nan")}
    cos = (
        float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        if np.linalg.norm(x) and np.linalg.norm(y)
        else float("nan")
    )
    return {"n_categories": int(len(x)), "spearman": float(st.spearmanr(x, y).statistic), "cosine": cos}

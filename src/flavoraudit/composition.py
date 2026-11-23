"""Composition, authenticity, stereotype, diversity and novelty metrics for cuisines (D2; plan §6.4–6.5).

* prevalence P_ic = n_ic / N_c (Ahn et al. 2011; Jain & Bagler 2018)
* relative prevalence (authenticity) p_ic = P_ic - mean_{c' != c} P_ic'  (Jain & Bagler 2018, Sec. IV)
* signature set = top-k ingredients of a REAL cuisine by relative prevalence
* Stereotype Amplification Index SAI = sum_{i in signature} P_i(candidate) / sum_{i in signature} P_i(real)
* prevalence JSD, category-composition JSD, frequency-rank MAE (Tuwani et al. 2019), recipe-size KS/Wasserstein
* diversity at equal N (rarefaction): unique ingredients, Shannon entropy, effective number, distinct-recipe share
* novelty: each recipe's maximum Jaccard similarity to any reference recipe (near-copy if >= 0.8)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats as st
from scipy.spatial.distance import jensenshannon

from .pairing import Cuisine


def prevalence(cuisine: Cuisine) -> pd.Series:
    """P_ic for every ingredient (S row) used in the cuisine."""
    counts = cuisine.ingredient_counts()
    return pd.Series({r: c / len(cuisine) for r, c in counts.items()}, dtype=float).sort_values(ascending=False)


def prevalence_table(cuisines: dict[str, Cuisine]) -> pd.DataFrame:
    """Rows = ingredients (S rows), columns = cuisines, values = P_ic (0 when absent)."""
    return pd.DataFrame({name: prevalence(c) for name, c in cuisines.items()}).fillna(0.0)


def relative_prevalence(table: pd.DataFrame) -> pd.DataFrame:
    """p_ic = P_ic minus the mean prevalence of i in all OTHER cuisines."""
    k = table.shape[1]
    others_mean = (table.sum(axis=1).to_numpy()[:, None] - table.to_numpy()) / (k - 1)
    return pd.DataFrame(table.to_numpy() - others_mean, index=table.index, columns=table.columns)


def signature_set(rel_prev: pd.DataFrame, cuisine: str, k: int = 10) -> list[int]:
    return [int(i) for i in rel_prev[cuisine].sort_values(ascending=False).head(k).index]


def sai(candidate: Cuisine, real: Cuisine, signature: list[int]) -> float:
    """Stereotype Amplification Index: > 1 means signature ingredients are over-used relative to the real cuisine."""
    return sai_from(prevalence(candidate), prevalence(real), signature)


def sai_from(pc: pd.Series, pr: pd.Series, signature: list[int]) -> float:
    """SAI from precomputed prevalence Series (candidate, real)."""
    num = sum(pc.get(i, 0.0) for i in signature)
    den = sum(pr.get(i, 0.0) for i in signature)
    return float(num / den) if den > 0 else float("nan")


def amplification(candidate: Cuisine, real: Cuisine, rows: list[int], alpha: float = 0.5) -> pd.Series:
    """Per-ingredient log2 ratio of prevalences with additive smoothing (alpha recipes)."""
    pc, pr = candidate.ingredient_counts(), real.ingredient_counts()
    nc, nr = len(candidate), len(real)
    return pd.Series(
        {i: np.log2(((pc.get(i, 0) + alpha) / (nc + alpha)) / ((pr.get(i, 0) + alpha) / (nr + alpha))) for i in rows}
    )


def prevalence_jsd(a: Cuisine, b: Cuisine) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between the two cuisines' normalized ingredient usage."""
    return distribution_jsd(prevalence(a), prevalence(b), normalize=True)


def distribution_jsd(pa: pd.Series, pb: pd.Series, normalize: bool = False) -> float:
    """JSD (base 2) between two non-negative Series aligned on the union of their indexes (missing = 0)."""
    idx = pa.index.union(pb.index)
    va, vb = pa.reindex(idx, fill_value=0.0).to_numpy(), pb.reindex(idx, fill_value=0.0).to_numpy()
    if normalize:
        va, vb = va / va.sum(), vb / vb.sum()
    return float(jensenshannon(va, vb, base=2) ** 2)


def category_shares(cuisine: Cuisine) -> pd.Series:
    """Mean share of each ingredient category per recipe (the category-composition profile)."""
    recs = [r for r in cuisine.recipes if len(r)]
    if not recs:
        return pd.Series(dtype=float)
    lens = np.fromiter((len(r) for r in recs), dtype=np.int64, count=len(recs))
    uniq, inv = np.unique(np.concatenate(recs), return_inverse=True)
    names = [cuisine.categories.get(int(u), "Unknown") for u in uniq]
    cats = sorted(set(names))
    code = np.array([cats.index(n) for n in names], dtype=np.int64)[inv]
    shares = np.bincount(code, weights=np.repeat(1.0 / lens, lens), minlength=len(cats)) / len(cuisine.recipes)
    return pd.Series(shares, index=cats).sort_values(ascending=False)


def category_jsd(a: Cuisine, b: Cuisine) -> float:
    return distribution_jsd(category_shares(a), category_shares(b))


def frequency_rank(cuisine: Cuisine) -> np.ndarray:
    """Ingredient frequencies sorted descending and normalized by the most popular (f(1) = 1)."""
    f = np.sort(np.fromiter(cuisine.ingredient_counts().values(), dtype=float))[::-1]
    return f / f[0]


def frequency_rank_mae(a: Cuisine, b: Cuisine) -> float:
    """MAE between normalized frequency-rank curves over the common ranks (Tuwani et al. 2019)."""
    return frequency_rank_mae_from(frequency_rank(a), frequency_rank(b))


def frequency_rank_mae_from(fa: np.ndarray, fb: np.ndarray) -> float:
    n = min(len(fa), len(fb))
    return float(np.abs(fa[:n] - fb[:n]).mean())


def size_distance(a: Cuisine, b: Cuisine) -> dict:
    sa, sb = a.sizes, b.sizes
    return {
        "mean_a": float(sa.mean()),
        "mean_b": float(sb.mean()),
        "ks": float(st.ks_2samp(sa, sb).statistic),
        "wasserstein": float(st.wasserstein_distance(sa, sb)),
    }


def diversity(cuisine: Cuisine, n: int | None = None, reps: int = 200, seed: int = 0) -> dict:
    """Diversity at equal N: rarefy to n recipes (mean over reps) so cuisines of different sizes are comparable."""
    rng = np.random.default_rng(seed)
    recs = cuisine.recipes
    n = n or len(recs)
    uniq, ent, distinct = [], [], []
    for _ in range(reps if n < len(recs) else 1):
        pick = [recs[i] for i in rng.choice(len(recs), size=n, replace=False)] if n < len(recs) else recs
        c = pd.Series(np.concatenate(pick)).value_counts()
        p = (c / c.sum()).to_numpy()
        uniq.append(len(c))
        ent.append(float(-(p * np.log(p)).sum()))
        distinct.append(len({tuple(r) for r in pick}) / len(pick))
    h = float(np.mean(ent))
    return {
        "n_recipes": n,
        "unique_ingredients": float(np.mean(uniq)),
        "entropy": h,
        "effective_ingredients": float(np.exp(h)),
        "distinct_recipe_share": float(np.mean(distinct)),
    }


def _binary(recipes: list[np.ndarray], n_cols: int) -> sparse.csr_matrix:
    rows = np.concatenate([np.full(len(r), k) for k, r in enumerate(recipes)])
    cols = np.concatenate(recipes)
    return sparse.csr_matrix((np.ones(len(cols), dtype=np.int32), (rows, cols)), shape=(len(recipes), n_cols))


def reference_matrix(reference: list[np.ndarray], n_cols: int) -> tuple[sparse.csc_matrix, np.ndarray]:
    """(B^T as CSC, recipe sizes) for a reference recipe list; build once and pass to ``novelty``."""
    B = _binary(reference, n_cols)
    return B.T.tocsc(), np.asarray(B.sum(axis=1)).ravel()


def novelty(
    candidate: Cuisine,
    reference: list[np.ndarray] | tuple[sparse.csc_matrix, np.ndarray],
    n_cols: int,
    chunk: int = 500,
    self_index: np.ndarray | None = None,
) -> np.ndarray:
    """For each candidate recipe, its maximum Jaccard similarity to any reference recipe.

    reference: a list of recipes, or the output of ``reference_matrix`` (faster when scoring many candidates).
    self_index[k] (optional) is the position of candidate recipe k inside ``reference``; that one comparison is
    skipped, so real recipes are scored against every OTHER real recipe (leave-one-out).
    """
    A = _binary(candidate.recipes, n_cols)
    a_size = np.asarray(A.sum(axis=1)).ravel()
    BT, b_size = reference if isinstance(reference, tuple) else reference_matrix(reference, n_cols)
    out = np.empty(A.shape[0])
    for s in range(0, A.shape[0], chunk):
        inter = (A[s : s + chunk] @ BT).toarray()
        union = a_size[s : s + chunk, None] + b_size[None, :] - inter
        jac = inter / np.maximum(union, 1)
        if self_index is not None:
            rows = np.arange(jac.shape[0])
            jac[rows, self_index[s : s + chunk]] = -1.0
        out[s : s + chunk] = jac.max(axis=1)
    return out

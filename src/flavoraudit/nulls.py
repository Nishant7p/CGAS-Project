"""The four randomized-cuisine (null) models of Singh & Bagler (2018), Sec. IV-B.

Every null keeps the real cuisine's own ingredient set and recipe-size distribution:

* random              - ingredients drawn uniformly (without replacement within a recipe)
* frequency           - ingredients drawn with probability proportional to their frequency of use in the cuisine
* category            - a real recipe is sampled as a template; its category composition is kept and each slot is
                        filled uniformly from the cuisine's ingredients of that category
* frequency_category  - as "category", with frequency-proportional draws inside each category

Weighted sampling without replacement uses the Gumbel-top-k trick, which matches sequential weighted sampling.
All generators return {recipe size: (m, size) array of S rows}, ready for pairing.ns_from_groups.
"""

from __future__ import annotations

import numpy as np

from .pairing import Cuisine

NULL_MODELS = ("random", "frequency", "category", "frequency_category")


def _draw(pool: np.ndarray, k: int, m: int, rng: np.random.Generator, logw: np.ndarray | None) -> np.ndarray:
    """m draws of k distinct items from pool (uniform if logw is None, else weighted via Gumbel-top-k)."""
    keys = rng.random((m, len(pool)), dtype=np.float32) if logw is None else logw + rng.gumbel(size=(m, len(pool)))
    idx = np.argpartition(-keys, k - 1, axis=1)[:, :k]
    return pool[idx]


def _chunks(n: int, size: int):
    for a in range(0, n, size):
        yield a, min(n, a + size)


def size_controlled(
    cuisine: Cuisine, n: int, rng: np.random.Generator, weighted: bool = False, chunk: int = 20_000
) -> dict[int, np.ndarray]:
    """Random (weighted=False) or frequency-preserving (weighted=True) null with n recipes."""
    real = cuisine.pairing_recipes()
    pool, freq = real.pool()
    logw = np.log(freq) if weighted else None
    sizes = rng.choice(real.sizes, size=n, replace=True)
    out: dict[int, np.ndarray] = {}
    for s in np.unique(sizes):
        m = int((sizes == s).sum())
        parts = [_draw(pool, int(s), b - a, rng, logw) for a, b in _chunks(m, chunk)]
        out[int(s)] = np.concatenate(parts)
    return out


def category_controlled(
    cuisine: Cuisine, n: int, rng: np.random.Generator, weighted: bool = False, chunk: int = 20_000
) -> dict[int, np.ndarray]:
    """Category-preserving (weighted=False) or frequency+category-preserving (weighted=True) null."""
    real = cuisine.pairing_recipes()
    pool, freq = real.pool()
    cat_of = np.array([real.categories.get(int(r), "Unknown") for r in pool])
    cats = sorted(set(cat_of))
    pools = {c: pool[cat_of == c] for c in cats}
    logws = {c: np.log(freq[cat_of == c]) for c in cats} if weighted else dict.fromkeys(cats)
    col = {r: cats.index(c) for r, c in zip(pool, cat_of, strict=True)}
    comp = np.zeros((len(real), len(cats)), dtype=np.int64)  # category composition of each template recipe
    for t, rec in enumerate(real.recipes):
        for r in rec:
            comp[t, col[int(r)]] += 1
    need = comp[rng.integers(0, len(real), size=n)]
    sizes = need.sum(axis=1)
    out_arr = np.full((n, int(sizes.max())), -1, dtype=np.int64)
    ptr = np.zeros(n, dtype=np.int64)
    for j, c in enumerate(cats):
        kcol = need[:, j]
        for k in np.unique(kcol[kcol > 0]):
            rows = np.flatnonzero(kcol == k)
            for a, b in _chunks(len(rows), chunk):
                rr = rows[a:b]
                sel = _draw(pools[c], int(k), len(rr), rng, logws[c])
                cols = ptr[rr][:, None] + np.arange(k)[None, :]
                out_arr[rr[:, None], cols] = sel
            ptr[rows] += k
    return {int(s): out_arr[sizes == s, :s] for s in np.unique(sizes)}


def generate(cuisine: Cuisine, model: str, n: int = 100_000, seed: int = 0) -> dict[int, np.ndarray]:
    """Generate a null cuisine of n recipes for one of NULL_MODELS."""
    rng = np.random.default_rng(seed)
    if model == "random":
        return size_controlled(cuisine, n, rng, weighted=False)
    if model == "frequency":
        return size_controlled(cuisine, n, rng, weighted=True)
    if model == "category":
        return category_controlled(cuisine, n, rng, weighted=False)
    if model == "frequency_category":
        return category_controlled(cuisine, n, rng, weighted=True)
    raise ValueError(f"unknown null model {model!r}; choose from {NULL_MODELS}")

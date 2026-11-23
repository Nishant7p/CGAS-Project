"""Ingredient contributions to a cuisine's food-pairing pattern.

Two published definitions:

* chi_ahn: Ahn et al. (2011) / Jain, Rakhi & Bagler (2015, eq. 4)
      chi_i = (1/N_c) sum_{R ∋ i} [2/(n_R(n_R-1)) sum_{j≠i in R} |C_i∩C_j|]
              - (2 f_i / (N_c <n_R>)) * sum_j f_j |C_i∩C_j| / sum_j f_j
  where f_i is the number of recipes that contain ingredient i.
* chi_percent: Singh & Bagler (2018, Sec. IV-C), "the percentage change in food pairing score in response to
  removal of the ingredient from the cuisine". Calibrated against their Fig. 5 (docs/replication_report.md):
  removing the ingredient from its recipes and measuring the % change in the cuisine's mean Ns (basis="ns"),
  ranking non-additive ingredients, reproduces their top-1 contributor in our top-3 for 22/22 regions and 63/66
  of their top-3 entries. basis="z" uses the % change in Z instead. Positive means the ingredient pushes the
  cuisine toward its observed pairing sign.

For the random (uniform) null the expected Ns is exactly the mean shared-compound count over all distinct pairs
in the ingredient pool (independent of recipe size), so only sd(Ns_null) needs simulation. This keeps chi_percent
free of Monte-Carlo noise in the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .nulls import size_controlled
from .pairing import Cuisine, ns_from_groups, ns_values


def pair_mean(pool: np.ndarray, S: np.ndarray) -> float:
    """Mean |C_i ∩ C_j| over all distinct pairs of the pool (= E[Ns] under the uniform random null)."""
    sub = S[np.ix_(pool, pool)].astype(np.float64)
    k = len(pool)
    return float((sub.sum() - np.trace(sub)) / (k * (k - 1)))


def random_null_z(cuisine: Cuisine, S: np.ndarray, n_sd: int = 20_000, seed: int = 0, n_ref: int = 100_000) -> float:
    """Z against the random null: exact null mean, simulated null SD."""
    real = cuisine.pairing_recipes()
    pool, _ = real.pool()
    sd = ns_from_groups(size_controlled(real, n_sd, np.random.default_rng(seed)), S).std(ddof=1)
    return float(np.sqrt(n_ref) * (ns_values(real, S).mean() - pair_mean(pool, S)) / sd)


def remove(cuisine: Cuisine, row: int, procedure: str = "ingredient") -> Cuisine:
    """Cuisine without one ingredient: drop it from every recipe ('ingredient') or drop its recipes ('recipes')."""
    if procedure == "ingredient":
        recs = [r[r != row] for r in cuisine.recipes]
    elif procedure == "recipes":
        recs = [r for r in cuisine.recipes if row not in r]
    else:
        raise ValueError("procedure must be 'ingredient' or 'recipes'")
    return Cuisine(cuisine.name, [r for r in recs if len(r) >= 2], cuisine.categories)


def chi_percent(
    cuisine: Cuisine,
    S: np.ndarray,
    rows: list[int] | None = None,
    top_k: int = 30,
    procedure: str = "ingredient",
    basis: str = "ns",
    exclude_categories: tuple[str, ...] = ("Additive",),
    n_sd: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Singh & Bagler contribution for the given rows (default: the top_k most frequent non-excluded ingredients).

    basis="ns": 100 * sign(Z) * (Ns - Ns_without_i) / Ns  (default; matches the published Fig. 5)
    basis="z":  100 * (Z - Z_without_i) / Z
    """
    real = cuisine.pairing_recipes()
    counts = real.ingredient_counts()
    if rows is None:
        ranked = [r for r, _ in counts.most_common()]
        rows = [r for r in ranked if real.categories.get(r) not in exclude_categories][:top_k]
    z0 = random_null_z(real, S, n_sd, seed)
    ns0 = float(ns_values(real, S).mean())
    out = []
    for r in rows:
        reduced = remove(real, r, procedure)
        rec = {"row": r, "frequency": counts[r] / len(real), "z": z0}
        if basis == "ns":
            ns1 = float(ns_values(reduced, S).mean())
            rec.update(ns=ns0, ns_without=ns1, chi_percent=100 * np.sign(z0) * (ns0 - ns1) / ns0)
        elif basis == "z":
            z1 = random_null_z(reduced, S, n_sd, seed)
            rec.update(z_without=z1, chi_percent=100 * (z0 - z1) / z0)
        else:
            raise ValueError("basis must be 'ns' or 'z'")
        out.append(rec)
    return pd.DataFrame(out).sort_values("chi_percent", ascending=False, ignore_index=True)


def chi_ahn(cuisine: Cuisine, S: np.ndarray) -> pd.DataFrame:
    """Ahn et al. (2011) contribution chi_i for every ingredient of the cuisine."""
    real = cuisine.pairing_recipes()
    n_c = len(real)
    pool, f = real.pool()
    pos = {int(r): k for k, r in enumerate(pool)}
    first = np.zeros(len(pool))
    for rec in real.recipes:
        n = len(rec)
        sub = S[np.ix_(rec, rec)].astype(np.float64)
        row_sums = sub.sum(axis=1) - np.diag(sub)  # sum_{j≠i in R} |C_i ∩ C_j|
        for i, val in zip(rec, row_sums, strict=True):
            first[pos[int(i)]] += 2.0 * val / (n * (n - 1))
    first /= n_c
    mean_n = real.sizes.mean()
    spp = S[np.ix_(pool, pool)].astype(np.float64)
    np.fill_diagonal(spp, 0.0)
    second = 2.0 * f / (n_c * mean_n) * (spp @ f) / f.sum()
    return pd.DataFrame({"row": pool, "frequency": f / n_c, "chi_ahn": first - second}).sort_values(
        "chi_ahn", ascending=False, ignore_index=True
    )

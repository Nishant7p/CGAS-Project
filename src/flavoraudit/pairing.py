"""Cuisines as sets of recipes, and flavor sharing Ns.

Ns(R) = 2 / (n (n - 1)) * sum_{i<j in R} |C_i ∩ C_j|  (Ahn et al. 2011; Jain, Rakhi & Bagler 2015, eq. 2).
The cuisine score is the mean of Ns(R) over its recipes with n >= 2.

Recipes are stored as integer arrays of rows into the shared-compound matrix S, so Ns for many recipes is one
vectorised gather per recipe size: S[X[:, :, None], X[:, None, :]] summed over the upper triangle.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cuisine:
    """A named collection of recipes. Each recipe is a sorted array of distinct S-matrix rows."""

    name: str
    recipes: list[np.ndarray]
    categories: dict[int, str] = field(default_factory=dict)  # S row -> ingredient category

    def __len__(self) -> int:
        return len(self.recipes)

    @property
    def sizes(self) -> np.ndarray:
        return np.fromiter((len(r) for r in self.recipes), dtype=np.int64, count=len(self.recipes))

    def ingredient_counts(self) -> Counter:
        """Number of recipes that contain each ingredient (S row)."""
        return Counter(int(i) for r in self.recipes for i in r)

    def pool(self) -> tuple[np.ndarray, np.ndarray]:
        """(ingredient rows, frequencies) for the cuisine's ingredient set, sorted by row."""
        c = self.ingredient_counts()
        rows = np.array(sorted(c), dtype=np.int64)
        return rows, np.array([c[r] for r in rows], dtype=np.float64)

    def pairing_recipes(self) -> Cuisine:
        """The sub-cuisine of recipes with at least 2 ingredients (Ns is undefined below that)."""
        return Cuisine(self.name, [r for r in self.recipes if len(r) >= 2], self.categories)


def build_cuisines(
    recipe_entities,
    recipes,
    index: dict[int, int],
    categories: dict[int, str],
    drop_entities: set[int] | frozenset[int] = frozenset(),
    labels: list[str] | None = None,
) -> dict[str, Cuisine]:
    """Build one Cuisine per label from long (recipe_id, entity_id) rows.

    recipe_entities: DataFrame with recipe_id and entity_id; recipes: DataFrame with recipe_id and cuisine.
    index: entity_id -> S row; categories: entity_id -> category; drop_entities are removed from every recipe.
    """
    re_ = recipe_entities[~recipe_entities["entity_id"].isin(drop_entities)]
    re_ = re_[re_["entity_id"].isin(index)]
    lab = dict(zip(recipes["recipe_id"], recipes["cuisine"], strict=True))
    rows_cat = {index[e]: c for e, c in categories.items() if e in index}
    grouped: dict[str, list[np.ndarray]] = {}
    for rid, ents in re_.groupby("recipe_id")["entity_id"]:
        cu = lab.get(rid)
        if cu is None or (labels is not None and cu not in labels):
            continue
        grouped.setdefault(cu, []).append(np.unique(np.fromiter((index[int(e)] for e in ents), dtype=np.int64)))
    return {cu: Cuisine(cu, recs, rows_cat) for cu, recs in sorted(grouped.items())}


def group_by_size(recipes: list[np.ndarray]) -> dict[int, np.ndarray]:
    """{size: (m, size) array} for recipes of size >= 2."""
    buckets: dict[int, list[np.ndarray]] = {}
    for r in recipes:
        if len(r) >= 2:
            buckets.setdefault(len(r), []).append(r)
    return {s: np.stack(v) for s, v in buckets.items()}


def ns_from_groups(groups: dict[int, np.ndarray], S: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Concatenated Ns values for size-grouped recipe arrays (order: by size, then row)."""
    out = []
    for s in sorted(groups):
        X = groups[s]
        iu, ju = np.triu_indices(s, 1)
        for a in range(0, len(X), chunk):
            x = X[a : a + chunk]
            out.append(S[x[:, iu], x[:, ju]].mean(axis=1, dtype=np.float64))
    return np.concatenate(out) if out else np.empty(0)


def ns_values(cuisine: Cuisine, S: np.ndarray) -> np.ndarray:
    """Ns for each recipe of size >= 2 (order follows group_by_size)."""
    return ns_from_groups(group_by_size(cuisine.recipes), S)


def ns_recipe(recipe: np.ndarray, S: np.ndarray) -> float:
    """Ns of one recipe (NaN if it has fewer than 2 ingredients)."""
    r = np.asarray(recipe)
    if len(r) < 2:
        return float("nan")
    iu, ju = np.triu_indices(len(r), 1)
    return float(S[r[iu], r[ju]].mean())

"""The flavor-fidelity scorecard (plan §6.6): every metric for one candidate cuisine against its real cuisine.

A candidate is any Cuisine claimed to be region c: an LLM cuisine, a copy-mutate cuisine or an equal-N subsample of
the real cuisine itself (the reference distribution). All of them go through ``scorecard_row``, so the numbers are
comparable by construction.

Columns (definitions in the plan, §6.1–6.6):

* pairing: mean Ns, ΔNs and d against the candidate's own random null, a percentile bootstrap CI of ΔNs (recipes
  resampled, null mean fixed), Z_SB (N_rand = 100,000; Jain, Rakhi & Bagler 2015 eq. 3), the real cuisine's Z,
  ``sign_match`` (sign of Z) and the pre-registered ``sign_error`` (the ΔNs CI excludes 0 with the opposite sign to
  the real cuisine's Z), and |ΔZ| / |Z_real|;
* fingerprint: Spearman and cosine similarity of the ΔNs_cat / Z_cat vectors (Jain et al. 2015) over shared
  categories;
* composition: prevalence JSD, category-composition JSD, SAI on the real top-10 signature set, frequency-rank MAE
  (Tuwani et al. 2019), recipe-size KS and Wasserstein distances;
* diversity: unique ingredients at the candidate's N and the ratio to the real cuisine rarefied to the same N;
* novelty: mean maximum Jaccard similarity to any real recipe and the near-copy rate (max Jaccard >= 0.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .composition import (
    category_shares,
    distribution_jsd,
    diversity,
    frequency_rank,
    frequency_rank_mae_from,
    novelty,
    prevalence,
    prevalence_table,
    reference_matrix,
    relative_prevalence,
    sai_from,
    size_distance,
)
from .fingerprint import fingerprint, fingerprint_similarity
from .pairing import Cuisine, ns_values
from .stats import food_pairing

NEAR_COPY = 0.8


@dataclass
class Reference:
    """The real cuisines and everything the scorecard compares a candidate with (real-side statistics cached)."""

    S: np.ndarray
    real: dict[str, Cuisine]
    real_z: dict[str, float]  # region -> Z_SB of the real cuisine against the random null
    signatures: dict[str, list[int]]  # region -> top-k S rows by relative prevalence
    fingerprints: dict[str, pd.DataFrame]  # region -> real ΔNs_cat / Z_cat table
    recipes: list[np.ndarray] = field(default_factory=list)  # all real recipes, region by region (novelty)
    offsets: dict[str, int] = field(default_factory=dict)  # region -> position of its first recipe in `recipes`
    _div_cache: dict = field(default_factory=dict, repr=False)
    _stats: dict = field(default_factory=dict, repr=False)
    _ref_matrix: tuple | None = field(default=None, repr=False)

    def stats(self, region: str) -> dict:
        """Cached prevalence, category shares and frequency-rank curve of the real cuisine."""
        if region not in self._stats:
            c = self.real[region]
            self._stats[region] = {"prev": prevalence(c), "cat": category_shares(c), "rank": frequency_rank(c)}
        return self._stats[region]

    def novelty_matrix(self) -> tuple:
        if self._ref_matrix is None:
            self._ref_matrix = reference_matrix(self.recipes, self.S.shape[0])
        return self._ref_matrix

    @classmethod
    def build(
        cls,
        real: dict[str, Cuisine],
        S: np.ndarray,
        real_z: dict[str, float] | None = None,
        fingerprints: dict[str, pd.DataFrame] | None = None,
        k_signature: int = 10,
        n_null: int = 100_000,
        fingerprint_null: int = 20_000,
        seed: int = 0,
    ) -> Reference:
        """Compute whatever is not supplied (real Z, fingerprints) and the signature sets."""
        if real_z is None:
            real_z = {r: food_pairing(c, S, "random", n_null=n_null, seed=seed).z_sb for r, c in real.items()}
        if fingerprints is None:
            fingerprints = {r: fingerprint(c, S, n_null=fingerprint_null, seed=seed) for r, c in real.items()}
        rel = relative_prevalence(prevalence_table(real))
        signatures = {r: [int(i) for i in rel[r].sort_values(ascending=False).head(k_signature).index] for r in real}
        recipes, offsets = [], {}
        for r, c in real.items():
            offsets[r] = len(recipes)
            recipes.extend(c.recipes)
        return cls(S, real, dict(real_z), signatures, dict(fingerprints), recipes, offsets)

    def real_diversity(self, region: str, n: int, reps: int = 50) -> float:
        """Unique ingredients of the real cuisine rarefied to n recipes (cached)."""
        key = (region, n)
        if key not in self._div_cache:
            self._div_cache[key] = diversity(self.real[region], n=min(n, len(self.real[region])), reps=reps, seed=1)[
                "unique_ingredients"
            ]
        return self._div_cache[key]


def scorecard_row(
    candidate: Cuisine,
    region: str,
    ref: Reference,
    *,
    n_null: int = 20_000,
    n_boot: int = 1000,
    fingerprint_null: int = 5_000,
    fingerprint_min_recipes: int = 10,
    with_fingerprint: bool = True,
    self_index: np.ndarray | None = None,
    max_novelty_recipes: int | None = None,
    seed: int = 0,
) -> dict:
    """All scorecard metrics for one candidate cuisine claimed to be ``region``.

    self_index: for subsamples of the real cuisine, each recipe's position in ``ref.recipes`` (leave-one-out novelty).
    max_novelty_recipes: estimate novelty on a seeded random subset of this many recipes (large cuisines only).
    """
    S, real = ref.S, ref.real[region]
    res = food_pairing(candidate, S, "random", n_null=n_null, seed=seed)
    ns = ns_values(candidate, S)
    rng = np.random.default_rng(seed + 1)
    boots = rng.choice(ns, size=(n_boot, len(ns)), replace=True).mean(axis=1) - res.ns_null
    lo, hi = (float(x) for x in np.quantile(boots, [0.025, 0.975]))
    z_real = float(ref.real_z[region])
    sign_error = bool((lo > 0 and z_real < 0) or (hi < 0 and z_real > 0))
    row = {
        "region": region,
        "n_recipes": len(candidate),
        "n_pairing_recipes": res.n_recipes,
        "mean_size": float(candidate.sizes.mean()),
        "ns": res.ns_cuisine,
        "ns_null": res.ns_null,
        "delta_ns": res.delta_ns,
        "delta_ns_lo": lo,
        "delta_ns_hi": hi,
        "d": res.d,
        "d_lo": lo / res.sd_null,
        "d_hi": hi / res.sd_null,
        "z_sb": res.z_sb,
        "z_real": z_real,
        "sign_match": bool(np.sign(res.z_sb) == np.sign(z_real)),
        "sign_error": sign_error,
        "rel_abs_dz": abs(res.z_sb - z_real) / abs(z_real) if z_real else float("nan"),
    }
    if with_fingerprint:
        f = fingerprint(candidate, S, n_null=fingerprint_null, seed=seed + 2, min_recipes=fingerprint_min_recipes)
        sim = (
            fingerprint_similarity(f, ref.fingerprints[region])
            if len(f)
            else {"n_categories": 0, "spearman": float("nan"), "cosine": float("nan")}
        )
        row.update(
            fingerprint_spearman=sim["spearman"], fingerprint_cosine=sim["cosine"], fingerprint_n=sim["n_categories"]
        )
    sizes = size_distance(candidate, real)
    uniq = diversity(candidate, reps=1)["unique_ingredients"]
    nov_cand, nov_self = candidate, self_index
    if max_novelty_recipes is not None and len(candidate) > max_novelty_recipes:
        pick = np.sort(np.random.default_rng(seed + 3).choice(len(candidate), max_novelty_recipes, replace=False))
        nov_cand = Cuisine(candidate.name, [candidate.recipes[i] for i in pick], candidate.categories)
        nov_self = None if self_index is None else self_index[pick]
    nov = novelty(nov_cand, ref.novelty_matrix(), S.shape[0], self_index=nov_self)
    rs, prev = ref.stats(region), prevalence(candidate)
    row.update(
        prevalence_jsd=distribution_jsd(prev, rs["prev"], normalize=True),
        category_jsd=distribution_jsd(category_shares(candidate), rs["cat"]),
        sai=sai_from(prev, rs["prev"], ref.signatures[region]),
        freq_rank_mae=frequency_rank_mae_from(frequency_rank(candidate), rs["rank"]),
        size_ks=sizes["ks"],
        size_wasserstein=sizes["wasserstein"],
        unique_ingredients=uniq,
        diversity_ratio=uniq / ref.real_diversity(region, len(candidate)),
        novelty_mean=float(nov.mean()),
        near_copy_rate=float((nov >= NEAR_COPY).mean()),
    )
    return row


def real_subsample(ref: Reference, region: str, n: int, rng: np.random.Generator) -> tuple[Cuisine, np.ndarray]:
    """An equal-N random subsample of the real cuisine and each recipe's position in ref.recipes."""
    real = ref.real[region]
    pick = rng.choice(len(real), size=min(n, len(real)), replace=False)
    sub = Cuisine(f"{region}:real@{n}", [real.recipes[i] for i in pick], real.categories)
    return sub, ref.offsets[region] + pick

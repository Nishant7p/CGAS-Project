"""Food-pairing statistics: Z-score (Jain, Rakhi & Bagler 2015, eq. 3), effect size, bootstrap CIs.

Z_SB = sqrt(N_rand) * (mean Ns_cuisine - mean Ns_null) / sd(Ns_null), with N_rand the number of null recipes
(100,000 in Singh & Bagler 2018). With N_rand fixed, Z_SB = sqrt(N_rand) * d, where d is the standardized
effect size, so Z_SB is comparable across cuisines of different sizes. The size-aware significance of the
real cuisine is reported separately (z_cuisine, p_value).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import nulls as _nulls
from .pairing import Cuisine, ns_from_groups, ns_values


@dataclass
class PairingResult:
    cuisine: str
    null_model: str
    n_recipes: int
    ns_cuisine: float
    ns_null: float
    sd_null: float
    delta_ns: float
    d: float  # standardized effect size
    z_sb: float  # Singh & Bagler Z with N_rand = n_ref
    z_cuisine: float  # size-aware: delta / (sd_null / sqrt(n_recipes))
    n_null: int

    def as_dict(self) -> dict:
        return asdict(self)


def z_sb(ns_cuisine_mean: float, null_ns: np.ndarray, n_ref: int | None = None) -> float:
    n = n_ref if n_ref is not None else len(null_ns)
    return float(np.sqrt(n) * (ns_cuisine_mean - null_ns.mean()) / null_ns.std(ddof=1))


def food_pairing(
    cuisine: Cuisine,
    S: np.ndarray,
    null_model: str = "random",
    n_null: int = 100_000,
    seed: int = 0,
    n_ref: int = 100_000,
) -> PairingResult:
    """Compare a cuisine's mean Ns with one null model (see flavoraudit.nulls.NULL_MODELS)."""
    real = ns_values(cuisine, S)
    null = ns_from_groups(_nulls.generate(cuisine, null_model, n=n_null, seed=seed), S)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    delta = float(real.mean()) - mu
    return PairingResult(
        cuisine=cuisine.name,
        null_model=null_model,
        n_recipes=len(real),
        ns_cuisine=float(real.mean()),
        ns_null=mu,
        sd_null=sd,
        delta_ns=delta,
        d=delta / sd,
        z_sb=float(np.sqrt(n_ref) * delta / sd),
        z_cuisine=float(np.sqrt(len(real)) * delta / sd),
        n_null=len(null),
    )


def bootstrap_delta_ci(
    cuisine: Cuisine, S: np.ndarray, null_mean: float, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI for ΔNs, resampling the cuisine's recipes with the null mean held fixed."""
    rng = np.random.default_rng(seed)
    vals = ns_values(cuisine, S)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1) - null_mean
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)

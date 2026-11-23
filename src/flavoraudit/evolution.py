"""Copy-mutate models of culinary evolution: the evolutionary baseline for the D2 audit.

Kinouchi et al. (2008) proposed copy-mutate. The Bagler lab adapted it to Indian regional cuisines (Jain & Bagler
2018: "CM-fitness random" and "CM-fitness ranked") and to CulinaryDB's world cuisines (Tuwani et al. 2019,
Algorithm 1: CM-R, CM-C and CM-M). This module implements Tuwani et al.'s Algorithm 1 with both fitness definitions:

1. **Fitness.** ``"random"``: f_i ~ U(0, 1) (Kinouchi 2008; Tuwani Step 1). ``"ranked"``: f_i = the ingredient's
   frequency in the real cuisine, Jain & Bagler's "frequency scaled fitness values". Only comparisons f_j > f_i
   matter, so any increasing transform of frequency gives the same model.
2. **Seed pools.** I0 = m ingredients drawn without replacement from the cuisine's ingredients. R0 = n = round(m / φ)
   recipes, each of s ingredients drawn without replacement from I0, where φ = (distinct ingredients) / (recipes) of
   the real cuisine. Tuwani et al. report m = 20 and n = I0 / φ. The papers use one size s = round(mean recipe size)
   (``sizes="mean"``). ``sizes="empirical"`` draws each seed recipe's size from the real size distribution instead
   (our size-matched variant, restricted to sizes <= |I0|).
3. **Each step.** If |I0| / |R| >= φ, a uniformly chosen mother recipe is copied and mutated M times: draw i from the
   copy and j from I0 (CM-C: from I0 ∩ category(i); CM-M: so for half of the draws); if f_j > f_i, j replaces i.
   The copy joins R. Otherwise a random unused ingredient of the cuisine joins I0. M = 4 for CM-R and 6 for CM-C and
   CM-M (Tuwani et al. 2019).
4. **Stop** at N recipes (default: the real cuisine's count).

Choices the papers leave open: a j already in the copy is not inserted (recipes stay ingredient sets); identical
copies are kept, as in the papers (no duplicate rejection); once all of the cuisine's ingredients are in the pool it
stops growing.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from .pairing import Cuisine

VARIANTS = ("CM-R", "CM-C", "CM-M")
FITNESS = ("random", "ranked")
DEFAULT_MUTATIONS = {"CM-R": 4, "CM-C": 6, "CM-M": 6}  # Tuwani et al. (2019), Sec. VI


def copy_mutate_with_stats(
    real: Cuisine,
    *,
    variant: str = "CM-R",
    fitness: str = "ranked",
    n_recipes: int | None = None,
    mutations: int | None = None,
    m0: int = 20,
    sizes: str = "mean",
    seed: int = 0,
    name: str | None = None,
) -> tuple[Cuisine, dict]:
    """Evolve one copy-mutate cuisine from a real cuisine's ingredients; returns (cuisine, run statistics).

    The first ``stats["seed_recipes"]`` recipes of the result are the seed pool R0, in creation order.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if fitness not in FITNESS:
        raise ValueError(f"unknown fitness {fitness!r}; choose from {FITNESS}")
    if sizes not in ("mean", "empirical"):
        raise ValueError("sizes must be 'mean' or 'empirical'")
    rng = np.random.default_rng(seed)
    py = random.Random(int(rng.integers(2**62)))
    rows, freq = real.pool()
    if len(rows) == 0:
        raise ValueError(f"{real.name}: the real cuisine has no ingredients")
    ingredients = rows.tolist()
    fit = dict(zip(ingredients, (rng.random(len(rows)) if fitness == "random" else freq).tolist(), strict=True))
    cat = {r: real.categories.get(r, "Unknown") for r in ingredients}
    n_target = n_recipes or len(real)
    phi = len(ingredients) / len(real)
    n_mut = DEFAULT_MUTATIONS[variant] if mutations is None else mutations

    order = ingredients[:]
    py.shuffle(order)
    m = min(m0, len(order))
    pool, remaining = order[:m], order[m:]  # remaining is shuffled, so pop() picks a uniformly random new ingredient
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in pool:
        by_cat[cat[r]].append(r)

    real_sizes = real.sizes
    allowed = real_sizes[real_sizes <= len(pool)]
    s_mean = max(1, min(round(float(real_sizes.mean())), len(pool)))
    n_seed = min(max(1, round(m / phi)), n_target)
    recipes: list[list[int]] = []
    for _ in range(n_seed):
        s = s_mean if sizes == "mean" or len(allowed) == 0 else int(allowed[py.randrange(len(allowed))])
        recipes.append(py.sample(pool, s))

    attempted = accepted = 0
    while len(recipes) < n_target:
        if remaining and len(pool) / len(recipes) < phi:
            p = remaining.pop()
            pool.append(p)
            by_cat[cat[p]].append(p)
            continue
        child = list(recipes[py.randrange(len(recipes))])
        for _ in range(n_mut):
            a = py.randrange(len(child))
            i = child[a]
            same = variant == "CM-C" or (variant == "CM-M" and py.random() < 0.5)
            cand = by_cat[cat[i]] if same else pool  # i is in the pool, so its category list is never empty
            j = cand[py.randrange(len(cand))]
            attempted += 1
            if fit[j] > fit[i] and j not in child:
                child[a] = j
                accepted += 1
        recipes.append(child)

    out = Cuisine(
        name or f"{real.name}:{variant}/{fitness}",
        [np.array(sorted(r), dtype=np.int64) for r in recipes],
        dict(real.categories),
    )
    stats = {
        "variant": variant,
        "fitness": fitness,
        "mutations_per_copy": n_mut,
        "phi": phi,
        "seed_ingredients": m,
        "seed_recipes": n_seed,
        "final_pool": len(pool),
        "mutation_attempts": attempted,
        "mutations_accepted": accepted,
        "acceptance_rate": accepted / attempted if attempted else float("nan"),
        "duplicate_share": 1 - len({tuple(r) for r in out.recipes}) / len(out),
    }
    return out, stats


def copy_mutate(real: Cuisine, **kwargs) -> Cuisine:
    """One copy-mutate cuisine (see copy_mutate_with_stats for the parameters)."""
    return copy_mutate_with_stats(real, **kwargs)[0]


def ensemble(real: Cuisine, n_instances: int = 10, seed: int = 0, **kwargs) -> list[Cuisine]:
    """n_instances independent copy-mutate cuisines with seeds seed, seed + 1, ... (Jain & Bagler used 24 sets)."""
    return [copy_mutate(real, seed=seed + k, **kwargs) for k in range(n_instances)]

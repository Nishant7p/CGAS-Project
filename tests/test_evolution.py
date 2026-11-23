"""Property tests for the copy-mutate baseline (flavoraudit.evolution; Tuwani et al. 2019, Algorithm 1)."""

from collections import Counter

import numpy as np
import pytest

from flavoraudit.composition import prevalence
from flavoraudit.evolution import DEFAULT_MUTATIONS, copy_mutate, copy_mutate_with_stats, ensemble
from flavoraudit.pairing import Cuisine


@pytest.fixture(scope="module")
def real() -> Cuisine:
    """A toy cuisine: 400 recipes over 80 ingredients in 5 categories, with skewed popularity and sizes 2-14."""
    rng = np.random.default_rng(42)
    n_ing = 80
    weights = 1.0 / np.arange(1, n_ing + 1)
    recipes = []
    for _ in range(400):
        s = int(rng.integers(2, 15))
        recipes.append(np.sort(rng.choice(n_ing, size=s, replace=False, p=weights / weights.sum())))
    cats = {i: f"cat{i % 5}" for i in range(n_ing)}
    return Cuisine("Toy", recipes, cats)


def composition(recipe, cats) -> tuple:
    return tuple(sorted(Counter(cats[int(i)] for i in recipe).items()))


@pytest.mark.parametrize("variant", ["CM-R", "CM-C", "CM-M"])
@pytest.mark.parametrize("fitness", ["random", "ranked"])
def test_basic_invariants(real, variant, fitness):
    cm, stats = copy_mutate_with_stats(real, variant=variant, fitness=fitness, seed=1)
    assert len(cm) == len(real)
    pool = set(real.pool()[0].tolist())
    for r in cm.recipes:
        assert len(np.unique(r)) == len(r) and np.all(np.diff(r) > 0)  # sorted ingredient sets
        assert set(r.tolist()) <= pool
    assert set(cm.sizes) == {round(float(real.sizes.mean()))}  # the papers' fixed recipe size
    assert stats["mutations_per_copy"] == DEFAULT_MUTATIONS[variant]
    assert stats["seed_recipes"] == round(20 / stats["phi"])
    assert abs(stats["final_pool"] - min(len(pool), stats["phi"] * len(real))) <= 1  # the pool keeps pace with φ


def test_seeded_and_reproducible(real):
    a, b, c = (copy_mutate(real, seed=s) for s in (3, 3, 4))
    assert all(np.array_equal(x, y) for x, y in zip(a.recipes, b.recipes, strict=True))
    assert any(not np.array_equal(x, y) for x, y in zip(a.recipes, c.recipes, strict=True))


def test_empirical_sizes_come_from_the_real_distribution(real):
    cm = copy_mutate(real, sizes="empirical", seed=5)
    assert set(cm.sizes) <= set(real.sizes) and len(set(cm.sizes)) > 3


def test_without_mutation_every_recipe_is_a_copy_of_a_seed_recipe(real):
    cm, stats = copy_mutate_with_stats(real, mutations=0, seed=2)
    seeds = {tuple(r) for r in cm.recipes[: stats["seed_recipes"]]}
    assert {tuple(r) for r in cm.recipes} <= seeds and stats["mutations_accepted"] == 0


def test_cm_c_keeps_each_lineage_category_composition(real):
    """CM-C replaces an ingredient only by one of the same category, so compositions are inherited from R0."""
    cm, stats = copy_mutate_with_stats(real, variant="CM-C", sizes="empirical", seed=7)
    seeds = {composition(r, real.categories) for r in cm.recipes[: stats["seed_recipes"]]}
    assert {composition(r, real.categories) for r in cm.recipes} <= seeds
    assert stats["mutations_accepted"] > 0
    cm_r, st_r = copy_mutate_with_stats(real, variant="CM-R", sizes="empirical", seed=7)
    seeds_r = {composition(r, real.categories) for r in cm_r.recipes[: st_r["seed_recipes"]]}
    assert not {composition(r, real.categories) for r in cm_r.recipes} <= seeds_r  # CM-R crosses categories


def test_ranked_fitness_favours_the_real_cuisines_popular_ingredients(real):
    """Jain & Bagler (2018): ranked fitness pulls recipes toward frequently used ingredients."""
    p_real = prevalence(real)

    def popularity(cm: Cuisine) -> float:  # mean real prevalence of the ingredients the model cuisine uses
        return float(np.mean([p_real.get(int(i), 0.0) for r in cm.recipes for i in r]))

    ranked = np.mean([popularity(c) for c in ensemble(real, 5, seed=10, fitness="ranked")])
    rand = np.mean([popularity(c) for c in ensemble(real, 5, seed=10, fitness="random")])
    assert ranked > 1.2 * rand


def test_custom_size_and_validation(real):
    assert len(copy_mutate(real, n_recipes=1000, seed=0)) == 1000
    assert len(copy_mutate(real, n_recipes=10, seed=0)) == 10  # fewer than the seed pool: R0 is truncated
    with pytest.raises(ValueError):
        copy_mutate(real, variant="CM-X")
    with pytest.raises(ValueError):
        copy_mutate(real, fitness="loved")
    with pytest.raises(ValueError):
        copy_mutate(real, sizes="random")

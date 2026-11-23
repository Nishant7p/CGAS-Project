"""Unit tests for flavoraudit core: Ns, null models, Z, contributions, synonym decoding (toy data, no I/O)."""

from collections import Counter

import numpy as np
import pytest

from flavoraudit.contribution import chi_ahn, pair_mean, remove
from flavoraudit.flavor import decode_synonym
from flavoraudit.nulls import NULL_MODELS, generate
from flavoraudit.pairing import Cuisine, group_by_size, ns_from_groups, ns_recipe, ns_values
from flavoraudit.stats import food_pairing, z_sb

# Toy shared-compound matrix for 6 ingredients (symmetric, diagonal = profile sizes).
S = np.array(
    [
        [10, 6, 0, 1, 2, 0],
        [6, 8, 1, 0, 3, 0],
        [0, 1, 5, 4, 0, 1],
        [1, 0, 4, 7, 0, 2],
        [2, 3, 0, 0, 9, 0],
        [0, 0, 1, 2, 0, 3],
    ],
    dtype=np.int32,
)
CATS = {0: "Spice", 1: "Spice", 2: "Vegetable", 3: "Vegetable", 4: "Dairy", 5: "Dairy"}


def toy_cuisine():
    recs = [[0, 1, 2], [0, 1], [2, 3, 5], [0, 4, 1, 3], [3, 5], [1, 2, 4], [0, 1, 4]]
    return Cuisine("Toy", [np.array(sorted(r)) for r in recs], CATS)


def test_ns_recipe_by_hand():
    assert ns_recipe(np.array([0, 1, 2]), S) == pytest.approx((6 + 0 + 1) / 3)
    assert ns_recipe(np.array([2, 0, 1]), S) == ns_recipe(np.array([0, 1, 2]), S)  # order-invariant
    assert np.isnan(ns_recipe(np.array([3]), S))


def test_ns_groups_match_single_recipe_function():
    cu = toy_cuisine()
    vals = ns_values(cu, S)
    expected = sorted(ns_recipe(r, S) for r in cu.recipes)
    assert sorted(vals) == pytest.approx(expected)
    assert set(group_by_size(cu.recipes)) == {2, 3, 4}


@pytest.mark.parametrize("model", NULL_MODELS)
def test_null_models_preserve_sizes_pool_and_distinctness(model):
    cu = toy_cuisine()
    groups = generate(cu, model, n=4000, seed=3)
    real_sizes = Counter(cu.sizes.tolist())
    assert set(groups) <= set(real_sizes)
    n = sum(len(v) for v in groups.values())
    assert n == 4000
    for s, arr in groups.items():
        assert arr.shape[1] == s
        assert all(len(set(row)) == s for row in arr)  # no ingredient twice in a recipe
        assert set(np.unique(arr)) <= {0, 1, 2, 3, 4, 5}
        assert abs(len(arr) / n - real_sizes[s] / len(cu)) < 0.05  # size distribution preserved


def test_category_nulls_keep_a_real_category_composition():
    cu = toy_cuisine()
    real_comps = {tuple(sorted(Counter(CATS[int(i)] for i in r).items())) for r in cu.recipes}
    for model in ("category", "frequency_category"):
        for arr in generate(cu, model, n=2000, seed=5).values():
            for row in arr[:300]:
                assert tuple(sorted(Counter(CATS[int(i)] for i in row).items())) in real_comps


def test_frequency_null_follows_ingredient_frequency():
    cu = toy_cuisine()
    groups = generate(cu, "frequency", n=20000, seed=1)
    use = Counter(int(i) for arr in groups.values() for i in arr.ravel())
    real = cu.ingredient_counts()
    assert use.most_common(1)[0][0] == real.most_common(1)[0][0]  # ingredient 0/1 dominate in both


def test_pair_mean_is_the_exact_random_null_expectation():
    cu = toy_cuisine()
    pool, _ = cu.pool()
    sim = ns_from_groups(generate(cu, "random", n=60000, seed=11), S).mean()
    assert sim == pytest.approx(pair_mean(pool, S), rel=0.01)


def test_z_sb_formula_and_food_pairing_fields():
    null = np.array([1.0, 2.0, 3.0, 4.0])
    assert z_sb(3.0, null, n_ref=100) == pytest.approx(10 * (3.0 - 2.5) / np.std(null, ddof=1))
    r = food_pairing(toy_cuisine(), S, "random", n_null=5000, seed=0, n_ref=100_000)
    assert r.n_null == 5000 and r.z_sb == pytest.approx(np.sqrt(100_000) * r.d)
    assert r.z_cuisine == pytest.approx(np.sqrt(r.n_recipes) * r.d)


def test_remove_procedures():
    cu = toy_cuisine()
    no_ing = remove(cu, 0, "ingredient")
    assert all(0 not in r and len(r) >= 2 for r in no_ing.recipes)
    no_rec = remove(cu, 0, "recipes")
    assert len(no_rec) == sum(0 not in r for r in cu.recipes)


def test_chi_ahn_covers_pool_and_is_finite():
    df = chi_ahn(toy_cuisine(), S)
    assert set(df["row"]) == {0, 1, 2, 3, 4, 5}
    assert np.isfinite(df["chi_ahn"]).all()


@pytest.mark.parametrize(
    "syn,expected",
    [
        ("pepper-bell", {"pepper bell", "bell pepper"}),
        ("bay=leaf", {"bay leaf"}),
        ("#cooking oil#", {"cooking oil"}),
        ("citrus-peel=oil", {"citrus peel oil", "peel oil citrus"}),
        ("kaali=mirch", {"kaali mirch"}),
    ],
)
def test_decode_synonym(syn, expected):
    assert set(decode_synonym(syn)) == expected

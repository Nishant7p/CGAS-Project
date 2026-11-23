"""Tests for composition / stereotype / diversity / novelty metrics and category fingerprints (toy data)."""

import numpy as np
import pytest

from flavoraudit.composition import (
    amplification,
    category_jsd,
    category_shares,
    diversity,
    frequency_rank,
    frequency_rank_mae,
    novelty,
    prevalence,
    prevalence_jsd,
    prevalence_table,
    relative_prevalence,
    sai,
    signature_set,
)
from flavoraudit.fingerprint import fingerprint, fingerprint_similarity
from flavoraudit.pairing import Cuisine

CATS = {0: "Spice", 1: "Spice", 2: "Vegetable", 3: "Vegetable", 4: "Dairy", 5: "Dairy"}
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


def cu(name, recs):
    return Cuisine(name, [np.array(sorted(r)) for r in recs], CATS)


A = cu("A", [[0, 1, 2], [0, 1], [0, 4], [2, 3, 5], [0, 1, 4]])
B = cu("B", [[2, 3], [3, 5], [2, 3, 5], [4, 5], [2, 4]])


def test_prevalence_and_relative_prevalence():
    p = prevalence(A)
    assert p[0] == pytest.approx(4 / 5) and p[3] == pytest.approx(1 / 5)
    rel = relative_prevalence(prevalence_table({"A": A, "B": B}))
    assert rel.loc[0, "A"] == pytest.approx(4 / 5 - 0.0)  # ingredient 0 never used in B
    assert signature_set(rel, "A", k=2) == [0, 1]


def test_sai_and_amplification():
    stereo = cu("LLM", [[0, 1], [0, 1, 4], [0, 1, 2], [0, 1], [0, 1, 3]])  # over-uses A's signature (0, 1)
    sig = [0, 1]
    assert sai(stereo, A, sig) > 1
    assert sai(A, A, sig) == pytest.approx(1.0)
    amp = amplification(stereo, A, sig)
    assert (amp > 0).all()


def test_divergences_are_zero_for_identical_and_positive_otherwise():
    assert prevalence_jsd(A, A) == pytest.approx(0.0, abs=1e-12)
    assert 0 < prevalence_jsd(A, B) <= 1
    assert category_jsd(A, A) == pytest.approx(0.0, abs=1e-12)
    assert category_shares(A).sum() == pytest.approx(1.0)
    assert frequency_rank(A)[0] == 1.0 and frequency_rank_mae(A, A) == 0.0


def test_diversity_rarefaction():
    d_full = diversity(A)
    d_small = diversity(A, n=3, reps=50)
    assert d_full["unique_ingredients"] == 6  # A uses every ingredient 0..5
    assert d_small["unique_ingredients"] <= d_full["unique_ingredients"]
    assert 0 < d_small["distinct_recipe_share"] <= 1


def test_novelty_exact_copy_and_disjoint():
    ref = [np.array([0, 1, 2]), np.array([3, 4])]
    cand = cu("C", [[0, 1, 2], [5, 3], [0, 4]])
    nov = novelty(cand, ref, n_cols=6)
    assert nov[0] == pytest.approx(1.0)  # exact copy
    assert nov[1] == pytest.approx(1 / 3)  # {3,5} vs {3,4}
    assert nov[2] == pytest.approx(1 / 3)  # {0,4} vs {0,1,2}: 1/4; vs {3,4}: 1/3 -> max 1/3
    assert np.all((nov >= 0) & (nov <= 1))


def test_fingerprint_runs_and_compares():
    fa = fingerprint(A, S, n_null=2000, seed=1, min_recipes=1)
    assert set(fa.columns) == {"n_recipes", "delta_ns_cat", "z_cat"}
    assert set(fa.index) <= {"Spice", "Vegetable", "Dairy"}
    sim = fingerprint_similarity(fa, fa)
    assert sim["n_categories"] <= 3
    if sim["n_categories"] == 3:
        assert sim["spearman"] == pytest.approx(1.0) and sim["cosine"] == pytest.approx(1.0)
    else:
        assert np.isnan(sim["spearman"])  # too few finite categories to compare


def test_fingerprint_of_tiny_cuisine_is_empty_not_an_error():
    f = fingerprint(A, S, n_null=500, seed=1, min_recipes=100)
    assert f.empty and list(f.columns) == ["n_recipes", "delta_ns_cat", "z_cat"]
    assert fingerprint_similarity(f, f)["n_categories"] == 0


def test_fingerprint_similarity_ignores_nan_categories():
    import pandas as pd

    a = pd.DataFrame({"z_cat": [1.0, 2.0, 3.0, float("nan")]}, index=["A", "B", "C", "D"])
    b = pd.DataFrame({"z_cat": [2.0, 4.0, 6.0, 1.0]}, index=["A", "B", "C", "D"])
    sim = fingerprint_similarity(a, b)
    assert sim["n_categories"] == 3 and sim["spearman"] == pytest.approx(1.0) and sim["cosine"] == pytest.approx(1.0)


def test_category_randomized_null_keeps_the_rest_of_each_template():
    """Jain et al. (2015): only category c is re-drawn; other ingredients and the c-count of the template are kept."""
    from collections import Counter

    from flavoraudit.fingerprint import category_randomized

    rng = np.random.default_rng(0)
    cats = {i: ("Spice" if i < 12 else f"other{i % 3}") for i in range(40)}
    recipes = [np.sort(rng.choice(40, size=int(rng.integers(3, 9)), replace=False)) for _ in range(200)]
    c = Cuisine("toy", recipes, cats)
    templates = Counter()
    for r in recipes:
        if any(cats[int(i)] == "Spice" for i in r):
            templates[
                (tuple(int(i) for i in r if cats[int(i)] != "Spice"), sum(cats[int(i)] == "Spice" for i in r))
            ] += 1
    used_spices = {int(i) for r in recipes for i in r if cats[int(i)] == "Spice"}
    null = category_randomized(c, "Spice", 3000, rng)
    assert len(null) == 3000
    for r in null:
        assert len(set(r.tolist())) == len(r)  # distinct ingredients
        rest = tuple(sorted(int(i) for i in r if cats[int(i)] != "Spice"))
        spices = [int(i) for i in r if cats[int(i)] == "Spice"]
        assert (rest, len(spices)) in templates and set(spices) <= used_spices
    assert category_randomized(c, "Nothing", 10, rng) == []

"""Tests for the D3 fidelity scorer (flavoraudit.scoring) on a toy world of three cuisines."""

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("sklearn")

from flavoraudit.pairing import Cuisine, ns_recipe  # noqa: E402
from flavoraudit.scoring import (  # noqa: E402
    Cooccurrence,
    Featurizer,
    FidelityScorer,
    expected_calibration_error,
    fit_temperature,
)

N_ING = 60


def toy_world(seed: int = 0, n: int = 300):
    """Cuisine k favours ingredients 20k..20k+19 and always pairs ingredient 20k with 20k+1."""
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 12, size=(N_ING, N_ING))
    S = np.triu(S, 1) + np.triu(S, 1).T + np.diag(rng.integers(10, 30, N_ING))
    cats = {i: f"cat{i % 4}" for i in range(N_ING)}
    out = {}
    for k, name in enumerate(["Alpha", "Beta", "Gamma"]):
        home = np.arange(20 * k, 20 * k + 20)
        recs = []
        for _ in range(n):
            s = int(rng.integers(4, 9))
            pick = set(rng.choice(home[2:], size=s - 3, replace=False).tolist())
            pick |= {int(home[0]), int(home[1]), int(rng.integers(0, N_ING))}  # signature pair + one random
            recs.append(np.array(sorted(pick), dtype=np.int64))
        out[name] = Cuisine(name, recs, cats)
    return S.astype(np.int32), out


def split(cuisines, frac, seed):
    rng = np.random.default_rng(seed)
    a, b = {}, {}
    for k, c in cuisines.items():
        idx = rng.permutation(len(c))
        cut = int(frac * len(c))
        a[k] = Cuisine(k, [c.recipes[i] for i in idx[:cut]], c.categories)
        b[k] = Cuisine(k, [c.recipes[i] for i in idx[cut:]], c.categories)
    return a, b


@pytest.fixture(scope="module")
def fitted():
    S, world = toy_world()
    train, rest = split(world, 0.7, 1)
    val, test = split(rest, 0.5, 2)
    return S, FidelityScorer.fit(train, S, val=val), test


def test_featurizer_indicators_and_category_counts():
    cats = {0: "Spice", 1: "Spice", 2: "Dairy", 5: "Dairy"}
    f = Featurizer(np.array([0, 1, 2]), ["Dairy", "Spice"], cats)
    X = f.transform([np.array([0, 1, 5]), np.array([2, 9])]).toarray()
    # columns: ingredient 0, 1, 2, then Dairy count, Spice count; 5 and 9 are unknown ingredients
    assert X.tolist() == [[1, 1, 0, 1, 2], [0, 0, 1, 1, 0]]


def test_shrinkage_pmi():
    X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]]))
    co = Cooccurrence.fit(X, prior=np.array([0.5, 0.4, 0.1]), kappa=2.0)
    pmi = co.pmi(np.array([0, 0]), np.array([1, 2]))
    assert pmi[0] > 0  # 0 and 1 co-occur more than independence predicts
    assert pmi[1] == pytest.approx(np.log(2.0 / (4 + 2.0)))  # never seen together: log(κ / (N + κ))
    assert np.isnan(co.mean_pmi(np.array([0])))


def test_temperature_and_ece():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, size=(4000, 5))
    p = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(5, p=row) for row in p])
    assert fit_temperature(3.0 * true_logits, y) == pytest.approx(3.0, rel=0.15)  # recovers over-confidence
    assert expected_calibration_error(p, y) < 0.03


def test_classifier_and_components(fitted):
    S, sc, test = fitted
    for k, region in enumerate(sc.regions):
        proba = sc.predict_proba(test[region].recipes)
        assert (proba.argmax(axis=1) == k).mean() > 0.9
        assert np.allclose(proba.sum(axis=1), 1)
    comp = sc.components(test["Alpha"].recipes, "Alpha")
    assert list(comp.columns) == ["classifier", "cooccurrence", "ns_consistency"]
    assert comp["ns_consistency"].between(0, 1).all()
    med = np.median([ns_recipe(r, S) for r in test["Alpha"].recipes])
    near = min(test["Alpha"].recipes, key=lambda r: abs(ns_recipe(r, S) - med))
    assert sc.components([near], "Alpha")["ns_consistency"].iloc[0] > 0.7


def test_composite_separates_genuine_from_other_cuisines(fitted):
    from sklearn.metrics import roc_auc_score

    _, sc, test = fitted
    pos = sc.score(test["Beta"].recipes, "Beta")
    neg = sc.score(test["Alpha"].recipes + test["Gamma"].recipes, "Beta")
    auc = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    assert auc > 0.95
    z = sc.zscores(test["Beta"].recipes, "Beta")
    assert abs(z["classifier"].mean()) < 0.5  # standardized on genuine validation recipes
    abl = sc.score(test["Beta"].recipes, "Beta", drop=("classifier",))
    assert abl.shape == pos.shape and not np.allclose(abl, pos)


def test_feedback_names_foreign_ingredients_and_missing_signature(fitted):
    _, sc, _ = fitted
    recipe = np.array([2, 3, 4, 45, 46])  # Alpha's own ingredients plus two from Gamma's home range (40-59)
    assert sc.atypical_cut["Alpha"] < 0  # the 5th percentile of genuine contributions is set by standardize()
    fb = sc.feedback(recipe, "Alpha", typical=[0, 1, 2])
    assert fb["atypical"] and set(fb["atypical"]) <= {45, 46}
    assert fb["missing"] and set(fb["missing"]) <= {0, 1}  # 2 is present, so it is never suggested
    assert fb["ns_verdict"] in {"high", "low", "typical"} and 0 <= fb["ns_percentile"] <= 1
    contrib = sc.contributions(recipe, "Alpha")
    assert contrib.index[0] in (45, 46) and contrib.iloc[-1] > contrib.iloc[0]
    genuine = sc.feedback(np.array([0, 1, 2, 3, 4]), "Alpha", typical=[0, 1, 2])
    assert genuine["atypical"] == [] and genuine["missing"] == []  # nothing to fix in a typical Alpha recipe

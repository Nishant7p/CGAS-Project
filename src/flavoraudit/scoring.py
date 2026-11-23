"""Fidelity scorer (D3; plan §9): how typical is a recipe of the cuisine it claims to be?

Three components, all fitted on genuine CulinaryDB recipes (a training split):

1. **Classifier**: log P(c | R) from a calibrated multinomial classifier on binary ingredient indicators plus
   ingredient-category counts (cuisine classification from ingredients, as in Sharma, Upadhyay & Bagler 2020).
2. **Co-occurrence**: the mean, over the recipe's ingredient pairs, of a shrinkage-smoothed PMI within cuisine c:
       p_c(i)   = (n_ic + κ q_i) / (N_c + κ)             q_i = prevalence of i over all training recipes
       p_c(i,j) = (n_ijc + κ p_c(i) p_c(j)) / (N_c + κ)  the joint shrinks toward independence
       PMI_c(i,j) = log p_c(i,j) / (p_c(i) p_c(j))
   A pair never seen in c scores log(κ / (N_c + κ)) < 0, more negative when c has more evidence; frequent pairs
   approach the plain PMI. (Additive smoothing would give unseen pairs a high PMI, the wrong way round.)
3. **Ns consistency**: 1 - |2 F_c(Ns(R)) - 1|, where F_c is the empirical CDF of recipe Ns in cuisine c (mid-rank).
   It is 1 at cuisine c's median flavor sharing and 0 in the tails (Ns: Ahn et al. 2011).

The composite is S = Σ_k w_k z_k, where z_k standardizes component k with the mean and SD of genuine validation
recipes of c. Ingredients outside the training vocabulary carry no classifier or PMI information and are skipped
there (they still count in Ns). scikit-learn is needed (``pip install flavoraudit[ml]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import log_softmax

from .pairing import Cuisine, ns_recipe

COMPONENTS = ("classifier", "cooccurrence", "ns_consistency")


def _sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("flavoraudit.scoring needs scikit-learn: pip install 'flavoraudit[ml]'") from e


@dataclass
class Featurizer:
    """Binary ingredient indicators over a fixed vocabulary of S rows, plus counts per ingredient category."""

    vocab: np.ndarray  # sorted S rows
    categories: list[str]
    category_of: dict[int, str]

    @classmethod
    def from_cuisines(cls, cuisines: dict[str, Cuisine]) -> Featurizer:
        vocab = np.unique(np.concatenate([np.concatenate(c.recipes) for c in cuisines.values() if c.recipes]))
        cat_of = next(iter(cuisines.values())).categories
        cats = sorted({cat_of.get(int(r), "Unknown") for r in vocab})
        return cls(vocab.astype(np.int64), cats, dict(cat_of))

    def known(self, recipe: np.ndarray) -> np.ndarray:
        """Positions in the vocabulary of the recipe's known ingredients."""
        pos = np.searchsorted(self.vocab, recipe)
        pos = np.clip(pos, 0, len(self.vocab) - 1)
        return pos[self.vocab[pos] == recipe]

    def transform(self, recipes: list[np.ndarray]) -> sparse.csr_matrix:
        """(n_recipes, V + n_categories) sparse features; duplicate category entries sum into counts."""
        n, V, K = len(recipes), len(self.vocab), len(self.categories)
        if n == 0:
            return sparse.csr_matrix((0, V + K))
        lens = np.fromiter((len(r) for r in recipes), dtype=np.int64, count=n)
        flat = np.concatenate([np.asarray(r, dtype=np.int64) for r in recipes])
        rid = np.repeat(np.arange(n), lens)
        pos = np.clip(np.searchsorted(self.vocab, flat), 0, V - 1)
        ok = self.vocab[pos] == flat
        ccode = {c: k for k, c in enumerate(self.categories)}
        unknown = ccode.get("Unknown", -1)
        codes = np.array([ccode.get(self.category_of.get(int(i), "Unknown"), unknown) for i in flat], dtype=np.int64)
        okc = codes >= 0
        rows = np.concatenate([rid[ok], rid[okc]])
        cols = np.concatenate([pos[ok], V + codes[okc]])
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, V + K))


@dataclass
class Cooccurrence:
    """Within-cuisine ingredient and pair counts with shrinkage-smoothed PMI (see the module docstring)."""

    n: int
    counts: np.ndarray  # (V,) recipes of the cuisine containing each vocabulary item
    pairs: sparse.csr_matrix  # (V, V) recipes containing both items
    prior: np.ndarray  # (V,) prevalence over all training recipes
    kappa: float

    @classmethod
    def fit(cls, X: sparse.csr_matrix, prior: np.ndarray, kappa: float) -> Cooccurrence:
        X = (X > 0).astype(np.int32)
        return cls(X.shape[0], np.asarray(X.sum(axis=0)).ravel(), (X.T @ X).tocsr(), prior, kappa)

    def marginal(self, idx: np.ndarray) -> np.ndarray:
        return (self.counts[idx] + self.kappa * self.prior[idx]) / (self.n + self.kappa)

    def pmi(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        pa, pb = self.marginal(a), self.marginal(b)
        nab = np.asarray(self.pairs[a, b]).ravel()
        pab = (nab + self.kappa * pa * pb) / (self.n + self.kappa)
        return np.log(pab / (pa * pb))

    def mean_pmi(self, pos: np.ndarray) -> float:
        if len(pos) < 2:
            return float("nan")
        iu, ju = np.triu_indices(len(pos), 1)
        return float(self.pmi(pos[iu], pos[ju]).mean())


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """ECE of the top-1 prediction (Guo et al. 2017): Σ_b |B_b|/n · |accuracy_b - confidence_b|."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimizing the validation NLL of softmax(logits / T) (Guo et al. 2017)."""
    from scipy.optimize import minimize_scalar

    def nll(log_t: float) -> float:
        return float(-log_softmax(logits / np.exp(log_t), axis=1)[np.arange(len(y)), y].mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


@dataclass
class FidelityScorer:
    regions: list[str]
    featurizer: Featurizer
    clf: object  # fitted scikit-learn classifier with decision_function (logits)
    temperature: float
    cooc: dict[str, Cooccurrence]
    ns_sorted: dict[str, np.ndarray]
    S: np.ndarray
    stats: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)  # region -> comp -> (mean, sd)
    weights: dict[str, float] = field(default_factory=lambda: dict.fromkeys(COMPONENTS, 1 / 3))
    atypical_cut: dict[str, float] = field(default_factory=dict)  # region -> 5th percentile of genuine contributions

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        train: dict[str, Cuisine],
        S: np.ndarray,
        val: dict[str, Cuisine] | None = None,
        C: float = 1.0,
        kappa: float = 10.0,
        clf=None,
        seed: int = 0,
    ) -> FidelityScorer:
        """Fit every component on ``train``; calibrate and standardize on ``val`` when given."""
        _sklearn()
        from sklearn.linear_model import LogisticRegression

        regions = sorted(train)
        feat = Featurizer.from_cuisines(train)
        recs = [r for c in regions for r in train[c].recipes]
        y = np.concatenate([np.full(len(train[c]), k) for k, c in enumerate(regions)])
        X = feat.transform(recs)
        model = clf if clf is not None else LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        model.fit(X, y)
        V = len(feat.vocab)
        prior = np.asarray((X[:, :V] > 0).mean(axis=0)).ravel()
        cooc = {c: Cooccurrence.fit(X[y == k, :V], prior, kappa) for k, c in enumerate(regions)}
        ns_sorted = {c: np.sort([ns_recipe(r, S) for r in train[c].recipes if len(r) >= 2]) for c in regions}
        scorer = cls(regions, feat, model, 1.0, cooc, ns_sorted, S)
        if val is not None:
            scorer.calibrate(val)
            scorer.standardize(val)
        return scorer

    def logits(self, recipes: list[np.ndarray]) -> np.ndarray:
        return np.asarray(self.clf.decision_function(self.featurizer.transform(recipes)))

    def calibrate(self, val: dict[str, Cuisine]) -> float:
        recs = [r for c in self.regions for r in val[c].recipes]
        y = np.concatenate([np.full(len(val[c]), k) for k, c in enumerate(self.regions)])
        self.temperature = fit_temperature(self.logits(recs), y)
        return self.temperature

    def standardize(self, val: dict[str, Cuisine]) -> None:
        """Mean and SD of each component over genuine validation recipes of each cuisine, and (for a linear
        classifier) the cuisine's atypical-ingredient cut: the 5th percentile of the classifier contributions of the
        ingredients in those genuine recipes."""
        for c in self.regions:
            comp = self.components(val[c].recipes, c)
            self.stats[c] = {k: (float(comp[k].mean()), float(comp[k].std(ddof=1) or 1.0)) for k in COMPONENTS}
            if hasattr(self.clf, "coef_") and len(val[c]):
                allc = np.concatenate([self.contributions(r, c).to_numpy() for r in val[c].recipes])
                self.atypical_cut[c] = float(np.quantile(allc, 0.05)) if len(allc) else 0.0

    # ------------------------------------------------------------------ scoring
    def predict_proba(self, recipes: list[np.ndarray]) -> np.ndarray:
        return np.exp(log_softmax(self.logits(recipes) / self.temperature, axis=1))

    def components(self, recipes: list[np.ndarray], region: str) -> pd.DataFrame:
        """The three raw component scores of each recipe as cuisine ``region``."""
        k = self.regions.index(region)
        logp = log_softmax(self.logits(recipes) / self.temperature, axis=1)[:, k]
        co = self.cooc[region]
        pmi = [co.mean_pmi(self.featurizer.known(np.asarray(r))) for r in recipes]
        ref = self.ns_sorted[region]
        ns = np.array([ns_recipe(r, self.S) for r in recipes])
        lo, hi = np.searchsorted(ref, ns, side="left"), np.searchsorted(ref, ns, side="right")
        F = (lo + hi) / (2 * len(ref))
        cons = np.where(np.isnan(ns), np.nan, 1 - np.abs(2 * F - 1))
        return pd.DataFrame({"classifier": logp, "cooccurrence": pmi, "ns_consistency": cons})

    def zscores(self, recipes: list[np.ndarray], region: str) -> pd.DataFrame:
        comp = self.components(recipes, region)
        if region not in self.stats:
            raise ValueError("standardize() on validation recipes first")
        return pd.DataFrame({k: (comp[k] - m) / s for k, (m, s) in self.stats[region].items()})

    def score(self, recipes: list[np.ndarray], region: str, drop: tuple[str, ...] = ()) -> np.ndarray:
        """Composite fidelity S = Σ w_k z_k (missing components count as 0, i.e. typical); ``drop`` for ablations."""
        z = self.zscores(recipes, region).fillna(0.0)
        w = {k: v for k, v in self.weights.items() if k not in drop}
        total = sum(w.values())
        return sum(z[k].to_numpy() * v for k, v in w.items()) / total

    # ------------------------------------------------------------------ explanation (D3 critique, D5 suggestions)
    def contributions(self, recipe: np.ndarray, region: str) -> pd.Series:
        """Each known ingredient's contribution to the classifier's log-odds for ``region`` vs the average cuisine.

        Uses the linear classifier's coefficients: w_{c,i} - mean_k w_{k,i}. Negative = atypical for the cuisine.
        """
        if not hasattr(self.clf, "coef_"):
            raise TypeError("contributions need a linear classifier (coef_)")
        k = self.regions.index(region)
        W = np.asarray(self.clf.coef_)
        pos = self.featurizer.known(np.asarray(recipe))
        contrib = W[k, pos] - W[:, pos].mean(axis=0)
        return pd.Series(contrib, index=self.featurizer.vocab[pos].astype(int)).sort_values()

    def feedback(
        self,
        recipe: np.ndarray,
        region: str,
        typical: list[int],
        atypical_below: float | None = None,
        max_items: int = 3,
    ) -> dict:
        """What makes a recipe atypical of ``region``, as S rows (plan §9.5, critique-and-revise).

        * atypical: up to ``max_items`` ingredients whose classifier contribution is below ``atypical_below``, by
          default the cuisine's own cut from ``standardize`` (the 5th percentile of contributions of ingredients in
          genuine validation recipes; 0 if not standardized), most atypical first;
        * missing: up to ``max_items`` of the ``typical`` ingredients (e.g. the cuisine's signature set) that the
          recipe lacks and that pair well (positive mean PMI) with its core, i.e. its non-atypical ingredients;
        * ns_verdict: 'high' or 'low' when the recipe's Ns lies above the 75th or below the 25th percentile of the
          cuisine's genuine recipes, else 'typical'.
        """
        contrib = self.contributions(recipe, region)
        cut = atypical_below if atypical_below is not None else self.atypical_cut.get(region, 0.0)
        atypical = [int(r) for r, v in contrib.items() if v < cut][:max_items]
        core = np.array([r for r in contrib.index if r not in atypical], dtype=np.int64)
        pos = self.featurizer.known(core) if len(core) else np.array([], dtype=np.int64)
        present = {int(r) for r in recipe}
        co = self.cooc[region]
        scored = []
        for r in typical:
            if r in present:
                continue
            cp = self.featurizer.known(np.array([r], dtype=np.int64))
            if len(cp) and len(pos):
                m = float(co.pmi(np.full(len(pos), cp[0]), pos).mean())
                if m > 0:
                    scored.append((m, int(r)))
        missing = [r for _, r in sorted(scored, reverse=True)[:max_items]]
        ns = ns_recipe(np.asarray(recipe), self.S)
        ref = self.ns_sorted[region]
        pct = float(np.searchsorted(ref, ns) / len(ref)) if np.isfinite(ns) else float("nan")
        verdict = "typical" if not np.isfinite(pct) or 0.25 <= pct <= 0.75 else ("high" if pct > 0.75 else "low")
        return {"atypical": atypical, "missing": missing, "ns": ns, "ns_percentile": pct, "ns_verdict": verdict}

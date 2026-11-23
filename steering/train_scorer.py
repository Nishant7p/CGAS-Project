"""D3, Phase 8 steps 1-4: train, select, calibrate and validate the fidelity scorer on CulinaryDB (plan §9).

Splits (per region, seeded): A_train 70% | A_val 10% | B_test 20%.
1. Classifier selection on A_val by macro-F1 among the LINEAR candidates, logistic regression (L2, class-balanced;
   C in {0.3, 1, 3}) and a linear SVM. The winner is frozen. The scorer needs a linear model because its
   per-ingredient contributions (w_c,i - mean_k w_k,i) drive the critique-and-revise feedback and the app's
   suggestions. Histogram gradient boosting (scikit-learn's LightGBM-style GBDT) is fitted as the non-linear
   comparator and its validation score is reported; it is also the independent evaluation classifier of step 5.
2. The scorer (flavoraudit.scoring.FidelityScorer) is fitted on A_train, temperature-calibrated and standardized on
   A_val. Component weights: the best of {equal, logistic fitted against other cuisines, logistic fitted against all
   three negative types} by the mean A_val macro AUC over the three negative types of step 4 (decided on A_val only,
   before any LLM recipe is scored; the first version judged weights against other cuisines only, which under-weighted
   co-occurrence, the component that detects unnatural combinations).
3. Test metrics on B_test: macro-F1, top-1/top-3 accuracy, ECE before/after temperature scaling, confusion matrix.
4. Scorer validation on B_test, per region c: AUC of genuine c recipes against (i) other cuisines' recipes scored as
   c, (ii) copy-mutate recipes evolved from c's training recipes (CM-R, ranked fitness), (iii) random-null recipes of
   c; for the composite and for each component.
5. The independent evaluation classifier for steering (plan §9.6) is a different model class trained on B_test;
   its macro-F1 is measured on A_val.

Usage (repo root): .venv/bin/python steering/train_scorer.py
Output: data/interim/steering/{scorer.pkl, eval_classifier.pkl, splits.json, scorer_metrics.json, confusion.csv}
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score, top_k_accuracy_score
from sklearn.svm import LinearSVC

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis" / "audit"))
from audit_io import load_reference  # noqa: E402

from flavoraudit import nulls  # noqa: E402
from flavoraudit.evolution import copy_mutate  # noqa: E402
from flavoraudit.pairing import Cuisine  # noqa: E402
from flavoraudit.scoring import COMPONENTS, FidelityScorer, expected_calibration_error  # noqa: E402

OUT = REPO / "data" / "interim" / "steering"
FRACTIONS = (0.7, 0.1, 0.2)
SEED = 0
MAX_OTHER = 3000  # other-cuisine negatives per region (sampled)


def split_cuisines(real: dict[str, Cuisine]) -> tuple[dict, dict, dict, dict]:
    parts: tuple[dict, dict, dict] = ({}, {}, {})
    positions = {}
    for region, c in real.items():
        rng = np.random.default_rng([SEED, zlib.crc32(region.encode())])
        idx = rng.permutation(len(c))
        a, b = int(FRACTIONS[0] * len(c)), int((FRACTIONS[0] + FRACTIONS[1]) * len(c))
        for part, sel in zip(parts, (idx[:a], idx[a:b], idx[b:]), strict=True):
            part[region] = Cuisine(region, [c.recipes[i] for i in sel], c.categories)
        positions[region] = {"train": idx[:a].tolist(), "val": idx[a:b].tolist(), "test": idx[b:].tolist()}
    return *parts, positions


def xy(scorer_feat, cuisines: dict[str, Cuisine], regions: list[str]):
    recs = [r for c in regions for r in cuisines[c].recipes]
    y = np.concatenate([np.full(len(cuisines[c]), k) for k, c in enumerate(regions)])
    return scorer_feat.transform(recs), y


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ref, *_ = load_reference()
    S = ref.S
    train, val, test, positions = split_cuisines(ref.real)
    regions = sorted(train)
    (OUT / "splits.json").write_text(json.dumps({"fractions": FRACTIONS, "seed": SEED, "positions": positions}))
    print("recipes:", {k: sum(len(p[r]) for r in regions) for k, p in (("train", train), ("val", val), ("test", test))})

    # 1. classifier selection on A_val (features from the training vocabulary)
    probe = FidelityScorer.fit(train, S, clf=LogisticRegression(C=1.0, class_weight="balanced", max_iter=50))
    feat = probe.featurizer
    Xtr, ytr = xy(feat, train, regions)
    Xva, yva = xy(feat, val, regions)
    candidates = {
        "logreg_C0.3": LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000),
        "logreg_C1": LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000),
        "logreg_C3": LogisticRegression(C=3.0, class_weight="balanced", max_iter=3000),
        "linear_svm": LinearSVC(C=0.1, class_weight="balanced", random_state=SEED),
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=SEED
        ),
    }
    selection = {}
    for name, model in candidates.items():
        t1 = time.time()
        Xfit, Xev = (Xtr.toarray(), Xva.toarray()) if name == "hist_gbdt" else (Xtr, Xva)
        model.fit(Xfit, ytr)
        pred = model.predict(Xev)
        selection[name] = {
            "val_macro_f1": float(f1_score(yva, pred, average="macro")),
            "val_top1": float((pred == yva).mean()),
            "fit_seconds": round(time.time() - t1, 1),
        }
        print(f"  {name:12s} {selection[name]}", flush=True)
    linear = [k for k in selection if k != "hist_gbdt"]
    best = max(linear, key=lambda k: selection[k]["val_macro_f1"])
    print("selected (linear):", best, "| non-linear comparator hist_gbdt:", selection["hist_gbdt"])

    # 2. the scorer with the selected classifier
    model = candidates[best]
    scorer = FidelityScorer.fit(train, S, val=val, clf=model)
    weights_equal = dict(scorer.weights)

    # Component weights, chosen on A_val only: z-scores of genuine validation recipes of c and of the three negative
    # types of plan §9.4 (other cuisines' validation recipes, copy-mutate and random-null recipes built from c's
    # training recipes) are computed once; each candidate weighting is judged by its mean macro AUC over the 3 types.
    rng_v = np.random.default_rng(SEED + 7)
    kinds = ("other_cuisines", "copy_mutate", "random_null")
    zval = {}
    for c in regions:
        others = [r for o in regions if o != c for r in val[o].recipes]
        others = [others[i] for i in rng_v.choice(len(others), size=min(MAX_OTHER, len(others)), replace=False)]
        cm = copy_mutate(
            train[c], variant="CM-R", fitness="ranked", sizes="empirical", n_recipes=len(train[c]), seed=11
        )
        cm_recs = [cm.recipes[i] for i in rng_v.choice(len(cm), size=min(len(val[c]), len(cm)), replace=False)]
        null_recs = [row for arr in nulls.generate(train[c], "random", n=len(val[c]), seed=12).values() for row in arr]
        zval[c] = {"pos": scorer.zscores(val[c].recipes, c).fillna(0.0)[list(COMPONENTS)].to_numpy()}
        for kind, recs in zip(kinds, (others, cm_recs, null_recs), strict=True):
            zval[c][kind] = scorer.zscores(recs, c).fillna(0.0)[list(COMPONENTS)].to_numpy()

    def val_aucs(w: dict) -> dict:
        ww = np.array([w[k] for k in COMPONENTS], dtype=float)
        ww = ww / ww.sum()
        res = {}
        for kind in kinds:
            res[kind] = float(np.mean([
                roc_auc_score(np.r_[np.ones(len(zval[c]["pos"])), np.zeros(len(zval[c][kind]))],
                              np.r_[zval[c]["pos"] @ ww, zval[c][kind] @ ww])
                for c in regions
            ]))  # fmt: skip
        res["mean"] = float(np.mean([res[k] for k in kinds]))
        return res

    def logistic_weights(neg_kinds: tuple[str, ...]) -> dict:
        X = np.vstack([zval[c]["pos"] for c in regions] + [zval[c][k] for c in regions for k in neg_kinds])
        n_pos = sum(len(zval[c]["pos"]) for c in regions)
        y = np.r_[np.ones(n_pos), np.zeros(len(X) - n_pos)]
        w = np.clip(LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y).coef_.ravel(), 0, None)
        return dict(zip(COMPONENTS, (w / w.sum()).tolist(), strict=True)) if w.sum() > 0 else dict(weights_equal)

    weightings = {
        "equal": dict(weights_equal),
        "logistic_other_cuisines": logistic_weights(("other_cuisines",)),
        "logistic_all_negatives": logistic_weights(kinds),
    }
    weight_selection = {name: {"weights": w, "val_auc": val_aucs(w)} for name, w in weightings.items()}
    chosen_weighting = max(weight_selection, key=lambda k: weight_selection[k]["val_auc"]["mean"])
    scorer.weights = weightings[chosen_weighting]
    for name, v in weight_selection.items():
        print(
            f"  weighting {name:24s} { ({k: round(x, 3) for k, x in v['weights'].items()}) } val AUC { ({k: round(x, 4) for k, x in v['val_auc'].items()}) }"
        )
    print("selected weighting:", chosen_weighting, flush=True)

    # 3. test metrics of the classifier
    Xte, yte = xy(feat, test, regions)
    logits = np.asarray(scorer.clf.decision_function(Xte))
    p_raw, p_cal = softmax_rows(logits), softmax_rows(logits / scorer.temperature)
    pred = p_cal.argmax(axis=1)
    clf_metrics = {
        "selected": best,
        "selection_on_val": selection,
        "temperature": scorer.temperature,
        "test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "test_top1": float((pred == yte).mean()),
        "test_top3": float(top_k_accuracy_score(yte, p_cal, k=3, labels=np.arange(len(regions)))),
        "test_ece_raw": expected_calibration_error(p_raw, yte),
        "test_ece_calibrated": expected_calibration_error(p_cal, yte),
        "test_logloss_raw": float(log_loss(yte, p_raw, labels=np.arange(len(regions)))),
        "test_logloss_calibrated": float(log_loss(yte, p_cal, labels=np.arange(len(regions)))),
        "per_region_f1": dict(zip(regions, f1_score(yte, pred, average=None).round(4).tolist(), strict=True)),
    }
    pd.DataFrame(confusion_matrix(yte, pred), index=regions, columns=regions).to_csv(OUT / "confusion.csv")
    print({k: v for k, v in clf_metrics.items() if k not in ("selection_on_val", "per_region_f1")})

    # 4. scorer validation on B_test: AUC against three kinds of negatives, composite and per component
    rng = np.random.default_rng(SEED)
    rows = []
    for c in regions:
        pos = test[c].recipes
        others = [r for o in regions if o != c for r in test[o].recipes]
        others = [others[i] for i in rng.choice(len(others), size=min(MAX_OTHER, len(others)), replace=False)]
        cm = copy_mutate(train[c], variant="CM-R", fitness="ranked", sizes="empirical", n_recipes=len(train[c]), seed=1)
        cm_recs = [cm.recipes[i] for i in rng.choice(len(cm), size=min(len(pos), len(cm)), replace=False)]
        groups = nulls.generate(train[c], "random", n=len(pos), seed=2)
        null_recs = [row for arr in groups.values() for row in arr]
        negs = {"other_cuisines": others, "copy_mutate": cm_recs, "random_null": null_recs}
        zpos = scorer.zscores(pos, c)
        spos = scorer.score(pos, c)
        for kind, neg in negs.items():
            zneg, sneg = scorer.zscores(neg, c), scorer.score(neg, c)
            yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            row = {"region": c, "negatives": kind, "n_pos": len(pos), "n_neg": len(neg)}
            row["composite"] = roc_auc_score(yy, np.r_[spos, sneg])
            for comp in COMPONENTS:
                v = np.r_[zpos[comp].fillna(0).to_numpy(), zneg[comp].fillna(0).to_numpy()]
                row[comp] = roc_auc_score(yy, v)
            rows.append(row)
    auc = pd.DataFrame(rows)
    auc.to_csv(OUT / "scorer_auc.csv", index=False)
    auc_summary = auc.groupby("negatives")[["composite", *COMPONENTS]].mean().round(4)
    print(auc_summary.to_string())

    # 5. independent evaluation classifier: a different model class, trained on B_test, checked on A_val
    eval_name = "hist_gbdt" if best.startswith(("logreg", "linear")) else "logreg_C1"
    ev = (
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=SEED)
        if eval_name == "hist_gbdt"
        else LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    )
    ev.fit(Xte.toarray() if eval_name == "hist_gbdt" else Xte, yte)
    ev_pred = ev.predict(Xva.toarray() if eval_name == "hist_gbdt" else Xva)
    eval_metrics = {
        "model": eval_name,
        "trained_on": "B_test",
        "macro_f1_on_A_val": float(f1_score(yva, ev_pred, average="macro")),
    }
    print("independent evaluation classifier:", eval_metrics)

    with open(OUT / "scorer.pkl", "wb") as fh:
        pickle.dump(scorer, fh)
    with open(OUT / "eval_classifier.pkl", "wb") as fh:
        pickle.dump({"model": ev, "featurizer": feat, "regions": regions, "name": eval_name}, fh)
    metrics = {
        "classifier": clf_metrics,
        "weights": scorer.weights,
        "weights_selected": chosen_weighting,
        "weights_selection": weight_selection,
        "scorer_auc_test": auc_summary.reset_index().to_dict(orient="records"),
        "eval_classifier": eval_metrics,
        "kappa": scorer.cooc[regions[0]].kappa,
        "vocab_size": int(len(feat.vocab)),
    }
    (OUT / "scorer_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D3 best-of-N: select recipes with the fidelity scorer and evaluate them on held-out metrics (plan §9.5-9.7).

Pool per model x region = the main run's plain-prompt recipes (full mode) plus the best-of-N pool calls
(steering/prompt_sets/steer_bon_v1.jsonl, run on Kaggle in mode "bon"; outputs in data/interim/steering/raw/).
Selection (pre-registered in steering/design.py): the pool is shuffled (seeded) and cut into groups of N; each group
contributes its best-scoring ADMISSIBLE recipe (not a near-copy of a CulinaryDB recipe, max Jaccard < 0.8, and not a
duplicate ingredient set of a recipe already selected) until N_SELECTED recipes are chosen. N = 1 is the unsteered
baseline under the same guards.

Evaluation never uses the scorer (plan §9.6): flavoraudit.audit.scorecard_row of the selected cuisine (Z sign and
ΔZ, fingerprint, prevalence JSD, SAI, diversity, novelty), the independent classifier's share of recipes assigned to
the claimed cuisine (gradient boosting trained on the scorer's held-out split), and output tokens per accepted
recipe. Ablations: selection by the composite with one component dropped.

Usage (repo root):
    .venv/bin/python steering/run_bon.py --models qwen3.5-9b llama-3.1-8b
    .venv/bin/python steering/run_bon.py --models openai-gpt-6-luna --n-values 1 2 4 --n-selected 10 --tag smoke
Output: data/interim/steering/bon_<tag>.parquet (one row per model x region x method)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "audit"))
from audit_io import load_reference  # noqa: E402

from flavoraudit.audit import scorecard_row  # noqa: E402
from flavoraudit.composition import novelty  # noqa: E402
from flavoraudit.pairing import Cuisine  # noqa: E402
from flavoraudit.scoring import COMPONENTS  # noqa: E402
from steering.design import N_SELECTED, N_VALUES, NEAR_COPY, STEER_REGIONS  # noqa: E402

STEER = REPO / "data" / "interim" / "steering"
LINES = REPO / "data" / "generated" / "aliased" / "lines.parquet"


def pool_lines(model: str) -> pd.DataFrame:
    """The main run's aliased lines plus the model's best-of-N pool lines (steering/collect.py), if collected."""
    parts = [pd.read_parquet(LINES)]
    bon = STEER / f"lines_bon_{model}.parquet"
    if bon.exists():
        parts.append(pd.read_parquet(bon))
    return pd.concat(parts, ignore_index=True)


def pool_for(
    model: str, lines: pd.DataFrame, idx: dict[int, int], regions=STEER_REGIONS, conditions=("plain",)
) -> pd.DataFrame:
    """One row per candidate recipe: region code, uid and S rows."""
    sel = lines[
        (lines["model_key"] == model)
        & (lines["condition"].isin(conditions))
        & (lines["region_code"].isin(regions))
        & (lines["recipe_idx"] < 5)
        & (lines["mode"].isin(["full", "bon"]))
    ]
    rows = []
    for (code, uid), g in sel.groupby(["region_code", "recipe_uid"], sort=True):
        ents = [idx[int(e)] for e in g["entity_id"].dropna() if int(e) in idx]
        rows.append({"region_code": code, "uid": uid, "rows": np.unique(np.array(ents, dtype=np.int64))})
    return pd.DataFrame(rows)


def select(
    pool: pd.DataFrame, scores: np.ndarray, admissible: np.ndarray, n: int, k: int, seed: int
) -> tuple[list[int], int]:
    """Best-of-n selection: (selected positions in ``pool``, candidates examined)."""
    order = np.random.default_rng(seed).permutation(len(pool))
    chosen, seen_sets, examined = [], set(), 0
    for start in range(0, len(order) - n + 1, n):
        group = order[start : start + n]
        examined += n
        for j in sorted(group, key=lambda t: -scores[t]):
            key = tuple(pool["rows"].iloc[j])
            if admissible[j] and key not in seen_sets and len(key) >= 2:
                chosen.append(int(j))
                seen_sets.add(key)
                break
        if len(chosen) == k:
            break
    return chosen, examined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-values", nargs="+", type=int, default=list(N_VALUES))
    ap.add_argument("--n-selected", type=int, default=N_SELECTED)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regions", nargs="+", default=list(STEER_REGIONS), help="region codes, or 'all'")
    ap.add_argument("--conditions", nargs="+", default=["plain"])
    args = ap.parse_args()
    t0 = time.time()
    ref, _fp, idx, _ = load_reference()
    with open(STEER / "scorer.pkl", "rb") as fh:
        scorer = pickle.load(fh)
    with open(STEER / "eval_classifier.pkl", "rb") as fh:
        ev = pickle.load(fh)
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs" / "regions.yaml").read_text()).items()}
    cats = next(iter(ref.real.values())).categories
    out, selected = [], []
    for model in args.models:
        regions = list(labels) if args.regions == ["all"] else args.regions
        pool_all = pool_for(model, pool_lines(model), idx, regions, tuple(args.conditions))
        if pool_all.empty:
            print(f"{model}: no pool recipes")
            continue
        nov = novelty(Cuisine("pool", list(pool_all["rows"]), cats), ref.novelty_matrix(), ref.S.shape[0])
        pool_all["admissible"] = nov < NEAR_COPY
        for code, pool in pool_all.groupby("region_code", sort=True):
            region = labels[code]
            pool = pool.reset_index(drop=True)
            recs = list(pool["rows"])
            methods = {"composite": scorer.score(recs, region)}
            for comp in COMPONENTS:
                methods[f"without_{comp}"] = scorer.score(recs, region, drop=(comp,))
            for method, scores in methods.items():
                for n in args.n_values:
                    if method != "composite" and n != max(args.n_values):
                        continue  # ablations only at the largest N
                    chosen, examined = select(
                        pool, scores, pool["admissible"].to_numpy(), n, args.n_selected, args.seed
                    )
                    if len(chosen) < 2:
                        continue
                    # canonical (uid) order: seeded nulls then give identical values for identical recipe sets,
                    # e.g. N = 1 here and round 0 of steering/run_revise.py
                    canon = sorted(chosen, key=lambda j: str(pool["uid"].iloc[j]))
                    cu = Cuisine(f"{model}:{code}:{method}:N{n}", [recs[j] for j in canon], cats)
                    row = scorecard_row(cu, region, ref, seed=args.seed, n_boot=500)
                    Xe = ev["featurizer"].transform(cu.recipes)
                    pred = ev["model"].predict(Xe.toarray() if ev["name"] == "hist_gbdt" else Xe)
                    row["indep_classifier_acc"] = float((np.array(ev["regions"])[pred] == region).mean())
                    row["mean_composite_selected"] = float(scores[chosen].mean())
                    out.append({"model": model, "code": code, "method": method, "N": n, "n_pool": len(pool),
                                "n_examined": examined, "n_selected": len(chosen), **row})  # fmt: skip
                    selected += [{"model": model, "code": code, "method": method, "N": n, "rank": k,
                                  "uid": pool["uid"].iloc[j], "score": float(scores[j])}
                                 for k, j in enumerate(chosen)]  # fmt: skip
            print(f"{model:22s} {code:5s} pool={len(pool):5d} | {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    path = STEER / f"bon_{args.tag}.parquet"
    df.to_parquet(path, index=False)
    pd.DataFrame(selected).to_parquet(STEER / f"bon_{args.tag}_selected.parquet", index=False)
    summ = (
        df[df["method"] == "composite"]
        .groupby(["model", "N"])[
            [
                "sign_match",
                "fingerprint_spearman",
                "prevalence_jsd",
                "indep_classifier_acc",
                "near_copy_rate",
                "diversity_ratio",
            ]
        ]
        .mean()
    )
    print(summ.round(3).to_string())
    (STEER / f"bon_{args.tag}_summary.json").write_text(
        json.dumps(summ.reset_index().to_dict(orient="records"), indent=1)
    )
    print(f"done in {time.time() - t0:.0f}s -> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D3 critique-and-revise (Self-Refine; Madaan et al. 2023), up to 2 rounds (plan §9.5; steering/design.py).

prepare  Round 1 revises the unsteered N = 1 selection of steering/run_bon.py (100 recipes per region); round r > 1
         revises round r-1's outputs. Each prompt carries the scorer's feedback (atypical ingredients, missing
         signature ingredients that pair well, the Ns verdict) and is written to
         steering/prompt_sets/steer_rev<r>_<model>.jsonl, to run on Kaggle in mode rev<r>.
evaluate Every round (0 = the unsteered parents) is scored on held-out metrics, as best-of-N: scorecard_row of each
         region's cuisine, the independent classifier's accuracy, and identity preservation (the Jaccard similarity of
         a revised recipe's ingredients with its parent's).

Usage (repo root):
    .venv/bin/python steering/run_revise.py prepare --model qwen3.5-9b --round 1
    .venv/bin/python generation/kaggle/push.py push --model qwen3.5-9b --mode rev1 \
        --prompts steering/prompt_sets/steer_rev1_qwen3.5-9b.jsonl
    .venv/bin/python generation/kaggle/push.py output --model qwen3.5-9b --mode rev1
    .venv/bin/python steering/collect.py --model qwen3.5-9b --mode rev1
    .venv/bin/python steering/run_revise.py evaluate --models qwen3.5-9b --rounds 2
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "audit"))
from audit_io import load_reference  # noqa: E402

from flavoraudit.audit import scorecard_row  # noqa: E402
from flavoraudit.pairing import Cuisine  # noqa: E402
from steering.design import feedback_items, revision_prompt, write_prompt_set  # noqa: E402

STEER = REPO / "data" / "interim" / "steering"
PARSED = REPO / "data" / "generated" / "parsed" / "recipes.parquet"
LINES = REPO / "data" / "generated" / "aliased" / "lines.parquet"


def rows_by_uid(lines: pd.DataFrame, idx: dict[int, int]) -> dict[str, np.ndarray]:
    out = {}
    for uid, g in lines.groupby("recipe_uid"):
        ents = [idx[int(e)] for e in g["entity_id"].dropna() if int(e) in idx]
        out[uid] = np.unique(np.array(ents, dtype=np.int64))
    return out


def round_inputs(model: str, r: int, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(recipes, lines) of the recipes that round r revises: the N = 1 selection for r = 1, else round r-1.

    Once round 1 exists, its parents are read from the revisions' own metadata (parent_uid), so round 0 stays the
    recipes that were actually revised even if best-of-N is re-run later and draws a different N = 1 selection.
    """
    if r == 1:
        rev1 = STEER / f"recipes_rev1_{model}.parquet"
        if rev1.exists():
            uids = {json.loads(m).get("parent_uid") for m in pd.read_parquet(rev1)["meta"]}
        else:
            sel = pd.read_parquet(STEER / f"bon_{tag}_selected.parquet")
            uids = set(sel.loc[(sel["model"] == model) & (sel["method"] == "composite") & (sel["N"] == 1), "uid"])
        parts = [pd.read_parquet(PARSED)]
        bon = STEER / f"recipes_bon_{model}.parquet"
        if bon.exists():
            parts.append(pd.read_parquet(bon))
        recipes = pd.concat(parts, ignore_index=True)
        recipes = recipes[recipes["recipe_uid"].isin(uids)]
        lparts = [pd.read_parquet(LINES)]
        if (STEER / f"lines_bon_{model}.parquet").exists():
            lparts.append(pd.read_parquet(STEER / f"lines_bon_{model}.parquet"))
        lines = pd.concat(lparts, ignore_index=True)
    else:
        recipes = pd.read_parquet(STEER / f"recipes_rev{r - 1}_{model}.parquet")
        lines = pd.read_parquet(STEER / f"lines_rev{r - 1}_{model}.parquet")
    return recipes, lines[lines["recipe_uid"].isin(set(recipes["recipe_uid"]))]


def cmd_prepare(args) -> int:
    ref, fp, idx, _ = load_reference()
    with open(STEER / "scorer.pkl", "rb") as fh:
        scorer = pickle.load(fh)
    inv = {r: e for e, r in idx.items()}
    names = {r: fp.names.get(inv.get(r), str(r)) for r in range(ref.S.shape[0])}
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs" / "regions.yaml").read_text()).items()}
    recipes, lines = round_inputs(args.model, args.round, args.tag)
    rows = rows_by_uid(lines, idx)
    prompts = []
    for rec in recipes.itertuples(index=False):
        region = labels[rec.region_code]
        r = rows.get(rec.recipe_uid, np.array([], dtype=np.int64))
        if len(r) < 2:
            continue
        items = feedback_items(scorer, r, region, names, ref.signatures[region])
        recipe = {"title": rec.title, "ingredients": list(rec.ingredients), "instructions": list(rec.instructions)}
        prompts.append(revision_prompt(recipe, items, rec.region_code, args.round, rec.recipe_uid))
    path = write_prompt_set(prompts, f"steer_rev{args.round}_{args.model}")
    print(f"{len(prompts)} revision prompts -> {path.relative_to(REPO)}")
    print(
        f"next: .venv/bin/python generation/kaggle/push.py push --model {args.model} --mode rev{args.round} "
        f"--prompts {path.relative_to(REPO)}"
    )
    return 0


def cmd_evaluate(args) -> int:
    ref, _fp, idx, _ = load_reference()
    with open(STEER / "eval_classifier.pkl", "rb") as fh:
        ev = pickle.load(fh)
    labels = {k: v["label"] for k, v in yaml.safe_load((REPO / "configs" / "regions.yaml").read_text()).items()}
    cats = next(iter(ref.real.values())).categories
    out = []
    for model in args.models:
        rounds = {0: round_inputs(model, 1, args.tag)}
        for r in range(1, args.rounds + 1):
            if (STEER / f"recipes_rev{r}_{model}.parquet").exists():
                rounds[r] = (
                    pd.read_parquet(STEER / f"recipes_rev{r}_{model}.parquet"),
                    pd.read_parquet(STEER / f"lines_rev{r}_{model}.parquet"),
                )
        parent_rows = rows_by_uid(rounds[0][1], idx)
        parent_of: dict[str, str] = {}  # every revised recipe -> the recipe it revised (round r -> r-1 -> ... -> 0)
        for r, (recipes, _lines) in rounds.items():
            if r >= 1:
                meta = recipes.set_index("recipe_uid")["meta"]
                parent_of.update({u: json.loads(m).get("parent_uid") for u, m in meta.items()})
        for r, (recipes, lines) in rounds.items():
            rows = rows_by_uid(lines, idx)
            for code, g in recipes.groupby("region_code"):
                g = g.sort_values("recipe_uid", key=lambda s: s.astype(str))  # canonical order, as in run_bon.py
                region = labels[code]
                recs = [rows[u] for u in g["recipe_uid"] if u in rows and len(rows[u]) >= 1]
                if len(recs) < 10:
                    continue
                cu = Cuisine(f"{model}:{code}:rev{r}", recs, cats)
                row = scorecard_row(cu, region, ref, seed=0, n_boot=500)
                Xe = ev["featurizer"].transform(recs)
                pred = ev["model"].predict(Xe.toarray() if ev["name"] == "hist_gbdt" else Xe)
                row["indep_classifier_acc"] = float((np.array(ev["regions"])[pred] == region).mean())
                if r >= 1:
                    jac = []
                    for u in g["recipe_uid"]:
                        root = parent_of.get(u)
                        while root is not None and root not in parent_rows:  # round 2 -> round 1 -> original
                            root = parent_of.get(root)
                        if u in rows and root in parent_rows:
                            a, b = set(rows[u].tolist()), set(parent_rows[root].tolist())
                            jac.append(len(a & b) / max(1, len(a | b)))
                    row["identity_jaccard"] = float(np.mean(jac)) if jac else np.nan
                out.append({"model": model, "round": r, "code": code, **row})
        print(f"{model}: rounds {sorted(rounds)}")
    df = pd.DataFrame(out)
    path = STEER / f"revise_{args.tag}.parquet"
    df.to_parquet(path, index=False)
    cols = ["sign_match", "fingerprint_spearman", "prevalence_jsd", "indep_classifier_acc", "identity_jaccard"]
    summ = df.groupby(["model", "round"])[[c for c in cols if c in df]].mean().round(4)
    print(summ.to_string())
    (STEER / f"revise_{args.tag}_summary.json").write_text(
        json.dumps(summ.reset_index().to_dict(orient="records"), indent=1)
    )
    print(f"-> {path.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--model", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--tag", default="main")
    e = sub.add_parser("evaluate")
    e.add_argument("--models", nargs="+", required=True)
    e.add_argument("--rounds", type=int, default=2)
    e.add_argument("--tag", default="main")
    args = ap.parse_args()
    return cmd_prepare(args) if args.cmd == "prepare" else cmd_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())

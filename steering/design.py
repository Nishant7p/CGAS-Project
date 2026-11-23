"""D3 steering design (plan §9.5): the pre-registered subset, the best-of-N candidate pool, scorer feedback and the
critique-and-revise prompts. Fixed here before any steering output exists.

* Regions: the 6 contrasting regions (Singh & Bagler 2018) and the 4 uniform regions with the most CulinaryDB recipes
  (USA, Italy, Indian Subcontinent, Mexico), which have the most reliable real fingerprints.
* Models: the two core open-weight models (free on Kaggle).
* Best-of-N, N in {1, 2, 4, 8, 16}: one pool of candidate recipes per model x region from the frozen plain prompt
  (extra replicates 16, 17, ... of prompt set v1, so the candidates follow the main run's distribution). The main
  run's 80 plain recipes join the pool. For each N, the shuffled pool is cut into groups of N and the best-scoring
  admissible recipe of each group is kept, until 100 are selected (plan: >= 100 per region and method). Admissible:
  not a near-copy of a CulinaryDB recipe (max Jaccard < 0.8) and not a duplicate of an already selected recipe.
* Critique-and-revise (Self-Refine; Madaan et al. 2023): up to 2 rounds on the 100 N = 1 recipes per region. The
  feedback (flavoraudit.scoring.FidelityScorer.feedback) names (a) up to 3 atypical ingredients, whose contribution
  to the classifier's log-odds for the cuisine is below the 5th percentile of contributions in genuine validation
  recipes of that cuisine, (b) up to 3 of the cuisine's top-10 signature ingredients that are missing but pair well
  (positive mean PMI) with the recipe's non-atypical core, and (c) whether its flavor sharing Ns lies outside the
  cuisine's interquartile range.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generation.common import N_REPS_FULL, build_prompt_set  # noqa: E402
from generation.config import load_regions, load_templates  # noqa: E402

STEER_REGIONS = ("BRI", "DACH", "EE", "JPN", "KOR", "SCND", "USA", "ITA", "INSC", "MEX")
STEER_MODELS = ("qwen3.5-9b", "llama-3.1-8b")
N_VALUES = (1, 2, 4, 8, 16)
N_SELECTED = 100  # recipes per region, method and model
POOL_TARGET = N_SELECTED * max(N_VALUES)  # 1,600 candidates per region
NEAR_COPY = 0.8
MAX_FEEDBACK_ITEMS = 3
PROMPT_DIR = REPO / "steering" / "prompt_sets"


def bon_pool_prompts(extra_calls_per_region: int, regions=STEER_REGIONS) -> list[dict]:
    """Extra plain-prompt calls (replicates 16, 17, ...) for the best-of-N pool; 5 recipes per call."""
    n_reps = N_REPS_FULL + extra_calls_per_region
    prompts = build_prompt_set(load_regions(), load_templates("v1"), n_reps=n_reps)
    out = [p for p in prompts if p["condition"] == "plain" and p["region_code"] in regions and p["rep"] >= N_REPS_FULL]
    for p in out:
        p["meta"] = {"purpose": "bon-pool"}
    return out


def default_extra_calls(main_run_plain_recipes: int = 80, recipes_per_call: int = 5) -> int:
    """Calls per region so that the pool reaches POOL_TARGET candidates with the main run's plain recipes."""
    return int(np.ceil((POOL_TARGET - main_run_plain_recipes) / recipes_per_call))


def feedback_items(scorer, recipe: np.ndarray, region: str, names: dict[int, str], signature: list[int]) -> dict:
    """The scorer's feedback for one recipe (flavoraudit.scoring.FidelityScorer.feedback), with ingredient names."""
    fb = scorer.feedback(recipe, region, signature, max_items=MAX_FEEDBACK_ITEMS)  # the cuisine's percentile cut
    comp = scorer.components([np.asarray(recipe)], region)
    return {
        "atypical": [names.get(r, str(r)) for r in fb["atypical"]],
        "missing": [names.get(r, str(r)) for r in fb["missing"]],
        "ns_verdict": fb["ns_verdict"],
        "ns_percentile": round(fb["ns_percentile"], 3) if np.isfinite(fb["ns_percentile"]) else None,
        "score_components": comp.iloc[0].round(4).to_dict(),
    }


def feedback_text(items: dict, label: str) -> str:
    lines = []
    if items["atypical"]:
        lines.append(f"- These ingredients are unusual in {label} cooking: {', '.join(items['atypical'])}.")
    if items["missing"]:
        lines.append(
            f"- These ingredients are typical of {label} cooking, go well with this dish, and are missing: "
            f"{', '.join(items['missing'])}."
        )
    if items["ns_verdict"] == "high":
        lines.append(
            f"- Its ingredients share more flavor compounds with each other than is usual in {label} recipes; "
            f"traditional {label} dishes more often combine ingredients with contrasting flavors."
        )
    elif items["ns_verdict"] == "low":
        lines.append(
            f"- Its ingredients share fewer flavor compounds with each other than is usual in {label} recipes; "
            f"traditional {label} dishes more often combine ingredients with similar flavors."
        )
    return "\n".join(lines) if lines else f"- It is already close to typical {label} home cooking; refine it lightly."


def revision_prompt(recipe: dict, items: dict, region_code: str, round_: int, parent_uid: str) -> dict:
    """One critique-and-revise prompt in the notebook's prompt format (see generation/kaggle/build_notebook.py)."""
    regions = load_regions()
    templates = load_templates("v1")
    tpl = (REPO / "configs" / "prompts" / "steer_v1" / "revise.txt").read_text(encoding="utf-8")
    reg = regions[region_code]
    shown = json.dumps(
        {
            "title": recipe["title"],
            "ingredients": list(recipe["ingredients"]),
            "instructions": list(recipe["instructions"]),
        },
        ensure_ascii=False,
        indent=1,
    )
    user = tpl.format(
        phrase=reg["phrase"], label=reg["label"], recipe=shown, feedback=feedback_text(items, reg["label"])
    )
    return {
        "key": f"rev{round_}|{parent_uid}",
        "version": "steer_v1",
        "region_code": region_code,
        "region_label": reg["label"],
        "condition": "revise",
        "rep": round_,
        "place": None,
        "system": templates["system"].strip(),
        "user": user.strip(),
        "meta": {"parent_uid": parent_uid, "round": round_, "feedback": items},
    }


def write_prompt_set(prompts: list[dict], name: str) -> Path:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    path = PROMPT_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
    return path


if __name__ == "__main__":
    extra = default_extra_calls()
    pool = bon_pool_prompts(extra)
    path = write_prompt_set(pool, "steer_bon_v1")
    print(f"best-of-N pool: {extra} extra calls x {len(STEER_REGIONS)} regions = {len(pool)} prompts -> {path}")

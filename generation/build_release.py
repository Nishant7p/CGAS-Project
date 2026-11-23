"""Assemble the LLMCuisine-22 release (D1) in CulinaryDB's file format (plan §2, D1; Phase 5).

Inputs: data/generated/parsed/{calls,recipes}.parquet, data/generated/aliased/{lines.parquet, coverage.csv} and the
raw call logs. Only full-mode calls are released, with at most the first 5 recipes of a call (the design asks for 5).

Output: data/release/LLMCuisine-22/
  01_Recipe_Details.csv               CulinaryDB columns (Recipe ID, Title, Source, Cuisine) + Model, Model Family,
                                      Condition, Persona Place, Replicate, Call ID, Recipe Index
  04_Recipe-Ingredients_Aliases.csv   CulinaryDB columns (Recipe ID, Original Ingredient Name, Aliased Ingredient Name,
                                      Entity ID) + Match (full | partial | curated | none); unmatched lines have no ID.
                                      Aliased Ingredient Name is the entity's CulinaryDB name (CulinaryDB itself stores
                                      the matched synonym phrase there).
  recipes_full.jsonl                  one recipe per line: title, ingredient lines, instructions and metadata
  generation_log.jsonl.gz             every full-mode call: prompts, model id and revision, parameters, seed, usage,
                                      timestamps, raw output
  calls.csv, coverage.csv             parse status per call; aliasing coverage per model x region
  MANIFEST.json                       counts, sha256 of every file, the prompt-set hash

Licence: CC BY-NC-SA 4.0 (DATA_LICENSE.md); the datasheet is docs/datasheet_LLMCuisine-22.md.
Usage (repo root): .venv/bin/python generation/build_release.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generation.parse_outputs import iter_records, mode_of  # noqa: E402

PARSED = REPO / "data" / "generated" / "parsed"
ALIASED = REPO / "data" / "generated" / "aliased"
RAW = REPO / "data" / "generated" / "raw"
OUT = REPO / "data" / "release" / "LLMCuisine-22"
RECIPES_PER_CALL = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    calls = pd.read_parquet(PARSED / "calls.parquet")
    recipes = pd.read_parquet(PARSED / "recipes.parquet")
    lines = pd.read_parquet(ALIASED / "lines.parquet")
    calls = calls[calls["mode"] == "full"]
    recipes = recipes[(recipes["mode"] == "full") & (recipes["recipe_idx"] < RECIPES_PER_CALL)]
    lines = lines[lines["recipe_uid"].isin(recipes["recipe_uid"])]
    OUT.mkdir(parents=True, exist_ok=True)

    details = pd.DataFrame(
        {
            "Recipe ID": recipes["recipe_uid"],
            "Title": recipes["title"],
            "Source": "LLM:" + recipes["model_key"],
            "Cuisine": recipes["region_label"],
            "Model": recipes["model_key"],
            "Model Family": recipes["family"],
            "Condition": recipes["condition"],
            "Persona Place": recipes["place"],
            "Replicate": recipes["rep"],
            "Call ID": recipes["call_id"],
            "Recipe Index": recipes["recipe_idx"],
        }
    ).sort_values(["Model", "Cuisine", "Condition", "Replicate", "Recipe Index"])
    details.to_csv(OUT / "01_Recipe_Details.csv", index=False)

    aliases = pd.DataFrame(
        {
            "Recipe ID": lines["recipe_uid"],
            "Original Ingredient Name": lines["line"],
            "Aliased Ingredient Name": lines["entity"],
            "Entity ID": lines["entity_id"].astype("Int64"),
            "Match": lines["how"],
        }
    )
    aliases.to_csv(OUT / "04_Recipe-Ingredients_Aliases.csv", index=False)

    with open(OUT / "recipes_full.jsonl", "w", encoding="utf-8") as fh:
        for r in recipes.itertuples(index=False):
            rec = {
                "recipe_id": r.recipe_uid,
                "title": r.title,
                "ingredients": list(r.ingredients),
                "instructions": list(r.instructions),
                "cuisine": r.region_label,
                "region_code": r.region_code,
                "model": r.model_key,
                "family": r.family,
                "condition": r.condition,
                "persona_place": r.place if isinstance(r.place, str) else None,  # null, not NaN (strict JSON)
                "replicate": int(r.rep),
                "call_id": r.call_id,
                "recipe_index": int(r.recipe_idx),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    paths = sorted(q for q in RAW.rglob("*.jsonl*") if q.suffix in (".jsonl", ".gz"))
    paths = [q for q in paths if not (q.suffix == ".gz" and q.with_suffix("") in paths) and mode_of(q) == "full"]
    seen, n_log = set(), 0
    with gzip.open(OUT / "generation_log.jsonl.gz", "wt", encoding="utf-8") as fh:
        for _path, rec in iter_records(paths):
            if rec["call_id"] in seen:
                continue
            seen.add(rec["call_id"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_log += 1

    calls[["call_id", "model_key", "region_code", "condition", "rep", "parse_status", "n_recipes", "issues",
           "finish_reason", "n_output_tokens"]].to_csv(OUT / "calls.csv", index=False)  # fmt: skip
    cov = pd.read_csv(ALIASED / "coverage.csv")
    cov[cov["mode"] == "full"].drop(columns="mode").to_csv(OUT / "coverage.csv", index=False)

    v1_sha = (REPO / "generation" / "prompt_sets" / "v1.sha256").read_text().strip()
    manifest = {
        "dataset": "LLMCuisine-22",
        "license": "CC BY-NC-SA 4.0",
        "prompt_set": {"version": "v1", "sha256": v1_sha},
        "counts": {
            "calls": int(len(calls)),
            "logged_calls": n_log,
            "recipes": int(len(details)),
            "ingredient_lines": int(len(aliases)),
            "by_model": details.groupby("Model").size().to_dict(),
        },
        "files": {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.name != "MANIFEST.json"},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["counts"], indent=1))
    size = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    print(f"release: {size:.1f} MB -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

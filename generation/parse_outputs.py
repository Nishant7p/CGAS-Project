"""Parse raw generation logs into tidy tables (one row per call, one row per recipe).

Input: every *.jsonl or *.jsonl.gz under data/generated/raw/ (Kaggle downloads and run_api.py logs alike).
Output: data/generated/parsed/calls.parquet and recipes.parquet, plus a per-model report printed to stdout.

Usage (repo root):  .venv/bin/python generation/parse_outputs.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generation.common import extract_json, validate_recipes  # noqa: E402

RAW = REPO / "data" / "generated" / "raw"
OUT = REPO / "data" / "generated" / "parsed"


def iter_records(paths: list[Path]):
    for p in paths:
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield p, json.loads(line)


def mode_of(path: Path) -> str:
    """'pilot' or 'full' from the log's file name (Kaggle: llmcuisine22_<model>_<mode>.jsonl; API: <mode>.jsonl).

    Pilot and full runs share call ids for replicate 0 (same model and prompt key), so mode is part of the key: the
    pilot was used to check the pipeline, and only full-mode calls enter the dataset and the analysis.
    """
    name = path.name.removesuffix(".gz").removesuffix(".jsonl")  # not split("."): model keys contain dots (qwen3.5)
    for mode in ("pilot", "full", "sens"):
        if name == mode or name.endswith(f"_{mode}"):
            return mode
    return "unknown"


def main() -> int:
    paths = sorted(q for q in RAW.rglob("*.jsonl*") if q.suffix in (".jsonl", ".gz"))
    # If both a .jsonl and its .gz copy exist, keep the .jsonl only.
    paths = [q for q in paths if not (q.suffix == ".gz" and q.with_suffix("") in paths)]
    if not paths:
        print(f"No logs under {RAW.relative_to(REPO)}/ yet.")
        return 0
    calls, recipes, seen = [], [], set()
    for path, r in iter_records(paths):
        mode = mode_of(path)
        if (mode, r["call_id"]) in seen:  # the same call logged twice (e.g. a re-downloaded file) is counted once
            continue
        seen.add((mode, r["call_id"]))
        obj, status = extract_json(r.get("output_text", ""))
        recs, issues = validate_recipes(obj) if obj is not None else ([], [status])
        base = {"mode": mode} | {
            k: r.get(k)
            for k in (
                "call_id",
                "model_key",
                "family",
                "region_code",
                "region_label",
                "condition",
                "rep",
                "place",
                "prompt_key",
            )
        }
        calls.append(
            {
                **base,
                "source_file": str(path.relative_to(REPO)),
                "parse_status": status,
                "n_recipes": len(recs),
                "issues": "; ".join(issues),
                "finish_reason": r.get("finish_reason"),
                "n_output_tokens": r.get("n_output_tokens"),
            }
        )
        for i, rec in enumerate(recs):
            recipes.append(
                {
                    **base,
                    "recipe_uid": f"{mode[0]}{r['call_id']}-{i}",  # f... (full) or p... (pilot): unique across modes
                    "recipe_idx": i,
                    "title": rec["title"],
                    "ingredients": rec["ingredients"],
                    "instructions": rec["instructions"],
                    "n_ingredient_lines": len(rec["ingredients"]),
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    cdf, rdf = pd.DataFrame(calls), pd.DataFrame(recipes)
    cdf.to_parquet(OUT / "calls.parquet", index=False)
    rdf.to_parquet(OUT / "recipes.parquet", index=False)
    rep = cdf.groupby(["mode", "model_key"]).agg(
        calls=("call_id", "size"),
        parsed_ok=("parse_status", lambda s: (s.isin(["ok", "repaired", "salvaged"])).mean()),
        exactly5=("n_recipes", lambda s: (s == 5).mean()),
        recipes=("n_recipes", "sum"),
        truncated=("finish_reason", lambda s: (s.astype(str).str.lower().str.contains("length|max_tokens")).mean()),
    )
    print(rep.round(3).to_string())
    if len(rdf):
        print(
            "\nmean ingredient lines per recipe:",
            rdf.groupby(["mode", "model_key"])["n_ingredient_lines"].mean().round(1).to_dict(),
        )
    print(f"\nwrote {OUT.relative_to(REPO)}/calls.parquet ({len(cdf)} calls) and recipes.parquet ({len(rdf)} recipes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

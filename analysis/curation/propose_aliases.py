"""Propose curated aliases for ingredient terms in LLM recipes that CulinaryDB's vocabulary does not match.

This automates the first half of the paper's curation step ("partial and unrecognized ingredients were labeled for
manual curation; n-grams were created ... to identify commonly occurring ingredients"). For every frequent unigram
or bigram in UNMATCHED LLM lines it looks up CulinaryDB's expert-aliased lines that contain the same term:
  * evidence-based proposal: the majority gold entity, when support >= 5 lines and precision >= 0.6;
  * otherwise the term is listed as NEEDS_MANUAL with example lines (a human picks an entity or "none").
The reviewed decisions go to curation/alias_overrides.csv (term, entity_id, decision, rationale), which
flavoraudit.aliasing applies only to LLM text (the terms are absent from CulinaryDB's own lines).

Inputs: data/interim/pilot/pilot_lines.parquet (or a full-run lines file passed with --lines).
Output: curation/alias_proposals.csv
Usage (repo root): .venv/bin/python analysis/curation/propose_aliases.py [--lines PATH] [--min-count 2]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from flavoraudit.aliasing import tokens
from flavoraudit.data import load_culinarydb

REPO = Path(__file__).resolve().parents[2]


def grams(toks: list[str]) -> set[tuple[str, ...]]:
    return {tuple(toks[i : i + n]) for n in (1, 2) for i in range(len(toks) - n + 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default=str(REPO / "data" / "interim" / "pilot" / "pilot_lines.parquet"))
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()
    lines = pd.read_parquet(args.lines)
    unmatched = lines[lines["how"] == "none"]
    term_count: Counter = Counter()
    examples: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for line in unmatched["line"]:
        for g in grams(tokens(line)):
            term_count[g] += 1
            if len(examples[g]) < 3:
                examples[g].append(line)
    terms = [t for t, c in term_count.items() if c >= args.min_count and not any(w.isdigit() for w in t)]
    cdb = load_culinarydb()
    names = dict(zip(cdb.ingredients["entity_id"], cdb.ingredients["name"], strict=True))
    gold_index: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    wanted = set(terms)
    for line, ent in zip(cdb.aliases["original"].astype(str), cdb.aliases["entity_id"], strict=True):
        for g in grams(tokens(line)) & wanted:
            gold_index[g][int(ent)] += 1
    rows = []
    for t in sorted(terms, key=lambda x: -term_count[x]):
        ev = gold_index.get(t, Counter())
        support = sum(ev.values())
        ent, k = ev.most_common(1)[0] if ev else (None, 0)
        precision = k / support if support else 0.0
        ok = support >= 5 and precision >= 0.6
        rows.append(
            {
                "term": " ".join(t),
                "llm_unmatched_lines": term_count[t],
                "proposal": "EVIDENCE" if ok else "NEEDS_MANUAL",
                "entity_id": ent if ok else None,
                "entity_name": names.get(ent) if ok else None,
                "culinarydb_support": support,
                "culinarydb_precision": round(precision, 2),
                "examples": " | ".join(examples[t]),
            }
        )
    out = pd.DataFrame(rows)
    path = REPO / "curation" / "alias_proposals.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print(out["proposal"].value_counts().to_string())
    print(
        out.head(30)[
            ["term", "llm_unmatched_lines", "proposal", "entity_name", "culinarydb_support", "culinarydb_precision"]
        ].to_string(index=False)
    )
    print(f"wrote {path.relative_to(REPO)} ({len(out)} terms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

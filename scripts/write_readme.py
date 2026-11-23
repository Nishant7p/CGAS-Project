"""Write README.md (the GitHub landing page) from the result files, so every number in it is current.

Usage (repo root): .venv/bin/python scripts/write_readme.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "presentation"))
import talk_script as T  # noqa: E402

GH = T.GITHUB


def main() -> int:
    d = T.D
    rows_models = "\n".join(
        f"| {T.NAMES[m]} | {d['models'][m]['family']} | {d['models'][m]['recipes']:,} | {d['models'][m]['regions']} |"
        for m in T.ORDER
    )
    readme = f"""# Do AI Chefs Respect Culinary Fingerprints?

**Auditing and steering LLM-generated recipes across 22 world cuisines**

BIO544 Computational Gastronomy · IIIT-Delhi · Monsoon 2026 · Instructor: Prof. Ganesh Bagler
Group 44: {T.SPEAKER_A} and {T.SPEAKER_B}

## Start here

| What | Where |
|---|---|
| Slides (talk) | [`deliverables/Group44_BIO544_slides.pdf`](deliverables/Group44_BIO544_slides.pdf) · [`.pptx`](deliverables/Group44_BIO544_slides.pptx) |
| Research report | [`deliverables/Group44_BIO544_report.pdf`](deliverables/Group44_BIO544_report.pdf) |
| Poster (A0) | [`deliverables/Group44_BIO544_poster.pdf`](deliverables/Group44_BIO544_poster.pdf) |
| Presenter guide (script, demo, Q&A) | [`deliverables/Group44_Presenter_Guide.pdf`](deliverables/Group44_Presenter_Guide.pdf) |
| Web app (FlavorPrint) | [Run the app](#run-the-flavorprint-app-d5) |

## The project in one paragraph

When an AI writes a “traditional” recipe, does it cook like the cuisine it claims? Every cuisine has a measurable
culinary fingerprint: how it combines ingredients and flavors (Ahn et al. 2011; Jain, Rakhi & Bagler 2015;
Singh & Bagler 2018). We first reproduced Singh & Bagler's food-pairing result for all 22 CulinaryDB regions, then
asked {d["n_models"]} large language models for recipes from every region ({d["total_recipes"]:,} recipes), measured
them with exactly the same tools as the real cuisines, and used the fingerprint to steer the models toward more
authentic recipes.

## Key results

| Measure | AI models | Real samples (same size) |
|---|---|---|
| Food-pairing direction matches the real cuisine | {T.span([T.SIGN[m] for m in T.ORDER], "{:.0%}")} of cuisines | {T.pct(T.GEN_SIGN)} |
| Category fingerprint similarity (Spearman ρ) | {T.span(T.FP.values())} | {T.GEN["fingerprint_spearman"]:.2f} |
| Signature ingredient pairs shared (top-100 NPMI, Jaccard) | {T.span(T.NETV.values())} | {T.NET["real@160"]["median"]:.2f} |
| Ingredient variety (real = 1.00) | {T.span(T.DIV.values())} | 1.00 |
| Stereotype index (1 = like real recipes) | {T.span(T.SAI.values())} | 1.00 |

- **Replication:** {T.REPL["sign_agreement"]}/{T.REPL["n"]} cuisines with the published sign, rank correlation
  {T.REPL["spearman"]:.2f}; Korea's top flavor driver is {T.KOREA_TOP["name"]}, as published.
- **Steering:** with fingerprint feedback, recipes recognised as their claimed cuisine rise from {T.pct(T.R0L)} to
  {T.pct(T.R2L)} (GPT-6-Luna, 2 rounds) and {T.pct(T.R0Q)} to {T.pct(T.R1Q)} (Qwen3.5-9B, 1 round), judged by an
  independent classifier; best-of-8 selection raises it for every model.
- **In short:** AI chefs get the skeleton of a cuisine right but only part of its fingerprint, and the fingerprint
  itself can guide them closer.

| Model | Developer | Recipes | Cuisines |
|---|---|---|---|
{rows_models}

## The five deliverables

| | Deliverable | Where |
|---|---|---|
| D1 | **LLMCuisine-22** dataset ({d["total_recipes"]:,} LLM recipes in CulinaryDB format, every call logged, datasheet) | [`data/release/LLMCuisine-22/`](data/release/LLMCuisine-22/) |
| D2 | **Food-pairing and fingerprint audit** (replication, equal-size reference, drivers, fingerprints, networks, copy-mutate baseline) | [`analysis/`](analysis/), [`figures/`](figures/), [`results/summary.json`](results/summary.json), report |
| D3 | **Fingerprint-guided steering** (fidelity scorer, best-of-N, critique-and-revise) | [`steering/`](steering/), [`results/steering_example.json`](results/steering_example.json) |
| D4 | **FlavorAudit** Python library (tested, documented) | [`src/flavoraudit/`](src/flavoraudit/), [`notebooks/`](notebooks/), [`docs/`](docs/) |
| D5 | **FlavorPrint** web app (audit any recipe, 22-cuisine dashboard) | [`app/`](app/) |

## Run the FlavorPrint app (D5)

Python 3.12 recommended. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r app/requirements.txt -e .
streamlit run app/streamlit_app.py
```

Open http://localhost:8501, pick an example (Kimchi jjigae, “Spaghetti, claimed Korean”, Chana masala) and press
**Audit recipe**. The app runs from the prepared artifacts in `app/artifacts/`; no database download is needed.
With [uv](https://docs.astral.sh/uv/): `uv sync --all-groups --extra ml && uv run streamlit run app/streamlit_app.py`.

## Use the FlavorAudit library (D4)

```bash
pip install -e ".[ml]"
flavoraudit --help
```

```python
from flavoraudit.world import load_world
from flavoraudit.stats import food_pairing

world = load_world()                      # CulinaryDB + FlavorDB (see "Data" below)
korea = world.cuisines["Korea"]
res = food_pairing(korea, world.S, "random")
print(res.z_sb, res.delta_ns)
```

Worked examples: [`notebooks/`](notebooks/) (replication, auditing recipes, copy-mutate and fingerprints).

## Reproduce the analysis

```bash
uv sync --all-groups --extra ml
uv run pytest -q                                          # tests
uv run python analysis/audit/llm_audit.py --tag full      # audit (D2)
uv run python analysis/audit/rq_tests.py --tag full
uv run python analysis/audit/networks.py
uv run python analysis/summarize.py                       # writes results/summary.json
uv run python report/build.py --include-f5                # report
uv run python presentation/story_deck.py                  # slides
```

Generation (D1) used `generation/` (Kaggle notebook for open models, API runners with every call logged). Hypotheses
and metrics were fixed in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) before the full LLM data were
analysed; later changes are dated in its decision log.

## Data and licences

- Code: MIT ([`LICENSE`](LICENSE)).
- CulinaryDB and FlavorDB (Prof. Bagler's lab) are CC BY-NC-SA 3.0 and are **not** redistributed here: download
  CulinaryDB from the lab's website into `data/raw/culinarydb/` and fetch FlavorDB with `scripts/fetch_flavordb.py`. Data derived from them (LLMCuisine-22, the app artifacts) are
  CC BY-NC-SA 4.0 with attribution ([`DATA_LICENSE.md`](DATA_LICENSE.md)). Ahn et al. (2011) data: CC BY 4.0.
- Model output terms are recorded in the dataset's datasheet.

## Acknowledgements

We thank Prof. Ganesh Bagler and the course TAs. CulinaryDB and FlavorDB are from the Complex Systems Lab, IIIT-Delhi.
The code and analyses were developed with the help of an AI coding assistant under our direction and review; the
report includes a disclosure.
"""
    (REPO / "README.md").write_text(readme, encoding="utf-8")
    print("README.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

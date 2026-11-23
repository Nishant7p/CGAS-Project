"""Assemble FINAL_BIO544_GROUP44_DELIVERABLES/ (next to the repo) from the repository's generated outputs.

Copies only; nothing is regenerated here. Re-run after any rebuild: the folder is replaced as a whole, a README index
and a MANIFEST (size + SHA-256 of every file) are written, and anything expected but missing is listed in the README.

Usage (repo root): .venv/bin/python scripts/export_final.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO.parent / "FINAL_BIO544_GROUP44_DELIVERABLES"
MISSING: list[str] = []

CODE_DIRS = ("src", "analysis", "generation", "steering", "app", "scripts", "tests", "configs", "docs", "report",
             "presentation", "notebooks", "curation")  # fmt: skip
CODE_FILES = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "DATA_LICENSE.md", "CITATION.cff", "CHANGELOG.md",
              "mkdocs.yml", ".gitignore")  # fmt: skip
SKIP_PARTS = {".venv", "__pycache__", "_build", "artifacts", "space", "screenshots", "site", ".DS_Store", "push"}


def copy(src: Path, dst: Path, what: str) -> bool:
    if not src.exists():
        MISSING.append(f"{what} ({src.relative_to(REPO.parent)})")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    else:
        shutil.copy2(src, dst)
    return True


def parquet_to_csv(src: Path, dst: Path, what: str) -> None:
    if not src.exists():
        MISSING.append(f"{what} ({src.relative_to(REPO.parent)})")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    for c in df.columns:  # lists/arrays -> readable strings
        if df[c].map(lambda v: hasattr(v, "__len__") and not isinstance(v, str)).any():
            df[c] = df[c].map(
                lambda v: "; ".join(map(str, v)) if hasattr(v, "__len__") and not isinstance(v, str) else v
            )
    df.to_csv(dst, index=False)


def source_zip(dst: Path) -> None:
    """Code, configs, docs and document sources only: no data, no environments, no build outputs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for name in CODE_FILES:
            if (REPO / name).exists():
                z.write(REPO / name, f"ai-chefs-fingerprints/{name}")
        for d in CODE_DIRS:
            for p in sorted((REPO / d).rglob("*")):
                rel = p.relative_to(REPO)
                keep = p.is_file() and not (set(rel.parts) & SKIP_PARTS) and p.suffix not in (".pyc", ".pdf", ".pptx")
                if keep and p.stat().st_size < 5_000_000:
                    z.write(p, f"ai-chefs-fingerprints/{rel}")


def write_readme(summary: dict) -> None:
    d = summary.get("dataset", {})
    gem = (d.get("models") or {}).get("gemini-3.5-flash-lite", {})
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# BIO544 Computational Gastronomy, Group 44: final deliverables",
        "",
        "**Do AI Chefs Respect Culinary Fingerprints? Auditing and Steering LLM-Generated Recipes across 22 World Cuisines**",
        "",
        "Group 44 (2023355, 2023477), IIIT-Delhi, Monsoon 2026. Course: BIO544 (Prof. Ganesh Bagler).",
        "",
        f"Exported {now} from `ai-chefs-fingerprints/` by `scripts/export_final.py`. Every number in these files comes "
        f"from `06_Analysis_results/summary.json` (generated {summary.get('generated_utc', '?')}) and the tables beside it.",
        "",
        "**Start with `HOW_TO_PRESENT_AND_DEMO.md`. To run the app: `RUN_THE_APP.md`.**",
        "",
        "## Where everything is",
        "",
        "| Folder | Contents |",
        "|---|---|",
        "| `01_Presentation/` | The talk: slides (`.pptx` with speaker notes, and `.pdf`) and the presenter guide PDF (script split between the two speakers, demo, deliverables, simple explanations, Q&A, numbers) |",
        "| `02_Report/` | Research report (PDF) and its LaTeX source |",
        "| `03_Poster/` | Research poster (A0 PDF) |",
        "| `04_Demo/` | How to launch and present the FlavorPrint demo, screenshots, a double-click launcher |",
        "| `05_Dataset_LLMCuisine-22/` | **D1**: LLM recipes in CulinaryDB format, full JSONL with generation metadata, the call log, coverage, datasheet, licence |",
        "| `06_Analysis_results/` | **D2** (and D3 numbers): `summary.json`, result tables (CSV), every figure (PNG 300 dpi and vector PDF) |",
        "| `07_FlavorAudit_library/` | **D4**: the `flavoraudit` wheel and sdist, quick start, example notebooks, HTML documentation |",
        "| `08_FlavorPrint_app/` | **D5**: the self-contained Streamlit app (Hugging Face Space folder, Docker SDK) |",
        "| `09_Steering_D3/` | **D3**: fidelity scorer and steering code, prompt sets, scorer metrics, the before/after example |",
        "| `10_Reproducibility/` | Environment lock, configs, the pre-registration and decision log, the Kaggle notebook, a source snapshot, how to reproduce |",
        "",
        "## Scope of this release",
        "",
        f"- LLMCuisine-22: {d.get('total_recipes', '?'):,} recipes from {d.get('n_models', '?')} LLMs "
        f"({d.get('n_proprietary', '?')} proprietary, {d.get('n_open_families', '?')} open-weight families), meeting "
        "the proposal's commitment (≥3 LLMs, ≥1 proprietary, ≥2 open families)."
        if d.get("meets_proposal")
        else f"- LLMCuisine-22: {d.get('total_recipes', '?'):,} recipes.",
        f"- GPT-6-Luna, Qwen3.5-9B and Llama-3.1-8B cover all 22 regions; Gemini 3.5 Flash-Lite covers "
        f"{gem.get('regions', '?')} regions ({gem.get('recipes', 0):,} recipes, set by the free API's daily quota).",
        "- Steering: best-of-N for GPT-6-Luna, Gemini 3.5 Flash-Lite and Qwen3.5-9B; critique-and-revise for GPT-6-Luna "
        "(2 rounds) and Qwen3.5-9B (1 round).",
        "- Extensions under way: the Qwen3.5-4B scale study and Ministral-3-8B, generated with the same notebook.",
        "- Everything is packaged and tested locally; public release (PyPI, Hugging Face, dataset archive) follows.",
    ]
    lines += [
        "",
        "## Run the live demo (about 1 minute to start)",
        "",
        "Double-click `04_Demo/run_demo.command`, or in a terminal:",
        "",
        "```bash",
        f'cd "{REPO}"',
        ".venv/bin/streamlit run app/streamlit_app.py",
        "```",
        "",
        "Without the project environment: `cd 08_FlavorPrint_app && pip install -r requirements.txt && streamlit run "
        "streamlit_app.py`.",
        "",
        "## Licences",
        "",
        "Code (FlavorAudit, FlavorPrint, analysis): MIT. LLMCuisine-22 and other data derived from CulinaryDB and "
        "FlavorDB: CC BY-NC-SA 4.0 with attribution (the sources are CC BY-NC-SA 3.0). Model output terms are in the "
        "datasheet.",
        "",
    ]
    if MISSING:
        lines += ["## Not included at export (not generated yet)", ""] + [f"- {m}" for m in MISSING] + [""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.txt":
            rows.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.stat().st_size:>11,}  {p.relative_to(OUT)}")
    (OUT / "MANIFEST.txt").write_text("sha256  bytes  path\n" + "\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    summary = json.loads((REPO / "results/summary.json").read_text())
    # 01 presentation: the story deck (presentation/story_deck.py) is the talk; its script comes from its own notes
    p = OUT / "01_Presentation"
    story = REPO / "presentation/story"
    copy(story / "Group44_BIO544_presentation.pptx", p / "Group44_BIO544_slides.pptx", "slides (pptx)")
    copy(story / "Group44_BIO544_presentation.pdf", p / "Group44_BIO544_slides.pdf", "slides (pdf)")
    copy(REPO / "presentation/guide/Group44_Presenter_Guide.pdf", p / "Group44_Presenter_Guide.pdf", "presenter guide")
    (OUT / "HOW_TO_PRESENT_AND_DEMO.md").write_text(GUIDE.replace("{REPO}", str(REPO)), encoding="utf-8")
    (OUT / "RUN_THE_APP.md").write_text(RUN_APP.replace("{REPO}", str(REPO)), encoding="utf-8")
    # 02 report, 03 poster
    copy(REPO / "report/main.pdf", OUT / "02_Report/Group44_BIO544_report.pdf", "report (pdf)")
    with zipfile.ZipFile(OUT / "02_Report/report_latex_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in [REPO / "report/main.tex", REPO / "report/refs.bib", *sorted((REPO / "report/generated").glob("*"))]:
            if f.is_file():
                z.write(f, f"report/{f.relative_to(REPO / 'report')}")
    copy(REPO / "presentation/poster.pdf", OUT / "03_Poster/Group44_BIO544_poster.pdf", "poster (pdf)")
    # 04 demo
    dm = OUT / "04_Demo"
    copy(REPO / "app/DEMO.md", dm / "DEMO.md", "demo instructions (app/DEMO.md)")
    copy(REPO / "app/screenshots", dm / "screenshots", "app screenshots")
    launcher = dm / "run_demo.command"
    launcher.write_text(f'#!/bin/zsh\ncd "{REPO}" || exit 1\nexec .venv/bin/streamlit run app/streamlit_app.py\n')
    launcher.chmod(0o755)
    # 05 dataset
    ds = OUT / "05_Dataset_LLMCuisine-22"
    copy(REPO / "data/release/LLMCuisine-22", ds / "LLMCuisine-22", "LLMCuisine-22 release")
    copy(REPO / "docs/datasheet_LLMCuisine-22.md", ds / "datasheet_LLMCuisine-22.md", "datasheet")
    copy(REPO / "DATA_LICENSE.md", ds / "DATA_LICENSE.md", "data licence")
    # 06 analysis results
    a = OUT / "06_Analysis_results"
    for f in ("summary.json", "steering_example.json"):
        copy(REPO / "results" / f, a / f, f)
    copy(REPO / "tables", a / "tables", "result tables")
    for f in sorted((REPO / "figures").glob("F*.png")):
        copy(f, a / "figures_png" / f.name, f.name)
    for f in sorted((REPO / "figures").glob("F*.pdf")):
        copy(f, a / "figures_pdf" / f.name, f.name)
    au = REPO / "data/interim/audit"
    copy(au / "full/scorecard.csv", a / "tables/llm_scorecard.csv", "LLM scorecard")
    parquet_to_csv(au / "full/networks.parquet", a / "tables/networks.csv", "networks")
    parquet_to_csv(au / "full/drivers.parquet", a / "tables/drivers_chi.csv", "drivers")
    parquet_to_csv(
        au / "copy_mutate/scorecard.parquet", a / "tables/copy_mutate_scorecard.csv", "copy-mutate scorecard"
    )
    copy(REPO / "data/interim/replication/results_v1_keep_empty.csv", a / "tables/replication_singh_bagler.csv",
         "replication results")  # fmt: skip
    copy(REPO / "docs/replication_report.md", a / "replication_report.md", "replication report")
    # 07 library
    lib = OUT / "07_FlavorAudit_library"
    for f in sorted((REPO / "dist").glob("flavoraudit-*")):
        copy(f, lib / f.name, f.name)
    copy(REPO / "notebooks", lib / "example_notebooks", "example notebooks")
    copy(REPO / "site", lib / "documentation_html", "HTML documentation (mkdocs build)")
    copy(REPO / "README.md", lib / "README.md", "README")
    # 08 app
    copy(REPO / "app/space", OUT / "08_FlavorPrint_app", "FlavorPrint Space folder")
    # 09 steering
    st = OUT / "09_Steering_D3"
    for f in sorted((REPO / "steering").glob("*.py")):
        copy(f, st / "code" / f.name, f.name)
    copy(REPO / "steering/prompt_sets", st / "prompt_sets", "steering prompt sets")
    copy(REPO / "results/steering_example.json", st / "before_after_example.json", "steering example")
    s_dir = REPO / "data/interim/steering"
    for f in ("scorer_metrics.json", "scorer_auc.csv", "confusion.csv", "splits.json", "bon_main_summary.json",
              "revise_main_summary.json"):  # fmt: skip
        copy(s_dir / f, st / "results" / f, f)
    parquet_to_csv(s_dir / "bon_main.parquet", st / "results/best_of_n.csv", "best-of-N results")
    parquet_to_csv(s_dir / "revise_main.parquet", st / "results/critique_and_revise.csv", "revise results")
    # 10 reproducibility
    rp = OUT / "10_Reproducibility"
    for f in ("pyproject.toml", "uv.lock"):
        copy(REPO / f, rp / f, f)
    copy(REPO / "configs", rp / "configs", "configs")
    copy(REPO / "docs/IMPLEMENTATION_PLAN.md", rp / "IMPLEMENTATION_PLAN_preregistration_and_decision_log.md", "plan")
    copy(REPO / "docs/STATUS.md", rp / "STATUS.md", "status log")
    copy(REPO / "docs/RUNBOOK_GENERATION.md", rp / "RUNBOOK_GENERATION.md", "generation runbook")
    copy(REPO / "generation/kaggle/llmcuisine22_generate.ipynb", rp / "llmcuisine22_generate.ipynb", "Kaggle notebook")
    copy(REPO / "generation/prompt_sets/v1.jsonl", rp / "prompt_set_v1.jsonl", "frozen prompt set")
    copy(REPO / "generation/prompt_sets/v1.sha256", rp / "prompt_set_v1.sha256", "prompt set hash")
    source_zip(rp / "ai-chefs-fingerprints_source.zip")
    (rp / "REPRODUCIBILITY.md").write_text(REPRO, encoding="utf-8")
    write_readme(summary)
    write_manifest()
    dl = REPO / "deliverables"  # the same final files, inside the repo for GitHub
    dl.mkdir(exist_ok=True)
    for src, name in ((story / "Group44_BIO544_presentation.pptx", "Group44_BIO544_slides.pptx"),
                      (story / "Group44_BIO544_presentation.pdf", "Group44_BIO544_slides.pdf"),
                      (REPO / "report/main.pdf", "Group44_BIO544_report.pdf"),
                      (REPO / "presentation/poster.pdf", "Group44_BIO544_poster.pdf"),
                      (REPO / "presentation/guide/Group44_Presenter_Guide.pdf", "Group44_Presenter_Guide.pdf")):  # fmt: skip
        if src.exists():
            shutil.copy2(src, dl / name)
    n = sum(1 for x in OUT.rglob("*") if x.is_file())
    size = sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file()) / 1e6
    print(f"{OUT} : {n} files, {size:.0f} MB; missing: {len(MISSING)}")
    for m in MISSING:
        print("  missing:", m)
    return 0


RUN_APP = """# Run the FlavorPrint app (D5)

## Easiest: double-click
Double-click **`04_Demo/run_demo.command`** in this folder. A terminal window opens and the app opens in your browser
at **http://localhost:8501**. The first page load takes about 5 seconds. Keep the terminal window open while
presenting; close it to stop the app.

(If macOS blocks the file the first time: right-click it, choose **Open**, then **Open** again.)

## Or from a terminal
```bash
cd "{REPO}"
.venv/bin/streamlit run app/streamlit_app.py
```
Then open **http://localhost:8501**. If that port is busy, add `--server.port 8502` and open http://localhost:8502.

## On another computer (no project environment)
```bash
cd 08_FlavorPrint_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What to click in the demo
**Kimchi jjigae → Audit recipe** (reads as Korea), then the **Flavor network** tab, then **Spaghetti as Korean**
(reads as Italy), then the **Suggestions** tab, **Chana masala** (stereotype flag), and the **22-cuisine dashboard**.
The full two-minute script is in `04_Demo/DEMO.md`.
"""

GUIDE = """# How to present and demo (read this first)

## 10 minutes before the talk
1. Open `01_Presentation/Group44_BIO544_slides.pptx` in PowerPoint or Keynote and start **Presenter View**: your
   speaker notes, with what to say and when to demo, appear under each slide. The same text is in
   `01_Presentation/SPEAKING_SCRIPT.md`. The PDF version is the backup if the projector laptop has no PowerPoint.
2. Start the app: double-click `04_Demo/run_demo.command`, or in a terminal:
   `cd "{REPO}" && .venv/bin/streamlit run app/streamlit_app.py`.
   A browser tab "FlavorPrint" opens at http://localhost:8501. Click the **Kimchi jjigae** example and
   **Audit recipe** once, so the first audit is already cached.
3. Optional tabs to keep ready (only if asked): the report PDF (`02_Report/`), the dataset CSV
   (`05_Dataset_LLMCuisine-22/LLMCuisine-22/01_Recipe_Details.csv`), and the library notebook
   (`07_FlavorAudit_library/example_notebooks/02_audit_recipes.ipynb`).

## The story in one breath
AI models now write "traditional" recipes. Computational gastronomy (Prof. Bagler's lab) measures how real cuisines
combine ingredients and flavors: their culinary fingerprint. We first reproduced the published food-pairing result
(22/22 cuisines), then measured 11,656 AI recipes from 4 models with the same tools. AI cuisines get the skeleton
right: recipe size, signature ingredients and often the pairing direction and key drivers. They miss part of the
fingerprint: distinctive ingredient combinations and variety. A fidelity scorer built from the fingerprint steers them
closer. Everything is packaged as a dataset, an audit, a steering method, a library and an app.

## Live demo (slide 13, about 2 minutes)
1. **Audit a recipe** tab, example **Kimchi jjigae**, cuisine Korea, **Audit recipe**. Point at the verdict ("reads as
   Korea"), the fidelity gauge and the ingredient table (each line mapped to CulinaryDB and FlavorDB).
2. **Flavor network** sub-tab: linked ingredients share flavor molecules.
3. Example **Spaghetti as Korean**, then **Audit recipe**. The headline turns to "reads as Italy" and the
   fidelity drops to the bottom percentiles.
4. **Suggestions** sub-tab: the atypical ingredients for Korea and what the fingerprint suggests instead. This is
   the same feedback our critique-and-revise steering gives the AI (slide 12).
5. **22-cuisine dashboard** tab: every model on every cuisine. Then return to the slides.
If the app fails to start, stay on slide 13: its screenshot shows the same app.

## What to show for each deliverable (if asked)
| Deliverable | Open | Say |
|---|---|---|
| D1 LLMCuisine-22 | `05_Dataset_LLMCuisine-22/` (CSV files in CulinaryDB format, `DATASHEET.md`) | "11,656 AI recipes in CulinaryDB's own format, with every model call logged." |
| D2 Audit | `02_Report/Group44_BIO544_report.pdf`, `06_Analysis_results/figures_png/` | "Same measures as for real cuisines, compared with genuine samples of equal size." |
| D3 Steering | slide 12; `09_Steering_D3/before_after_example.json` | "Feedback from the fingerprint makes recipes more recognisable while keeping the dish." |
| D4 FlavorAudit | backup slide 16; the notebook above, or `flavoraudit --help` in the terminal | "Every method as a tested, installable library." |
| D5 FlavorPrint | the live demo | "Anyone can audit a recipe in the browser." |

## Questions
Slide 15 has every link (dataset, report, code, app). Backup slides 16-25 cover the library code, the pairing-direction matrix, flavor drivers, variety, equal-size
comparison, copy-mutate, the persona prompt, the scorecard, best-of-N and networks. Prepared answers, the methods in plain words and the full
script split between the two speakers are in `01_Presentation/Group44_Presenter_Guide.pdf`.
"""

REPRO = """# Reproducing the results

Environment: Python 3.12 with uv (`uv sync --all-groups` in the repository root, then `.venv/bin/python`). The exact
versions are pinned in `uv.lock`. Fixed seeds are set in every script; configuration lives in `configs/`.

Data (not redistributed with the code; CC BY-NC-SA 3.0): CulinaryDB and FlavorDB, fetched by the repository's
`scripts/` into `data/raw/`; Ahn et al. (2011) ingredient-compound data (CC BY 4.0).

Pipeline (repository root):

1. Replication gate: `analysis/replication/replicate_singh_bagler.py`, then `analysis/figures/f2_replication.py`.
2. Real-cuisine references: `analysis/audit/real_subsamples.py`, `analysis/audit/equal_n_pairing.py`,
   `analysis/audit/copy_mutate_baseline.py`.
3. Generation (D1): Kaggle notebook `generation/kaggle/llmcuisine22_generate.ipynb` (open-weight models, pushed by
   `generation/kaggle/push.py`), `generation/run_api.py` (Gemini), `generation/openai_batch.py` (GPT-6-Luna). Every
   call is logged as JSONL with prompt, model id and revision, sampling parameters, timestamps, usage and raw output.
   The frozen prompt set is `prompt_set_v1.jsonl` (sha256 in `prompt_set_v1.sha256`).
4. Parsing and aliasing: `generation/parse_outputs.py`, `generation/alias_lines.py`; release:
   `generation/build_release.py`, `generation/write_datasheet.py`.
5. Audit (D2): `analysis/audit/llm_audit.py --tag full`, `analysis/audit/rq_tests.py --tag full`,
   `analysis/audit/networks.py`, `analysis/audit/drivers.py --tag full`; figures: `analysis/figures/d2_figures.py`,
   `analysis/figures/d2_more_figures.py`, `analysis/figures/equal_n_figures.py`.
6. Steering (D3): `steering/train_scorer.py`, `steering/run_bon.py --models gemini-3.5-flash-lite openai-gpt-6-luna
   qwen3.5-9b --regions all --conditions plain persona --n-values 1 2 4 8 --n-selected 20` (the deadline-mode design
   in the decision log; the script defaults are the original pre-registered design), `steering/run_revise.py`
   (prepare / evaluate),
   `steering/example.py`, `analysis/figures/d3_figures.py`.
7. Summary and documents: `analysis/summarize.py` (writes `results/summary.json`), `report/build.py --include-f5`,
   `presentation/build.py`, `app/build_artifacts.py`, then `scripts/export_final.py`.

`scripts/autopilot.py` chains steps 4-7 as generation runs finish. Hypotheses, metrics and tests were fixed in the
implementation plan before the full LLM data were analysed; every later change is dated in its decision log
(`IMPLEMENTATION_PLAN_preregistration_and_decision_log.md`).
"""


if __name__ == "__main__":
    raise SystemExit(main())

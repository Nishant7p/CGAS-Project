# Implementation plan: *Do AI Chefs Respect Culinary Fingerprints?*

Group 44 · CGAS BIO544 (IIIT-Delhi, Monsoon 2026) · plan v1.0, written 2026-11-22.
This is the master plan. `docs/STATUS.md` tracks progress against it, and `docs/RESEARCH_NOTES.md` holds the verified facts and sources it relies on.

---

## 0. One-paragraph summary

We test whether large language models, asked for "traditional" recipes from each of the 22 CulinaryDB world regions, reproduce the measurable **culinary fingerprints** of those cuisines. These are the flavor-pairing sign and strength (Ns and Z against four null models), the ingredients that drive the pairing, the category fingerprints (ΔNs_cat), the ingredient popularity structure and the co-occurrence patterns. The alternative is that they flatten cuisines into stereotypes. The method is Prof. Bagler's framework applied without changes: Singh & Bagler 2018, Jain et al. 2015 and Ahn et al. 2011. The baselines are the real CulinaryDB recipes and a copy-mutate evolution model. We then use the fingerprints to **steer** generation (best-of-N reranking and critique-and-revise), and release a dataset (D1), an audit (D2), a steering model (D3), an open-source library (D4) and a public web app (D5).

## 1. Research questions and pre-registered hypotheses

We freeze these, with the metrics in Section 6, **before any full-scale LLM data is analysed**. Any later change goes into the decision log (Section 13) with a reason.

| ID | Question | Hypothesis (direction) | Primary metric | Decision rule |
|---|---|---|---|---|
| RQ1 | Do LLM cuisines reproduce each region's food-pairing **sign and strength**? | H1: LLMs push cuisines toward **uniform (positive) pairing**, so sign errors cluster in the 6 contrasting regions (JPN, KOR, SCND, DACH, BRI, EE). | Sign agreement of Z_random with the real cuisine (22 regions × model); ΔZ and ΔNs effect size with 95% bootstrap CI | Sign error = CI of the LLM's ΔNs excludes 0 with the opposite sign to the real cuisine's. Test H1 with Fisher's exact test of sign errors in contrasting vs uniform regions. |
| RQ2a | Do LLM recipes rely on the **same flavor drivers**? | H2a: the top contributors differ (e.g. soybean sauce's share of Korea's negative pairing shrinks or grows). | Overlap of each region's top-k χ% contributors (Jaccard@5); Spearman correlation of χ% | Report the per-region distributions; no single threshold. |
| RQ2b | Do LLMs **amplify stereotypes**? | H2b: signature (overrepresented) ingredients are over-used, giving an SAI above 1, and diversity is lower. | Stereotype Amplification Index (SAI), unique-ingredient ratio, ingredient entropy, frequency-rank MAE | One-sample test of log SAI > 0 across regions (Wilcoxon signed-rank), BH-FDR. |
| RQ2c | Do LLMs keep the **category fingerprint**? | H2c: the ΔNs_cat fingerprints of LLM cuisines correlate less with real fingerprints than real cuisines correlate with themselves under resampling. | Cosine and Spearman between ΔNs_cat vectors | Compare with a bootstrap null built from real recipes. |
| RQ3 | Can fingerprint feedback **steer** generation toward authenticity without copying? | H3: best-of-N and critique-and-revise lower fingerprint distance and sign errors vs plain prompting, and novelty stays high. | Held-out evaluation metrics (Section 9.4) and the near-copy rate | Paired comparison per region (Wilcoxon), with effect size and CI. |
| RQ4 (secondary) | Do prompt framing, open vs proprietary models, and scale matter? | H4: the home-cook persona slightly improves fidelity. Neither the proprietary models (Gemini, GPT-6-Luna) nor the larger model of a family (Qwen3.5-9B vs 4B) are consistently more faithful than the open or smaller models (consistent with Carichon et al. 2026). *(The scale pair was Sol vs Luna until Sol became optional; see the decision log.)* | All primary metrics by condition, model and the Qwen 9B/4B pair | Paired comparison across the 22 regions (Wilcoxon), with region as the unit. |

## 2. Deliverables and definitions of done

A deliverable counts as **done** only when every box is ticked **and** it is reachable from the README's "Deliverables" table.

### D1: LLMCuisine-22 dataset
- [ ] ≥ 10,000 parsed recipes covering all 22 regions × ≥ 3 LLMs (≥ 1 proprietary, ≥ 2 open-weight families), with equal targets per model × region × prompt condition.
- [ ] Two prompt conditions (plain, home-cook persona), frozen after the pilot. Every call is logged as JSONL: prompt, system prompt, model id, version and date, all parameters, usage, latency, raw output and parse status.
- [ ] Every ingredient line is aliased with our implementation of the CulinaryDB protocol. Unmatched lines are curated and logged in `curation/alias_overrides.csv`. A coverage report (per model and region) is published.
- [ ] The release follows **CulinaryDB's file format**: `01_Recipe_Details.csv` (plus Model, Condition and Call ID columns) and `04_Recipe-Ingredients_Aliases.csv`, plus `generation_log.jsonl.gz` and `recipes_full.jsonl` (title, ingredients, instructions).
- [ ] A **datasheet** (Gebru et al. 2021 template), a license (CC BY-NC-SA 4.0) and the LLM output terms.
- [ ] Published on **Kaggle** and **Hugging Face Datasets**, with a Zenodo DOI. Each publication needs the user's go-ahead.
- [ ] QA: parse rate, refusal rate, duplicate report, and a manual audit of 100 random recipes (plausibility and aliasing precision).

### D2: food-pairing and culinary-fingerprint audit
- [ ] **Replication gate passed** (Section 5, Phase 2) and written up as `docs/replication_report.md`.
- [ ] For every model × region (and plain vs persona): N̄s, ΔNs, Z against all 4 nulls with bootstrap CIs, χ contributions, ΔNs_cat fingerprint, prevalence and SAI, diversity, novelty, size distribution and network metrics (Section 6).
- [ ] Copy-mutate baseline cuisines for each region, run through the same metrics (Phase 7).
- [ ] **Flavor-fidelity scorecard**: a model × region matrix plus a one-page summary per model. Figures F1–F15 (Section 11), generated by `analysis/` scripts.
- [ ] Results answer RQ1–RQ4 with pre-registered tests. Sensitivity analyses: flavor-profile perturbation (molecule dropout), Ahn-2011 profiles for shared ingredients, duplicates removed, additive handling, equal-N subsampling.

### D3: fingerprint-guided generator
- [ ] Fidelity scorer with 3 components (classifier, co-occurrence likelihood, Ns consistency), trained on an 80% CulinaryDB split. Classifier quality reported (macro-F1, top-3 accuracy, calibration). The scorer is validated by its AUC for separating real held-out recipes from other-cuisine, copy-mutate and null recipes.
- [ ] Steering: best-of-N (N ∈ {1, 2, 4, 8, 16}) and critique-and-revise (≤ 2 rounds), run on a pre-registered subset: all 6 contrasting regions, 4 uniform regions and ≥ 2 LLMs.
- [ ] Evaluated on **held-out metrics that were not optimized**: cuisine-level Z sign and ΔZ, fingerprint distance, prevalence JSD, an independent classifier and novelty/near-copy rate. Cost per accepted recipe reported. Ablation of each scorer component.

### D4: FlavorAudit library (`flavoraudit`)
- [ ] `pip install flavoraudit` works from PyPI (TestPyPI first). Python ≥ 3.10. MIT code. Data downloads on first use with a license notice.
- [ ] Modules: aliasing, flavor profiles and shared-compound matrix, Ns, the 4 nulls, Z and statistics, contributions (both definitions), fingerprints, copy-mutate models, distance metrics, plots and a CLI.
- [ ] Tests: ≥ 85% line coverage, property tests, and a slow regression test that reproduces Fig. 4 signs. CI on GitHub Actions for Python 3.10–3.13.
- [ ] Documentation site (MkDocs Material + mkdocstrings) with a quick start, "Replicate Singh & Bagler in 10 lines", "Audit your own recipes" and an API reference.
- [ ] **Example notebooks reproduce every D2 and D3 headline number.** Includes CITATION.cff and a CHANGELOG.

### D5: FlavorPrint web app
- [ ] Public Space on Hugging Face (Docker SDK + Streamlit template).
- [ ] **Audit tab**: paste a recipe and pick the claimed cuisine. Shows:
  - aliased ingredients (matched and unmatched);
  - the flavor-sharing network;
  - the recipe's Ns against the cuisine's real distribution (percentile);
  - predicted cuisine (top-5) and fidelity score breakdown;
  - stereotype flags;
  - fingerprint-guided suggestions (add or swap).
- [ ] **Dashboard tab**: interactive D2 results (pairing dot-plot, scorecard heatmap, fingerprint comparisons, SAI). **About tab**: methods, data and licenses.
- [ ] Cold start under 30 s and an audit under 3 s. Playwright smoke test. Works on a phone-width screen.

## 3. Architecture

```
CulinaryDB (4 CSV) ─┐                         ┌─> real cuisines (22 regions) ─────────────┐
FlavorDB v1/v2 JSON ├─> flavor layer ─────────┤                                            │
Ahn 2011 (CAS)  ────┘   entity→molecule sets  └─> shared-compound matrix S (≈1,000²) ──┐   │
                                                                                        │   │
LLM APIs / Kaggle ─> generation logs (JSONL) ─> parsed recipes ─> ALIASER ─> LLM cuisines ─┤
                                                                                        │   │
                     copy-mutate model ──────────────────────────────> CM cuisines ─────┤   │
                                                                                        ▼   ▼
             FlavorAudit core: Ns · 4 nulls · Z/ΔNs/CI · χ · ΔNs_cat · prevalence/SAI · diversity · networks
                                                                                        │
                       ┌────────────────────────────────────────────────────────────────┼──────────────┐
                       ▼                                                                ▼              ▼
                 D2 scorecard & figures                          D3 fidelity scorer ─> steering   D5 FlavorPrint
```

## 4. Repository layout (target)

```
ai-chefs-fingerprints/
├── README.md                 # deliverables table with links, quick start, citation
├── pyproject.toml  uv.lock   # library + dependency groups (dev, research, llm, app, docs)
├── src/flavoraudit/          # D4 library (MIT); no research-code imports
├── tests/                    # unit, property and regression tests
├── configs/                  # regions.yaml, models.yaml, prompts/*.txt, analysis.yaml
├── scripts/                  # data fetchers, one-off utilities
├── generation/               # D1 generation clients (openai/gemini/groq), batch builders, Kaggle notebook
├── curation/                 # alias_overrides.csv, manual audit sheets
├── analysis/                 # D2 scripts → figures/ and tables/
├── steering/                 # D3 scorer training + steering experiments
├── app/                      # D5 Streamlit app (+ Dockerfile for HF Space)
├── notebooks/                # example notebooks reproducing D2/D3 numbers
├── data/
│   ├── raw/                  # downloads (gitignored; CC BY-NC-SA / CC BY)
│   ├── reference/            # published targets (tracked)
│   ├── interim/              # flavor matrix, aliased tables (gitignored, rebuildable)
│   ├── generated/            # raw LLM logs + parsed recipes (gitignored; released separately)
│   └── release/LLMCuisine-22 # D1 release files (tracked if < 50 MB)
├── figures/  tables/         # generated outputs
├── report/                   # LaTeX report, poster, slides
└── docs/                     # plan, status, research notes, datasheet, replication report, mkdocs site
```

## 5. Phases

Effort is given in focused person-days. Phases 1–3 are a strict chain. Phase 4 can start right after the aliaser exists. Two people can run **Stream A** (science: 1 → 2 → 3 → 5 → 6 → 7) and **Stream B** (generation and product: 4 → 8 → 10) in parallel, and share Phases 9 and 11.

### Phase 0: foundations (mostly done 2026-11-22)
- [x] Read the course material, proposal, assignments and grading. Download CulinaryDB, Ahn 2011, the reference figures and 18 papers. Python 3.12 environment via uv.
- [x] FlavorDB entity JSON fetched from both endpoints (`flavordb`, `flavordb2`) by `scripts/fetch_flavordb.py`. The content is **identical**: 936 entities, IDs 0–978, 60,263 entity–molecule links and 1,788 molecules. The site now serves only current data, so no 2018 snapshot can be downloaded.
- [ ] `git init`, first commit (needs the user's OK), GitHub repo (private until release), issue board mirroring this plan.
- [ ] `.env` with API keys (the user fills it in from `.env.example`). Set an OpenAI project budget cap in the dashboard.

### Phase 1: flavor data layer (1–2 days). **DONE 2026-11-22**
`src/flavoraudit/{data,flavor}.py`:
- all 1,033 entities profiled;
- 0 unmatched compound constituents;
- S is 1,033 × 1,033, symmetric, with the diagonal equal to profile size;
- covered by tests.
1. Parse the FlavorDB entity JSON into `entities.parquet` (id, name, synonyms, category) and `entity_molecules.parquet` (id, pubchem_id).
2. Build the **CulinaryDB entity table** (930 basic + 103 compound ingredients) and attach profiles. Exactly **24 CulinaryDB ingredients have no FlavorDB entry**: the paper's 13 coarse-grained + 4 Ahn + 7 additives, which cover 17,910 of 456,279 rows (3.9%). The heaviest are Cayenne (5,834 rows), Lemon Juice (4,047), Baking Powder (3,644) and Yeast (936).
   - IDs 0–978 present in FlavorDB, including the CulinaryDB-added dishes 937–978: FlavorDB molecules, joined directly on the same ID.
   - IDs 979–991 (13 "coarse-grained" items such as Apple Juice, Coconut Milk, Lemon Juice, Tomato Paste and Lard): inherit the profile of their FlavorDB parent. Document every mapping as an assumption.
   - IDs 992–995 (Cayenne, Yeast, Tequila, Sauerkraut): **Ahn 2011 compounds**, CAS-mapped to FlavorDB PubChem IDs through FlavorDB's `cas_id` field. Unmapped CAS numbers become separate IDs.
   - IDs 996–998 (Baking Powder, MSG, Citric Acid): the molecule itself where FlavorDB has it. IDs 999–1002 (Cooking Spray, Gelatin, Food Coloring, Liquid Smoke) get no profile, as in the paper.
   - IDs 2000–2102 (compound ingredients): the union of their constituents' profiles. Constituent names map to IDs through the synonyms.
3. Build the shared-compound matrix `S[i, j] = |C_i ∩ C_j|` (int32, about 1,000 × 1,000) with bitsets, plus the self-check `S[i, i] = |C_i|`.
4. Checks: profile counts, the share of recipe-ingredient rows with an empty profile (by region), and the Ahn CAS→PubChem mapping rate.
- **Output:** `data/interim/flavor/{entities,profiles,S.npy}` and `flavoraudit.flavor` with tests.

### Phase 2: replication gate (2–3 days). **Nothing about LLMs is analysed before this passes.** **PASSED 2026-11-22**
- Canonical `v1_keep_empty`. Random null 22/22 signs. ρ = 0.980 / 0.980 / 0.942 / 0.979 (random / frequency / category / frequency+category).
- Contributions: top-1 in our top-3 for 22/22, 63/66 top-3 overlap.
- Details in `docs/replication_report.md`; regression guard `pytest -m slow`.
1. Implement Ns (recipe and cuisine) and the 4 nulls (Section 6.2). Use 100,000 recipes per null and fixed seeds.
2. Reproduce Table 1 counts, the recipe-size distribution (mean about 9), the frequency-rank curves, the category heatmap (without Additive) and **Fig. 4 (all 4 nulls × 22 regions)**.
3. Implement contributions: χ (Ahn eq.) and χ% ("% change in Z on removal"). Calibrate the removal procedure (drop the ingredient from its recipes and rebuild the nulls) against the published top-3 contributors.
4. **Calibration grid.** The paper under-specifies a few choices, so test each and pick by agreement with the targets, documenting the choice:
   - (a) keep or drop ingredients with an empty profile in Ns;
   - (b) count recipe size as all ingredients or only profiled ones;
   - (c) reconstruction variants for the 24 missing ingredients (e.g. Cayenne from Ahn's data vs a Capsicum proxy; Lemon Juice → Lemon; Baking Powder empty vs not);
   - (d) deduplicate (recipe, entity) or not;
   - (e) whether compound ingredients use the union rule.
5. **Acceptance:**
   - sign agreement **22/22** for the random null;
   - Spearman ρ ≥ 0.90 with the published Z for the random and category nulls, and ≥ 0.70 for the frequency and frequency+category nulls;
   - the published top-1 negative contributor is in our top-3 for ≥ 5 of 6 contrasting regions (Korea's Soybean Sauce must be #1);
   - if the numbers differ, the differences are explained in the replication report.
- **Output:** `docs/replication_report.md`, figure F2 (ours vs published), and a regression test (`pytest -m slow`).

### Phase 3: aliasing engine (2–3 days). **DONE 2026-11-22**
- `src/flavoraudit/aliasing.py` reconstructs CulinaryDB's protocol:
  - singularization with `inflect`;
  - '-' marks any-order parts, '=' joinable words, '#' disabled entries;
  - partial matches;
  - tie-breaks learned from expert lines.
- **Held-out test: 97.9% of unique lines, 99.0% of occurrences, 99.98% coverage** (target ≥ 90%). Evidence: `analysis/aliasing/evaluate_aliaser.py`, `data/interim/aliasing/metrics.json`.
- On the LLM pilots, 95.7–95.9% of ingredient lines alias. Unmatched terms (mirin, dashi, gochugaru, …) go to `analysis/curation/propose_aliases.py`, which proposes mappings from CulinaryDB evidence; the final curation runs on the full-run vocabulary.
1. Implement the protocol (Singh & Bagler, Section IV-A):
   - lower-case; strip punctuation, digits and fractions;
   - singularize each word (inflect, plus an exception list: molasses, couscous, hummus, asparagus, swiss and so on);
   - remove English and **culinary stop-words** (units, sizes, preparation verbs, freshness words);
   - match n-grams (n ≤ 6) against entity names and synonyms, **longest match first**;
   - handle multi-ingredient lines ("salt and pepper") and label partial or unmatched lines for curation.
2. **Evaluate on the CulinaryDB gold standard**: 456,279 rows and 161,208 unique raw phrases, each with an expert entity ID. Hold out 20% of phrases (stratified by source). Report exact-entity accuracy, coverage and per-source accuracy. Tune the stop-word lists on the training split only. **Target ≥ 90% held-out accuracy.**
3. Curation workflow:
   - rank unmatched phrases by frequency and build n-gram statistics (as in the paper);
   - an LLM-assisted suggester (cheap model, constrained to choose from the entity list or "none") pre-fills `alias_overrides.csv`;
   - **a human accepts or rejects every suggestion**;
   - the aliaser applies overrides first.
- **Output:** `flavoraudit.aliasing` and an aliasing evaluation table and figure (an appendix figure in the report).

### Phase 4: generation pipeline and pilot (2–3 days, plus 1–3 days wall-clock for batches)
1. `configs/regions.yaml`: the 22 regions (exact CulinaryDB labels), adjectives, and country lists for personas.
2. `configs/models.yaml`: the **planned roster** (user decisions 2026-11-22: "do we need this many models?", then "use OpenAI as little as possible, use more free models"). The instructor asked for no model count. The proposal promises ≥ 3 LLMs (1 proprietary plus 2 open-weight families).

   | Tier | Models (keys) | Who does the work | Why |
   |---|---|---|---|
   | **core** (the proposal, at $0) | `llama-3.1-8b`, `qwen3.5-9b` (Kaggle) + `gemini-3.5-flash-lite` (free API) | Claude via the Kaggle CLI and the API runner | exactly what D1 promises: 3 × 3,520 = 10,560 recipes, 2 open families + 1 proprietary |
   | **extra** | `qwen3.5-4b`, `ministral-3-8b` (Kaggle, free); `openai-gpt-6-luna` (Batch, **$0.19 actual**) | Claude | scale contrast (Qwen 9B vs 4B), a 3rd open family, a 2nd proprietary vendor |
   | optional | GPT-6.2-Sol (about $10), Olmo-3-7B, Phi-4-mini, Gemma-3-4B, Llama-3.2-3B, EXAONE-3.5-7.8B (Kaggle); Groq Qwen3.8-27B and gpt-oss-120b | not planned | researched and configured; use only if the user asks or all 5 deliverables are finished |

   Total planned: 6 × 3,520 = **21,120 recipes**, about 4–20 Kaggle GPU-hours (the slow transformers fallback is the upper bound) and **$0.19 cash**. No HF token is needed (Llama loads from the verified byte-identical mirror).
   - [x] **Built (2026-11-22):** `configs/regions.yaml`, `configs/prompts/v1/*`, `configs/models.yaml`, `generation/common.py` (prompt set, ids, seeds, robust parser), `generation/kaggle/build_notebook.py` → `llmcuisine22_generate.ipynb`, `generation/run_api.py`, `generation/parse_outputs.py`, and 18 tests. Prompt set v1 = 704 prompts, sha256 `468c7d7f4c79…`. The notebook loop (checkpoint, resume, summary) was verified locally with a fake engine. The user's steps are in `docs/RUNBOOK_GENERATION.md`.
3. Prompts (drafts in Appendix A):
   - **plain** ("Give 5 different traditional {adjective} recipes…") and **persona** ("You are a home cook who grew up in {country} …");
   - JSON output schema: title; ingredients as raw lines plus a short name; ordered instructions;
   - nothing that hints at flavor pairing;
   - English only.
4. Design: per model × region × condition, 16 calls × 5 recipes = 80, so 160 per model × region and **3,520 per model**.
   - **Sampling policy (pre-registered; revised before any data existed):**
     - **All open-weight models: temperature 0.7, top_p 0.95, no penalties, thinking off.**
     - **Proprietary APIs: provider defaults.** Reasoning models fix temperature; reasoning is at its lowest allowed level.
     - max_tokens is 6,144 for all.
     - Why uniform: developer recommendations range from T < 0.1 (Mistral) and T = 0 in Phi's examples to 0.7 (Qwen). Near-greedy decoding returns near-duplicate recipes, so per-model settings would confound decoding with culinary knowledge.
   - A **sensitivity subset** (1 replicate per region and condition at a common T = 1.0, top_p = 1.0) checks that conclusions don't hinge on the settings.
   - The per-call seed and the pinned HF model revision (commit sha) are logged.
5. Clients: a resumable job runner with a rate limiter, retries with backoff and idempotent call IDs (hash of model, region, condition and replicate). Batch-file builders for OpenAI, Gemini and Groq, and a Kaggle notebook for Llama.
6. **Pilot:** 1 call per model × region × condition (44 calls per model). Check:
   - parse rate ≥ 98% and refusal rate;
   - alias coverage ≥ 90% of lines;
   - mean recipe size and duplicates;
   - tokens and cost per recipe.

   Fix the prompts **once**, then **freeze them** (git tag `prompts-v1`) and record the decision.
7. Full run. Monitor progress, then re-run failed calls only.
- **Output:** `data/generated/raw/<model>/*.jsonl` and `data/generated/parsed/recipes.parquet`.

### Phase 5: dataset assembly and release (1–2 days)
Alias every line, apply the overrides and compute coverage. Write the CulinaryDB-format CSVs, the datasheet (`docs/datasheet_LLMCuisine-22.md`), the coverage report and a dedup report, and run the manual audit of 100 recipes. Prepare the Kaggle, HF and Zenodo metadata. **Publish only after the user approves.**

### Phase 6: audit analyses, D2 (4–6 days). **Harness built 2026-11-22; waiting for the full runs**
- **Built:**
  - `flavoraudit.audit.scorecard_row`, the one scoring function for LLM, copy-mutate and real-subsample cuisines;
  - `analysis/audit/{audit_io, real_subsamples, llm_audit, rq_tests}.py`, covering pooled and per-condition scorecards and the pre-registered tests with BH-FDR.
- **Still to do:** χ% drivers (RQ2a), PMI networks, figures F3–F15, and the sensitivity analyses.

Compute every metric in Section 6 for: real; each model × condition; pooled per model; and copy-mutate. Use equal-N comparisons by subsampling the real cuisine to the LLM's N (1,000 bootstrap draws). Then:
- run the RQ tests (Section 7);
- run the sensitivity analyses: profile perturbation, Ahn profiles, deduplication, additives, persona vs plain, and recipes with ≥ 80% coverage only;
- produce figures F3–F15, the scorecard and per-model summaries.

### Phase 7: copy-mutate baseline (1 day). **Built 2026-11-22**
`flavoraudit.evolution` (Tuwani 2019 Algorithm 1, ranked or random fitness, CM-R/C/M; 12 property tests) and `analysis/audit/copy_mutate_baseline.py`.
For each region, run **CM-fitness-ranked** (Jain & Bagler 2018; Tuwani 2019; our Assignment 3 code) with the nature basket set to the region's ingredients, fitness set to their real popularity, and sizes set to the region's distribution. Generate a cuisine of the region's size and one at the LLM's N, with 10 instances. Use CM-random as a secondary model. Run all D2 metrics on these cuisines. This baseline shows what a simple evolutionary model achieves, so the LLMs can be judged against it.

### Phase 8: fidelity scorer and steering, D3 (4–6 days). **Scorer built 2026-11-22**
`flavoraudit.scoring.FidelityScorer` (classifier, shrinkage PMI, Ns consistency; temperature calibration; ECE) and `steering/train_scorer.py` (model selection on validation, test metrics, validation AUCs against 3 negative types, independent evaluation classifier).
Details are in Section 9. Order: classifier, then PMI tables, then Ns-consistency, then scorer validation (AUC), then best-of-N, then critique-and-revise, then held-out evaluation, then ablations, then cost. The budget is capped by a pre-registered subset.

### Phase 9: library hardening, D4 (3–4 days, ongoing)
Freeze the API (Section 10), write docstrings citing equations, raise coverage to ≥ 85%, set up CI, build the docs site, write the example notebooks (they must reproduce the D2/D3 headline numbers), then release TestPyPI → PyPI **v0.1.0** (after the user's OK) and mint a Zenodo software DOI.

### Phase 10: FlavorPrint app, D5 (3–4 days)
Pages: Audit, Dashboard, About. Precompute the artifacts (S matrix, classifier, PMI tables, region distributions) so the Space stays CPU-only. Use `st.cache_resource` and `st.cache_data`. Then deploy on HF Spaces (Docker, Streamlit template, `app_port: 8501`), run Playwright smoke tests and add a demo GIF to the README.

### Phase 11: report, poster, slides and video (4–5 days)
- **Report:** IEEE two-column style, as in the ICDE workshop papers of the lab, about 8 pages plus appendix. **Poster** (A0) for the symposium. **Slides:** 12–15 for Week 14. A **3-minute demo video**.
- A "proposal vs delivered" traceability table, an explicit "what is new vs baseline" box, and an AI-assistance disclosure.
- **Q&A preparation:**
  - Why Z inflates with N_rand.
  - Why the frequency null explains the pattern.
  - How we differ from Group 23.
  - Why the copy-mutate model is a baseline.

### Phase 12: final QA (1–2 days)
Rebuild everything from a clean clone (`make all` or `uv run …`), check every link in the README, verify the license files, and confirm that no secret appears in the git history. Final read-through by both team members.

**If a deadline appears, the minimum path is:** Phases 1–2–3, then the pilot plus the minimum 3-model run, then D2 core metrics (Z, ΔNs, χ, SAI and scorecard), then D3 best-of-N only, then the D4 release, then a minimal D5 audit tab, then the report.

## 6. Metric catalogue (exact definitions)

### 6.1 Flavor pairing
- Recipe flavor sharing: **Ns(R) = 2/(n(n−1)) · Σ_{i<j∈R} |C_i ∩ C_j|** for n ≥ 2 (Ahn 2011; Jain 2015 eq. 2). Recipes with n < 2 are excluded and counted.
- Cuisine: **N̄s = (1/N_R) Σ_R Ns(R)**. **ΔNs = N̄s_cuisine − N̄s_null**.
- **Z_SB = √N_rand · ΔNs / σ_rand**, with N_rand = 100,000 and σ_rand the SD of Ns(R) over the null cuisine (Jain 2015 eq. 3). Report it for comparability with the published values.
- Size-aware significance: the empirical p-value from the distribution of the mean Ns over many null cuisines with **the real cuisine's N**. Also report the effect size d = ΔNs / σ_rand. Since Z_SB = √N_rand · d, it is comparable across cuisines of any size.
- Bootstrap CI: resample the cuisine's recipes 1,000 times and recompute ΔNs against a fixed null.

### 6.2 Null models (each preserves the cuisine's own ingredient set and recipe-size distribution)
- **Random:** size s ~ the empirical size distribution; draw s ingredients uniformly without replacement.
- **Frequency:** as random, with weights ∝ the ingredient's recipe count in the cuisine (weighted sampling without replacement via the Gumbel top-k trick).
- **Category:** pick a real recipe as a template and keep its category multiset; fill each category slot uniformly from the cuisine's ingredients in that category.
- **Frequency + category:** as category, with weights ∝ frequency within each category.
- **Category-randomized (for ΔNs_cat):** randomize only category c's ingredients and keep all others (Jain 2015). **Fingerprint = vector of Z_cat over the 21 categories.**

### 6.3 Drivers
- **χᵢ (Ahn):** (1/N_c) Σ_{R∋i} [2/(n_R(n_R−1)) Σ_{j≠i∈R} |C_i ∩ C_j|] − (2f_i/(N_c⟨n_R⟩)) · Σ_j f_j|C_i ∩ C_j| / Σ_j f_j.
- **χ%ᵢ (Singh & Bagler):** 100 · (Z − Z₋ᵢ)/Z, where Z₋ᵢ is computed after removing i from the cuisine. The exact removal procedure is calibrated in Phase 2.

### 6.4 Composition, authenticity and stereotypes
- Prevalence **P_ic = n_ic / N_c**. Relative prevalence (authenticity) **p_ic = P_ic − mean_{c'≠c} P_ic'** (Jain & Bagler 2018). Overrepresentation (Tuwani 2019) is used as a check.
- **Signature set** Σ_c = the top-10 ingredients by p_ic in the **real** cuisine.
- **Stereotype Amplification Index:** SAI_c = Σ_{i∈Σ_c} P_ic^LLM / Σ_{i∈Σ_c} P_ic^real. Per-ingredient amplification is log(P^LLM/P^real) with additive smoothing.
- Prevalence distance: **Jensen–Shannon divergence** between normalized prevalence vectors (real vs LLM), plus Spearman ρ over the union of the top-50 ingredients.
- Category composition: the mean category share per recipe, compared with JSD, plus the category-importance heatmap (as in Singh & Bagler Fig. 2).

### 6.5 Diversity, novelty and structure
- Diversity: unique ingredients (at equal N via rarefaction), Shannon entropy of the ingredient distribution, the share of distinct ingredient sets, title duplicates and **frequency-rank MAE** against real (Tuwani 2019).
- Recipe size: mean and SD, KS statistic and Wasserstein distance against real.
- Novelty: for each LLM recipe, the **max Jaccard similarity to any CulinaryDB recipe** (all regions; computed with a MinHash index and an exact recheck). The near-copy rate is the share with max Jaccard ≥ 0.8.
- Co-occurrence: PMI networks per cuisine. Compare the top-k edges (Jaccard@100) and the overlap of the disparity-filter backbones.

### 6.6 Flavor-fidelity scorecard (per model × region)
Columns: sign match; |ΔZ_SB| / |Z_real|; d with CI; fingerprint cosine; prevalence JSD; SAI; frequency-rank MAE; size KS; diversity ratio; near-copy rate. The composite **Fidelity Index** is the mean of rank-normalized components. **The components are always shown next to it.**

## 7. Statistical analysis plan
- Seeds are fixed per analysis (`configs/analysis.yaml`). Nulls use 100,000 recipes. Bootstraps use 1,000 resamples, and CIs are percentile-based.
- Equal-N comparisons: subsample real cuisines to the LLM's N (or both to min(N)). Repeat 1,000 times and report the median and 95% interval.
- Multiple comparisons: BH-FDR at q = 0.05 across the 22 regions within each model and metric. Report adjusted and raw p-values.
- Tests: Fisher's exact test (H1), Wilcoxon signed-rank (H2b, H3), permutation tests for differences between conditions or models, and a mixed-effects or stratified analysis for RQ4.
- Robustness:
  - flavor-profile perturbation: drop 10% or 20% of each ingredient's molecules at random, 100 replicates, and report sign stability;
  - Ahn-2011 profiles substituted for ingredients present in both databases. This answers Singh & Bagler's open question about robustness to changes in flavor profiles;
  - deduplication, additive handling, coverage ≥ 80% only, and leave-one-model-out.
- Negative or null results are reported in full.

## 8. Generation protocol details
- **Call ID** = sha256("model_key|v1|region|condition|replicate")[:16], and the per-call seed is a separate sha256 of the same key (`generation/common.py`). Calls are idempotent and every runner resumes after failures. Pilot and full runs share call ids for replicate 0, so the parsed tables carry a `mode` column and only full-mode calls enter D1 and D2.
- Logged per call: UTC timestamp, provider, model id, any returned model version or fingerprint, every request parameter (temperature, top_p, max tokens, reasoning effort, seed, schema), system and user prompts, raw response, usage tokens (prompt, completion, reasoning), latency, finish reason and parse status.
- Parsing: strict JSON schema validation, one repair pass (JSON extraction only, never content edits) and a language check. Refusals or off-topic answers are logged and **not** retried with a modified prompt.
- Budget guard: cost is estimated from the pilot. The runner refuses to start if the projected cost is above the cap set in `.env` (`BUDGET_USD`).
- Fairness: identical prompt templates across models. The same 5-per-call design everywhere. The persona's country is sampled uniformly per call from the region's list, seeded and logged.

## 9. D3 design: scorer and steering
1. **Classifier.** Multinomial logistic regression (L2, class-balanced) on binary ingredient vectors plus category counts, trained on an 80% CulinaryDB split stratified by region. Compare with a linear SVM and LightGBM, then pick and freeze one. Report macro-F1, top-1 and top-3 accuracy and calibration (reliability diagram, ECE). Apply temperature scaling if needed.
2. **Co-occurrence likelihood.** Smoothed PMI of ingredient pairs within cuisine c, from the training split. Score = the mean PMI over the recipe's pairs, so it does not depend on recipe size.
3. **Ns consistency.** Let F_c be the empirical CDF of recipe Ns in real cuisine c. Score = 1 − |2F_c(Ns(R)) − 1| (1 at the median, 0 at the tails). Also check cuisine-level sign consistency.
4. **Composite score.** S = Σ_k w_k · z(score_k), with weights fixed before steering (equal weights, or logistic weights fitted on the validation split). **Validation:** AUC for real held-out recipes of c against (i) other cuisines' recipes, (ii) copy-mutate recipes and (iii) null-model recipes.
5. **Steering methods.**
   - Best-of-N: sample N candidates from the plain prompt and keep the top-S one, N ∈ {1, 2, 4, 8, 16}.
   - Critique-and-revise (Self-Refine style): feed back the most atypical ingredients (lowest classifier contribution), the region's highly typical ingredients that are absent, and whether Ns is too high or low. Ask for a revision that keeps the dish's identity. Up to 2 rounds.
   - The combination of both.
6. **Held-out evaluation, so the optimized score is never also the yardstick:**
   - Ns-based: cuisine-level Z sign and ΔZ (≥ 100 steered recipes per region and method).
   - Composition: fingerprint and prevalence JSD.
   - **An independent classifier** (different model class, trained on the other 20% split).
   - Novelty: near-copy rate and mean nearest-neighbor Jaccard. Diversity: unique ingredients.
   - Practical: cost and tokens per accepted recipe.
   - Optional: an LLM-judge authenticity rating (blind) and a small blind human rating (a "Turing test for chefs"-style mini-study, which is a stretch goal).
7. **Ablations:** drop one component at a time. **Guard against reward hacking:** a novelty penalty and a duplicate filter.

## 10. FlavorAudit public API (target for v0.1.0)

```python
import flavoraudit as fa

db = fa.datasets.load_culinarydb()  # cached download + license notice
flav = fa.datasets.load_flavordb()  # FlavorProfiles (current FlavorDB + the 24 reconstructed profiles)
kor = db.cuisine("Korea")
res = fa.food_pairing(kor, flav, nulls="all", n_null=100_000, seed=0)
res.table()  # Ns, ΔNs, Z_SB, d, CI per null
fa.contributions(kor, flav, method="percent")  # χ% ; method="ahn" for χ
fa.fingerprint(kor, flav)  # ΔNs_cat / Z_cat vector
al = fa.Aliaser.from_culinarydb(db)
al.alias("2 small Jalapeno Peppers, roasted and slit")  # [Match('Jalapeno', id=…, how='longest')]
audit = fa.audit(my_recipes, reference=db, flavor=flav)  # scorecard DataFrame
cm = fa.evolution.copy_mutate(kor, fitness="ranked", n_instances=10, seed=0)
fa.plots.pairing_dotplot(res)
```
CLI: `flavoraudit audit recipes.csv --cuisine-col cuisine --out out/`. Modules: `datasets, io, aliasing, flavor, pairing, nulls, stats, contribution, fingerprint, composition, network, evolution, scoring (extra: ml), plots, cli`.

## 11. Figure plan (number them as in the report)
- **F1** Pipeline overview.
- **F2** Replication: our Z against the published Z, 4 nulls × 22 regions.
- **F3** Fig. 4-style dot plot of real vs each LLM's Z by region.
- **F4** ΔNs effect sizes with CIs.
- **F5** Sign-agreement matrix (model × region).
- **F6** Category fingerprints (real vs LLM) and their distances.
- **F7** Driver comparison: χ% top contributors (Korea soybean sauce focus).
- **F8** SAI by region and model.
- **F9** Diversity and frequency-rank against real and copy-mutate.
- **F10** Recipe size distributions.
- **F11** Co-occurrence backbones for 2 showcase regions.
- **F12** Scorecard heatmap.
- **F13** Copy-mutate baseline comparison.
- **F14** Persona vs plain.
- **F15** Steering curves: metric against N and against revision rounds, with novelty.
- **Appendix:** aliasing accuracy, classifier confusion matrix, calibration, cost table.

## 12. Risks and mitigations
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Replication doesn't match: today's FlavorDB is not the 2018 version, and 24 profiles plus the compounds are reconstructed | M | H | Calibration grid (Phase 2); report rank agreement and explain gaps; optionally look for an archived 2018–2019 FlavorDB scrape (Wayback, Kaggle, GitHub) for a closer replication. |
| R2 | FlavorDB or CulinaryDB sites go down or change | M | H | Raw JSON (fetched 2026-11-22, gz) and the ZIP are cached locally. Archive them with the release under CC BY-NC-SA. |
| R3 | Low aliasing coverage for LLM text | M | M | Raw line plus name fallback, curated overrides, LLM-suggested mappings with human checks, and analyses restricted to high-coverage recipes. |
| R4 | Model ids or pricing change mid-run | M | M | Log exact ids and versions, finish each model's run within days, keep raw outputs. |
| R5 | Free-tier rate limits | H | L | Batch or paid fallback (costs are tiny), resumable runner. |
| R6 | vLLM won't run on Kaggle T4, or fp16 overflows | L–M | M | vLLM 0.34 docs list T4 (CC 7.5) as supported, and fp16 is set automatically. The notebook falls back to transformers automatically. Prefer a GPU L4 (bf16) if offered. The pilot catches gibberish or empty outputs before the full run. |
| R7 | Cost overrun | L | M | Budget guard, OpenAI project cap, cost projected from the pilot. |
| R8 | Statistical artefacts (Z inflated, unequal N) | H | H | Equal-N subsampling, effect sizes, bootstrap, size-aware p-values. |
| R9 | Steering games the scorer or copies recipes | M | H | Held-out metrics, independent classifier, novelty constraint, ablations. |
| R10 | Overlap with Group 23 is perceived | M | M | Differentiators stated in the report and slides; avoid their distinctive components. |
| R11 | Disk space (about 12 GB free) | M | L | No local models, compressed raw data, ask before cleaning caches. |
| R12 | Licence violations | L | H | CC BY-NC-SA for derived data, MIT for code, data downloaded at runtime, attributions. |
| R13 | Coordinating 2 people | M | M | STATUS.md, GitHub issues mirroring the phases, owners decided by the team. |

## 13. Decision log
| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | Library renamed CuisineKit → **FlavorAudit**; title revised to "…Auditing and Steering…" | Group 18 already uses CuisineKit; steering is our differentiator. |
| 2026-11-22 | Repo `ai-chefs-fingerprints/` holds all 5 deliverables; library at `src/flavoraudit` | One link for the graders; the library is still packaged on its own. |
| 2026-11-22 | Use the **current FlavorDB** (both endpoints serve identical data). Robustness comes from molecule-dropout perturbation and Ahn-2011 profiles | No 2018 snapshot is online. Robustness to flavor profiles is an open question the paper itself raises. |
| 2026-11-22 | Derived data under CC BY-NC-SA 4.0, code MIT | Source licences (share-alike). |
| 2026-11-22 | Plan for quality, no fixed deadline (the user said time is not an issue); keep the minimum path ready | User instruction. |
| 2026-11-22 | The user has an OpenAI API key plus AI Studio, Groq, Kaggle, HF and GitHub accounts; model roster still to be confirmed before spending | User answers. |
| 2026-11-22 | Task owners are decided later by the team | User answer. |
| 2026-11-22 | **Free models first**: Llama-3.1-8B and Qwen3.5-9B on Kaggle (HF weights) plus Gemini 3.5 Flash-Lite free API as the proprietary model; Olmo-3-7B and Groq Qwen3.8-27B / gpt-oss-120b optional; OpenAI disabled | User: "use free models for now … I will use Kaggle". This still meets the proposal (≥ 3 LLMs, 1 proprietary + 2 open families). |
| 2026-11-22 | ~~Sampling = each developer's recommended settings~~ → **superseded the same day**: uniform T 0.7 / top_p 0.95 for open-weight models, provider defaults for proprietary APIs, plus a T = 1.0 sensitivity subset | While adding models we found Mistral recommends T < 0.1 and Phi's examples use T = 0. Near-greedy decoding would make near-duplicate recipes and confound decoding with knowledge. No data had been generated yet. |
| 2026-11-22 | ~~Expanded roster of 14 models~~ → **superseded the same day** (next row) | User first said "use more free hugging face models", "I will provide OpenAI API". |
| 2026-11-22 | **Roster re-tiered to minimise OpenAI spend.** Core = Llama-3.1-8B + Qwen3.5-9B + **Gemini 3.5 Flash-Lite** (free proprietary model), so the proposal is met at $0. Extra = Qwen3.5-4B (free; scale), Ministral-3-8B (free; 3rd open family), GPT-6-Luna (about $0.2; the only planned OpenAI spend). GPT-6.2-Sol is optional. `BUDGET_USD = 5` | User: "use OpenAI as little as possible, use more free models". |
| 2026-11-22 | **Pilot decision: Gemini JSON mode** (`response_mime_type: application/json`) for its full run; the parser also gained a bracket-balancing repair | 3 of the first 10 Gemini pilot replies omitted a closing bracket. The repair recovers them, and JSON mode prevents it. Content is unaffected. |
| 2026-11-22 | **Aliasing = CulinaryDB-faithful protocol + learned tie-breaks** (97.9% / 99.0% held-out agreement) | Mapping LLM text exactly as CulinaryDB mapped human text keeps the real-vs-LLM comparison unbiased. |
| 2026-11-22 | **Llama weights load from `RedHatAI/Llama-3.1-8B-Instruct`**, an ungated mirror verified byte-identical to `meta-llama/Llama-3.1-8B-Instruct` (weights, tokenizer, chat template and configs: 10/10 files; `scripts/verify_mirror.py`). The report cites Meta's model and the licence still applies | The user didn't want to fill Meta's access form or attach HF secrets. A byte-identical copy gives exactly the same model, and the evidence is stored in `data/reference/`. (The Unsloth copy was rejected: same weights, but an edited tokenizer and template.) |
| 2026-11-22 | **Lean planned roster: 5 models.** Core = Llama-3.1-8B + Qwen3.5-9B + GPT-6.2-Sol (exactly the proposal). Extra = Gemini 3.5 Flash-Lite + GPT-6-Luna (API-only, zero user effort). The other 9 are kept as `optional` | User: "do we need this many models? it will take so much time". No model count was required, and marks come from deliverable quality, so the time goes into D2/D3/D5 instead. |
| 2026-11-22 | Ingredients are requested as plain recipe lines (no separate "name" field) | Mirrors CulinaryDB's raw "Original Ingredient Name", so one aliaser serves both, and it keeps the format easy for small models. |
| 2026-11-22 | **Prompts frozen** as prompt set v1 (sha256 `468c7d7f4c79…`) after the Luna, Gemini and Qwen pilots (98–100% parsed after structural repair, 94–96% of lines aliased). The git tag `prompts-v1` waits for the user's OK to commit | The pilot criteria (§5 Phase 4 step 6) were met; no prompt change was needed. |
| 2026-11-22 | **Gemini full run uses `thinking_level: minimal`** (the pilot used `low`); decided before any full-run data | Both levels were accepted in a probe; `minimal` is the closest to "thinking off", as for Qwen (`enable_thinking: false`) and Luna (`reasoning_effort: none`). A rejected level now stops the run instead of silently falling back to dynamic thinking. |
| 2026-11-22 | **Gemini stays on the free synchronous API** with `--wait-daily` and 8 h of retries per call | Batch is rejected on the free tier (`FAILED_PRECONDITION`); enabling billing (about $1) is the user's call. The free tier has long 503 "high demand" spells. |
| 2026-11-22 | **Parsing repairs are structural only**: trailing commas, bracket balancing, recipe-boundary normalisation, and reading a reply as a stream of JSON values; then salvage of complete recipe objects. Content is never edited, and every call's status (ok, repaired, salvaged) is kept | Real slips seen: Gemini closes the document after recipe 1, Qwen and Gemini drop or duplicate one bracket at a recipe boundary. |
| 2026-11-22 | **Dataset and analysis use full-mode calls only, at most the first 5 recipes per call**; recipes with no matched ingredient are dropped; under-production (fewer than 5 recipes) is kept and reported | The pilot only tested the pipeline (and shares call ids with replicate 0). The design asks for exactly 5 per call; equal-N comparisons handle shortfalls. |
| 2026-11-22 | **Copy-mutate baseline = Tuwani et al. 2019, Algorithm 1** (m = 20, n = m/φ, M = 4) **with Jain & Bagler's ranked fitness** (empirical frequency) as the primary model and random fitness as the secondary. Seed recipe sizes follow the region's distribution; identical copies are kept; 10 instances per region; scored at full size and at n = 160 | The lab's exact algorithm on CulinaryDB, plus the plan's size-matching. Sanity check: rank-frequency MAE 0.013–0.05, as in the paper. |
| 2026-11-22 | **Equal-N reference = 1,000 random 160-recipe subsamples per real cuisine** (fingerprints on 200), scored by the same `flavoraudit.audit.scorecard_row` as the LLM and copy-mutate cuisines; novelty of real recipes is leave-one-out | Gives the sampling-noise floor of every metric at the LLMs' N, the false-positive rate of the sign-error rule, and the H2c resampling null. |
| 2026-11-22 | **RQ4 scale pair: Qwen3.5-9B vs Qwen3.5-4B** (was Sol vs Luna) | Sol became optional when the roster was re-tiered to minimise OpenAI spend, before any full run. Recorded after the Luna full run was fetched, whose pipeline-test summary was printed; the change follows from the roster, not from results. |
| 2026-11-22 | **Amendment to RQ1, made before the full LLM analysis: equal-N food pairing.** The pre-registered sign-error rule (the LLM's ΔNs CI vs the *full* real cuisine's sign) is still reported, together with its false-positive rate on genuine samples. The primary inference adds an **equal-N comparison**: each LLM cuisine's own-pool d is placed in the distribution of d over genuine subsamples of the same region at the same N (`analysis/audit/equal_n_pairing.py`), with an empirical two-sided p and BH across regions. "Too uniform" means above the 97.5% quantile and "too contrasting" below the 2.5% quantile. **H1 (equal-N)** asks whether LLM d sits above the genuine median, especially in the 6 contrasting regions (Wilcoxon of the shift; Fisher on direction × region type). A fixed-pool ΔNs (against the full real cuisine's pool) is reported as secondary. Fig. 4 is also re-checked at matched size (every cuisine cut to 290 pairing recipes) | Calibration on 1,000 genuine 160-recipe subsamples per region found that the own-pool random null is **N-dependent through its ingredient pool**. A small sample loses the cuisine's long tail of rare, low-sharing ingredients, so the null's expected Ns rises while the real mean Ns does not. For USA, ΔNs falls from 12.7 to about 0 and the pool from 595 to 238 ingredients; against the full pool it stays 12.8. Genuine 160-recipe samples of Canada, China, France and South East Asia have the *opposite* sign to their full cuisine, and the pre-registered rule fires on 8.6% of genuine samples (55–61% for Canada and China). Only the Luna pipeline-test summary had been printed when this was decided; the change follows from the genuine-sample calibration, not from LLM results. |
| 2026-11-22 (evening) | **Deadline mode** (user: finish all deliverables much sooner, without fabricating results).<br>**D3 best-of-N:** reranks the *already generated* recipes (160 per model × region, plain + persona, all 22 regions; N ∈ {1, 2, 4, 8}; 20 selected per region and N; the same guards and held-out metrics) instead of the pre-registered 1,600-candidate pool (N up to 16, 100 selected, 10 regions).<br>**Critique-and-revise:** 20 recipes per region × 22 regions = 440 per model. GPT-6-Luna runs via Batch, for 2 rounds (estimated ≤ $0.6); Qwen3.5-9B runs 1 round on Kaggle.<br>**Deferred:** the extra models Qwen3.5-4B and Ministral-3-8B, and the T = 1.0 sensitivity subset. They are not proposal commitments; without Qwen3.5-4B, the RQ4 scale test is reported as not run.<br>**Documents:** report, slides and poster are generated from `results/summary.json` by scripts and rebuilt automatically by `scripts/autopilot.py` as runs finish. | vLLM 0.34 fails on the Kaggle T4 (engine-core error) and the transformers fallback runs at about 50 output tokens/s, so the pre-registered pool would cost about 20 GPU-hours per model. Every result shown comes from executed runs. Nothing unexecuted is presented as measured, and the deviations are stated in the report. |
| 2026-11-22 | **D3 scorer details**: splits 70/10/20 (train/validation/test, per region, seeded); the co-occurrence component uses shrinkage-smoothed PMI (marginals shrink to global prevalence, the joint to independence, κ = 10) instead of additive smoothing; the gradient-boosting comparator is scikit-learn's histogram GBDT (LightGBM is not installed); **the scorer's classifier is chosen among linear models** (linear SVM won, validation macro-F1 0.401) because its per-ingredient contributions drive the critique feedback and the app's suggestions. The GBDT (0.419) is reported as the comparator and is the independent evaluation classifier. **Component weights** are the best of {equal, logistic vs other cuisines, logistic vs all three negative types} by the mean *validation* AUC over the three negative types of §9.4. All classifiers are seeded | Additive smoothing gives unseen pairs a high PMI, the wrong way round. A first run chose weights against other cuisines only, which under-weighted co-occurrence: the composite AUC against copy-mutate recipes was 0.76, while co-occurrence alone reached 0.95. The criterion was corrected to the plan's three negative types, judged on validation data only. All choices are made before any LLM recipe is scored. |

## 14. Budget (as of 2026-11-22; estimates from `generation/openai_batch.py --dry-run`)
| Item | Cost | Resource |
|---|---|---|
| Llama-3.1-8B, Qwen3.5-9B, Qwen3.5-4B, Ministral-3-8B: 3,520 recipes each | $0 | Kaggle GPU (weekly quota about 30 h; 2 batch sessions at a time) |
| Gemini 3.5 Flash-Lite | $0 | free tier (Batch needs billing); multi-day because of 503 overload spells |
| OpenAI gpt-6-luna: pilot + full | **$0.20 actual** (ledger) | Batch API |
| OpenAI gpt-6.2-sol | not planned (optional, about $10) | |
| D3 steering | $0 (Kaggle) | best-of-N and critique-and-revise on the open models |
| **Total cash** | **$0.20 spent** | `BUDGET_USD = 5` cap enforced by the batch tool and ledger |

## Appendix A: prompt drafts (superseded)

**The frozen prompts are the files in `configs/prompts/v1/`** (prompt set v1: 704 prompts, sha256 `468c7d7f4c79…`, embedded in every Kaggle notebook and logged with every call). The drafts below are kept for the record; the final versions dropped the separate "name" field (see the decision log).

**System (both conditions):** "You are a helpful cooking assistant. Respond only with JSON that follows the given schema. Write in English."

**Plain:** "Give me 5 different traditional {adjective} recipes. For each recipe, provide a title, the full list of ingredients as they would appear in a recipe (with quantities), and step-by-step instructions."

**Persona:** "You are a home cook who grew up in {country} ({region_label}) and cooks the food of your family and region every day. Share 5 different traditional dishes you actually cook at home. For each, give a title, the full ingredient list as you would write it for a friend (with quantities), and step-by-step instructions."

**Schema:** `{"recipes": [{"title": str, "ingredients": [{"line": str, "name": str}], "instructions": [str]}]}`, with exactly 5 recipes.

## Appendix B: region labels
22 analysis regions (exact CulinaryDB labels): Africa, Australia & NZ, British Isles, Canada, Caribbean, China, DACH Countries, Eastern Europe, France, Greece, Indian Subcontinent, Italy, Japan, Korea, Mexico, Middle East, Scandinavia, South America, South East Asia, Spain, Thailand, USA. Four aggregate-only labels are excluded from region analyses but kept for WORLD: Misc.: Portugal, Misc.: Dutch, Misc.: Belgian, Misc.: Central America (207 recipes).

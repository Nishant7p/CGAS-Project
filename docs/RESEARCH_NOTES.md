# Research notes: verified facts, formulas, data quirks and sources

Started 2026-11-22. Every fact carries the date it was checked. Provider facts change fast, so re-verify anything older than about 2 weeks before acting on it.

## 1. Course and grading (sources: Week01 deck slides 33–37, Classroom comments, proposal)
- BIO544 Computational Gastronomy, Monsoon 2026, Prof. Ganesh Bagler (CoSyLab, IIIT-Delhi). Lectures Tue/Fri 11:00–12:30, A007 R&D Block.
- **Weights:** mid-sem 20%, end-sem 30%, assignments 30%, **mini-project 20%**.
- Mini-project FAQ (Week01 slide 36): "Teams of 1-2 students working on projects for at least 3 weeks …".
  - **Evaluation: based on project-specific deliverables.**
  - Opportunities: hackathon, project-and-poster presentations at the Symposium on Computational Gastronomy.
- Schedule (slide 37): Weeks 12–13 "Mini Project", **Week 14 "Mini Project Presentations"**. Weeks 10–11 cover cuisine classification, identity of a cuisine, and transforming a recipe from one cuisine to another. That supports D3. Course outcome CO3 is "design classification and prediction models from culinary data".
- Proposal call: 1 student ≥ 3 deliverables; 2 students ≥ 4; max 5; each "unique and non-trivial". TA (Madhvi Kumari Sinha) defined a deliverable as "a specific output/result … dataset, analysis, model, visualisation, web application …", each **distinct and non-trivial**.
- The proposal was submitted 2026-10-02 (just after the 1 Oct deadline), from nishant23355@iiitd.ac.in. The user reported on 2026-11-22 that there is no time pressure.
- Week02 deck, "Mini-Project Ideas" slide, lists "Create a python library for the analysis of cuisine, starting with cuisine data imported in a certain format":
  - recipe size and popularity;
  - category composition;
  - itemsets;
  - food pairing;
  - random cuisine generation.

  It also suggests Kaggle challenges (predict cuisine from recipes, and others). This is the course's own endorsement of D4.
- Assignments:
  - A1: Food.com scraping, NER, size and rank-frequency.
  - A2: frequent itemsets; size- and frequency-controlled random cuisines.
  - **A3 (due 24 Oct): copy-mutate models** (Q1 modification; Q2 add/delete; Q3 popularity fitness). Our A3 code is in `../../Assignment3/Group-44_Assignment3/`.
  - The rubric style rewards exact naming, labeled axes and ticks, saved PNGs, mean ± SD across instances and documentation.

## 2. Data sources (all downloaded 2026-11-22 into `data/raw/`)
### CulinaryDB (CoSyLab): https://cosylab.iiitd.edu.in/culinarydb/ (zip at `static/data/CulinaryDB.zip`)
- **License: CC BY-NC-SA 3.0 Unported.** Files dated 2018-03-16.
  - `01_Recipe_Details.csv`: Recipe ID, Title, Source, Cuisine. 45,772 recipes. Sources: AllRecipes 16,177, Food Network 15,917, Epicurious 11,069, TarlaDalal 2,609.
  - `02_Ingredients.csv`: Aliased Ingredient Name, Ingredient Synonyms (";"-separated), Entity ID, Category. 930 rows, IDs 0–1002, 21 categories.
  - `03_Compound_Ingredients.csv`: name, synonyms, entity_id (2000–2102), "Contituent Ingredients" (sic, comma list), Category. 103 rows.
  - `04_Recipe-Ingredients_Aliases.csv`: Recipe ID, **Original Ingredient Name (the raw recipe line, e.g. "12 ounces white onions, peeled and sliced paper thin")**, Aliased Ingredient Name, Entity ID. 456,279 rows and **161,208 unique raw phrases**. This is the gold standard for evaluating our aliaser.
- 26 cuisine labels: the 22 regions plus "Misc.: Portugal" (138), "Misc.: Dutch" (40), "Misc.: Belgian" (15) and "Misc.: Central America" (14). Those 4 total the paper's 207 aggregate-only recipes.
- Our counts match Table 1 within a few (e.g. ITA 7,502 recipes with ingredients vs 7,504; USA 595 unique ingredients vs 612). The mean recipe size (unique entities) is 9.02 with SD 4.01, min 1, max 38. The paper says "average of nine".
- Entity ID bands in recipe rows (row counts):

  | Band | Rows | What it is |
  |---|---|---|
  | ≤ 936 | 360,469 | FlavorDB IDs |
  | 937–1999 | 18,489 | CulinaryDB additions |
  | ≥ 2000 | 33,602 | compound ingredients |

- The top entities are Salt, Pepper, Garlic, Onion, **Olive (olive oil maps here)**, Sugar, Butter, Egg, Water, Flour, Tomato, Chicken, Cream, Capsicum, Cooking Oil, Parsley, Cayenne, Coriander, Milk, Vinegar.
- The site also has `basic_stats.html` and `food_pairing_analysis.html`, whose figures are saved in `data/raw/culinarydb_site/figures/`. The latter shows Z per null model (2FP-R0..R3) and the top-contributor charts. They are digitized into `data/reference/singh_bagler_2018_targets.json`.

### FlavorDB (CoSyLab): https://cosylab.iiitd.edu.in/flavordb/ (also /flavordb2/)
- **License: CC BY-NC-SA 3.0.** No robots.txt (404). The JSON endpoint is `/<db>/entities_json?id=<id>`.
  - Entity fields: entity_id, category, category_readable, entity_alias, entity_alias_readable, entity_alias_synonyms, natural_source_name, molecules[] and others.
  - Molecule fields: pubchem_id, common_name, **cas_id** ("@"-separated when there are several), fema_flavor_profile, odor, taste, flavor_profile, fooddb_flavor_profile, bitter, super_sweet and others.
- **Both `flavordb` and `flavordb2` endpoints return identical data**:

  | Item | Value |
  |---|---|
  | Entities | 936 (OK for IDs 0–978) |
  | Not found | 75 IDs (35 inside 0–936, plus gaps and everything ≥ 979) |
  | Entity–molecule links | 60,263 |
  | Unique PubChem IDs | 1,788 |
  | Entities with 0 molecules | none |

  The 2018 version is therefore not available online. The raw gz JSON is cached in `data/raw/flavordb_v1|v2/`.
- Small profiles: Salt 4, Sugar 5, Water 4, Flour 3, Cooking oil 4, Soybean Sauce 19. Pepper (black, id 339) has 223.
- **CulinaryDB IDs join to FlavorDB IDs directly** (Egg 0, Bread 2, Rye Bread 3, Soybean Sauce 291, Pepper 339, Capsicum 362).
- **24 CulinaryDB ingredients have no FlavorDB entry.** These are exactly the paper's 13 + 4 + 7, and they cover 17,910 of 456,279 rows (3.9%):

  | Group | Ingredients (CulinaryDB row counts) |
  |---|---|
  | Ahn | **Cayenne 5,834**, Yeast 936, Tequila 220, Sauerkraut 142 |
  | Coarse-grained | **Lemon Juice 4,047**, Coconut Milk 679, Coriander Seed 424, Apple Juice 207, Brown Rice 204, Lard 204, Coconut Oil 99; Anise Oil, Tomato Juice/Paste/Puree, Hops Beer and Cured Ham have 0 rows |
  | Additives | **Baking Powder 3,644**, Cooking Spray 560, Gelatin 283, Food Coloring 261, Liquid Smoke 124, MSG 36, Citric Acid 6 |

  Paper: the 13 coarse-grained profiles come from FlavorDB parents; Cayenne, Yeast, Tequila and Sauerkraut from Ahn et al.; the 7 additives were added manually, and Cooking Spray, Gelatin, Food Coloring and Liquid Smoke have no profile.

### Ahn et al. 2011 data: Zenodo DOI 10.5281/zenodo.11449657 (`flavor_network_data.zip`, **CC BY 4.0**)
- `ingr_comp/ingr_info.tsv`: 1,530 ingredients (id, name, category).
- `comp_info.tsv`: 1,107 compounds (id, name, **CAS**).
- `ingr_comp.tsv`: 36,781 ingredient–compound pairs.
- `scirep-cuisines-detail/`: allrecipes, epicurious and menupan recipes with cuisine labels. `flavor_network_backbone/` holds nodes and edges.
- It contains cayenne, yeast, tequila and sauerkraut. Map CAS to PubChem through FlavorDB's `cas_id`.

## 3. Methods: exact definitions (cite these in code docstrings)
- **Ns(R) = 2/(n(n−1)) Σ_{i≠j∈R} |C_i ∩ C_j|** (Ahn 2011; Jain, Rakhi & Bagler 2015 PLoS ONE eq. 2), where C_i is the flavor profile. The cuisine N̄s is the mean over its recipes, and ΔNs = N̄s_cuisine − N̄s_rand.
- **Z = √N_Rand · (N̄s_cuisine − N̄s_Rand) / σ_Rand**, where N_Rand is the number of recipes in the randomized cuisine and σ_Rand the SD of recipe Ns in it (Jain 2015 eq. 3; Singh & Bagler 2018 use N_rand = 100,000).
  - The class slide (Week02) writes Z = (N̄s_cuisine − N̄s_cuisine-Rand)/σ_Ran without the √N.
  - Implication: Z_SB = √N_rand × (standardized effect d), so the published Z is about 316·d. That explains values like ITA ≈ 263, SCND ≈ −148.
- **Null models** (Singh & Bagler §IV-B; Jain 2015 for details): each keeps the cuisine's exact ingredient set and recipe-size distribution.
  - Random: uniform selection.
  - Ingredient frequency: selection weighted by frequency.
  - Ingredient category: the recipe's category composition is preserved and ingredients are drawn uniformly within each category.
  - Frequency + category: both.
  - Jain 2015 used 10,000 recipes per null (8 sets for the category nulls). Singh & Bagler used **100,000**.
- **Fig. 4 reading:** the blue dots are the real cuisine's Z against the random null. Red, yellow and green are the real cuisine's Z against the frequency, category and frequency+category nulls.
  - Frequency-preserving nulls bring |Z| close to 0, so popularity explains the pattern.
  - The category null does not.
  - 16 uniform regions: ITA, AFR, CBN, GRC, ESP, USA, INSC, ME, MEX, ANZ, SAM, FRA, THA, CHN, SEA, CAN. 6 contrasting: EE, KOR, BRI, DACH, JPN, SCND.
- **Ingredient contribution:**
  - (a) Ahn/Jain eq. 4: χ_i = (1/N_R) Σ_{R∋i} [2/(n(n−1)) Σ_{j≠i∈R} |F_i∩F_j|] − (2f_i/(N_R⟨n⟩)) · Σ_{j∈c} f_j|F_i∩F_j| / Σ_{j∈c} f_j.
  - (b) Singh & Bagler §IV-C: "percentage change in food pairing score in response to removal of the ingredient from the cuisine". The published magnitudes are KOR Soybean Sauce 16.85%, Beef 7.7%, Vinegar 6.15%; CHN Ginger 5.29%; SEA Garlic 4.76%.
- **Category contribution and culinary fingerprint (Jain 2015):** randomize the ingredients within one category while preserving category and frequency for all others, giving ΔNs_cat per category. "The pattern of category contributions presents itself as a 'culinary fingerprint' of the cuisine." In Indian cuisine, spice and dairy drive the negative pairing, and vegetable and fruit the positive.
- **Authenticity (Jain & Bagler 2018):** prevalence P_ic = n_ic/N_c. The relative prevalence p_ic = P_ic − the average P_i over the other cuisines.
- **Ingredient Overrepresentation (Tuwani et al. 2019):** P in cuisine ς minus the pooled proportion across all cuisines. Tuwani compares models by the **MAE between rank-frequency distributions** and uses 158,544 recipes from 25 regions (a RecipeDB precursor, not CulinaryDB).
- **Copy-mutate models:**
  - Kinouchi 2008. Jain & Bagler 2018: seed pool 20 recipes, S = 7, initial pool 10 ingredients, pool/recipe ratio r kept empirical, M mutations, 24 instances. Variants: CM-fitness random, CM-fitness ranked (frequency fitness, which fits better) and CMAD (add/delete).
  - Tuwani 2019: CM-R, CM-C (category-constrained), CM-M (mixture) and a null model, with 100 sets.
  - Our A3 code implements modification, add/delete and popularity fitness.
- **Aliasing protocol (Singh & Bagler §IV-A; Week05 slide):**
  1. Lower-case; remove punctuation and special characters.
  2. Singularize each word (NLTK + inflect).
  3. Remove stopwords and culinary stopwords.
  4. Match database ingredients in order of length (longest first).
  5. Label partial or unrecognized matches for manual curation, using n-grams up to 6.
  6. Add synonyms (bun → bread, lager → beer, curd → yogurt) and spelling variants (whiskey/whisky, asafoetida/hing, chili/chile).

  Slide example: "2 small Jalapeno Peppers, roasted and slit perpendicularly" → "jalapeno".
- **Shortcomings the authors list** (Week05 slide 22): recipes might not be "traditional", the ingredient database is not exhaustive, unmatched ingredients need manual curation, and duplicates appear across sources.
- **Open questions in Singh & Bagler's conclusion, which our project addresses:**
  - robustness to flavor-profile changes;
  - generating novel palatable recipes;
  - using culinary fingerprints to design food (that is D3).

## 4. LLM access landscape (checked 2026-11-22 on official docs via fetch; re-verify)
- **OpenAI** (developers.openai.com):

  | Model | Tier | Price per 1M in / out | Batch price per 1M in / out | Reasoning effort |
  |---|---|---|---|---|
  | `gpt-6.2-astra` | flagship | $8 / $40 | half | low…max |
  | `gpt-6.2-sol` | mid | $2 / $10 | $1 / $5 | low…max; rejects "none" |
  | `gpt-6.1-terra` | | $1 / $5 | half | |
  | `gpt-6-luna` | mini-class | $0.10 / $0.50 | $0.05 / $0.25 | none…max; default medium |

  - Older models are still listed (gpt-5-mini $0.25 / $2, gpt-5-nano $0.05 / $0.40).
  - Batch: 50% off, 24 h window, ≤ 50k requests and ≤ 200 MB per batch. `structured_outputs` is listed for these models.
  - **Measured: the GPT-6-Luna full run** (704 calls, Batch, `reasoning_effort: none`) finished in under 3 h. It used 102,068 input and 727,327 output tokens with 0 reasoning tokens, at **$0.19 actual** (estimate $0.52). 703 of 704 calls had exactly 5 recipes.
  - Support for temperature, top_p and seed on the reasoning models is NOT stated; verify in the API reference before relying on it.
- **Google Gemini** (redirects to aistudio.google.com/docs):
  - Free tier for `gemini-4-hydrogen`, `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` and `gemini-2.5-flash-lite`. Free-tier data is used to improve products.
  - **Free-tier RPM/RPD are not published** (third-party reports: Flash about 5 RPM / 20 RPD, Flash-Lite about 15 RPM / 500 RPD). Check https://aistudio.google.com/rate-limit.
  - Paid prices per 1M: 3.8 Flash $0.75 / $3.75 (batch $0.375 / $1.875) until 2026-12-31, then double. 3.5 Flash-Lite $0.30 / $2.50.
  - Batch is paid-only, except that 3.5 Flash-Lite batch shows free. **Verified 2026-11-22: on our free-tier key, `batches.create` fails with `400 FAILED_PRECONDITION`, so Batch needs billing.** This also confirms that the key is unbilled.
  - Deprecations: gemini-2.5-flash routes to 3.5-flash-lite and 2.5-pro routes to 3.8-flash on 2026-12-01.
  - **Measured on `gemini-3.5-flash-lite` (2026-11-22):**
    - `models.get` reports thinking = True, an output limit of 65,536, and default T 1.0 / top_p 0.95.
    - `thinking_level` `minimal` and `low` are both accepted. The full run uses `minimal` (0 reasoning tokens).
    - The free tier returns long spells of `503 UNAVAILABLE` ("high demand"): 11 of the first 21 full-run attempts, and one call failed for over 20 minutes. The runner now retries for up to 8 h per call.
    - **JSON mode does not guarantee one valid document.** In about 8% of replies, Gemini closed the document after recipe 1 and listed the rest as bare objects; there are also bracket slips at recipe boundaries. Both are repaired structurally in `generation/common.py`.
    - About 7% of calls return fewer than the 5 recipes asked for. This is genuine under-production and is kept as data.
    - **The free-tier daily cap is 500 requests per project and model** (429 `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue 500, hit at 17:04 UTC on 2026-11-22). **Failed 503 attempts count toward it**: 300 successful calls plus 168 failed attempts plus a few probes reached 500. On an overloaded day this leaves about 250–450 useful calls. The quota resets at midnight Pacific, and the runner sleeps until then (`--wait-daily`).
- **Groq:**
  - **Free plan per model: gpt-oss-120b, gpt-oss-20b and qwen3.8-27b at 30 RPM, 1K RPD, 8K TPM, 200K TPD** (the page's tab labelling was ambiguous; verify in the console).
  - Paid: gpt-oss-120b $0.15 / $0.60, gpt-oss-20b $0.075 / $0.30, qwen3.8-27b $0.80 / $4.00.
  - llama-3.1-8b-instant and llama-3.3-70b-versatile are now "ContactSales" (enterprise).
  - Batch is 50% off for gpt-oss and llama models; qwen is not batch-supported.
- **Others:**
  - OpenRouter ":free" models: 50 req/day, or 1,000/day after buying ≥ 10 credits.
  - Cerebras: only a $5, 30-day trial that needs a card.
  - **GitHub Models was retired on 2026-07-30.**
  - NVIDIA build.nvidia.com: "free inference" for a few Nemotron/Kimi models, with no stated limits.
  - The Mistral tier page returned 404.
- **Kaggle:** the CLI docs ("Accelerators available as of Sep 2026") list:
  - `NvidiaTeslaT4` (GPU T4 ×2, the default), `NvidiaL4`, `NvidiaL4X1`, `NvidiaTeslaA100`, `NvidiaH100`, `NvidiaRtxPro6000`, `TpuV5E8` and `TpuV6E8`;
  - "some only for specific competitions or admins";
  - P100 is not listed.
  - Weekly quota (third-party reports: about 30 h) and session length (about 12 h) were not readable from the JS-only docs; check your account.
  - Kernel metadata fields: `enable_gpu`, `enable_internet`, `machine_shape`. CLI: `kaggle kernels push -p DIR [--accelerator ID]`, `kaggle kernels status`, `kaggle kernels output`.
  - **Auth:** generate a token at https://www.kaggle.com/settings/api and store it in `~/.kaggle/access_token` or `KAGGLE_API_TOKEN`; the legacy `kaggle.json` still works.
  - Secrets: in the editor, **Add-ons → Secrets**, then `kaggle_secrets.UserSecretsClient().get_secret()`. Whether they persist across CLI pushes is not documented.
- **vLLM (corrected 2026-11-22):** the latest is **0.34.0** (torch 2.14, Python ≥ 3.11). Its docs say "GPU: compute capability 7.5 or higher (e.g., T4, RTX20xx, A100, L4, H100, B200)", **so the T4 is supported**. The earlier reports that V1 needs CC ≥ 8.0 are outdated (2025). T4 has no bf16, so use `dtype=float16`. Gemma 3/4 are reported unstable on T4.
- **Seen on Kaggle T4 ×2 (Qwen3.5-9B pilot log, 2026-11-22):**
  - **Backends:** vLLM picks the `TRITON_ATTN` attention backend (FA2 needs CC ≥ 8.0), and the FlashInfer top-p/top-k sampler is unsupported on CC 7.5, so the PyTorch sampler is used.
  - **The failure:** for Qwen3.5, vLLM calls FlashInfer's `gemma_rmsnorm`, which is JIT-compiled (FlashInfer 0.7.3). The link step `c++ … -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda` fails with `cannot find -lcuda`, because the image has only the driver's versioned `libcuda.so.1`. Llama uses vLLM's own RMSNorm (`vllm_c`), so it should not hit this.
  - **The fix (notebook nb-v2):** symlink `libcuda.so` to the driver library and prepend it to `LIBRARY_PATH` before `LLM()`.
  - **Fallback speed:** transformers 5.20.2 ran Qwen3.5-9B at 49 output tok/s with batch 8 (44 calls in 20.5 min).
- **Kaggle limits seen:**
  - "Maximum batch GPU session count of 2". A QUEUED kernel counts toward it, and it can stay queued for hours while a later push starts at once.
  - `kaggle kernels logs` is empty until the run ends.
  - There is no CLI cancel, so a stuck queued kernel can only be deleted. That needs the user's OK.
- **Open-weight models on HF (checked 2026-11-22 via the HF API):**
  - Qwen3.5 (0.8B/2B/4B/9B, Feb 2026), Qwen3.6-27B/35B-A3B and Qwen3.8-27B (Aug 2026) are all ungated Apache-2.0 and multimodal (`image-text-to-text`; use text-only mode).
  - Qwen3.5 thinks by default, so disable it with `chat_template_kwargs={"enable_thinking": False}`. vLLM `--language-model-only` skips the vision encoder. Recommended non-thinking sampling: T 0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
  - Meta's newest small model is still **Llama-3.1-8B-Instruct** (gated). Llama-4 Scout/Maverick (17B×16E/128E MoE), Muse-Spark-Open-816B and Muse-Glimmer-30B are too big for free GPUs.
  - Gemma 4 (26B-A4B, 31B; Mar 2026) is too big at fp16.
  - Olmo-3-7B-Instruct (Ai2, Nov 2025, Apache-2.0, recommended T 0.6, top_p 0.95).
  - Mistral's recent releases are huge; Ministral-3-3B is small.
  - Phi-4-mini-instruct (3.8B, MIT).
- **Hugging Face Spaces:** the Streamlit SDK was deprecated on 2025-04-30. Use the **Docker SDK + Streamlit template** (`sdk: docker`, `app_port: 8501`, `src/streamlit_app.py`) and pin the Python and dependency versions.
- PyPI names `flavoraudit`, `flavor-audit`, `flavorprint` and `llmcuisine` were all free (HTTP 404) on 2026-11-22.

## 5. Related work (PDFs in `../references/`)

**Food pairing and cuisine methods (CoSyLab and others)**
- Ahn et al. 2011, Sci Rep, arXiv 1111.6074: the flavor network; food pairing positive in North American and Western European cuisines, negative in East Asian and Southern European; χ_i.
- Jain, Rakhi & Bagler 2015, PLoS ONE, arXiv 1505.00890 (and 1502.03815, "Spices form the basis…"): negative pairing in Indian regional cuisines; exact Z and χ formulas; ΔNs_cat as the culinary fingerprint.
- Singh & Bagler 2018, ICDEW, arXiv 1803.04343: CulinaryDB with 22 regions, 16+/6−, 4 nulls, contributions. The core paper we replicate.
- Jain & Bagler 2018, Physica A, arXiv 1505.00155: copy-mutate for Indian cuisines; ranked fitness reproduces Ns better; authenticity via relative prevalence.
  - Their model: a seed pool R0 of 20 recipes, each of S = 7 ingredients from a random pool I0 of 10. Copies are mutated M times (replace i by a random pool ingredient j if f_j > f_i).
  - The pool's ingredient/recipe ratio is kept at the empirical value, and they run 24 instances.
  - "Fitness ranked" means "frequency scaled fitness values": fitness is the ingredient's empirical frequency.
- Kinouchi et al. 2008, NJP, arXiv 0802.4393: the original copy-mutate model.
- Tuwani, Sahoo, Singh & Bagler 2019, ICDEW, arXiv 1904.10138: copy-mutate for world cuisines (25 regions, 158k recipes); overrepresentation metric; rank-frequency MAE.
  - **Algorithm 1 (read from the PDF image; the text layer drops the math):** fitness ~ U(0,1); I0 = m ingredients; R0 = n recipes of size s̄ (the mean recipe size). Each step: if |I0|/|R| ≥ ∅ (the cuisine's ingredients/recipes ratio), copy a random recipe and mutate it M times; otherwise add a random ingredient to I0.
  - **Parameters: m = 20, n = I0/∂, M = 4 (CM-R) and 6 (CM-C and CM-M).** CM-C replaces within i's category; CM-M does so half the time.
  - Their rank-frequency MAEs (Fig. 4) are about 0.01–0.13.
  - Our `flavoraudit.evolution` implements this. On real CulinaryDB regions it gives MAEs of 0.013–0.05, and ranked fitness is closer to real Ns than random fitness in 3 of 4 regions checked (Indian Subcontinent: 50.5 vs 38.2, real 56.9).
- Varshney et al. 2013, arXiv 1307.7982: food pairing with dirty data; robustness ideas.
- Zhu et al. 2013, PLoS ONE, arXiv 1307.3185: geography and similarity of Chinese regional cuisines.
- Grover et al. 2022, arXiv 2205.05451: FlavorDB2.
- Bagler & Goel 2024, npj Syst Biol Appl 10:72: the perspective with the "untested hitherto" quote, Ratatouille and the Turing Test for Chef (24 chefs, F1 69.88%).
- Bagler 2022, arXiv 2211.09059: a generative grammar of cooking.
- Sharma, Upadhyay & Bagler 2020, arXiv 2004.14165: cuisine classification (D3 classifier baseline).

**LLMs, recipes and culture**
- Carichon, Rampa & Farnadi 2026, arXiv 2602.10964 (LLMFusion; the closest work): textual JSD novelty; 8 open models via vLLM at T = 0 and repetition penalty 1.4; LLMs over-produce divergence and are not grounded in culturally salient ingredients. **They study no flavor chemistry**, so our gap is intact. Code: github.com/fcarichon/LLMCokingNovelty.
- Cao et al. 2024, TACL, arXiv 2310.17353: cultural adaptation of recipes (Chinese↔English).
- Lee et al. 2024, arXiv 2411.01996, "Culinary Class Wars": the ASH (authenticity, sensitivity, harmony) LLM-judge for cuisine transfer.
- Hu et al. 2025, arXiv 2507.21934, CARRIAGE: RAG for diversity in cross-cultural recipe adaptation; Recipe Cultural Appropriateness Score.
- Epicure 2026, arXiv 2604.22776: ingredient embeddings capture chemical affinity and co-occurrence; chemistry explains only part of pairing.
- RecipeMT 2025 (ScienceDirect S2772566925001338): authentic recipe ideation with a flavor-sharing score and transformers. The closest steering-like prior; cite in D3.

**Methodology for our own pipeline**
- Gebru et al. 2021, arXiv 1803.09010: the datasheets template for D1.
- Madaan et al. 2023, arXiv 2303.17651: Self-Refine, the critique-and-revise basis for D3.

**Novelty check (2026-11-22):** a web search found **no study that audits LLM-generated recipes against flavor-compound sharing (Ns or Z) or culinary fingerprints.** Group 23 (the same class) is the only overlapping effort known.

## 6. Things to verify or decide later
- The exact χ% removal procedure: drop the ingredient from its recipes, or drop the recipes containing it. Calibrate against the KOR/CHN/SEA targets.
- How Singh & Bagler handled ingredients with tiny or empty profiles and duplicate (recipe, entity) rows.
- Whether a 2018–2019 FlavorDB scrape exists (Wayback, GitHub, Kaggle) for a closer replication.
- OpenAI reasoning models: support for temperature, top_p and seed, and the structured-output parameter name in the Responses API.
- Gemini live free-tier limits (AI Studio rate-limit page) and Groq free vs developer tab values (console).
- ~~vLLM version that runs on Kaggle T4~~: 0.34.0 installs and starts. The Qwen3.5 JIT link failure has a fix in nb-v2 (§4); whether the fix works is checked from the Qwen full-run log.
- The Week03 lecture PDF on disk is corrupted (6.1M U+FFFD replacement characters); re-download it from Classroom.

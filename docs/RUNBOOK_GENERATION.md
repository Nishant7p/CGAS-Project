# Runbook: generating the LLM recipes (D1)

Last updated: 2026-11-22 (lean roster). The roster is in `configs/models.yaml`; the rationale is in the plan's §5 Phase 4.

## Why we need LLMs at all
The project asks: *when an AI writes a "traditional" recipe, does it follow the real flavor logic of that cuisine?*
- The **real recipes** (CulinaryDB, 45,772) are the ground truth. We already have them.
- The **AI recipes** don't exist yet. The LLMs write them (that's D1), and D2 compares them with the real ones.
- D3 (steering) also calls LLMs. The analysis code, the library (D4) and the app (D5) need no LLM.

## How many models? (decided 2026-11-22)
The instructor didn't ask for a number. Our proposal promises **≥ 3 LLMs: 1 proprietary and 2 open-weight families (e.g. Llama, Qwen)**. We plan **5**:

| Tier | Model | Runs on | Your effort | Cost |
|---|---|---|---|---|
| core | Llama-3.1-8B-Instruct (`llama-3.1-8b`) | Kaggle GPU | none (Claude pushes it via the Kaggle API) | free |
| core | Qwen3.5-9B (`qwen3.5-9b`) | Kaggle GPU | none (Claude pushes it via the Kaggle API) | free |
| core | GPT-6.2 Sol (`openai-gpt-6.2-sol`) | OpenAI Batch API | none (Claude runs it) | about $10.4 |
| extra | Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | Gemini free API | none | free |
| extra | GPT-6 Luna (`openai-gpt-6-luna`) | OpenAI Batch API | none | about $0.5 |

- Each model gets the same 704 prompts: 22 regions × 2 prompt styles × 16 repeats, 5 recipes each, so 3,520 recipes per model and **17,600 in total**.
- The 3 core models alone (10,560 recipes) already meet the proposal.
- Nine more models (Mistral, Olmo, Phi, Gemma, smaller Qwen and Llama, EXAONE, Groq) are configured as **optional**. They're used only if time remains.

## One-time setup (about 5 minutes: three keys, pasted to Claude in chat)
**No Hugging Face token is needed.**
- Qwen is ungated.
- Llama loads from `RedHatAI/Llama-3.1-8B-Instruct`, a copy verified byte-identical to Meta's official repo (10/10 files, see `data/reference/Llama-3.1-8B-Instruct_mirror_check.json`).

| Key | Exact page | What to click | Looks like |
|---|---|---|---|
| Kaggle API token | https://www.kaggle.com/settings/api | **Generate New Token** (under "API"), then copy it. If only **Create Legacy API Key** is offered, paste the downloaded `kaggle.json` contents. | long string |
| OpenAI API key | https://platform.openai.com/api-keys | **+ Create new secret key**, name it `llmcuisine`, click **Create secret key**, then copy it | `sk-…` |
| Gemini API key | https://aistudio.google.com/apikey | **Create API key** (choose "in new project" if asked), then copy it | `AIza…` |

Also check:
- **OpenAI credit:** https://platform.openai.com/settings/organization/billing/overview needs about $12 of prepaid balance.
- **OpenAI budget cap (optional):** https://platform.openai.com/settings/organization/limits

Claude stores the keys only in `.env` and `~/.kaggle/access_token`, which are excluded from git and never printed. `scripts/check_access.py` verifies them with free calls. **After the project, revoke all three keys on the same pages.**

Note (2026-11-22): the old `~/.kaggle/kaggle.json` on this Mac is rejected by Kaggle's servers ("Authentication required"), so a new token is required.

## What happens next (Claude does it)
1. **Pilots** (44 calls per model): the Qwen and Llama notebooks on Kaggle, the GPT-6.2 Sol batch (about $0.65) and Gemini. Claude checks the parse rate, recipe count, ingredient matching and duplicates.
2. **Freeze the prompts** (git tag `prompts-v1`), then the **full runs**: Kaggle about 1–3 h per model, OpenAI batches within 24 h, Gemini over a few days of free quota.
3. Parse everything: `.venv/bin/python generation/parse_outputs.py`.

## Manual Kaggle route (if you skip the token in step 2)
1. Kaggle → **Create → New Notebook → File → Import Notebook**, then upload `generation/kaggle/llmcuisine22_generate.ipynb`.
2. Right-hand panel: **Accelerator = GPU T4 x2** (or GPU L4 if offered), **Internet = On**. For Llama, also attach the `HF_TOKEN` secret.
3. Settings cell: `MODEL_KEY = "qwen3.5-9b"` (or `"llama-3.1-8b"`) and `MODE = "pilot"`, then **Run All** and send Claude the last cell's printout.
4. After approval: `MODE = "full"`, then **Save Version → Save & Run All (Commit)**. Download the `.jsonl` from the version's **Output** tab into `data/generated/raw/<model>/`.

## Commands Claude runs (for reference)
```bash
.venv/bin/python generation/openai_batch.py submit --model openai-gpt-6.2-sol --mode pilot   # about $0.65
.venv/bin/python generation/openai_batch.py status --model openai-gpt-6.2-sol --mode pilot
.venv/bin/python generation/openai_batch.py fetch  --model openai-gpt-6.2-sol --mode pilot
.venv/bin/python generation/run_api.py --model gemini-3.5-flash-lite --mode pilot             # free
.venv/bin/python generation/parse_outputs.py
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Cannot access meta-llama/...` | Licence not approved yet, or the `HF_TOKEN` secret isn't attached to the notebook. |
| `vLLM could not start` | It falls back to transformers automatically. If that runs out of memory: **Run → Restart & clear outputs**, set `FORCE_ENGINE = "transformers"`. |
| Gibberish or empty outputs | Tell Claude. Likely an fp16 overflow: use an L4 if offered, or Claude switches the model to float32. |
| Session died | Run again. It resumes from the saved file and never repeats finished calls. |
| `Daily quota reached` (Gemini) | Normal on the free tier. Claude re-runs it the next day. |
| `REFUSED: this would exceed BUDGET_USD` | The spending guard worked as intended. |

"""Build the self-contained Kaggle notebook(s) that generate LLMCuisine-22 recipes with open-weight models.

The notebook embeds (1) the exact prompt set (gzip + base64, sha256-checked), (2) the Kaggle model registry
and (3) the source of generation/common.py, so it needs no uploaded files and builds the same call ids and
seeds as the rest of the pipeline.

Usage (from the repo root):
    .venv/bin/python generation/kaggle/build_notebook.py                         # generic notebook
    .venv/bin/python generation/kaggle/build_notebook.py --kaggle-user NAME      # + per-model folders for `kaggle kernels push`
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from generation.common import PROMPT_SET_VERSION, build_prompt_set  # noqa: E402
from generation.config import load_models, load_regions, load_templates  # noqa: E402

NOTEBOOK_VERSION = "nb-v2"  # v2: libcuda link shim for FlashInfer JIT, eager retry, OOM-adaptive transformers batches
VLLM_VERSION = "0.34.0"
KAGGLE_DIR = REPO / "generation" / "kaggle"
PROMPT_DIR = REPO / "generation" / "prompt_sets"

HEADER = """# LLMCuisine-22: recipe generation on Kaggle GPUs
Group 44 · CGAS BIO544 mini-project · Deliverable D1. This notebook asks **one open-weight LLM** (Hugging Face weights)
for *traditional* recipes from the **22 CulinaryDB world regions** (5 recipes per call, plain and home-cook-persona prompts)
and logs every call (prompt, model revision, sampling settings, seed, raw output) for the dataset.

**Before you run (right-hand panel → Session options):**
1. **Accelerator → GPU T4 ×2** (or a GPU L4 option if your account shows one).
2. **Internet → On** (needs a phone-verified account).
3. **No token is needed for the planned models.** Qwen is ungated, and Llama loads from `RedHatAI/Llama-3.1-8B-Instruct`, a byte-identical copy of Meta's repo. Only the optional gated models (Gemma, Llama-3.2) need **Add-ons → Secrets → `HF_TOKEN`**.

**Run:** set `MODEL_KEY` and `MODE` in the next cell.
- **`MODE = "pilot"`**: 44 calls, about 15–30 min. Use **Run All** and watch the log.
- **`MODE = "full"`**: 704 calls, about 1–3 h. Use **Save Version → Save & Run All (Commit)** so it keeps running in the background.

**Output:** `/kaggle/working/llmcuisine22_<model>_<mode>.jsonl` (plus a `.gz` copy). Download it from the notebook's **Output** tab.
Results are saved after every batch. If a session dies, run the notebook again and it resumes where it stopped.
"""

SETTINGS = f"""# ==== SETTINGS: change these ====
MODEL_KEY = "{{model_key}}"   # core: "qwen3.5-9b", "llama-3.1-8b" | extra: "qwen3.5-4b", "ministral-3-8b" | optional: "olmo-3-7b", "phi-4-mini", "gemma-3-4b", "llama-3.2-3b", "exaone-3.5-7.8b"
MODE = "{{mode}}"            # "pilot" = 44 calls (220 recipes) | "full" = 704 calls (3,520 recipes) | "sens" = 44 calls at T 1.0, top_p 1.0
MAX_CALLS = None             # e.g. 4 for a quick smoke test; None = every call in the mode
BATCH_SIZE = 64              # prompts per checkpoint (results are written after each batch)
FORCE_ENGINE = {{force_engine}}          # None = vLLM with automatic fallback; or "transformers"
VLLM_VERSION = "{VLLM_VERSION}"     # pinned for reproducibility
NOTEBOOK_VERSION = "{NOTEBOOK_VERSION}"
"""

INSTALL = """# Install vLLM BEFORE anything imports torch (2-5 min the first time). Falls back to transformers if it fails.
import importlib.util, subprocess, sys

def _pip(*args):
    print("pip install", *args, flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    print((r.stdout or "")[-1500:], (r.stderr or "")[-3000:])
    return r.returncode == 0

VLLM_OK = False
if FORCE_ENGINE != "transformers":
    VLLM_OK = importlib.util.find_spec("vllm") is not None or _pip(f"vllm=={VLLM_VERSION}")
print("vLLM available:", VLLM_OK)
"""

ENV = """import os, json, time, gzip, base64, hashlib, platform, datetime, subprocess

# Read GPU facts from nvidia-smi so the notebook process never starts CUDA itself (keeps vLLM's multi-GPU start clean).
def _gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except Exception as e:
        raise RuntimeError("No GPU visible. Settings -> Accelerator -> GPU T4 x2, then restart the session.") from e
    rows = [[x.strip() for x in line.split(",")] for line in out if line.strip()]
    return [r[0] for r in rows], [round(float(r[1]) / 1024, 1) for r in rows], tuple(int(x) for x in rows[0][2].split("."))

M = MODELS[MODEL_KEY]
GPU_NAMES, GPU_MEM_GB, CC = _gpus()
N_GPU = len(GPU_NAMES)
if CC >= (8, 0):
    DTYPE = "bfloat16"                                      # L4 / A100 / H100
elif M.get("fp16_unsafe"):
    DTYPE = "float32"                                       # e.g. Gemma overflows in fp16; T4 has no bf16
else:
    DTYPE = "float16"                                       # T4 (sm75)
BYTES = 4 if DTYPE == "float32" else 2
NEED_GB = 1.1 * BYTES * M["params_b"] + 4                   # weights + KV-cache headroom
if NEED_GB > sum(GPU_MEM_GB):
    raise RuntimeError(f"{MODEL_KEY} needs about {NEED_GB:.0f} GB in {DTYPE}, but the GPUs have {sum(GPU_MEM_GB):.0f} GB. Pick a larger accelerator.")
TP = 1 if GPU_MEM_GB[0] >= NEED_GB else N_GPU
print(f"GPUs: {GPU_NAMES} {GPU_MEM_GB} GB | compute capability {CC} | dtype {DTYPE} | tensor_parallel {TP}")
print(f"Model: {MODEL_KEY} -> {M['hf_id']} | mode {MODE} | python {platform.python_version()}")

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    print("No HF_TOKEN secret attached (fine for Qwen/OLMo):", type(e).__name__)

from huggingface_hub import model_info
try:
    REVISION = model_info(M["hf_id"], token=os.environ.get("HF_TOKEN")).sha
except Exception as e:
    raise RuntimeError(
        f"Cannot access {M['hf_id']}: {e}. For gated models (Llama) accept the licence on its Hugging Face page "
        "and attach your HF_TOKEN secret."
    )
print("Pinned model revision:", REVISION)
"""

ENGINE = """from transformers import AutoTokenizer
import transformers

TRUST = bool(M.get("engine_args", {}).get("trust_remote_code"))
tok = AutoTokenizer.from_pretrained(M["hf_id"], revision=REVISION, trust_remote_code=TRUST)
llm, hf_model, ENGINE, ENGINE_KW, ENGINE_NOTES = None, None, None, {}, []


def expose_libcuda_for_jit(shim="/tmp/llmc_libcuda"):
    \"\"\"Make `-lcuda` linkable for FlashInfer's JIT kernels (vLLM builds FlashInfer's gemma_rmsnorm for Qwen3.5).

    Kaggle images ship only the driver's versioned libcuda.so.1 (no libcuda.so, no stub), so nb-v1 failed with
    "/usr/bin/ld: cannot find -lcuda" and fell back to the slow transformers engine. The c++ linker driver reads
    LIBRARY_PATH, which the vLLM worker processes (and their ninja builds) inherit.
    \"\"\"
    import glob
    found = []
    for pat in ("/usr/local/cuda*/lib64/stubs/libcuda.so", "/usr/local/cuda*/targets/*/lib/stubs/libcuda.so",
                "/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/local/nvidia/lib64/libcuda.so.1", "/usr/lib64/libcuda.so.1"):
        found += sorted(glob.glob(pat))
    if not found:
        try:
            out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
            found = [ln.split("=>")[-1].strip() for ln in out.splitlines() if "libcuda.so" in ln and "=>" in ln]
        except Exception:
            pass
    if not found:
        return "libcuda not found; JIT kernels may fail to link"
    target = os.path.realpath(found[0])
    os.makedirs(shim, exist_ok=True)
    link = os.path.join(shim, "libcuda.so")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(target, link)
    old = os.environ.get("LIBRARY_PATH")
    os.environ["LIBRARY_PATH"] = shim + (":" + old if old else "")
    return f"LIBRARY_PATH shim: libcuda.so -> {target}"


if VLLM_OK and FORCE_ENGINE != "transformers":
    ENGINE_NOTES.append(expose_libcuda_for_jit())
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # no FlashInfer sampler on sm75; PyTorch sampler is used
    print(ENGINE_NOTES[-1])
    try:
        import vllm
        from vllm import LLM, SamplingParams
        base = dict(model=M["hf_id"], revision=REVISION, dtype=DTYPE, tensor_parallel_size=TP,
                    max_model_len=8192, gpu_memory_utilization=0.90, seed=0)
        variants = [dict(base, **M.get("engine_args", {}))]
        if M.get("engine_args", {}).get("language_model_only"):
            variants.append(dict(base, limit_mm_per_prompt={"image": 0, "video": 0}))
        variants.append(base)
        last_err, tried_eager, i = None, False, 0
        while llm is None and i < len(variants):
            kw = variants[i]
            try:
                llm = LLM(**kw)
                ENGINE, ENGINE_KW = f"vllm-{vllm.__version__}", {k: v for k, v in kw.items() if k != "model"}
            except (TypeError, ValueError) as e:  # unsupported engine argument: try the next variant
                last_err, i = e, i + 1
                print("vLLM init variant failed:", repr(e)[:300])
            except Exception as e:  # e.g. CUDA-graph capture on T4: retry once without CUDA graphs
                if tried_eager:
                    raise
                last_err, tried_eager = e, True
                print("vLLM init failed, retrying once with enforce_eager=True:", repr(e)[:600])
                variants[i] = dict(kw, enforce_eager=True)
                ENGINE_NOTES.append("enforce_eager retry after: " + repr(e)[:200])
        if llm is None:
            raise last_err
    except Exception as e:
        print("vLLM could not start, falling back to transformers:", repr(e)[:800])
        print("If the fallback runs out of GPU memory: Run > Restart & clear outputs, set FORCE_ENGINE = 'transformers', run again.")
        llm = None

if llm is None:
    import torch
    from transformers import AutoModelForCausalLM
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(M["hf_id"], revision=REVISION, trust_remote_code=TRUST,
                                                        torch_dtype=getattr(torch, DTYPE), device_map="auto")
    except Exception as e:
        print("AutoModelForCausalLM failed, trying AutoModelForImageTextToText:", repr(e)[:300])
        from transformers import AutoModelForImageTextToText
        hf_model = AutoModelForImageTextToText.from_pretrained(M["hf_id"], revision=REVISION, trust_remote_code=TRUST,
                                                               torch_dtype=getattr(torch, DTYPE), device_map="auto")
    hf_model.eval()
    ENGINE = f"transformers-{transformers.__version__}"
print("Engine:", ENGINE)
"""

GENERATE = """SAMPLING = sampling_for(M["sampling"], MODE)   # MODE "sens": the common T 1.0 / top_p 1.0 sensitivity setting
print("Sampling:", SAMPLING)

def render(p):
    msgs = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **M.get("chat_template_kwargs", {}))

def sha16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def run_vllm(batch):
    prompts = [render(p) for p in batch]
    sps = [SamplingParams(max_tokens=M["max_tokens"], seed=seed_for(MODEL_KEY, p["key"]), **SAMPLING) for p in batch]
    outs = llm.generate(prompts, sps, use_tqdm=True)
    return [dict(text=o.outputs[0].text, finish_reason=o.outputs[0].finish_reason,
                 n_prompt_tokens=len(o.prompt_token_ids or []), n_output_tokens=len(o.outputs[0].token_ids),
                 rendered_sha16=sha16(pr)) for pr, o in zip(prompts, outs)]

HF_SUB = 16   # transformers fallback: prompts per generate() call (nb-v1 used 8: 49 tok/s); halves on CUDA OOM

def run_hf(batch):
    global HF_SUB
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    s, res, i = SAMPLING, [], 0
    while i < len(batch):
        chunk = batch[i:i + HF_SUB]
        prompts = [render(p) for p in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(hf_model.device)
        torch.manual_seed(seed_for(MODEL_KEY, chunk[0]["key"]))
        try:
            with torch.no_grad():
                gen = hf_model.generate(**enc, do_sample=True, temperature=s.get("temperature", 1.0),
                                        top_p=s.get("top_p", 1.0), top_k=s.get("top_k", 0),
                                        max_new_tokens=M["max_tokens"], pad_token_id=tok.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise
            HF_SUB = len(chunk) // 2
            print("CUDA out of memory: retrying this chunk with sub-batch", HF_SUB, flush=True)
            continue
        for j in range(len(chunk)):
            out_ids = gen[j, enc["input_ids"].shape[1]:]
            n_out = int((out_ids != tok.pad_token_id).sum())
            res.append(dict(text=tok.decode(out_ids, skip_special_tokens=True),
                            finish_reason="length" if n_out >= M["max_tokens"] else "stop",
                            n_prompt_tokens=int(enc["attention_mask"][j].sum()), n_output_tokens=n_out,
                            rendered_sha16=sha16(prompts[j]), engine_batch=len(chunk)))
        i += len(chunk)
    return res

utcnow = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
OUT = f"/kaggle/working/llmcuisine22_{MODEL_KEY}_{MODE}.jsonl"
todo_all = (list(PROMPT_SET) if CUSTOM_PROMPTS else select_mode(PROMPT_SET, MODE))[: MAX_CALLS or None]
done = set()
if os.path.exists(OUT):
    with open(OUT) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["call_id"])
            except Exception:
                pass
todo = [p for p in todo_all if call_id(MODEL_KEY, p["key"]) not in done]
print(f"{len(todo_all)} calls in mode '{MODE}': {len(done)} already done, {len(todo)} to run -> {OUT}")

META = dict(model_key=MODEL_KEY, hf_id=M["hf_id"], canonical_id=M.get("canonical_id", M["hf_id"]), revision=REVISION, family=M["family"], engine=ENGINE,
            engine_args=ENGINE_KW, engine_notes=ENGINE_NOTES, dtype=DTYPE, tensor_parallel=TP, gpus=GPU_NAMES, sampling=SAMPLING, mode=MODE,
            sampling_note=("presence_penalty not applied (transformers)" if llm is None and "presence_penalty" in SAMPLING else ""),
            chat_template_kwargs=M.get("chat_template_kwargs", {}), max_tokens=M["max_tokens"],
            prompt_set_version=PROMPT_SET_LABEL, prompt_set_sha256=PROMPT_SET_SHA256, notebook_version=NOTEBOOK_VERSION)
t0, n_tok = time.time(), 0
with open(OUT, "a") as fh:
    for b in range(0, len(todo), BATCH_SIZE):
        batch = todo[b:b + BATCH_SIZE]
        started = utcnow()
        results = run_vllm(batch) if llm is not None else run_hf(batch)
        finished = utcnow()
        for p, r in zip(batch, results):
            rec = {"call_id": call_id(MODEL_KEY, p["key"]), "prompt_key": p["key"], "region_code": p["region_code"],
                   "region_label": p["region_label"], "condition": p["condition"], "rep": p["rep"], "place": p["place"],
                   "system": p["system"], "user": p["user"], "seed": seed_for(MODEL_KEY, p["key"]),
                   "output_text": r["text"], "finish_reason": r["finish_reason"], "n_prompt_tokens": r["n_prompt_tokens"],
                   "n_output_tokens": r["n_output_tokens"], "rendered_prompt_sha16": r["rendered_sha16"],
                   "engine_batch": r.get("engine_batch"), "batch_started_utc": started, "batch_finished_utc": finished,
                   **({"meta": p["meta"]} if "meta" in p else {}), **META}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\\n")
            n_tok += r["n_output_tokens"]
        fh.flush()
        el = time.time() - t0
        n_done = b + len(batch)
        print(f"[{utcnow()}] {n_done}/{len(todo)} calls | {n_tok / max(el, 1):.0f} output tok/s | "
              f"ETA {el / n_done * (len(todo) - n_done) / 60:.1f} min", flush=True)
print("Generation finished in", round((time.time() - t0) / 60, 1), "min")
"""

SUMMARY = """from collections import Counter
rows = [json.loads(l) for l in open(OUT)]
status, n_rec, n5, n_ing, fin = Counter(), 0, 0, [], Counter(r["finish_reason"] for r in rows)
for r in rows:
    obj, st = extract_json(r["output_text"])
    status[st] += 1
    recs, issues = validate_recipes(obj) if obj is not None else ([], ["unparsed"])
    n_rec += len(recs)
    n5 += len(recs) == 5
    n_ing += [len(x["ingredients"]) for x in recs]
print(f"calls: {len(rows)} | parse status: {dict(status)} | finish: {dict(fin)}")
print(f"recipes parsed: {n_rec} (calls with exactly 5: {n5}/{len(rows)}) | mean ingredients/recipe: {sum(n_ing) / max(len(n_ing), 1):.1f}")
example, _ = extract_json(rows[0]["output_text"])
print(json.dumps((validate_recipes(example)[0] or [{}])[0], indent=1, ensure_ascii=False)[:1500])
with open(OUT, "rb") as src, gzip.open(OUT + ".gz", "wb") as dst:
    dst.write(src.read())
print("Saved:", OUT, "and", OUT + ".gz", "-> download from the Output tab")
if M["family"].startswith("Llama"):
    print("Reminder: Llama outputs must carry the 'Built with Llama' attribution in the dataset release.")
"""


REQUIRED_PROMPT_FIELDS = ("key", "region_code", "region_label", "condition", "rep", "place", "system", "user")


def build(
    model_key: str = "qwen3.5-9b",
    mode: str = "pilot",
    prompts: list[dict] | None = None,
    label: str | None = None,
    force_engine: str | None = None,
) -> tuple[nbf.NotebookNode, dict]:
    """The notebook for one model and mode.

    Default: the frozen v1 prompt set, with MODE selecting pilot, full or sens. With ``prompts`` (e.g. a D3 steering
    pool or a revision round), that list is embedded instead under ``label``, and every prompt in it is run; each
    prompt needs REQUIRED_PROMPT_FIELDS and may carry a "meta" dict that is copied into its record.
    ``force_engine="transformers"`` skips the vLLM install and start-up (vLLM 0.34 fails on the T4; see RESEARCH_NOTES).
    """
    if force_engine not in (None, "transformers"):
        raise ValueError(f"force_engine must be None or 'transformers', not {force_engine!r}")
    custom = prompts is not None
    if custom:
        missing = [f for p in prompts for f in REQUIRED_PROMPT_FIELDS if f not in p]
        if missing or not label:
            raise ValueError(
                f"custom prompts need a label and the fields {REQUIRED_PROMPT_FIELDS}; missing {missing[:5]}"
            )
        prompt_set = prompts
    else:
        prompt_set, label = build_prompt_set(load_regions(), load_templates(PROMPT_SET_VERSION)), PROMPT_SET_VERSION
    raw = "\n".join(json.dumps(p, ensure_ascii=False, sort_keys=True) for p in prompt_set).encode()
    sha = hashlib.sha256(raw).hexdigest()
    if not custom:  # keep the canonical v1 files in the repo current; custom sets live in their own source file
        PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        (PROMPT_DIR / f"{label}.jsonl").write_bytes(raw + b"\n")
        (PROMPT_DIR / f"{label}.sha256").write_text(sha + "\n")

    kaggle_models = {k: m for k, m in load_models().items() if m["runner"] == "kaggle"}
    keep = (
        "hf_id",
        "canonical_id",
        "family",
        "tier",
        "params_b",
        "gated",
        "license",
        "sampling",
        "chat_template_kwargs",
        "engine_args",
        "fp16_unsafe",
        "max_tokens",
    )
    # Omit absent fields (never write None): the notebook relies on M.get(field, default).
    registry = {k: {f: m[f] for f in keep if m.get(f) is not None} for k, m in kaggle_models.items()}
    b64 = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    common_src = (REPO / "generation" / "common.py").read_text(encoding="utf-8")

    data_cell = (
        "# Embedded inputs (do not edit): prompt set + model registry. The sha256 check guarantees the exact prompts.\n"
        "import base64, gzip, hashlib, json\n"
        f'PROMPT_SET_B64 = "{b64}"\n'
        f'PROMPT_SET_SHA256 = "{sha}"\n'
        f'PROMPT_SET_LABEL = "{label}"\n'
        f"CUSTOM_PROMPTS = {custom}  # True: run every embedded prompt (D3 steering sets); False: MODE selects from v1\n"
        f"MODELS = json.loads(r'''{json.dumps(registry, indent=1)}''')\n"
        "_raw = gzip.decompress(base64.b64decode(PROMPT_SET_B64))\n"
        "assert hashlib.sha256(_raw).hexdigest() == PROMPT_SET_SHA256, 'prompt set corrupted'\n"
        "PROMPT_SET = [json.loads(l) for l in _raw.decode().splitlines()]\n"
        "print(len(PROMPT_SET), 'prompts loaded; models:', list(MODELS))\n"
    )
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(HEADER),
        nbf.v4.new_code_cell(SETTINGS.format(model_key=model_key, mode=mode, force_engine=repr(force_engine))),
        nbf.v4.new_code_cell(INSTALL),
        nbf.v4.new_code_cell(data_cell),
        nbf.v4.new_markdown_cell(
            "Shared helpers: an exact copy of `generation/common.py` from the project repository."
        ),
        nbf.v4.new_code_cell(common_src),
        nbf.v4.new_code_cell(ENV),
        nbf.v4.new_code_cell(ENGINE),
        nbf.v4.new_code_cell(GENERATE),
        nbf.v4.new_code_cell(SUMMARY),
    ]
    nbf.validate(nb)
    return nb, {"prompt_set_sha256": sha, "n_prompts": len(prompt_set), "label": label}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-user", help="Kaggle username: also write per-model folders for `kaggle kernels push`")
    args = ap.parse_args()
    nb, info = build()
    out = KAGGLE_DIR / "llmcuisine22_generate.ipynb"
    nbf.write(nb, out)
    print(
        f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 1024:.0f} KB); prompts={info['n_prompts']} sha256={info['prompt_set_sha256'][:12]}"
    )
    if args.kaggle_user:
        for mk in [k for k, m in load_models().items() if m["runner"] == "kaggle"]:
            for mode in ("pilot", "full", "sens"):
                slug = f"llmcuisine22-{mk.replace('.', '-')}-{mode}"
                d = KAGGLE_DIR / "push" / slug
                d.mkdir(parents=True, exist_ok=True)
                nbm, _ = build(mk, mode)
                nbf.write(nbm, d / f"{slug}.ipynb")
                meta = {
                    "id": f"{args.kaggle_user}/{slug}",
                    "title": slug,
                    "code_file": f"{slug}.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": "true",
                    "enable_gpu": "true",
                    "enable_internet": "true",
                    "machine_shape": "NvidiaTeslaT4",
                    "dataset_sources": [],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                }
                (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
                print("wrote", d.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

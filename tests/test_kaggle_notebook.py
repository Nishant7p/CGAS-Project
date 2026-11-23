"""Run the real Kaggle notebook cells with a fake GPU engine (no GPU, no network).

Covers prompt loading and its sha check, model registry fields, seeds and sampling, checkpointing,
resume without duplicate calls, and the summary/parse step. Two models are exercised: one with chat
template kwargs (Qwen: thinking disabled) and one without (Llama), which guards against None fields.
"""

import ast
import contextlib
import datetime
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import types

import pytest

from generation.config import load_models
from generation.kaggle.build_notebook import build

GOOD = {
    "recipes": [
        {
            "title": f"Dish {i}",
            "ingredients": ["2 cups rice", "1 tsp salt", "1 onion, chopped"],
            "instructions": ["Cook."],
        }
        for i in range(5)
    ]
}


class _Comp:
    def __init__(self, text):
        self.text, self.finish_reason, self.token_ids = text, "stop", list(range(len(text) // 4))


class _Req:
    def __init__(self, text):
        self.outputs, self.prompt_token_ids = [_Comp(text)], list(range(300))


@pytest.mark.parametrize("model_key,mode", [("qwen3.5-9b", "pilot"), ("llama-3.1-8b", "pilot"), ("qwen3.5-9b", "sens")])
def test_notebook_generation_loop_with_fake_engine(tmp_path, model_key, mode):
    nb, info = build(model_key, mode)
    assert info["n_prompts"] == 704
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    settings, _install, data, common, _env, _engine, generate, summary = code
    expected_kwargs = load_models()[model_key].get("chat_template_kwargs", {})
    expected_sampling = {"temperature": 1.0, "top_p": 1.0} if mode == "sens" else {"temperature": 0.7, "top_p": 0.95}
    calls = {"n": 0}

    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            assert kw == expected_kwargs
            return f"<s>{msgs[0]['content']}|{msgs[1]['content']}</s>"

    class LLM:
        def generate(self, prompts, sps, use_tqdm=True):
            calls["n"] += len(prompts)
            assert len(prompts) == len(sps)
            return [_Req("<think>\n\n</think>\n\n" + json.dumps(GOOD)) for _ in prompts]

    class SamplingParams:
        def __init__(self, **kw):
            assert kw["max_tokens"] == 6144
            assert {k: kw[k] for k in ("temperature", "top_p")} == expected_sampling
            assert 0 <= kw["seed"] < 2**31

    ns = {}
    exec(settings.replace("MAX_CALLS = None", "MAX_CALLS = 6").replace("BATCH_SIZE = 64", "BATCH_SIZE = 4"), ns)
    exec(data, ns)
    exec(common, ns)
    assert model_key in ns["MODELS"] and all(v is not None for v in ns["MODELS"][model_key].values())
    ns.update(
        M=ns["MODELS"][model_key],
        tok=Tok(),
        llm=LLM(),
        SamplingParams=SamplingParams,
        ENGINE="vllm-fake",
        ENGINE_KW={},
        ENGINE_NOTES=[],
        DTYPE="float16",
        TP=2,
        GPU_NAMES=["Tesla T4", "Tesla T4"],
        REVISION="abc123",
        os=os,
        time=time,
        gzip=gzip,
        hashlib=hashlib,
        datetime=datetime,
    )
    src = generate.replace("/kaggle/working/", str(tmp_path) + "/")
    exec(src, ns)
    exec(src, ns)  # second run must resume and make no new calls
    assert calls["n"] == 6
    exec(summary, ns)
    with open(ns["OUT"]) as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 6 and len({r["call_id"] for r in rows}) == 6
    assert rows[0]["model_key"] == model_key and rows[0]["revision"] == "abc123"
    assert rows[0]["prompt_set_sha256"] == info["prompt_set_sha256"]
    assert rows[0]["mode"] == mode and rows[0]["sampling"] == expected_sampling
    assert all(r["rep"] == 0 for r in rows)  # pilot and sens both use replicate 0
    assert ns["OUT"].endswith(f"_{mode}.jsonl") and os.path.exists(ns["OUT"] + ".gz")


def test_transformers_fallback_halves_sub_batch_on_oom(tmp_path, monkeypatch):
    """The fallback engine (used by the nb-v1 Qwen pilot) must survive CUDA OOM by splitting the chunk and log it."""
    import numpy as np

    nb, _ = build("llama-3.1-8b", "pilot")
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    settings, _install, data, common, _env, _engine, generate, summary = code

    class OOM(Exception):
        pass

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None)
    fake_torch.manual_seed = lambda seed: None
    fake_torch.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sizes = []

    class Enc(dict):
        def to(self, device):
            return self

    class Tok:
        pad_token, eos_token, pad_token_id = None, "</s>", 0

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            return msgs[1]["content"]

        def __call__(self, prompts, **kw):
            n = len(prompts)
            return Enc(input_ids=np.ones((n, 7), dtype=int), attention_mask=np.ones((n, 7), dtype=int))

        def decode(self, ids, skip_special_tokens=True):
            return json.dumps(GOOD)

    class HF:
        device = "cpu"

        def generate(self, input_ids, attention_mask, **kw):
            sizes.append(input_ids.shape[0])
            if input_ids.shape[0] > 2:
                raise OOM()
            out = np.zeros((input_ids.shape[0], 17), dtype=int)
            out[:, 7:] = 5  # 10 generated (non-pad) tokens per prompt
            return out

    ns = {}
    exec(settings.replace("MAX_CALLS = None", "MAX_CALLS = 6").replace("BATCH_SIZE = 64", "BATCH_SIZE = 4"), ns)
    exec(data, ns)
    exec(common, ns)
    ns.update(
        M=ns["MODELS"]["llama-3.1-8b"],
        tok=Tok(),
        llm=None,
        hf_model=HF(),
        ENGINE="transformers-fake",
        ENGINE_KW={},
        ENGINE_NOTES=["note"],
        DTYPE="float16",
        TP=2,
        GPU_NAMES=["Tesla T4", "Tesla T4"],
        REVISION="abc123",
        os=os,
        time=time,
        gzip=gzip,
        hashlib=hashlib,
        datetime=datetime,
    )
    exec(generate.replace("/kaggle/working/", str(tmp_path) + "/"), ns)
    assert sizes == [4, 2, 2, 2]  # first chunk OOMs once, then the halved size sticks for the rest of the run
    exec(summary, ns)
    with open(ns["OUT"]) as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 6 and {r["engine_batch"] for r in rows} == {2}
    assert all(r["n_output_tokens"] == 10 and r["finish_reason"] == "stop" for r in rows)
    assert rows[0]["engine"] == "transformers-fake" and rows[0]["engine_notes"] == ["note"]


def test_libcuda_shim_links_driver_library_for_jit(tmp_path, monkeypatch):
    import glob

    nb, _ = build("qwen3.5-9b", "pilot")
    engine = next(c.source for c in nb.cells if c.cell_type == "code" and "def expose_libcuda_for_jit" in c.source)
    fn = next(
        n for n in ast.parse(engine).body if isinstance(n, ast.FunctionDef) and n.name == "expose_libcuda_for_jit"
    )
    ns = {"os": os, "subprocess": subprocess}
    exec(compile(ast.Module([fn], []), "engine_cell", "exec"), ns)
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"")
    monkeypatch.setattr(
        glob, "glob", lambda pat: [str(driver)] if pat.endswith("x86_64-linux-gnu/libcuda.so.1") else []
    )
    monkeypatch.setenv("LIBRARY_PATH", "/opt/lib")
    note = ns["expose_libcuda_for_jit"](shim=str(tmp_path / "shim"))
    assert os.readlink(tmp_path / "shim" / "libcuda.so") == str(driver.resolve())
    assert os.environ["LIBRARY_PATH"] == f"{tmp_path / 'shim'}:/opt/lib"
    assert "libcuda.so ->" in note
    ns["expose_libcuda_for_jit"](shim=str(tmp_path / "shim"))  # idempotent: replaces the link, no error
    monkeypatch.setattr(glob, "glob", lambda pat: [])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    assert "not found" in ns["expose_libcuda_for_jit"](shim=str(tmp_path / "shim2"))


def _custom_prompts(n=3):
    return [
        {
            "key": f"rev1|abc{i}",
            "region_code": "KOR",
            "region_label": "Korea",
            "condition": "revise",
            "rep": 1,
            "place": None,
            "system": "You are a helpful cooking assistant.",
            "user": f"Revise recipe {i}.",
            "meta": {"parent_uid": f"fabc-{i}", "round": 1},
        }
        for i in range(n)
    ]


def test_notebook_runs_every_prompt_of_a_custom_steering_set(tmp_path):
    nb, info = build("llama-3.1-8b", "rev1", prompts=_custom_prompts(), label="steer-rev1-test")
    assert info["n_prompts"] == 3 and info["label"] == "steer-rev1-test"
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    settings, _install, data, common, _env, _engine, generate, _summary = code

    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            return msgs[1]["content"]

    class LLM:
        def generate(self, prompts, sps, use_tqdm=True):
            return [_Req(json.dumps(GOOD)) for _ in prompts]

    ns = {}
    exec(settings, ns)
    exec(data, ns)
    exec(common, ns)
    assert ns["CUSTOM_PROMPTS"] is True and ns["PROMPT_SET_LABEL"] == "steer-rev1-test"
    ns.update(
        M=ns["MODELS"]["llama-3.1-8b"], tok=Tok(), llm=LLM(), SamplingParams=lambda **kw: kw, ENGINE="vllm-fake",
        ENGINE_KW={}, ENGINE_NOTES=[], DTYPE="float16", TP=2, GPU_NAMES=["T4"], REVISION="r", os=os, time=time,
        gzip=gzip, hashlib=hashlib, datetime=datetime,
    )  # fmt: skip
    exec(generate.replace("/kaggle/working/", str(tmp_path) + "/"), ns)
    with open(ns["OUT"]) as fh:
        rows = [json.loads(line) for line in fh]
    assert [r["prompt_key"] for r in rows] == ["rev1|abc0", "rev1|abc1", "rev1|abc2"]
    assert (
        rows[1]["meta"] == {"parent_uid": "fabc-1", "round": 1} and rows[0]["prompt_set_version"] == "steer-rev1-test"
    )
    assert ns["OUT"].endswith("_rev1.jsonl")


def test_custom_prompt_sets_are_validated():
    bad = _custom_prompts(1)
    del bad[0]["system"]
    with pytest.raises(ValueError):
        build("llama-3.1-8b", "rev1", prompts=bad, label="x")
    with pytest.raises(ValueError):
        build("llama-3.1-8b", "rev1", prompts=_custom_prompts(1), label=None)


def test_notebook_registry_matches_config_and_sampling_policy():
    nb, _ = build()
    data = next(c.source for c in nb.cells if c.cell_type == "code" and "PROMPT_SET_B64" in c.source)
    ns = {}
    exec(data, ns)
    kaggle = {k for k, m in load_models().items() if m["runner"] == "kaggle"}
    assert set(ns["MODELS"]) == kaggle
    for k, m in ns["MODELS"].items():
        assert m["sampling"] == {"temperature": 0.7, "top_p": 0.95}, k
        assert m["hf_id"] and m["params_b"] > 0

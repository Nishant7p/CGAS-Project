"""Run the Kaggle generation notebook without a browser: push (create + run), check status, download output.

Each model x mode gets its own private notebook "<user>/llmcuisine22-<model>-<mode>" on GPU T4 x2 with
internet on. The Kaggle username is read from the token (~/.kaggle/access_token or KAGGLE_API_TOKEN).

Usage (repo root):
    .venv/bin/python generation/kaggle/push.py push   --model qwen3.5-9b --mode pilot
    .venv/bin/python generation/kaggle/push.py status --model qwen3.5-9b --mode pilot
    .venv/bin/python generation/kaggle/push.py output --model qwen3.5-9b --mode pilot
D3 steering sets (any other mode name) embed a prompt file instead of the v1 set; their output goes to
data/interim/steering/raw/<model>/, outside the D1 dataset:
    .venv/bin/python generation/kaggle/push.py push --model qwen3.5-9b --mode bon \
        --prompts steering/prompt_sets/bon.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from generation.config import load_models  # noqa: E402
from generation.kaggle.build_notebook import build  # noqa: E402

KAGGLE = str(REPO / ".venv" / "bin" / "kaggle")
ACCELERATOR = "NvidiaTeslaT4"  # GPU T4 x2 (Kaggle CLI docs, Sep 2026)


def username() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api.config_values["username"]


def slug(model: str, mode: str) -> str:
    return f"llmcuisine22-{model.replace('.', '-')}-{mode}"


STANDARD_MODES = ("pilot", "full", "sens")


def load_prompts(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(model: str, mode: str, user: str, prompts: str | None = None, engine: str | None = None) -> Path:
    s = slug(model, mode)
    d = REPO / "generation" / "kaggle" / "push" / s
    d.mkdir(parents=True, exist_ok=True)
    if mode in STANDARD_MODES:
        nb, info = build(model, mode, force_engine=engine)
    else:
        nb, info = build(model, mode, prompts=load_prompts(prompts), label=Path(prompts).stem, force_engine=engine)
    nbf.write(nb, d / f"{s}.ipynb")
    meta = {
        "id": f"{user}/{s}",
        "title": s,
        "code_file": f"{s}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": ACCELERATOR,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"prepared {d.relative_to(REPO)} (prompt set {info['prompt_set_sha256'][:12]})")
    return d


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd[1:]) if cmd[0] == KAGGLE else " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "status", "output"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="pilot", help="pilot | full | sens, or a D3 steering mode name with --prompts")
    ap.add_argument("--prompts", help="JSONL prompt file for a steering mode (embedded in the notebook)")
    ap.add_argument(
        "--engine",
        choices=["transformers"],
        help="skip vLLM and use the transformers engine directly (vLLM 0.34 fails on the T4; RESEARCH_NOTES)",
    )
    args = ap.parse_args()
    m = load_models().get(args.model)
    if not m or m["runner"] != "kaggle":
        print("Kaggle models:", [k for k, v in load_models().items() if v["runner"] == "kaggle"])
        return 2
    steering = args.mode not in STANDARD_MODES
    if steering and args.action == "push" and not args.prompts:
        print(f"mode {args.mode!r} is a steering mode: pass --prompts FILE")
        return 2
    user = username()
    ref = f"{user}/{slug(args.model, args.mode)}"
    if args.action == "push":
        d = prepare(args.model, args.mode, user, args.prompts, args.engine)
        return run([KAGGLE, "kernels", "push", "-p", str(d), "--accelerator", ACCELERATOR])
    if args.action == "status":
        return run([KAGGLE, "kernels", "status", ref])
    out = REPO / "data" / ("interim/steering/raw" if steering else "generated/raw") / args.model
    out.mkdir(parents=True, exist_ok=True)
    rc = run([KAGGLE, "kernels", "output", ref, "-p", str(out), "--file-pattern", r".*\.jsonl(\.gz)?$"])
    print("files:", sorted(p.name for p in out.glob("*.jsonl*")))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

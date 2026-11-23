"""Check every credential the pipeline needs, with free calls only. Secrets are never printed.

Reads .env (OPENAI_API_KEY, GEMINI_API_KEY, optional HF_TOKEN) and the Kaggle token (~/.kaggle/access_token or
KAGGLE_API_TOKEN). Exit code 0 when all required checks pass.

Usage (repo root): .venv/bin/python scripts/check_access.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _planned_ids(models: dict, provider: str) -> list[str]:
    return [m["model_id"] for m in models.values() if m.get("provider") == provider and m["tier"] in ("core", "extra")]


def _report(available: set[str], wanted: list[str]) -> tuple[bool, str]:
    missing = [w for w in wanted if w not in available]
    if missing:
        return False, f"key works, but these models are not visible to it: {missing}"
    return True, "models available: " + ", ".join(wanted)


def check_kaggle() -> tuple[bool, str]:
    """Authenticate AND make a real server call: reading a stale kaggle.json alone looks fine but is rejected."""
    import subprocess

    kaggle = str(REPO / ".venv" / "bin" / "kaggle")
    r = subprocess.run([kaggle, "kernels", "list", "--mine", "--page-size", "1"], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "Authentication required" in out:
        return False, "credential rejected by Kaggle: generate a new token at https://www.kaggle.com/settings/api"
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return True, f"server call OK, user '{api.config_values.get('username')}'"
    except BaseException as e:  # the kaggle package may raise SystemExit when no credentials exist
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def check_openai(models: dict) -> tuple[bool, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return False, "OPENAI_API_KEY missing in .env"
    try:
        from openai import OpenAI

        ids = {m.id for m in OpenAI(api_key=key).models.list()}
        return _report(ids, _planned_ids(models, "openai"))
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def check_gemini(models: dict) -> tuple[bool, str]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return False, "GEMINI_API_KEY missing in .env"
    try:
        from google import genai

        client = genai.Client(api_key=key)  # keep a reference: the pager needs an open client while iterating
        names = {m.name.split("/")[-1] for m in client.models.list()}
        return _report(names, _planned_ids(models, "gemini"))
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def check_hf() -> tuple[bool, str]:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return True, "not set (not needed for the planned models)"
    try:
        from huggingface_hub import whoami

        return True, f"token valid for '{whoami(token=tok).get('name')}'"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def main() -> int:
    from dotenv import load_dotenv

    from generation.config import load_models

    load_dotenv(REPO / ".env")
    models = load_models()
    checks = {
        "Kaggle (required)": check_kaggle(),
        "OpenAI (required)": check_openai(models),
        "Gemini (required)": check_gemini(models),
        "Hugging Face (optional)": check_hf(),
    }
    for name, (ok, msg) in checks.items():
        print(f"{'OK  ' if ok else 'FAIL'} {name:26s} {msg}")
    budget = os.environ.get("BUDGET_USD", "25")
    print(f"     OpenAI budget cap BUDGET_USD = {budget}")
    return 0 if all(ok for ok, _ in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

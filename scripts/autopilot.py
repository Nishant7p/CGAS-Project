"""Autopilot: finish the running jobs and rebuild every deliverable from their outputs, without a human or an agent.

Every POLL seconds it checks each job; when a job's output is ready it runs the downstream steps once (state in
data/interim/autopilot_state.json, log in data/interim/autopilot.log). All steps are the repository's own scripts and
are idempotent, so re-running (or restarting the autopilot) never repeats finished work or loses outputs.

Jobs:
  llama_full    Kaggle llama-3.1-8b full -> download -> refresh_d2
  gemini_full   generation/run_api.py (already running with --wait-daily) reaches 704 calls -> refresh_d2
  luna_rev1     OpenAI batch -> fetch -> collect -> prepare round 2 -> submit luna_rev2
  luna_rev2     OpenAI batch -> fetch -> collect -> evaluate revisions
  qwen_rev1     Kaggle qwen3.5-9b rev1 -> download -> collect -> evaluate revisions
  qwen4b_full   Kaggle qwen3.5-4b full (the pre-registered RQ4 scale pair) -> download -> refresh_d2
  ministral_pilot  Kaggle ministral-3-8b pilot -> download -> pilot gate (fixed before any Ministral output was seen:
                every one of the 44 calls returned, >= 80% parse into exactly 5 recipes, <= 10% hit max_tokens; the
                existing pilots scored 95-100% and 0%) -> push ministral full, or log why not
  ministral_full   Kaggle ministral-3-8b full -> download -> refresh_d2
refresh_d2: parse -> alias -> llm_audit -> rq_tests -> networks -> figures -> drivers -> release -> datasheet -> summary
-> app artifacts -> report/build.py and presentation/build.py (when they exist).

Usage (repo root):  nohup .venv/bin/python scripts/autopilot.py > /dev/null 2>&1 &
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
STATE = REPO / "data" / "interim" / "autopilot_state.json"
LOG = REPO / "data" / "interim" / "autopilot.log"
POLL = 300
CALLS_PER_MODEL = 704
PILOT_CALLS = 44
PILOT_MIN_EXACTLY5 = 0.80
PILOT_MAX_TRUNCATED = 0.10


def log(msg: str) -> None:
    line = f"[{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(*args: str, timeout: int = 7200) -> tuple[int, str]:
    try:
        p = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        log(f"$ {' '.join(a.replace(str(REPO) + '/', '') for a in args)} -> exit {p.returncode} | {out.strip()[-300:]}")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        log(f"$ {' '.join(args)} -> TIMEOUT after {timeout}s")
        return 124, ""


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1))


def kaggle_status(model: str, mode: str) -> str:
    _, out = run(PY, "generation/kaggle/push.py", "status", "--model", model, "--mode", mode, timeout=180)
    for tok in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if tok in out:
            return tok
    return "UNKNOWN"


def batch_status(model: str, mode: str) -> str:
    _, out = run(PY, "generation/openai_batch.py", "status", "--model", model, "--mode", mode, timeout=180)
    for tok in ("completed", "failed", "expired", "cancelled", "in_progress", "validating", "finalizing"):
        if f": {tok}" in out:
            return tok
    return "unknown"


def lines_in(path: Path) -> int:
    return sum(1 for x in path.read_text(encoding="utf-8").splitlines() if x.strip()) if path.exists() else 0


def refresh_d2(reason: str) -> None:
    log(f"refresh_d2 ({reason})")
    steps = [
        (PY, "generation/parse_outputs.py"),
        (PY, "generation/alias_lines.py"),
        (PY, "analysis/audit/llm_audit.py", "--tag", "full"),
        (PY, "analysis/audit/rq_tests.py", "--tag", "full"),
        (PY, "analysis/audit/networks.py"),
        (PY, "analysis/figures/d2_figures.py", "--tag", "full"),
        (PY, "analysis/figures/equal_n_figures.py"),
        (PY, "analysis/audit/drivers.py", "--tag", "full", "--reps", "20", "--skip-real"),
        (PY, "generation/build_release.py"),
        (PY, "generation/write_datasheet.py"),
    ]
    for s in steps:
        run(*s)
    rebuild_documents()


def rebuild_documents() -> None:
    run(PY, "analysis/summarize.py")
    run(PY, "app/build_artifacts.py")
    run(PY, "analysis/figures/d2_more_figures.py", timeout=3600)
    # the talk deck is presentation/story_deck.py; presentation/build.py only rebuilds the poster (its old deck and
    # prep pack are superseded and must not be regenerated over the edited rehearsal pack)
    for script, extra in (("report/build.py", ("--include-f5",)), ("presentation/build.py", ("--poster-only",)),
                          ("presentation/story_deck.py", ()), ("scripts/export_final.py", ())):  # fmt: skip
        if (REPO / script).exists():
            run(PY, script, *extra, timeout=3600)


def evaluate_revisions() -> None:
    models = [
        m
        for m in ("openai-gpt-6-luna", "qwen3.5-9b")
        if (REPO / f"data/interim/steering/recipes_rev1_{m}.parquet").exists()
    ]
    if models:
        run(PY, "steering/run_revise.py", "evaluate", "--models", *models, "--rounds", "2")
        rebuild_documents()


def pilot_gate(model: str) -> tuple[bool, str]:
    """The pre-stated pilot gate (see the module docstring) on the downloaded pilot JSONL."""
    import sys

    sys.path.insert(0, str(REPO))
    from generation.common import extract_json, validate_recipes

    path = REPO / "data" / "generated" / "raw" / model / f"llmcuisine22_{model}_pilot.jsonl"
    if not path.exists():
        return False, f"no pilot output at {path.relative_to(REPO)}"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    five = trunc = 0
    for r in recs:
        obj, _ = extract_json(r.get("output_text") or "")
        ok = validate_recipes(obj)[0] if obj is not None else []
        five += len(ok) == 5
        trunc += r.get("finish_reason") == "length"
    n = max(1, len(recs))
    msg = f"calls {len(recs)}/{PILOT_CALLS}, exactly-5 {five / n:.2f}, truncated {trunc / n:.2f}"
    passed = len(recs) >= PILOT_CALLS and five / n >= PILOT_MIN_EXACTLY5 and trunc / n <= PILOT_MAX_TRUNCATED
    return passed, msg


def kaggle_job(state: dict, key: str, model: str, mode: str, on_done) -> bool:
    """Poll one Kaggle run; on completion download its output and call on_done(). Returns True while pending."""
    if state.get(key) and state.get(key) not in ("pushed",):
        return False
    st = kaggle_status(model, mode)
    if st == "COMPLETE":
        run(PY, "generation/kaggle/push.py", "output", "--model", model, "--mode", mode, timeout=1800)
        state[key] = "done"
        save(state)
        on_done()
    elif st in ("ERROR", "CANCEL"):
        state[key] = f"failed:{st}"
        save(state)
        log(f"{model} {mode} FAILED on Kaggle; needs attention (log: kaggle kernels output, or the Kaggle web page)")
    return True


def after_ministral_pilot(state: dict) -> None:
    passed, msg = pilot_gate("ministral-3-8b")
    log(f"ministral pilot gate: {'PASS' if passed else 'FAIL'} ({msg})")
    if not passed:
        state["ministral_full"] = f"not pushed: pilot gate failed ({msg})"
    else:
        code, _ = run(PY, "generation/kaggle/push.py", "push", "--model", "ministral-3-8b", "--mode", "full",
                      "--engine", "transformers", timeout=600)  # fmt: skip
        state["ministral_full"] = "pushed" if code == 0 else f"push-failed:{code}"
    save(state)


def step(state: dict) -> bool:
    """One pass over all jobs; returns True when nothing is left to wait for."""
    pending = False
    # RQ4 extras on Kaggle: Qwen3.5-4B full, and Ministral-3-8B pilot -> gate -> full
    pending |= kaggle_job(state, "qwen4b_full", "qwen3.5-4b", "full", lambda: refresh_d2("qwen3.5-4b full complete"))
    pending |= kaggle_job(state, "ministral_pilot", "ministral-3-8b", "pilot", lambda: after_ministral_pilot(state))
    if state.get("ministral_full") == "pushed":
        pending |= kaggle_job(
            state, "ministral_full", "ministral-3-8b", "full", lambda: refresh_d2("ministral-3-8b full complete")
        )
    # Llama full on Kaggle
    if not state.get("llama_full"):
        pending = True
        st = kaggle_status("llama-3.1-8b", "full")
        if st == "COMPLETE":
            run(PY, "generation/kaggle/push.py", "output", "--model", "llama-3.1-8b", "--mode", "full", timeout=1800)
            state["llama_full"] = "done"
            save(state)
            refresh_d2("llama full complete")
        elif st in ("ERROR", "CANCEL"):
            state["llama_full"] = f"failed:{st}"
            save(state)
            log(
                "llama full FAILED on Kaggle; needs attention (re-push: generation/kaggle/push.py push --model llama-3.1-8b --mode full)"
            )
    # Gemini full (the runner keeps going by itself; we only refresh when it completes)
    if not state.get("gemini_full"):
        pending = True
        n = lines_in(REPO / "data/generated/raw/gemini-3.5-flash-lite/full.jsonl")
        if n >= CALLS_PER_MODEL:
            state["gemini_full"] = "done"
            save(state)
            refresh_d2("gemini full complete")
    # Luna critique-and-revise, rounds 1 and 2 (OpenAI Batch)
    for rnd in (1, 2):
        key = f"luna_rev{rnd}"
        if state.get(key) == "done":
            continue
        pending = True
        if rnd == 2 and state.get("luna_rev2") != "submitted":
            break  # round 2 is submitted after round 1 is collected
        bs = batch_status("openai-gpt-6-luna", f"rev{rnd}")
        if bs == "completed":
            run(PY, "generation/openai_batch.py", "fetch", "--model", "openai-gpt-6-luna", "--mode", f"rev{rnd}")
            run(PY, "steering/collect.py", "--model", "openai-gpt-6-luna", "--mode", f"rev{rnd}")
            state[key] = "done"
            if rnd == 1:
                run(PY, "steering/run_revise.py", "prepare", "--model", "openai-gpt-6-luna", "--round", "2")
                code, _ = run(PY, "generation/openai_batch.py", "submit", "--model", "openai-gpt-6-luna", "--mode", "rev2",
                              "--prompts", "steering/prompt_sets/steer_rev2_openai-gpt-6-luna.jsonl")  # fmt: skip
                state["luna_rev2"] = "submitted" if code == 0 else f"submit-failed:{code}"
            save(state)
            evaluate_revisions()
        elif bs in ("failed", "expired", "cancelled"):
            state[key] = f"failed:{bs}"
            save(state)
        break
    # Qwen critique-and-revise round 1 (Kaggle)
    if not state.get("qwen_rev1"):
        pending = True
        st = kaggle_status("qwen3.5-9b", "rev1")
        if st == "COMPLETE":
            run(PY, "generation/kaggle/push.py", "output", "--model", "qwen3.5-9b", "--mode", "rev1", timeout=1800)
            run(PY, "steering/collect.py", "--model", "qwen3.5-9b", "--mode", "rev1")
            state["qwen_rev1"] = "done"
            save(state)
            evaluate_revisions()
        elif st in ("ERROR", "CANCEL"):
            state["qwen_rev1"] = f"failed:{st}"
            save(state)
    return not pending


FREEZE = REPO / "data" / "interim" / "FREEZE"  # while it exists: no downloads, refreshes or rebuilds (Kaggle keeps outputs)


def main() -> int:
    log("autopilot started")
    state = load()
    last_note = 0.0
    while True:
        try:
            if FREEZE.exists():
                if time.time() - last_note > 3600:
                    log(f"paused: {FREEZE.relative_to(REPO)} exists (documents frozen); delete it to resume")
                    last_note = time.time()
            elif step(state):
                log("all jobs finished; final rebuild done. autopilot exiting")
                return 0
        except Exception as e:  # keep going: one failing job must not stop the others
            log(f"error in step: {e!r}")
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())

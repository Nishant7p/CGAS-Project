"""Run LLMCuisine-22 generation against hosted APIs (Gemini, Groq, OpenAI) from the Mac.

It uses the same prompt set, call ids, seeds and record format as the Kaggle notebook. It is resumable
(it skips call ids already logged) and respects free-tier limits: requests and tokens per minute, plus
requests and tokens per day, tracked in a small state file.

Free-tier daily caps are unpublished, so the server is authoritative: a 429 whose QuotaFailure names a per-day
quota is recorded (quota id and value) in the state file. Without --wait-daily the run then stops cleanly (run it
again to resume); with --wait-daily it sleeps until the provider's daily reset (Gemini: midnight Pacific) and
continues. Transient errors (5xx, per-minute 429, network) are retried with backoff (capped at 5 min) until 8 hours
of waiting have accumulated for one call.

Usage (repo root):
    .venv/bin/python generation/run_api.py --model gemini-3.5-flash-lite --mode pilot --dry-run
    .venv/bin/python generation/run_api.py --model gemini-3.5-flash-lite --mode full --wait-daily
    .venv/bin/python generation/run_api.py --model groq-qwen3.8-27b --mode full
Keys come from .env (GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY) and are never printed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import re
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generation.common import (  # noqa: E402
    PROMPT_SET_VERSION,
    build_prompt_set,
    call_id,
    seed_for,
    select_mode,
)
from generation.config import load_models, load_regions, load_templates  # noqa: E402

RUNNER_VERSION = "api-v2"  # v2: server-driven daily quota handling (--wait-daily), longer transient retries
MAX_WAIT_PER_CALL_S = 8 * 3600  # transient failures (5xx "high demand", per-minute 429, network): keep retrying with
# backoff 10, 20, 40, 80, 160, then 300 s until this much waiting has accumulated for one call. Free-tier overload
# spells last hours (2026-11-22: 11 of the first 21 Gemini full-run attempts returned 503).
EST_OUTPUT_TOKENS_PER_CALL = 2600  # rough planning value; the pilot measures the real number
EST_PROMPT_TOKENS_PER_CALL = 250
REQUEST_TIMEOUT_S = 240  # per HTTP request. Without it, one hung Gemini request blocked the run for 40+ minutes
# (2026-11-22); a timeout is then retried like any transient error. Normal calls take 5-30 s.


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def quota_day(provider: str, now: dt.datetime | None = None) -> str:
    """The provider's quota day: Gemini counts days in Pacific time, others in UTC."""
    tz = ZoneInfo("America/Los_Angeles") if provider == "gemini" else dt.timezone.utc
    return (now or dt.datetime.now(dt.timezone.utc)).astimezone(tz).date().isoformat()


class Quota:
    """Sliding one-minute windows (requests, tokens) plus daily counters persisted to disk."""

    def __init__(self, limits: dict, state_path: Path, provider: str = ""):
        self.rpm, self.tpm = limits.get("rpm"), limits.get("tpm")
        self.rpd, self.tpd = limits.get("rpd"), limits.get("tpd")
        self.state_path, self.provider = state_path, provider
        self.req_times: deque[float] = deque()
        self.tok_times: deque[tuple[float, int]] = deque()
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        self._roll()

    def _roll(self) -> None:
        """Start new daily counters when the quota day changes (keeps any recorded server limit)."""
        today = quota_day(self.provider)
        if self.state.get("date") != today:
            self.state = {**self.state, "date": today, "requests": 0, "tokens": 0}

    def daily_left(self) -> bool:
        self._roll()
        if self.rpd and self.state["requests"] >= self.rpd:
            return False
        return not (self.tpd and self.state["tokens"] + EST_OUTPUT_TOKENS_PER_CALL > self.tpd)

    def wait(self, est_tokens: int) -> None:
        while True:
            now = time.time()
            while self.req_times and now - self.req_times[0] > 60:
                self.req_times.popleft()
            while self.tok_times and now - self.tok_times[0][0] > 60:
                self.tok_times.popleft()
            used_tok = sum(t for _, t in self.tok_times)
            if (not self.rpm or len(self.req_times) < self.rpm) and (not self.tpm or used_tok + est_tokens <= self.tpm):
                return
            time.sleep(1.0)

    def record(self, tokens: int) -> None:
        now = time.time()
        self.req_times.append(now)
        self.tok_times.append((now, tokens))
        self._roll()
        self.state["requests"] += 1
        self.state["tokens"] += tokens
        self.save()

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state))


def status_of(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


_QUOTA_ID = re.compile(r"""["']quotaId["']\s*:\s*["']([^"']+)""")
_QUOTA_VALUE = re.compile(r"""["']quotaValue["']\s*:\s*["']?(\d+)""")
_RETRY_DELAY = re.compile(r"""["']retryDelay["']\s*:\s*["'](\d+(?:\.\d+)?)s""")


def quota_violation(exc: Exception) -> tuple[str | None, int | None, float | None]:
    """(quota_id, quota_value, retry_delay_s) from a 429 body (Gemini: google.rpc QuotaFailure and RetryInfo)."""
    details = getattr(exc, "details", None)
    text = json.dumps(details) if isinstance(details, dict | list) else str(exc)
    qid, val, delay = _QUOTA_ID.search(text), _QUOTA_VALUE.search(text), _RETRY_DELAY.search(text)
    return (
        qid.group(1) if qid else None,
        int(val.group(1)) if val else None,
        float(delay.group(1)) if delay else None,
    )


def is_daily_quota(quota_id: str | None, message: str) -> bool:
    """Gemini names the quota (e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier); Groq says 'per day (TPD)'."""
    return bool(quota_id and "perday" in quota_id.lower()) or "per day" in message.lower()


def seconds_until_daily_reset(provider: str, now: dt.datetime | None = None) -> float:
    """Gemini resets requests-per-day quotas at midnight Pacific time; other providers use rolling windows (15 min)."""
    if provider != "gemini":
        return 900.0
    tz = ZoneInfo("America/Los_Angeles")
    local = (now or dt.datetime.now(dt.timezone.utc)).astimezone(tz)
    reset = dt.datetime.combine(local.date() + dt.timedelta(days=1), dt.time(0, 0), tzinfo=tz)
    return reset.timestamp() - local.timestamp()  # not reset - local: same-tzinfo subtraction ignores DST shifts


class StopRun(Exception):
    """Stop the run cleanly; rerunning the same command resumes it."""

    def __init__(self, message: str, exit_code: int = 0):
        super().__init__(message)
        self.exit_code = exit_code


def call_with_retries(
    call: Callable[[str, str, int], dict],
    p: dict,
    seed: int,
    provider: str,
    quota: Quota,
    wait_daily: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict, str, float, int]:
    """One call with the retry policy in the module docstring: (result, started_utc, latency_s, n_retries).

    started_utc and latency_s describe the successful attempt only. Raises StopRun to end the run.
    """
    attempt, waited = 0, 0.0
    while True:
        started, t0 = utcnow(), time.time()
        try:
            return call(p["system"], p["user"], seed), started, time.time() - t0, attempt
        except Exception as e:
            code, msg = status_of(e), str(e)
            if code in (401, 403):
                raise StopRun(f"Auth error ({code}); check the API key in .env. Stopping.", 1) from e
            if code == 400:
                raise StopRun(f"Request rejected (400: {msg[:300]}). Stopping; fix the request first.", 1) from e
            qid, qval, delay = quota_violation(e) if code == 429 else (None, None, None)
            if code == 429 and is_daily_quota(qid, msg):
                quota.state["server_daily_quota"] = {"quota_id": qid, "limit": qval, "hit_utc": utcnow()}
                quota.save()
                if not wait_daily:
                    raise StopRun(f"Server daily quota reached ({qid}, limit {qval}). Rerun later to resume.") from e
                secs = seconds_until_daily_reset(provider) + 120
                print(
                    f"  daily quota reached ({qid}, limit {qval}); sleeping {secs / 3600:.1f} h until reset", flush=True
                )
                sleep(secs)
                continue
            attempt += 1
            wait = max(delay + 1 if delay else 0, min(300, 10 * 2 ** min(attempt - 1, 5)))
            if waited + wait > MAX_WAIT_PER_CALL_S:
                raise StopRun(
                    f"Still failing after {waited / 3600:.1f} h of retries ({code}: {msg[:160]}); stopping "
                    "(rerun later to resume).",
                    1,
                ) from e
            print(f"  call {p['key']} failed ({code}: {msg[:120]}); retry {attempt} in {wait:.0f}s", flush=True)
            sleep(wait)
            waited += wait


# ----------------------------------------------------------------------------- provider adapters
def make_caller(m: dict):
    """Return call(system, user, seed) -> dict, plus a dict of the parameters actually sent."""
    provider, sent = m["provider"], {"max_tokens": m["max_tokens"], **m.get("sampling", {})}

    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000)
        )
        thinking = dict(m.get("thinking_config") or {})
        if thinking:
            sent["thinking_config"] = thinking

        def call(system: str, user: str, seed: int) -> dict:
            kw = dict(system_instruction=system, max_output_tokens=m["max_tokens"], seed=seed, **m.get("sampling", {}))
            if "thinking_config" in sent:
                kw["thinking_config"] = types.ThinkingConfig(**sent["thinking_config"])
            if m.get("response_mime_type"):  # JSON mode: guarantees syntactically valid JSON (pilot decision)
                kw["response_mime_type"] = m["response_mime_type"]
                sent["response_mime_type"] = m["response_mime_type"]
            # No silent fallback if the thinking level is rejected: dropping it would switch to the default (dynamic)
            # thinking mid-run. A 400 stops the run instead (call_with_retries).
            r = client.models.generate_content(
                model=m["model_id"], contents=user, config=types.GenerateContentConfig(**kw)
            )
            u = r.usage_metadata
            cand = (r.candidates or [None])[0]
            return dict(
                text=r.text or "",
                finish_reason=str(getattr(cand, "finish_reason", "") or ""),
                n_prompt_tokens=getattr(u, "prompt_token_count", None),
                n_output_tokens=getattr(u, "candidates_token_count", None),
                n_reasoning_tokens=getattr(u, "thoughts_token_count", None),
                model_version_returned=getattr(r, "model_version", None),
            )

        return call, sent

    if provider in ("groq", "openai"):
        if provider == "groq":
            from groq import Groq

            client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=REQUEST_TIMEOUT_S)
        else:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=REQUEST_TIMEOUT_S)
        if m.get("reasoning_effort"):
            sent["reasoning_effort"] = m["reasoning_effort"]

        def call(system: str, user: str, seed: int) -> dict:
            kw = dict(
                model=m["model_id"],
                max_completion_tokens=m["max_tokens"],
                seed=seed,
                **m.get("sampling", {}),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            if "reasoning_effort" in sent:
                kw["reasoning_effort"] = sent["reasoning_effort"]
            try:
                r = client.chat.completions.create(**kw)
            except Exception as e:  # unsupported reasoning_effort value: drop it once, log it, retry
                if "reasoning_effort" in sent and status_of(e) == 400 and "reasoning" in str(e).lower():
                    print("  note: model rejected reasoning_effort; continuing without it (recorded in logs)")
                    sent.pop("reasoning_effort")
                    return call(system, user, seed)
                raise
            ch, u = r.choices[0], r.usage
            details = getattr(u, "completion_tokens_details", None)
            return dict(
                text=ch.message.content or "",
                finish_reason=ch.finish_reason,
                n_prompt_tokens=getattr(u, "prompt_tokens", None),
                n_output_tokens=getattr(u, "completion_tokens", None),
                n_reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
                model_version_returned=getattr(r, "model", None),
                system_fingerprint=getattr(r, "system_fingerprint", None),
            )

        return call, sent

    raise ValueError(f"unknown provider {provider!r}")


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--dry-run", action="store_true", help="show the plan; make no API calls")
    ap.add_argument("--allow-disabled", action="store_true", help="run a model marked enabled: false (paid)")
    ap.add_argument("--wait-daily", action="store_true", help="sleep through daily quota resets instead of stopping")
    args = ap.parse_args()

    models = load_models()
    if args.model not in models or models[args.model]["runner"] != "api":
        print("API models:", [k for k, v in models.items() if v["runner"] == "api"])
        return 2
    m = models[args.model]
    if not m.get("enabled", True) and not args.allow_disabled:
        print(f"{args.model} is disabled in configs/models.yaml (paid). Use --allow-disabled only with the user's OK.")
        return 2

    prompts = select_mode(build_prompt_set(load_regions(), load_templates(PROMPT_SET_VERSION)), args.mode)
    if args.max_calls:
        prompts = prompts[: args.max_calls]
    out_dir = REPO / "data" / "generated" / "raw" / args.model
    out_path = out_dir / f"{args.mode}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                done.add(json.loads(line)["call_id"])
    todo = [p for p in prompts if call_id(args.model, p["key"]) not in done]
    lim = m.get("rate_limits", {})
    per_day = min(
        x
        for x in [
            lim.get("rpd") or 10**9,
            (lim.get("tpd") or 10**12) // (EST_OUTPUT_TOKENS_PER_CALL + EST_PROMPT_TOKENS_PER_CALL),
        ]
    )
    print(
        f"{args.model} ({m['provider']}:{m['model_id']}) mode={args.mode}: "
        f"{len(prompts)} calls, {len(done)} done, {len(todo)} to run"
    )
    print(
        f"limits {lim} -> about {min(per_day, len(todo))} calls/day, "
        f"so about {len(todo) / max(per_day, 1):.1f} day(s) of quota"
    )
    if args.dry_run or not todo:
        return 0

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    key_name = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}[m["provider"]]
    if not os.environ.get(key_name):
        print(f"Missing {key_name} in .env; add it (see .env.example). Nothing was called.")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    quota = Quota(lim, out_dir / "_quota_state.json", m["provider"])
    call, sent = make_caller(m)
    n_ok, exit_code = 0, 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            while not quota.daily_left():
                if not args.wait_daily:
                    print(f"Local daily limit reached after {n_ok} calls today. Run again tomorrow to resume.")
                    break
                secs = seconds_until_daily_reset(m["provider"]) + 120
                print(f"  local daily limit reached; sleeping {secs / 3600:.1f} h", flush=True)
                time.sleep(secs)
            if not quota.daily_left():
                break
            quota.wait(EST_OUTPUT_TOKENS_PER_CALL + EST_PROMPT_TOKENS_PER_CALL)
            seed = seed_for(args.model, p["key"])
            try:
                res, started, latency, n_retries = call_with_retries(
                    call, p, seed, m["provider"], quota, args.wait_daily
                )
            except StopRun as e:
                print(e, flush=True)
                exit_code = e.exit_code
                break
            used = (
                (res.get("n_prompt_tokens") or 0)
                + (res.get("n_output_tokens") or 0)
                + (res.get("n_reasoning_tokens") or 0)
            )
            quota.record(used)
            rec = {
                "call_id": call_id(args.model, p["key"]),
                "prompt_key": p["key"],
                "region_code": p["region_code"],
                "region_label": p["region_label"],
                "condition": p["condition"],
                "rep": p["rep"],
                "place": p["place"],
                "system": p["system"],
                "user": p["user"],
                "seed": seed,
                "output_text": res.pop("text"),
                **res,
                "model_key": args.model,
                "provider": m["provider"],
                "model_id": m["model_id"],
                "family": m["family"],
                "params_sent": dict(sent),
                "prompt_set_version": PROMPT_SET_VERSION,
                "runner_version": RUNNER_VERSION,
                "started_utc": started,
                "latency_s": round(latency, 2),
                "n_retries": n_retries,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += 1
            print(
                f"[{i}/{len(todo)}] {p['key']} ok ({rec.get('n_output_tokens')} out tokens, {rec['latency_s']}s)",
                flush=True,
            )
    print(f"Done for now: {n_ok} new calls -> {out_path.relative_to(REPO)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

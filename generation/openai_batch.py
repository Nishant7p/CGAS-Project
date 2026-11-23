"""Generate LLMCuisine-22 recipes with OpenAI models through the Batch API (half price, results within 24 h).

Same prompts, call ids, seeds and record format as the Kaggle notebook and run_api.py. A spend ledger
(data/generated/spend_ledger.json) plus BUDGET_USD in .env stop any submission whose projected total
would exceed the cap.

Usage (repo root):
    .venv/bin/python generation/openai_batch.py submit --model openai-gpt-6.2-sol --mode pilot --dry-run
    .venv/bin/python generation/openai_batch.py submit --model openai-gpt-6.2-sol --mode pilot
    .venv/bin/python generation/openai_batch.py status --model openai-gpt-6.2-sol --mode pilot
    .venv/bin/python generation/openai_batch.py fetch  --model openai-gpt-6.2-sol --mode pilot
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generation.common import PROMPT_SET_VERSION, build_prompt_set, call_id, seed_for, select_mode  # noqa: E402
from generation.config import load_models, load_regions, load_templates  # noqa: E402

RUNNER_VERSION = "openai-batch-v1"
EST_PROMPT_TOKENS = 250
EST_OUTPUT_TOKENS = 2900  # about 5 recipes x 520 tokens, plus low-effort reasoning; the pilot measures the truth
LEDGER = REPO / "data" / "generated" / "spend_ledger.json"


STANDARD_MODES = ("pilot", "full", "sens")


def raw_dir(model_key: str, mode: str = "full") -> Path:
    """D1 logs go to data/generated/raw; D3 steering modes (custom prompt files) to data/interim/steering/raw."""
    base = ("data", "generated", "raw") if mode in STANDARD_MODES else ("data", "interim", "steering", "raw")
    return REPO.joinpath(*base) / model_key


def log_path(model_key: str, mode: str) -> Path:
    name = f"{mode}.jsonl" if mode in STANDARD_MODES else f"llmcuisine22_{model_key}_{mode}.jsonl"
    return raw_dir(model_key, mode) / name


def build_requests(
    model_key: str, m: dict, mode: str, max_calls: int | None = None, prompts: list[dict] | None = None
) -> list[dict]:
    if prompts is None:
        prompts = select_mode(build_prompt_set(load_regions(), load_templates(PROMPT_SET_VERSION)), mode)
    prompts = prompts[:max_calls] if max_calls else prompts
    reqs = []
    for p in prompts:
        body = {
            "model": m["model_id"],
            "messages": [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}],
            "max_completion_tokens": m["max_tokens"],
            "seed": seed_for(model_key, p["key"]),
            **m.get("sampling", {}),
        }
        if m.get("reasoning_effort"):
            body["reasoning_effort"] = m["reasoning_effort"]
        reqs.append(
            {
                "custom_id": call_id(model_key, p["key"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
                "_prompt": p,
            }
        )
    return reqs


def estimate_cost(m: dict, n_calls: int, batch: bool = True) -> float:
    price = m["price_per_m"]
    factor = 0.5 if batch else 1.0
    return factor * n_calls * (EST_PROMPT_TOKENS * price["input"] + EST_OUTPUT_TOKENS * price["output"]) / 1e6


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(e.get("actual_usd") or e.get("estimated_usd", 0.0) for e in json.loads(LEDGER.read_text()))


def ledger_add(entry: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    entries.append(entry)
    LEDGER.write_text(json.dumps(entries, indent=1))


def client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(REPO / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY in .env (see .env.example). Nothing was sent.")
    return OpenAI()


def cmd_submit(args, m: dict) -> int:
    d = raw_dir(args.model, args.mode)
    state_path = d / f"_batch_{args.mode}.json"
    if state_path.exists() and not args.force:
        print(f"A batch already exists for {args.model}/{args.mode}: {json.loads(state_path.read_text())['batch_id']}")
        print("Use `status` / `fetch`, or --force to submit another (only for calls not yet logged).")
        return 1
    custom = None
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as fh:
            custom = [json.loads(x) for x in fh if x.strip()]
    reqs = build_requests(args.model, m, args.mode, args.max_calls, prompts=custom)
    log = log_path(args.model, args.mode)
    done = {json.loads(x)["call_id"] for x in log.read_text().splitlines() if x.strip()} if log.exists() else set()
    reqs = [r for r in reqs if r["custom_id"] not in done]
    est = estimate_cost(m, len(reqs))
    from dotenv import dotenv_values

    budget = float(dotenv_values(REPO / ".env").get("BUDGET_USD") or os.environ.get("BUDGET_USD") or 25)
    spent = ledger_total()
    print(
        f"{args.model} ({m['model_id']}) {args.mode}: {len(reqs)} requests, estimated ${est:.2f} with Batch "
        f"(ledger so far ${spent:.2f}; budget ${budget:.2f})"
    )
    if not reqs:
        print("Nothing to submit.")
        return 0
    if spent + est > budget:
        print("REFUSED: this would exceed BUDGET_USD. Raise it in .env only with the user's OK.")
        return 2
    if args.dry_run:
        print("dry run: first request body:", json.dumps(reqs[0]["body"])[:600])
        return 0
    cl = client()
    buf = io.BytesIO("".join(json.dumps({k: v for k, v in r.items() if k != "_prompt"}) + "\n" for r in reqs).encode())
    buf.name = f"llmcuisine22_{args.model}_{args.mode}.jsonl"
    f = cl.files.create(file=buf, purpose="batch")
    b = cl.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "LLMCuisine-22", "model_key": args.model, "mode": args.mode},
    )
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "batch_id": b.id,
        "input_file_id": f.id,
        "n_requests": len(reqs),
        "estimated_usd": round(est, 4),
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "prompts": {r["custom_id"]: r["_prompt"] for r in reqs},
        "params_sent": {k: v for k, v in reqs[0]["body"].items() if k not in ("messages", "seed")},
        "label": Path(args.prompts).stem if args.prompts else PROMPT_SET_VERSION,
    }
    state_path.write_text(json.dumps(state))
    ledger_add(
        {
            "model_key": args.model,
            "mode": args.mode,
            "batch_id": b.id,
            "estimated_usd": round(est, 4),
            "submitted_utc": state["submitted_utc"],
        }
    )
    print(f"Submitted batch {b.id} ({len(reqs)} requests). Check with: status --model {args.model} --mode {args.mode}")
    return 0


def cmd_status(args, m: dict) -> int:
    state = json.loads((raw_dir(args.model, args.mode) / f"_batch_{args.mode}.json").read_text())
    b = client().batches.retrieve(state["batch_id"])
    rc = b.request_counts
    print(f"{state['batch_id']}: {b.status} | completed {rc.completed}/{rc.total}, failed {rc.failed}")
    return 0


def cmd_fetch(args, m: dict) -> int:
    d = raw_dir(args.model, args.mode)
    state = json.loads((d / f"_batch_{args.mode}.json").read_text())
    cl = client()
    b = cl.batches.retrieve(state["batch_id"])
    if b.status != "completed":
        print(f"Batch status is {b.status}; fetch again later.")
        return 1
    log = log_path(args.model, args.mode)
    done = {json.loads(x)["call_id"] for x in log.read_text().splitlines() if x.strip()} if log.exists() else set()
    lines = cl.files.content(b.output_file_id).text.splitlines()
    n_new, in_tok, out_tok = 0, 0, 0
    with log.open("a", encoding="utf-8") as fh:
        for line in lines:
            item = json.loads(line)
            cid = item["custom_id"]
            if cid in done or item.get("error") or (item.get("response") or {}).get("status_code") != 200:
                continue
            body = item["response"]["body"]
            p, ch, u = state["prompts"][cid], body["choices"][0], body.get("usage") or {}
            details = u.get("completion_tokens_details") or {}
            in_tok += u.get("prompt_tokens") or 0
            out_tok += u.get("completion_tokens") or 0
            rec = {
                "call_id": cid,
                "prompt_key": p["key"],
                "region_code": p["region_code"],
                "region_label": p["region_label"],
                "condition": p["condition"],
                "rep": p["rep"],
                "place": p["place"],
                "system": p["system"],
                "user": p["user"],
                "seed": seed_for(args.model, p["key"]),
                "output_text": ch["message"].get("content") or "",
                "finish_reason": ch.get("finish_reason"),
                "n_prompt_tokens": u.get("prompt_tokens"),
                "n_output_tokens": u.get("completion_tokens"),
                "n_reasoning_tokens": details.get("reasoning_tokens"),
                "model_version_returned": body.get("model"),
                "system_fingerprint": body.get("system_fingerprint"),
                "model_key": args.model,
                "provider": "openai",
                "model_id": m["model_id"],
                "family": m["family"],
                "params_sent": state["params_sent"],
                "prompt_set_version": state.get("label", PROMPT_SET_VERSION),
                "runner_version": RUNNER_VERSION,
                "batch_id": state["batch_id"],
                **({"meta": p["meta"]} if "meta" in p else {}),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_new += 1
    actual = 0.5 * (in_tok * m["price_per_m"]["input"] + out_tok * m["price_per_m"]["output"]) / 1e6
    ledger_add(
        {
            "model_key": args.model,
            "mode": args.mode,
            "batch_id": state["batch_id"],
            "actual_usd": round(actual, 4),
            "estimated_usd": 0.0,
            "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
    )
    print(
        f"Fetched {n_new} new records -> {log.relative_to(REPO)} | actual cost about ${actual:.2f} "
        f"({in_tok} in / {out_tok} out tokens)"
    )
    if b.error_file_id:
        print("Some requests failed; error file:", b.error_file_id, "(resubmit with `submit --force`).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["submit", "status", "fetch"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="pilot", help="pilot | full, or a D3 steering mode name with --prompts")
    ap.add_argument("--prompts", help="JSONL prompt file for a steering mode (e.g. steering/prompt_sets/...)")
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    models = load_models()
    m = models.get(args.model)
    if not m or m.get("provider") != "openai":
        print("OpenAI models:", [k for k, v in models.items() if v.get("provider") == "openai"])
        return 2
    return {"submit": cmd_submit, "status": cmd_status, "fetch": cmd_fetch}[args.command](args, m)


if __name__ == "__main__":
    raise SystemExit(main())

"""OpenAI Batch helpers: request bodies, cost estimate, budget guard (no network)."""

import json

from generation import openai_batch as ob
from generation.common import call_id
from generation.config import load_models


def test_build_requests_bodies_match_prompt_set():
    m = load_models()["openai-gpt-6.2-sol"]
    reqs = ob.build_requests("openai-gpt-6.2-sol", m, "pilot")
    assert len(reqs) == 44
    r = reqs[0]
    assert r["url"] == "/v1/chat/completions" and r["method"] == "POST"
    assert r["custom_id"] == call_id("openai-gpt-6.2-sol", r["_prompt"]["key"])
    b = r["body"]
    assert b["model"] == "gpt-6.2-sol" and b["max_completion_tokens"] == 6144
    assert b["reasoning_effort"] == "low" and "temperature" not in b
    assert [msg["role"] for msg in b["messages"]] == ["system", "user"]
    json.dumps({k: v for k, v in r.items() if k != "_prompt"})  # serializable for the batch file


def test_cost_estimates_are_sane():
    models = load_models()
    sol, luna = models["openai-gpt-6.2-sol"], models["openai-gpt-6-luna"]
    assert 5 < ob.estimate_cost(sol, 704) < 15  # about $10 for 3,520 recipes with Batch
    assert ob.estimate_cost(luna, 704) < 1
    assert ob.estimate_cost(sol, 44) < 1  # the pilot costs well under a dollar
    assert abs(ob.estimate_cost(sol, 10, batch=False) - 2 * ob.estimate_cost(sol, 10)) < 1e-9


def test_ledger_total(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "LEDGER", tmp_path / "ledger.json")
    assert ob.ledger_total() == 0.0
    ob.ledger_add({"estimated_usd": 0.7})
    ob.ledger_add({"actual_usd": 0.5, "estimated_usd": 0.0})
    assert abs(ob.ledger_total() - 1.2) < 1e-9

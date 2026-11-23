"""Tests for the shared generation helpers (prompt set, ids/seeds, robust output parsing)."""

from collections import Counter

import pytest

from generation.common import (
    build_prompt_set,
    call_id,
    extract_json,
    seed_for,
    select_mode,
    validate_recipes,
)
from generation.config import load_models, load_regions, load_templates


@pytest.fixture(scope="module")
def prompt_set():
    return build_prompt_set(load_regions(), load_templates("v1"))


def test_prompt_set_shape_and_uniqueness(prompt_set):
    assert len(prompt_set) == 22 * 2 * 16
    assert len({p["key"] for p in prompt_set}) == len(prompt_set)
    assert {p["region_code"] for p in prompt_set} == set(load_regions())


def test_prompt_set_is_deterministic(prompt_set):
    again = build_prompt_set(load_regions(), load_templates("v1"))
    assert again == prompt_set


def test_persona_places_are_balanced_and_plain_has_none(prompt_set):
    for code in load_regions():
        persona = [p for p in prompt_set if p["region_code"] == code and p["condition"] == "persona"]
        counts = Counter(p["place"] for p in persona)
        assert len(counts) == 8 and set(counts.values()) == {2}
        plain = [p for p in prompt_set if p["region_code"] == code and p["condition"] == "plain"]
        assert all(p["place"] is None for p in plain)


def test_prompt_text_contains_region_and_format(prompt_set):
    ita_plain = next(p for p in prompt_set if p["region_code"] == "ITA" and p["condition"] == "plain")
    assert "traditional Italian recipes" in ita_plain["user"]
    assert '"recipes"' in ita_plain["user"] and "exactly 5 recipes" in ita_plain["user"]
    assert "{" not in ita_plain["system"]
    kor_persona = next(p for p in prompt_set if p["region_code"] == "KOR" and p["condition"] == "persona")
    assert "South Korea" in kor_persona["user"] and "home cook" in kor_persona["user"]
    # No prompt may hint at the hypothesis under test.
    for p in prompt_set:
        assert "flavor" not in p["user"].lower() and "pairing" not in p["user"].lower()


def test_pilot_selection(prompt_set):
    pilot = select_mode(prompt_set, "pilot")
    assert len(pilot) == 44 and all(p["rep"] == 0 for p in pilot)
    with pytest.raises(ValueError):
        select_mode(prompt_set, "huge")


def test_ids_and_seeds_are_deterministic_and_model_specific():
    k = "v1|KOR|plain|00"
    assert call_id("qwen3.5-9b", k) == call_id("qwen3.5-9b", k)
    assert call_id("qwen3.5-9b", k) != call_id("llama-3.1-8b", k)
    s = seed_for("qwen3.5-9b", k)
    assert s == seed_for("qwen3.5-9b", k) and 0 <= s < 2**31


GOOD = (
    '{"recipes": [{"title": "Bibimbap", "ingredients": ["2 cups rice", "1 tbsp gochujang"], '
    '"instructions": ["Cook rice."]}]}'
)


@pytest.mark.parametrize(
    "text,status",
    [
        (GOOD, "ok"),
        ("```json\n" + GOOD + "\n```", "ok"),
        ("Sure! Here are the recipes:\n" + GOOD + "\nEnjoy!", "ok"),
        ("<think>\nlet me think\n</think>\n\n" + GOOD, "ok"),
        (GOOD.replace('"]}]}', '",]},]}'), "repaired"),
    ],
)
def test_extract_json_variants(text, status):
    obj, st = extract_json(text)
    assert st == status
    recs, _ = validate_recipes(obj)
    assert recs[0]["title"] == "Bibimbap" and len(recs[0]["ingredients"]) == 2


@pytest.mark.parametrize(
    "tail",
    [
        '"serve immediately."}]}',  # pilot: Gemini omitted the ']' closing the last instructions list
        '"slice."}]}]}',  # pilot: missing ']' plus a stray "]}" at the end
        '"serve warm."}]}\n]}',
    ],
)
def test_extract_json_balances_gemini_brackets(tail):
    head = '{"recipes": [' + ", ".join(
        f'{{"title": "R{i}", "ingredients": ["1 egg", "salt"], "instructions": ["Mix."]}}' for i in range(4)
    )
    text = head + ', {"title": "R4", "ingredients": ["2 eggs", "milk"], "instructions": ["Cook.", ' + tail
    obj, st = extract_json(text)
    assert st == "repaired"
    recs, issues = validate_recipes(obj)
    assert [r["title"] for r in recs] == ["R0", "R1", "R2", "R3", "R4"] and issues == []
    assert recs[4]["instructions"][-1].startswith("serve") or recs[4]["instructions"][-1] == "slice."


def test_extract_json_closes_unclosed_recipe_objects():
    # Pilot (Qwen): '..."last step."], {"title": ...' - the first recipe object was never closed.
    text = (
        '{"recipes": [{"title": "Fiskesuppe", "ingredients": ["1 lb cod", "milk"], "instructions": ["Simmer."],'
        ' {"title": "Frikadeller", "ingredients": ["5 oz fish", "egg"], "instructions": ["Fry."]}]}'
    )
    obj, st = extract_json(text)
    assert st == "repaired"
    recs, _ = validate_recipes(obj)
    assert [r["title"] for r in recs] == ["Fiskesuppe", "Frikadeller"]


@pytest.mark.parametrize(
    "boundary",
    [
        '"]}, "title"',  # Gemini full run: the next recipe's "{" is missing
        '"]]}, {"title"',  # an extra "]"
        '"]}}, {"title"',  # an extra "}"
        '"],\n  {\n    "title"',  # Qwen pilot: the recipe's "}" is missing
    ],
)
def test_extract_json_normalizes_slipped_recipe_boundaries(boundary):
    text = (
        '{"recipes": [{"title": "A", "ingredients": ["1 egg", "salt"], "instructions": ["Mix.'
        + boundary
        + ': "B", "ingredients": ["2 eggs", "milk"], "instructions": ["Fry."]}]}'
    )
    obj, st = extract_json(text)
    assert st == "repaired"
    assert [r["title"] for r in validate_recipes(obj)[0]] == ["A", "B"]


@pytest.mark.parametrize("tail", ["]}", "]}]}]", ""])
def test_extract_json_reads_recipes_written_after_an_early_close(tail):
    # Gemini full run (JSON mode): the document is closed after recipe 1 and recipes 2..5 follow as bare objects.
    rec = '{{"title": "R{i}", "ingredients": ["1 egg", "salt"], "instructions": ["Mix ] and }} stir."]}}'
    text = '{"recipes": [' + rec.format(i=0) + "]}, " + ", ".join(rec.format(i=i) for i in range(1, 5)) + tail
    obj, st = extract_json(text)
    assert st == "repaired"
    recs, issues = validate_recipes(obj)
    assert [r["title"] for r in recs] == ["R0", "R1", "R2", "R3", "R4"] and issues == []
    assert recs[3]["instructions"] == ["Mix ] and } stir."]  # brackets inside strings are untouched


def test_balance_brackets_never_touches_string_content():
    from generation.common import balance_brackets

    s = '{"a": ["x ] y } z"'
    assert balance_brackets(s) == '{"a": ["x ] y } z"]}'


def test_extract_json_salvages_truncated_reply():
    truncated = (
        '{"recipes": [{"title": "A", "ingredients": ["1 egg", "salt"], "instructions": ["x"]}, '
        '{"title": "B", "ingredients": ["2 eggs", "milk"], "instructions": ["y"]}, {"title": "C", "ingred'
    )
    obj, st = extract_json(truncated)
    assert st in ("repaired", "salvaged")  # bracket balancing or object salvage: both recover the complete recipes
    recs, issues = validate_recipes(obj)
    assert [r["title"] for r in recs] == ["A", "B"]
    assert any("expected 5" in i for i in issues)


@pytest.mark.parametrize("text,status", [("", "empty"), ("   ", "empty"), ("I cannot help with that.", "unparseable")])
def test_extract_json_failures(text, status):
    obj, st = extract_json(text)
    assert obj is None and st == status


def test_validate_normalizes_dict_ingredients_and_string_steps():
    obj = {
        "recipes": [
            {
                "name": "Kimchi Jjigae",
                "ingredients": [{"quantity": "1 cup", "item": "kimchi"}, {"name": "pork belly", "amount": "200 g"}],
                "instructions": "Fry pork.\nAdd kimchi.\nSimmer.",
            }
        ]
    }
    recs, issues = validate_recipes(obj)
    assert recs[0]["title"] == "Kimchi Jjigae"
    assert recs[0]["ingredients"] == ["1 cup kimchi", "200 g pork belly"]
    assert recs[0]["instructions"] == ["Fry pork.", "Add kimchi.", "Simmer."]
    assert issues == ["expected 5 recipes, got 1"]


def test_validate_rejects_incomplete_recipes():
    recs, issues = validate_recipes(
        {"recipes": [{"title": "", "ingredients": ["a", "b"]}, {"title": "X", "ingredients": ["a"]}]}
    )
    assert recs == [] and len(issues) == 3


def test_planned_roster_meets_the_proposal_cheaply():
    """D1 promises >= 10,000 recipes from >= 3 LLMs: >= 1 proprietary and >= 2 open-weight families.

    User decisions (2026-11-22): keep scope lean; use OpenAI as little as possible (the core is free).
    """
    from generation.openai_batch import estimate_cost

    planned = {k: m for k, m in load_models().items() if m["tier"] in ("core", "extra")}
    core = {k: m for k, m in planned.items() if m["tier"] == "core"}
    assert set(core) == {"llama-3.1-8b", "qwen3.5-9b", "gemini-3.5-flash-lite"}
    assert len({m["family"] for m in core.values() if m["runner"] == "kaggle"}) >= 2
    assert any(m.get("provider") == "gemini" for m in core.values())  # free proprietary model
    assert not any(m.get("provider") == "openai" for m in core.values())  # the proposal is met at $0
    assert len(core) * 3520 >= 10_000
    assert len(planned) <= 6
    openai_spend = sum(estimate_cost(m, 704) for m in planned.values() if m.get("provider") == "openai")
    assert openai_spend < 1.0  # only GPT-6-Luna is planned (about $0.5)


def test_model_registry_is_consistent():
    models = load_models()
    for key, m in models.items():
        assert m["runner"] in ("kaggle", "api"), key
        assert m["max_tokens"] == 6144
        if m["runner"] == "kaggle":
            assert m["hf_id"] and "sampling" in m
        else:
            assert m["provider"] in ("gemini", "groq", "openai") and m["model_id"]
    assert sum(1 for m in models.values() if m["runner"] == "kaggle") >= 2

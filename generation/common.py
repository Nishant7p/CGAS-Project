"""Shared helpers for LLMCuisine-22 generation (D1).

Pure standard library on purpose: this file is used by the Mac-side API runner and is also
inlined verbatim into the Kaggle notebook, so both paths build identical prompts, call ids and
seeds, and parse outputs the same way.

Design (see docs/IMPLEMENTATION_PLAN.md, Phase 4):
* 22 regions x 2 conditions (plain, persona) x n_reps replicates. Each call asks for 5 recipes.
  The full run uses 16 replicates (704 calls, 3,520 recipes per model). The pilot uses replicate 0
  only (44 calls).
* Prompts are model-agnostic and matched across models (a paired design). The persona place
  rotates through the region's 8 places so each is used equally often.
* call_id and seed are deterministic functions of (model_key, prompt key).
"""

from __future__ import annotations

import hashlib
import json
import re

PROMPT_SET_VERSION = "v1"
CONDITIONS = ("plain", "persona")
N_REPS_FULL = 16
# Pre-registered decoding-sensitivity subset (plan §5, Phase 4 step 4), open-weight models only: replicate 0 of every
# region x condition again at a common T = 1.0, top_p = 1.0. Same prompts, call ids and seeds as the main run (a
# paired design); parse_outputs keeps it apart through its mode ("sens").
SENSITIVITY_SAMPLING = {"temperature": 1.0, "top_p": 1.0}
SENSITIVITY_REPS = 1


def prompt_key(version: str, region_code: str, condition: str, rep: int) -> str:
    return f"{version}|{region_code}|{condition}|{rep:02d}"


def build_prompt_set(
    regions: dict, templates: dict, n_reps: int = N_REPS_FULL, version: str = PROMPT_SET_VERSION
) -> list[dict]:
    """Build the model-agnostic prompt set.

    regions: {code: {"label", "phrase", "places"}}; templates: {"system", "format", "plain", "persona"}.
    """
    fmt = templates["format"].strip()
    system = templates["system"].strip()
    out = []
    for code in sorted(regions):
        reg = regions[code]
        places = list(reg["places"])
        # Region-specific rotation offset so rep 0 does not always use the first-listed place.
        offset = int(hashlib.sha256(code.encode()).hexdigest(), 16) % len(places)
        for cond in CONDITIONS:
            for rep in range(n_reps):
                place = places[(rep + offset) % len(places)] if cond == "persona" else None
                user = templates[cond].format(phrase=reg["phrase"], place=place, format=fmt).strip()
                out.append(
                    {
                        "key": prompt_key(version, code, cond, rep),
                        "version": version,
                        "region_code": code,
                        "region_label": reg["label"],
                        "condition": cond,
                        "rep": rep,
                        "place": place,
                        "system": system,
                        "user": user,
                    }
                )
    return out


def select_mode(prompt_set: list[dict], mode: str) -> list[dict]:
    if mode == "pilot":
        return [p for p in prompt_set if p["rep"] == 0]
    if mode == "full":
        return list(prompt_set)
    if mode == "sens":
        return [p for p in prompt_set if p["rep"] < SENSITIVITY_REPS]
    raise ValueError(f"unknown mode {mode!r} (use 'pilot', 'full' or 'sens')")


def sampling_for(model_sampling: dict, mode: str) -> dict:
    """The decoding parameters of a run: the model's own, or the common sensitivity setting in mode 'sens'."""
    return dict(SENSITIVITY_SAMPLING) if mode == "sens" else dict(model_sampling)


def call_id(model_key: str, key: str) -> str:
    return hashlib.sha256(f"{model_key}|{key}".encode()).hexdigest()[:16]


def seed_for(model_key: str, key: str) -> int:
    return int(hashlib.sha256(f"seed|{model_key}|{key}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


# ---------------------------------------------------------------- output parsing
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks. If a closing tag exists without an opening tag, keep only what follows it."""
    t = _THINK_BLOCK.sub("", text or "")
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return t.strip()


def _repair(s: str) -> str:
    s = _TRAILING_COMMA.sub(r"\1", s)
    return s.replace("“", '"').replace("”", '"')


_RECIPE_BOUNDARY = re.compile(r'\][\s\]}]*,\s*\{?\s*("title"\s*:)')


def normalize_recipe_boundaries(s: str) -> str:
    """Rewrite every boundary between two recipes to the canonical '...], {"title": ...' + closing brace.

    A recipe's last field is a list, so a boundary is ']' + '}' + ',' + '{"title":'. Slips seen in the logs:
    Qwen pilot '], {"title"' (missing '}'); Gemini full run ']}, "title"' (missing '{'), ']]}, {"title"' and
    ']}}, {"title"' (an extra closer), and ']}]}, {"title"' (the document closed after the first recipe).
    Only tried after strict parsing fails.
    """
    return _RECIPE_BOUNDARY.sub(r"]}, {\1", s)


def balance_brackets(s: str) -> str:
    """Fix mismatched or missing JSON closers without touching any text inside strings.

    Seen in the pilot: Gemini sometimes omits the ']' that closes the last instructions list ('..."}]}').
    When a closer does not match the innermost open bracket, the missing closers are inserted; a stray closer with
    nothing open is dropped; open brackets left at the end are closed.
    """
    pairs = {"{": "}", "[": "]"}
    out, stack, in_str, esc = [], [], False, False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(ch)
        elif ch in "]}":
            if ch not in (pairs[b] for b in stack):
                continue  # stray closer
            while stack and pairs[stack[-1]] != ch:
                out.append(pairs[stack.pop()])
            stack.pop()
        out.append(ch)
    if in_str:
        out.append('"')
    out.extend(pairs[b] for b in reversed(stack))
    return "".join(out)


def _recipe_stream(text: str) -> list | None:
    """Recipes from a reply written as several JSON values instead of one document.

    Seen in the Gemini full run (JSON mode): the "recipes" list and the top-level object are closed after the first
    recipe and the remaining recipes follow as bare objects: {"recipes": [r1]}, r2, r3, r4, r5]}. Every value is
    decoded in turn; recipe lists and recipe-like objects (with "ingredients") are collected; separators and stray
    closing brackets between values are skipped. Nothing inside a value is changed.
    """
    dec = json.JSONDecoder()
    i, items, n_values = text.find("{"), [], 0
    while 0 <= i < len(text):
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        n_values += 1
        if isinstance(obj, dict) and isinstance(obj.get("recipes"), list):
            items.extend(obj["recipes"])
        elif isinstance(obj, dict) and "ingredients" in obj:
            items.append(obj)
        while i < len(text) and text[i] in " \r\n\t,]}":
            i += 1
        if i < len(text) and text[i] != "{":
            break
    return items if n_values > 1 and items else None


def _salvage_recipes(text: str) -> list | None:
    """Recover complete recipe objects from truncated JSON (e.g. the output hit max_tokens mid-recipe)."""
    m = re.search(r'"recipes"\s*:\s*\[', text)
    if not m:
        return None
    dec = json.JSONDecoder()
    i, items = m.end(), []
    while i < len(text):
        while i < len(text) and text[i] in " \r\n\t,":
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        items.append(obj)
        i = j
    return items or None


def extract_json(text: str) -> tuple[object | None, str]:
    """Return (parsed_object, status). Status is ok | repaired | salvaged | empty | unparseable."""
    if not text or not text.strip():
        return None, "empty"
    t = strip_reasoning(text)
    candidates = [mm.group(1) for mm in _FENCE.finditer(t)] + [t]
    for c in candidates:
        start, end = c.find("{"), c.rfind("}")
        if start == -1 or end <= start:
            continue
        s = c[start : end + 1]
        attempts = (
            ("ok", s),
            ("repaired", _repair(s)),
            ("repaired", balance_brackets(_repair(s))),
            ("repaired", balance_brackets(normalize_recipe_boundaries(_repair(s)))),
        )
        for status, attempt in attempts:
            try:
                return json.loads(attempt), status
            except json.JSONDecodeError:
                continue
    streamed = _recipe_stream(t)
    if streamed:
        return {"recipes": streamed}, "repaired"
    salvaged = _salvage_recipes(t)
    if salvaged:
        return {"recipes": salvaged}, "salvaged"
    return None, "unparseable"


def _ingredient_line(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        keys = ("quantity", "amount", "unit", "item", "ingredient", "name", "preparation", "note")
        return " ".join(str(item[k]).strip() for k in keys if item.get(k)).strip()
    return str(item).strip()


def validate_recipes(obj: object) -> tuple[list[dict], list[str]]:
    """Normalize a parsed reply into [{"title", "ingredients": [str], "instructions": [str]}] and list any issues."""
    issues: list[str] = []
    if isinstance(obj, dict):
        recs = obj.get("recipes")
        if recs is None and "ingredients" in obj:
            recs = [obj]
    elif isinstance(obj, list):
        recs = obj
    else:
        return [], ["reply is not a JSON object"]
    if not isinstance(recs, list):
        return [], ["no 'recipes' list"]
    good = []
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            issues.append(f"recipe {i}: not an object")
            continue
        title = str(r.get("title") or r.get("name") or r.get("dish") or "").strip()
        ings = r.get("ingredients")
        if isinstance(ings, str):
            ings = [x for x in ings.split("\n")]
        lines = [s for s in (_ingredient_line(x) for x in (ings or [])) if s] if isinstance(ings, list) else []
        steps = r.get("instructions") or r.get("steps") or r.get("method") or []
        if isinstance(steps, str):
            steps = re.split(r"\n+", steps)
        steps = [str(s).strip() for s in steps if str(s).strip()] if isinstance(steps, list) else []
        if not title or len(lines) < 2:
            issues.append(f"recipe {i}: missing title or fewer than 2 ingredients")
            continue
        good.append({"title": title, "ingredients": lines, "instructions": steps})
    if len(recs) != 5:
        issues.append(f"expected 5 recipes, got {len(recs)}")
    return good, issues

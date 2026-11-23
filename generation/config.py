"""Load the generation configs (regions, prompt templates, model registry) from configs/."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


def load_regions(path: Path = CONFIGS / "regions.yaml") -> dict:
    regions = yaml.safe_load(path.read_text(encoding="utf-8"))
    for code, reg in regions.items():
        if len(reg["places"]) != 8:
            raise ValueError(f"{code}: need exactly 8 persona places for balanced replicates")
    return regions


def load_templates(version: str = "v1") -> dict:
    d = CONFIGS / "prompts" / version
    return {name: (d / f"{name}.txt").read_text(encoding="utf-8") for name in ("system", "format", "plain", "persona")}


def load_models(path: Path = CONFIGS / "models.yaml") -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {})
    models = {}
    for key, m in cfg["models"].items():
        merged = {**defaults, **m}
        merged.setdefault("enabled", True)
        models[key] = merged
    return models

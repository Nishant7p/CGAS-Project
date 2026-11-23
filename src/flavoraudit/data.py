"""Loaders for CulinaryDB, FlavorDB (cached entity JSON) and the Ahn et al. (2011) ingredient-compound data.

The raw files are third-party data (CulinaryDB and FlavorDB: CC BY-NC-SA 3.0; Ahn et al.: CC BY 4.0) and are
never bundled with this MIT-licensed package. Point ``data_dir`` (or the ``FLAVORAUDIT_DATA`` environment
variable) at a folder with this layout::

    culinarydb/extracted/01_Recipe_Details.csv ... 04_Recipe-Ingredients_Aliases.csv
    flavordb_v1/<entity_id>.json.gz            (from scripts/fetch_flavordb.py)
    ahn2011/extracted/ingr_comp/{ingr_info,comp_info,ingr_comp}.tsv
"""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_CAS = re.compile(r"\d{2,7}-\d{2}-\d")


def default_data_dir() -> Path:
    env = os.environ.get("FLAVORAUDIT_DATA")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "data" / "raw"
        if (cand / "culinarydb").exists():
            return cand
    cand = Path.cwd() / "data" / "raw"
    if not (cand / "culinarydb").exists():
        raise FileNotFoundError(
            "FlavorAudit needs the CulinaryDB, FlavorDB and Ahn et al. (2011) data, which are not bundled (CulinaryDB "
            "and FlavorDB are CC BY-NC-SA 3.0). Put them in a folder (layout: culinarydb/, flavordb_v1/, ahn2011/; the "
            "project's scripts/ fetch them) and set FLAVORAUDIT_DATA=/path/to/folder or pass data_dir=..."
        )
    return cand


def _split(s: object, sep: str) -> list[str]:
    if not isinstance(s, str):
        return []
    return [x.strip() for x in s.split(sep) if x.strip()]


@dataclass
class CulinaryDB:
    """CulinaryDB (Singh & Bagler 2018): 45,772 recipes, 22 regions (+4 aggregate-only labels)."""

    recipes: pd.DataFrame  # recipe_id, title, source, cuisine
    ingredients: pd.DataFrame  # entity_id, name, synonyms[list], category, is_compound, constituents[list]
    recipe_entities: pd.DataFrame  # recipe_id, entity_id  (one row per distinct entity per recipe)
    aliases: pd.DataFrame  # recipe_id, original (raw line), aliased, entity_id  (gold aliasing data)

    @property
    def regions(self) -> list[str]:
        """The 22 analysis regions (labels starting with 'Misc.' are aggregate-only)."""
        return sorted(c for c in self.recipes["cuisine"].unique() if not str(c).startswith("Misc."))


def load_culinarydb(data_dir: str | Path | None = None) -> CulinaryDB:
    d = Path(data_dir or default_data_dir()) / "culinarydb" / "extracted"
    rec = pd.read_csv(d / "01_Recipe_Details.csv").rename(
        columns={"Recipe ID": "recipe_id", "Title": "title", "Source": "source", "Cuisine": "cuisine"}
    )
    basic = pd.read_csv(d / "02_Ingredients.csv").rename(
        columns={
            "Aliased Ingredient Name": "name",
            "Ingredient Synonyms": "synonyms",
            "Entity ID": "entity_id",
            "Category": "category",
        }
    )
    basic["is_compound"] = False
    basic["constituents"] = [[] for _ in range(len(basic))]
    comp = pd.read_csv(d / "03_Compound_Ingredients.csv").rename(
        columns={
            "Compound Ingredient Name": "name",
            "Compound Ingredient Synonyms": "synonyms",
            "Contituent Ingredients": "constituents",  # sic, as in the source file
            "Category": "category",
        }
    )
    comp["is_compound"] = True
    comp["constituents"] = comp["constituents"].map(lambda s: _split(s, ","))
    ing = pd.concat([basic, comp], ignore_index=True)
    ing["synonyms"] = ing["synonyms"].map(lambda s: _split(s, ";"))
    ing["entity_id"] = ing["entity_id"].astype(int)
    ali = pd.read_csv(d / "04_Recipe-Ingredients_Aliases.csv").rename(
        columns={
            "Recipe ID": "recipe_id",
            "Original Ingredient Name": "original",
            "Aliased Ingredient Name": "aliased",
            "Entity ID": "entity_id",
        }
    )
    ali["aliased"] = ali["aliased"].astype(str).str.strip()
    ali["entity_id"] = ali["entity_id"].astype(int)
    re_ = ali.drop_duplicates(["recipe_id", "entity_id"])[["recipe_id", "entity_id"]].reset_index(drop=True)
    return CulinaryDB(rec, ing[["entity_id", "name", "synonyms", "category", "is_compound", "constituents"]], re_, ali)


@dataclass
class FlavorDB:
    """FlavorDB (Garg et al. 2018; current content = FlavorDB2): entities and their flavor molecules."""

    entities: pd.DataFrame  # entity_id, name, category, synonyms, n_molecules
    molecules: pd.DataFrame  # pubchem_id, common_name, cas_ids[list]
    entity_molecules: dict[int, frozenset[int]]  # entity_id -> PubChem ids

    def cas_to_pubchem(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for pid, cas_list in zip(self.molecules["pubchem_id"], self.molecules["cas_ids"], strict=True):
            for cas in cas_list:
                out.setdefault(cas, int(pid))
        return out


def load_flavordb(data_dir: str | Path | None = None, folder: str = "flavordb_v1") -> FlavorDB:
    d = Path(data_dir or default_data_dir()) / folder
    ents, mols, emap = [], {}, {}
    for p in sorted(d.glob("*.json.gz"), key=lambda q: int(q.name.split(".")[0])):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            e = json.load(fh)
        eid = int(e["entity_id"])
        ms = e.get("molecules") or []
        pids = set()
        for m in ms:
            pid = int(m["pubchem_id"])
            pids.add(pid)
            if pid not in mols:
                mols[pid] = (m.get("common_name") or "", _CAS.findall(m.get("cas_id") or ""))
        emap[eid] = frozenset(pids)
        ents.append(
            {
                "entity_id": eid,
                "name": e.get("entity_alias_readable"),
                "category": e.get("category_readable"),
                "synonyms": e.get("entity_alias_synonyms"),
                "n_molecules": len(pids),
            }
        )
    mol_df = pd.DataFrame([{"pubchem_id": k, "common_name": v[0], "cas_ids": v[1]} for k, v in sorted(mols.items())])
    return FlavorDB(pd.DataFrame(ents), mol_df, emap)


@dataclass
class AhnData:
    """Ahn et al. (2011) flavor-network data: ingredients and their compounds identified by CAS number."""

    ingredients: pd.DataFrame  # ingredient_id, name, category
    compounds: pd.DataFrame  # compound_id, name, cas
    ingredient_cas: dict[str, frozenset[str]]  # ingredient name -> CAS numbers

    def profile(self, name: str) -> frozenset[str]:
        return self.ingredient_cas.get(name, frozenset())


def load_ahn(data_dir: str | Path | None = None) -> AhnData:
    d = Path(data_dir or default_data_dir()) / "ahn2011" / "extracted" / "ingr_comp"

    def read(name: str, cols: list[str]) -> pd.DataFrame:
        return pd.read_csv(d / name, sep="\t", comment="#", header=None, names=cols, dtype=str)

    ing = read("ingr_info.tsv", ["ingredient_id", "name", "category"])
    comp = read("comp_info.tsv", ["compound_id", "name", "cas"])
    links = read("ingr_comp.tsv", ["ingredient_id", "compound_id"])
    id2name = dict(zip(ing["ingredient_id"], ing["name"], strict=True))
    id2cas = dict(zip(comp["compound_id"], comp["cas"], strict=True))
    prof: dict[str, set[str]] = {}
    for i, c in zip(links["ingredient_id"], links["compound_id"], strict=True):
        cas = id2cas.get(c)
        if cas and i in id2name:
            prof.setdefault(id2name[i], set()).add(cas.strip())
    return AhnData(ing, comp, {k: frozenset(v) for k, v in prof.items()})

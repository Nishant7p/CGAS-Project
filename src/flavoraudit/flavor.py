"""Flavor profiles for CulinaryDB entities and the shared-compound matrix.

Reconstructs the ingredient -> flavor-molecule mapping described in Singh & Bagler (2018), Sec. III-B/C:

* CulinaryDB entity ids 0-978 are FlavorDB entity ids, so the profile is the FlavorDB molecule set.
* 13 "coarse-grained" ingredients (ids 979-991) take the profile of their FlavorDB parent (e.g. Lemon Juice -> Lemon).
* 4 ingredients (Cayenne, Yeast, Tequila, Sauerkraut; ids 992-995) come from Ahn et al. (2011). Their compounds
  (CAS numbers) are mapped to FlavorDB PubChem ids where possible and kept as separate CAS tokens otherwise.
* 7 additives (ids 996-1002) were "manually added": MSG -> L-glutamic acid, Citric Acid -> citric acid; Baking Powder
  has no FlavorDB flavor molecule; Cooking Spray, Gelatin, Food Coloring and Liquid Smoke have no profile (as in
  the paper).
* 103 compound ingredients (ids >= 2000): the union of their constituents' profiles.

Molecules are identified by string tokens, "P<pubchem>" or "CAS<number>", so both sources share one key space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .data import AhnData, CulinaryDB, FlavorDB

COARSE_PARENTS = {
    979: 323,  # Anise Oil -> Anise
    980: 162,  # Apple Juice -> Apple
    981: 172,  # Coconut Milk -> Coconut
    982: 172,  # Coconut Oil -> Coconut
    983: 9,  # Hops Beer -> Beer
    984: 240,  # Lemon Juice -> Lemon
    985: 55,  # Brown Rice -> Rice
    986: 364,  # Tomato Juice -> Tomato
    987: 364,  # Tomato Paste -> Tomato
    988: 364,  # Tomato Puree -> Tomato
    989: 254,  # Coriander Seed -> Coriander
    990: 278,  # Lard (pork fat) -> Pork
    991: 274,  # Cured Ham -> Ham
}
AHN_INGREDIENTS = {992: "cayenne", 993: "yeast", 994: "tequila", 995: "sauerkraut"}
ADDITIVE_MOLECULES = {996: (), 997: (33032,), 998: (311,)}  # Baking Powder, MSG, Citric Acid
NO_PROFILE = frozenset({999, 1000, 1001, 1002})  # Cooking Spray, Gelatin, Food Coloring, Liquid Smoke


@dataclass
class FlavorProfiles:
    """Entity -> molecule-token sets, with provenance and a dense index for matrix work."""

    profiles: dict[int, frozenset[str]]
    provenance: dict[int, str]
    names: dict[int, str]
    categories: dict[int, str]
    unmatched_constituents: dict[int, list[str]] = field(default_factory=dict)

    @property
    def entity_ids(self) -> np.ndarray:
        return np.array(sorted(self.profiles), dtype=np.int64)

    def index(self) -> dict[int, int]:
        return {int(e): i for i, e in enumerate(self.entity_ids)}

    def size(self, entity_id: int) -> int:
        return len(self.profiles.get(entity_id, ()))

    def shared_matrix(self) -> tuple[np.ndarray, dict[int, int]]:
        """S[i, j] = |C_i ∩ C_j| over all entities (S[i, i] = |C_i|). Returns (S as int32, entity_id -> row)."""
        idx = self.index()
        vocab: dict[str, int] = {}
        rows, cols = [], []
        for e, r in idx.items():
            for tok in self.profiles[e]:
                rows.append(r)
                cols.append(vocab.setdefault(tok, len(vocab)))
        a = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(idx), len(vocab)))
        return np.asarray((a @ a.T).todense(), dtype=np.int32), idx

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "entity_id": e,
                    "name": self.names.get(e),
                    "category": self.categories.get(e),
                    "n_molecules": len(p),
                    "source": self.provenance[e],
                }
                for e, p in sorted(self.profiles.items())
            ]
        )


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def decode_synonym(syn: str) -> list[str]:
    """Decode CulinaryDB's synonym notation into plain phrases.

    '=' joins words of one name ('bay=leaf' -> 'bay leaf'); '-' separates a base from its qualifier, written in
    either natural order ('pepper-bell' -> 'bell pepper', 'mustard-oil' -> 'mustard oil'); '#..#' wraps some names.
    Both orders are returned for '-' forms.
    """
    s = syn.strip().strip("#").strip()
    parts = [p.replace("=", " ").strip() for p in s.split("-") if p.strip()]
    if not parts:
        return []
    out = [" ".join(parts), " ".join(reversed(parts))]
    return list(dict.fromkeys(_norm(o) for o in out if _norm(o)))


# Constituent names in 03_Compound_Ingredients.csv that no rule can resolve (typos or generic words).
CONSTITUENT_ALIASES = {"oil": 901, "black mustard seed oil": 317}  # generic "oil" -> Cooking oil


def _name_index(cdb: CulinaryDB) -> dict[str, int]:
    """Decoded names and synonyms -> entity id; basic ingredients take precedence over compounds on collisions."""
    idx: dict[str, int] = {}
    ing = cdb.ingredients.sort_values("is_compound", kind="stable")  # basic first
    for e, name, syns in zip(ing["entity_id"], ing["name"], ing["synonyms"], strict=True):
        for phrase in [_norm(name), *(p for s in syns for p in decode_synonym(s))]:
            if phrase:
                idx.setdefault(phrase, int(e))
    return idx


def _singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and w[-3] in "sxz":
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _lookup(name: str, idx: dict[str, int]) -> int | None:
    """Exact, then longest contiguous word run (left to right), then a close fuzzy match."""
    n = _norm(name)
    if n in CONSTITUENT_ALIASES:
        return CONSTITUENT_ALIASES[n]
    words = [_singular(w) for w in n.split()]
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            cand = " ".join(words[start : start + length])
            if cand in idx:
                return idx[cand]
    import difflib

    close = difflib.get_close_matches(n, list(idx), n=1, cutoff=0.88)
    return idx[close[0]] if close else None


def build_profiles(
    cdb: CulinaryDB,
    fdb: FlavorDB,
    ahn: AhnData,
    *,
    cayenne_source: str = "ahn",
    baking_powder_empty: bool = True,
) -> FlavorProfiles:
    """Attach a flavor profile to every CulinaryDB entity (basic and compound). See the module docstring for rules."""
    ing = cdb.ingredients
    names = dict(zip(ing["entity_id"].astype(int), ing["name"], strict=True))
    cats = dict(zip(ing["entity_id"].astype(int), ing["category"], strict=True))
    cas2pid = fdb.cas_to_pubchem()

    def fdb_tokens(eid: int) -> frozenset[str]:
        return frozenset(f"P{p}" for p in fdb.entity_molecules.get(eid, ()))

    def ahn_tokens(name: str) -> frozenset[str]:
        return frozenset(f"P{cas2pid[c]}" if c in cas2pid else f"CAS{c}" for c in ahn.profile(name))

    profiles: dict[int, frozenset[str]] = {}
    prov: dict[int, str] = {}
    for eid in sorted(int(e) for e in ing.loc[~ing["is_compound"], "entity_id"]):
        if eid in fdb.entity_molecules:
            profiles[eid], prov[eid] = fdb_tokens(eid), "flavordb"
        elif eid in COARSE_PARENTS:
            parent = COARSE_PARENTS[eid]
            profiles[eid], prov[eid] = fdb_tokens(parent), f"flavordb-parent:{parent}"
        elif eid in AHN_INGREDIENTS:
            if eid == 992 and cayenne_source == "capsicum":
                profiles[eid], prov[eid] = fdb_tokens(362), "flavordb-parent:362 (cayenne proxy)"
            else:
                profiles[eid], prov[eid] = ahn_tokens(AHN_INGREDIENTS[eid]), "ahn2011"
        elif eid in ADDITIVE_MOLECULES:
            mols = ADDITIVE_MOLECULES[eid]
            if eid == 996 and not baking_powder_empty:
                mols = ("baking-powder",)
            profiles[eid] = frozenset(f"P{m}" if isinstance(m, int) else m for m in mols)
            prov[eid] = "manual-additive"
        elif eid in NO_PROFILE:
            profiles[eid], prov[eid] = frozenset(), "no-profile (paper)"
        else:
            profiles[eid], prov[eid] = frozenset(), "missing"

    name_idx = _name_index(cdb)
    comp = ing[ing["is_compound"]]
    resolved = {
        int(eid): [(c, _lookup(c, name_idx)) for c in constituents]
        for eid, constituents in zip(comp["entity_id"], comp["constituents"], strict=True)
    }
    # Two passes so a compound made from another compound (e.g. Coleslaw <- Mayonnaise) gets its full profile.
    for _ in range(2):
        for eid, parts in resolved.items():
            union: set[str] = set()
            for _c, ce in parts:
                if ce is not None and ce != eid:
                    union |= profiles.get(ce, frozenset())
            profiles[eid], prov[eid] = frozenset(union), "compound-union"
    unmatched = {eid: [c for c, ce in parts if ce is None] for eid, parts in resolved.items()}
    unmatched = {k: v for k, v in unmatched.items() if v}
    return FlavorProfiles(profiles, prov, names, cats, unmatched)

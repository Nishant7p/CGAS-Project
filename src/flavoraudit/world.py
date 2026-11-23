"""The high-level entry point: one object holding CulinaryDB, the flavor layer and the 22 real cuisines.

    import flavoraudit as fa
    world = fa.load_world()                      # CulinaryDB + FlavorDB + Ahn 2011 from the data directory
    korea = world.cuisine("Korea")
    fa.food_pairing(korea, world.S).z_sb         # Singh & Bagler's Z against the random null (negative for Korea)
    mine = world.cuisine_from_lines("my Korean set", [["2 cups kimchi", "1 tbsp soy sauce", ...], ...])
    world.scorecard(mine, "Korea")               # every D2 audit metric against the real Korean cuisine

The reference statistics for scorecards (real Z for all 22 regions, real fingerprints) take about a minute to compute
and are cached in memory; pass a precomputed ``audit.Reference`` to skip that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .aliasing import Aliaser
from .audit import Reference, scorecard_row
from .data import CulinaryDB, load_ahn, load_culinarydb, load_flavordb
from .flavor import FlavorProfiles, build_profiles
from .pairing import Cuisine, build_cuisines


@dataclass
class World:
    cdb: CulinaryDB
    profiles: FlavorProfiles
    S: np.ndarray
    index: dict[int, int]  # entity id -> S row
    cuisines: dict[str, Cuisine]
    _aliaser: Aliaser | None = field(default=None, repr=False)
    _reference: Reference | None = field(default=None, repr=False)

    @property
    def regions(self) -> list[str]:
        return sorted(self.cuisines)

    def cuisine(self, name: str) -> Cuisine:
        if name not in self.cuisines:
            raise KeyError(f"unknown cuisine {name!r}; choose from {self.regions}")
        return self.cuisines[name]

    @property
    def aliaser(self) -> Aliaser:
        """The production aliaser (CulinaryDB protocol, learned tie-breaks), built on first use."""
        if self._aliaser is None:
            self._aliaser = Aliaser.reference(self.cdb)
        return self._aliaser

    def alias(self, line: str) -> int | None:
        """CulinaryDB entity id of one raw ingredient line (None if unmatched)."""
        return self.aliaser.alias(line).entity_id

    def recipe_from_lines(self, lines: list[str]) -> np.ndarray:
        """S rows of the matched ingredients of one recipe (duplicates and unmatched lines dropped)."""
        ents = (self.alias(line) for line in lines)
        return np.unique(np.array([self.index[e] for e in ents if e is not None and e in self.index], dtype=np.int64))

    def cuisine_from_lines(self, name: str, recipes: list[list[str]]) -> Cuisine:
        """A Cuisine from raw recipes (each a list of ingredient lines), aliased like CulinaryDB."""
        cats = next(iter(self.cuisines.values())).categories
        recs = [r for r in (self.recipe_from_lines(lines) for lines in recipes) if len(r)]
        return Cuisine(name, recs, cats)

    def reference(self) -> Reference:
        """Real Z and fingerprints of all 22 regions for scorecards (computed once, about a minute)."""
        if self._reference is None:
            self._reference = Reference.build(self.cuisines, self.S, n_null=100_000, fingerprint_null=20_000)
        return self._reference

    def scorecard(self, candidate: Cuisine, region: str, **kwargs) -> dict:
        """All D2 audit metrics of a candidate cuisine against the real cuisine ``region``."""
        return scorecard_row(candidate, region, self.reference(), **kwargs)


def load_world(data_dir: str | Path | None = None) -> World:
    """Load CulinaryDB, FlavorDB and the Ahn et al. 2011 data, build flavor profiles and the 22 real cuisines.

    The data is licensed CC BY-NC-SA 3.0 (CulinaryDB, FlavorDB) and CC BY 4.0 (Ahn et al.); see DATA_LICENSE.md.
    """
    cdb, fdb, ahn = load_culinarydb(data_dir), load_flavordb(data_dir), load_ahn(data_dir)
    profiles = build_profiles(cdb, fdb, ahn)
    S, index = profiles.shared_matrix()
    cuisines = build_cuisines(cdb.recipe_entities, cdb.recipes, index, profiles.categories, labels=cdb.regions)
    return World(cdb, profiles, S, index, cuisines)

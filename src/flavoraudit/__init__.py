"""FlavorAudit: audit any recipe collection against the flavor-pairing fingerprints of world cuisines.

Methods: flavor sharing and food pairing (Ahn et al. 2011; Jain, Rakhi & Bagler 2015; Singh & Bagler 2018), category
fingerprints and authenticity (Jain et al. 2015; Jain & Bagler 2018), copy-mutate evolution (Tuwani et al. 2019),
on CulinaryDB and FlavorDB (CoSyLab, IIIT-Delhi).

    import flavoraudit as fa
    world = fa.load_world()
    fa.food_pairing(world.cuisine("Korea"), world.S).z_sb
"""

from .aliasing import Aliaser
from .audit import Reference, scorecard_row
from .contribution import chi_ahn, chi_percent
from .evolution import copy_mutate
from .fingerprint import fingerprint, fingerprint_similarity
from .nulls import NULL_MODELS
from .pairing import Cuisine, ns_recipe, ns_values
from .stats import PairingResult, bootstrap_delta_ci, food_pairing
from .world import World, load_world

__version__ = "0.1.0.dev0"

__all__ = [
    "NULL_MODELS",
    "Aliaser",
    "Cuisine",
    "PairingResult",
    "Reference",
    "World",
    "bootstrap_delta_ci",
    "chi_ahn",
    "chi_percent",
    "copy_mutate",
    "fingerprint",
    "fingerprint_similarity",
    "food_pairing",
    "load_world",
    "ns_recipe",
    "ns_values",
    "scorecard_row",
]

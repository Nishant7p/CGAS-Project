# Data provenance and licenses

| Path | What | Source | License | How to rebuild |
|---|---|---|---|---|
| `raw/culinarydb/CulinaryDB.zip` → `extracted/` | 45,772 recipes, 22 regions plus 4 misc labels; 930 ingredients, 103 compound ingredients, 456,279 recipe–ingredient alias rows | https://cosylab.iiitd.edu.in/culinarydb/static/data/CulinaryDB.zip (files dated 2018-03-16, fetched 2026-11-22) | CC BY-NC-SA 3.0 | `curl` the URL, then unzip |
| `raw/culinarydb_site/` | CulinaryDB web pages and published figures (Z per null model, top contributors, size, rank, heatmap) | cosylab.iiitd.edu.in/culinarydb/ | CC BY-NC-SA 3.0 | see `docs/RESEARCH_NOTES.md` |
| `raw/flavordb_v1/`, `raw/flavordb_v2/` | One gzipped JSON per FlavorDB entity (936 each, identical content), plus `_status.csv` | `/flavordb/entities_json?id=` and `/flavordb2/entities_json?id=` (fetched 2026-11-22) | CC BY-NC-SA 3.0 | `python -I scripts/fetch_flavordb.py --db flavordb --out data/raw/flavordb_v1` |
| `raw/ahn2011/` | Ahn et al. 2011 ingredient–compound data (CAS ids), recipes, flavor-network backbone | Zenodo DOI 10.5281/zenodo.11449657 | CC BY 4.0 | Zenodo API, record 11449657 |
| `reference/singh_bagler_2018_targets.json` | Published Z (4 nulls × 22 regions) and top-3 contributions, read from figures; used as replication targets | CulinaryDB site figures | derived (CC BY-NC-SA) | manual reading (documented in file) |
| `interim/` | Flavor layer, matrices, aliased tables | built by our scripts | derived | Phase 1 scripts |
| `generated/` | LLM generation logs and parsed recipes (D1) | our API calls | CC BY-NC-SA 4.0 + LLM terms | Phase 4 runner |
| `release/LLMCuisine-22/` | Public D1 release in CulinaryDB format | built from `generated/` | CC BY-NC-SA 4.0 | Phase 5 |

Attribution to use: *CulinaryDB and FlavorDB, Complex Systems Laboratory (Prof. Ganesh Bagler), IIIT-Delhi, http://cosylab.iiitd.edu.in. Licensed CC BY-NC-SA 3.0.* Also cite Singh & Bagler (2018), Garg et al. (2018) and Ahn et al. (2011).

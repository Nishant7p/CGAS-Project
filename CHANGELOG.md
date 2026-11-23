# Changelog

All notable changes to FlavorAudit (`flavoraudit`). The format follows Keep a Changelog; versions follow SemVer.

## [Unreleased] (0.1.0.dev0)

### Added
- `data`: loaders for CulinaryDB, FlavorDB and the Ahn et al. 2011 data.
- `flavor`: flavor profiles for all 1,033 CulinaryDB entities (24 missing ones reconstructed) and the shared-compound
  matrix S.
- `pairing`, `nulls`, `stats`: recipe and cuisine Ns, the four null models of Singh & Bagler 2018, Z_SB, effect size
  and bootstrap CIs. Reproduces Fig. 4 (22/22 signs; ρ = 0.98).
- `contribution`: χ% (Singh & Bagler) and χ (Ahn et al. 2011).
- `aliasing`: the CulinaryDB aliasing protocol with learned tie-breaks (held-out 97.9% / 99.0%).
- `composition`, `fingerprint`: prevalence, relative prevalence, SAI, JSDs, frequency-rank MAE, diversity, novelty,
  and ΔNs_cat category fingerprints with a vectorised category-randomized null.
- `evolution`: the copy-mutate models of Tuwani et al. 2019 (CM-R, CM-C, CM-M) with random or ranked fitness.
- `audit`: the flavor-fidelity scorecard, one scoring function for LLM, copy-mutate and real-subsample cuisines.
- `scoring`: the D3 fidelity scorer (classifier, shrinkage PMI, Ns consistency; calibration and ECE).

# Task B — SEA-AD Cell-Loss Extraction (Working Spec for Claude Code)

**Goal:** Produce data-derived, severity-graded cell-loss/survival curves for the four model
populations (PYR/SST/PV/VIP equivalents) from the SEA-AD atlas, to replace the hardcoded
late-stage fractions in `ad_modifiers.py` (`apply_ad_cell_loss`: SST 0.85, PV 0.90, PYR 0.97,
VIP 1.00).

**Read first:** `SEA_AD_transcriptomic_integration.md` (the governing guide). Rule #0 still
applies, but note: **cell-loss is the P0/HUMAN_DIRECT exception** — proportions/counts are a
direct, quantitative use (unlike channel transcripts). This task IS allowed to produce magnitudes,
because cell *abundance* is a real measured quantity, not a conductance proxy.

**Run location:** Locally on Mac. This is analysis, not simulation — no HPC, no NEURON. Use a
plain Python env with `scanpy`, `anndata`, `pandas`, `abc_atlas_access`.

**Scope discipline (critical for laptop):** You need cell-type PROPORTIONS, not gene expression.
That means you primarily need the **`.obs` metadata table** (per-nucleus: cell-type label + donor
ID + pathology score), NOT the full expression matrix. The full MTG h5ad is many GB (1.4M cells ×
~36k genes). **Do NOT load the full expression matrix.** Pull only obs/metadata via the ABC Atlas
API, or load the h5ad in backed mode (`anndata.read_h5ad(path, backed='r')`) and access only
`.obs`. If you find yourself reading `.X`, stop — you've over-scoped.

---

## Data access (confirmed open — no DUA needed)

- **Source:** SEA-AD processed snRNA-seq AnnData, MTG, ~1.4M QC'd nuclei, 84 donors.
  Taxonomy: 3 classes / 24 subclasses / 139 supertypes.
- **Access:** OPEN on AWS Registry of Open Data (`registry.opendata.aws/allen-sea-ad-atlas/`).
  No Synapse account / DUA required for processed data (controlled access is only for RAW seq).
- **Recommended API:** `abc_atlas_access` (`AbcProjectCache`), per the Allen notebook
  `alleninstitute.github.io/abc_atlas_access/notebooks/asap_pmdbs_seaad_taxonomy.html`.
  This lets you fetch the metadata/obs table without downloading the full matrix.
- **Severity variables available in donor metadata:**
  - `ADNC` (AD Neuropathologic Change) and `Braak` stage — DEFINITELY present, use for B1.
  - **CPS (Continuous Pseudo-progression Score)** — NOW published (was not in 2023). Confirm the
    exact column name and that it joins at donor level. Use for B2.
  - NeuN+ density (MTG quantitative neuropathology) — optional cross-check on neuronal loss.

---

## B1 — Proportions vs ADNC/Braak (DO FIRST — definitely-public, ~1 day)

1. **Pull the obs/metadata table** via ABC Atlas API (or backed-mode `.obs`). Columns needed:
   per-nucleus cell-type label (subclass level is right granularity), donor ID, and donor-level
   `ADNC` + `Braak`.
2. **Map SEA-AD subclasses → model populations.** SEA-AD MTG subclasses include (verify exact
   names in the data):
   - `L2/3 IT` excitatory → **HL23PYR** (this is the supragranular IT population, the late-loss
     excitatory type)
   - `SST` (and `SST Chodl`) → **HL23SST**
   - `Pvalb` → **HL23PV**
   - `Vip` → **HL23VIP**
   Record the exact subclass strings used in a mapping dict; do not guess — print the unique
   subclass labels first and map explicitly.
3. **Compute per-donor cell-type proportions:** for each donor, fraction of neurons of each model
   population (proportion of that subclass among all neurons, or among all nuclei — pick one
   denominator and state it; "fraction of neurons" is usually the right normalization for a loss
   curve).
4. **Aggregate vs severity:** group donors by ADNC stage (and separately by Braak), compute
   mean ± SD/SEM proportion per population per stage. Normalize to the healthiest bin (= survival
   fraction, so the lowest-pathology bin = 1.0). This gives a survival curve per population.
5. **Output:** `data/transcriptomic/cell_loss_vs_ADNC.csv` (and `_vs_Braak.csv`) with columns:
   `population, severity_bin, mean_proportion, sem, survival_fraction (normalized to healthiest)`.
6. **Sanity check against known biology:** SST should drop EARLY; PV and L2/3 IT (PYR) should drop
   LATE; VIP relatively spared. If the curves don't show SST-early / PV+PYR-late, something is
   wrong with the mapping or normalization — flag it, don't ship it. (This pattern is the SEA-AD
   headline finding and is independently replicated; β≈−0.48 for SST, −0.45 for IT in a separate
   1,373-donor study.)

## B2 — True CPS curves (UPGRADE — do after B1, ~1–2 extra days)

Only after B1 works:
1. **Locate the CPS column** in donor metadata. Confirm: (a) exact name, (b) it's donor-level,
   (c) whether it's the MTG pseudoprogression CPS used in Gabitto et al. or a region-specific
   variant (the published CPS file is associated with Caudate neuropathology imaging — verify the
   donor-level CPS joins correctly to MTG cell counts). **Report what you find before proceeding** —
   if the CPS↔MTG-cells join is ambiguous, stop and surface it rather than forcing a join.
2. Re-express the B1 proportions against continuous CPS instead of discrete ADNC/Braak bins: fit a
   smooth survival curve (e.g. LOWESS or a monotonic spline) of proportion vs CPS per population.
3. **Output:** `data/transcriptomic/cell_loss_vs_CPS.csv` and a figure (survival fraction vs CPS,
   4 populations). This is the version that plugs directly into `f(CPS)`.

---

## How the output feeds the model (NOT in this task — just so you build the right shape)

The curves will later parameterize `apply_ad_cell_loss()`: instead of hardcoded late-stage
fractions, the cell-loss factor for each population becomes a function of CPS (or stage). So the
CSV must be queryable as `survival_fraction[population][severity]`. **Do NOT modify
`ad_modifiers.py` in this task** — just produce the data tables + figure. Wiring them in is a
later, separate step gated on the AD-layer checkpoint.

---

## Deliverables

1. `scripts/transcriptomic/extract_seaad_cellloss.py` — the extraction (obs-table-only, documented).
2. `data/transcriptomic/cell_loss_vs_ADNC.csv`, `cell_loss_vs_Braak.csv` (B1).
3. `data/transcriptomic/cell_loss_vs_CPS.csv` + survival-curve figure (B2, if CPS join confirmed).
4. A short `data/transcriptomic/README.md` recording: which SEA-AD file/version, the subclass→pop
   mapping dict used, the denominator choice, and the CPS-join verification result.
5. **Keep the raw h5ad OUT of git** (it's GB-scale). Only the derived CSVs + script are committed.

## Guardrails

- Obs-table-only; do not load `.X` / full expression matrix.
- Print and explicitly map subclass labels — no guessed strings.
- State the normalization denominator and the healthiest-bin reference explicitly.
- B1 before B2; if CPS join is ambiguous, ship B1 and flag B2.
- Sanity-check against SST-early / PV+PYR-late / VIP-spared; refuse to ship curves that contradict
  it without explanation.
- Do not touch `ad_modifiers.py`.

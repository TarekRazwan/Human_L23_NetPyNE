# Transcriptomic Data — SEA-AD Cell-Loss Extraction

## Data Source

- **Dataset:** Seattle Alzheimer's Disease Brain Cell Atlas (SEA-AD)
- **Publication:** Gabitto et al. 2024 (PMC11577961)
- **Region:** Middle Temporal Gyrus (MTG)
- **Release:** 2024-02-13
- **S3 bucket:** `sea-ad-single-cell-profiling` (AWS Open Data, no auth)
- **Donors:** 69 with snRNA-seq (of 84 total in SEA-AD)
- **Nuclei:** 915,512 neuronal nuclei (170 libraries, 109 supertypes)

## Files

| File | Description |
|------|-------------|
| `donor_composition.csv` | Per-donor neuron counts, proportions, and severity variables |
| `cell_loss_vs_ADNC.csv` | Survival curves by AD Neuropathological Change (4 bins) |
| `cell_loss_vs_Braak.csv` | Survival curves by Braak stage (6 levels) |
| `cell_loss_vs_CPS.csv` | Survival curves by CPS quartile (4 bins) |
| `figures/survival_vs_*.png` | Corresponding plots |

## Scientific Choices

### Subclass → Model Population Mapping

```python
{
    'L2/3 IT':   'PYR',    # supragranular excitatory IT
    'Sst':       'SST',    # all 16 Sst supertypes
    'Sst Chodl': 'SST',    # included: SST-expressing, rare (~0.1%)
    'Pvalb':     'PV',     # all 13 Pvalb supertypes
    'Vip':       'VIP',    # all 16 Vip supertypes
    # Chandelier: EXCLUDED from PV (distinct subclass, targets AIS)
    # All other subclasses: denominator only
}
```

### Denominator

Fraction of **all neurons** — population count / total neuronal nuclei per donor.
This avoids confounding by reactive gliosis (astrocyte/microglia proliferation
in AD would inflate an "all nuclei" denominator).

### Normalization Anchors

- **ADNC:** "Not AD" (9 donors) — cleanest healthy baseline
- **Braak:** "Braak II" (4 donors) — Braak 0 has only 2 donors (too noisy)
- **CPS:** Lowest quartile (17 donors)

### Compositional Closure Caveat

Proportions sum to 1, so when lost populations (SST) shrink, spared
populations (PV, VIP) can show apparent fractional *increase* — this is
a mathematical artifact, not biological growth.  Survival fractions > 1.0
are flagged as `closure_artifact = True` in the CSVs.

## Raw vs scCODA Cross-Check (CPS axis)

| Pop | Raw survival (CPS Q4) | scCODA direction | scCODA median effect |
|-----|----------------------|------------------|---------------------|
| SST | 0.724 (loss) | LOSS | -0.950 |
| PV  | 0.984 (stable) | LOSS | -1.019 |
| PYR | 0.868 (loss) | LOSS | -1.485 |
| VIP | 0.890 (loss) | LOSS | -0.546 |

scCODA detects credible loss in ALL populations vs CPS, including PV (which
appears stable in raw proportions).  This is because scCODA models the
compositional structure: PV's raw stability is partly a closure artifact
from SST's decline.  Both analyses agree on the *direction* for SST, PYR,
and VIP.

## Direction 1 — f(CPS) Curve Shapes

Best-fit functional forms and onset CPS per population (fit to 69 per-donor
values, compared by AIC):

| Pop | Best model | Onset CPS | Slope | R² |
|-----|-----------|-----------|-------|----|
| SST | linear | 0.33 | -0.48 | 0.113 |
| PYR | linear | 0.53 | -0.20 | 0.053 |
| PV  | linear (flat) | n/a | ~0 | 0.001 |
| VIP | hinge | 0.86 | flat→steep | 0.094 |

**Caveat (read before citing R²):** The low R² values (0.05–0.11) reflect
per-donor compositional scatter — individual donors vary 3-fold in SST
fraction even within the same ADNC group.  The *population-level trends*
are solid and cross-validated against ADNC bins, scCODA compositional
modeling, and the external Barker et al. 1,373-donor replication.  The
deliverable is the **onset CPS values and linear-shape classification**,
not the R².  With n=69, AIC correctly rejects complex forms (sigmoid/hinge
don't earn their extra parameters except marginally for VIP).

Files: `cps_curve_fits.csv`, `figures/cps_curve_fits.png`

## Direction 2 — Supertype-Resolved Vulnerability

Within SST (18 supertypes) and PV (13 supertypes), cell loss is NOT
uniform — it is concentrated in specific molecular subtypes:

- **SST:** 7 supertypes (55% of nuclei) show >35% loss at CPS Q4;
  the remaining 11 are stable or show closure-artifact gains.
  Sst_22, _20, _11 are the most vulnerable (survival <0.40).
- **PV:** 2 supertypes (Pvalb_8, _14) lose cells at SST-like rates
  (survival 0.59–0.64), but the dominant Pvalb_15 (28k nuclei, 37%
  of all PV) is completely stable — masking the subtype loss in the
  aggregate.

**Caveat:** The robust finding is the *pattern* (loss is subtype-concentrated;
dominant supertypes spared), NOT the precise survival value of any individual
rare supertype.  Several (Sst_22: 590 nuclei, Sst Chodl_1: 688) are noisy
at this sample size.  The model uses the aggregate curves; the supertype
result is a biological finding about within-class heterogeneity, not a
structural change to the model.

Files: `supertype_survival_Sst.csv`, `supertype_survival_Pvalb.csv`,
`figures/supertype_vulnerability_Sst.png`, `figures/supertype_vulnerability_Pvalb.png`

## Direction 4 — APOE4 Stratification (EXPLORATORY)

**Status: Exploratory, underpowered.  Not a model parameter — hypothesis-generating only.**

### Power limitation

69 donors split into 20 APOE4 carriers (14 e3/e4, 4 e4/e4, 2 e2/e4) and
49 non-carriers.  e4/e4 homozygotes (n=4) are too few to separate from
heterozygotes; all carriers are grouped together.

The fundamental confounder: APOE4 carriers are heavily enriched at high
severity.  All 20 carriers have ADNC Intermediate or High; **zero** carriers
fall in "Not AD" or "Low."  By CPS, carriers cluster in Q3–Q4 (17 of 20),
with only 1 in Q1 and 2 in Q2.  This means the carrier "baseline" (Q1)
rests on a single donor — any carrier-vs-non-carrier comparison at low CPS
is uninterpretable.  Bins with < 5 donors are flagged as underpowered
(red n in the figure).

### Observations (NOT conclusions)

- **SST:** Carriers show a suggestive earlier/steeper decline (survival 0.50
  at Q3 vs 0.88 for non-carriers), but the carrier Q1 baseline is n=1 and
  Q2 is n=2, so the carrier curve's normalization is unstable.
- **VIP:** Carriers may show more VIP loss (0.67 at Q3, 0.82 at Q4 vs ~1.0
  for non-carriers), but the same baseline caveat applies.
- **PYR, PV:** No clear genotype difference at the points where both groups
  have adequate n (Q3–Q4).

### Interpretation

**Does the data motivate a future APOE4 modifier? — Maybe, but not from
this dataset alone.**  The suggestive SST/VIP patterns could reflect:
(a) a real APOE4-accelerated vulnerability, (b) the severity confound
(carriers are sicker, so their cells at any CPS quartile are drawn from a
different part of the disease process), or (c) noise from n=1–2 baselines.
Distinguishing these requires a larger cohort with healthy APOE4 carriers —
SEA-AD's 69-donor MTG subset does not have them.  The proposal correctly
frames genotype-conditioned modifiers as future work; this analysis confirms
that framing is appropriate.

Files: `apoe_stratified_loss.csv`, `figures/apoe_stratified_survival.png`

## Sanity Gate

- **ADNC:** PASSED — SST drops most, PV stable, VIP/PYR spared
- **CPS:** PASSED — SST drops most (0.724), PV stable, VIP/PYR moderate
- **Braak:** FAILED — noisy (2 donors at Braak 0, non-monotonic pattern);
  Braak staging is NFT-based and doesn't capture the full AD severity
  spectrum as cleanly as ADNC or CPS for MTG cell composition

## Pipeline

- `transcriptomic/scripts/seaad_pipeline.py` — core module
- `transcriptomic/scripts/run_ad_cellloss.py` — AD config driver
- `transcriptomic/scripts/ad_cellloss_walkthrough.ipynb` — teaching notebook

Generated 2026-06-06.

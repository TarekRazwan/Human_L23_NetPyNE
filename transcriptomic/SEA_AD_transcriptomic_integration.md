# SEA-AD & Transcriptomic Data → f(CPS) Integration Guide

**Purpose:** Instructions for Claude Code on how to use SEA-AD multimodal atlas data and the
Zielonka et al. 2026 L2/3 degeneration-trajectory paper within this project. This is a
**scoping + data-handling** guide, not a license to plug transcript counts into conductances.
Read this fully before writing any extraction or mapping code.

**Anchoring document:** `Thesis_Proposal_Razwan_Full_v2.md` (the authoritative plan). The
`f(CPS)` framework, the fifteen mechanism-tiered targets, and the evidence tiers (P0/P1/P2;
HUMAN_DIRECT → EXPLORATORY) defined there govern everything below.

---

## 0. The single most important rule

**Transcript abundance is NOT conductance.** mRNA level ≠ protein level ≠ functional channel
density ≠ maximal conductance (ḡ). There is post-transcriptional regulation, trafficking, and
compensation between every arrow. Therefore:

- Transcriptomic data may set the **direction** (up/down) and **timing** (which CPS stage) and
  **which cell type** of a perturbation.
- Transcriptomic data must **NOT** set the **magnitude** of a `gbar` scaling factor as if a
  fold-change were the parameter value.
- Any magnitude use must be flagged explicitly as an EXPLORATORY assumption with a stated
  mRNA→function conversion, and MUST be tested in the Aim 3 sensitivity analysis — never treated
  as ground truth.

This is the project's credibility line. The thesis Background (lines 74–77) is explicit that
`f(CPS)` parameterizes the *consequences* of molecular pathology at the synaptic/channel/intrinsic
level — it does not read magnitudes off transcriptomics.

---

## 1. Two data sources, two distinct roles

### 1a. SEA-AD (Gabitto et al.) — the PARAMETER source
SEA-AD is the atlas the thesis's `f(CPS)` axis is **anchored to** (the Continuous
Pseudo-progression Score, CPS). It provides four classes of modeling input, at very different
evidence tiers:

| # | SEA-AD product | Use in this project | Evidence tier | Status |
|---|----------------|---------------------|---------------|--------|
| 1 | **Cell-type vulnerability / loss curves** (early SST + PVALB loss in L2/3; later L5/6 IT loss) | Drives `ad_cell_loss` modifiers and late-stage population reduction | **HUMAN_DIRECT, P0** — strong, quantitative | Usable now |
| 2 | **CPS pseudo-progression axis** | The empirical backbone of `f(CPS)`; maps model stages → data severity | **HUMAN_DIRECT** — the axis itself | Core to framework |
| 3 | **Ion-channel / receptor transcript levels** (SCN, KCN, GRIN, GRIA, GABRA families) | Direction/timing prior ONLY for intrinsic + synaptic modifiers | **EXPLORATORY** — transcript ≠ conductance | Cautious use |
| 4 | **Spatial co-localization / distances** (MERFISH/Xenium relative to tau/Aβ) | NOT for the current circuit model — relevant to future reaction-diffusion/proteopathy work | Out of current scope | Park |

### 1b. Zielonka et al. 2026 (bioRxiv 2026.04.14.718430) — the CORROBORATION source
851,682 L2/3 excitatory-neuron transcriptomes, 557 individuals, **four cohorts including
SEA-AD/Gabitto** + Gazestani + Mathys + Lu. Reconstructs a **continuous pseudotime trajectory**
with **three changepoint-defined stages** (Stage 1 Homeostasis/Early Dysfunction → Stage 2
Functional Decline → Stage 3 Chronic Stress & Tau Pathology).

**What it is good for (and not):**
- **STRONG use — validates the framework axis.** Same cell type (human L2/3 excitatory = PYR),
  same continuous-severity logic as `f(CPS)`, built partly from the same SEA-AD data, across four
  cohorts. This independently supports that a continuous staged axis is data-grounded. This is the
  single most defensible use.
- **MODERATE use — direction of decline.** Gene programs GP_3 ("ion & lipid regulation",
  including "voltage-gated ion channel activity" GO_MF), GP_15 ("synapse assembly"), GP_18
  ("neuronal signaling") all **decline early** (Stage 1→2). This corroborates the *direction* of
  the intrinsic-excitability and synaptic-weakening modifiers — as GO-program trends, NOT
  magnitudes.
- **DO NOT** use it to set channel magnitudes. Its molecular depth is tau-kinase / phosphatase /
  proteostasis / mitochondria (CDK5, CAMK2B, TTBK1, GSK3A, etc.) — this is the **upstream cascade**
  the thesis parameterizes the consequences of, not the channel layer itself. There is no headline
  KCNC1/KCNN/HCN/SCN finding.

---

## 2. What to actually build (in priority order)

### Task A (HIGHEST PRIORITY, fully defensible): Staging-axis alignment figure + mapping table
Produce the figure the thesis calls Fig. 3 ("f(CPS) framework architecture") support material:
a mapping between **this project's AD stages** (healthy / early / intermediate / late, per
`ad_modifiers.py` STAGE_PRESETS) and the **empirical severity axes** (SEA-AD CPS bins; Zielonka
Stage 1/2/3 pseudotime).

- Build a table: model stage → SEA-AD CPS bin → Zielonka pseudotime stage → key declining gene
  programs at that stage (GP_3/15/18 early; tau-kinase GP_7/19 late) → which `f(CPS)` modifiers are
  active at that stage and in which direction.
- This visually justifies *why* each modifier exists and *when* it turns on, grounded in two
  independent data-derived trajectories over the matched cell type.
- **No magnitude claims** — this is alignment and direction only.

### Task B (HIGH PRIORITY, strong evidence): Cell-loss matrix extraction (SEA-AD)
Extract SEA-AD cell-type-specific loss/survival as a function of CPS for the four populations
(PYR/SST/PV/VIP equivalents; note SEA-AD resolves L2/3 SST, PVALB, VIP, and L2/3 IT excitatory).
- Source: annotated AnnData (h5ad) from the AD Knowledge Portal / ABC Atlas API (see §3).
- Output: a `cell_loss_vs_CPS` table that parameterizes `apply_ad_cell_loss()` — replacing the
  current hardcoded late-stage fractions (SST 0.85, PV 0.90, PYR 0.97, VIP 1.00) with
  **data-derived, CPS-continuous** survival curves.
- This is P0 / HUMAN_DIRECT — the strongest quantitative use of SEA-AD in the whole framework.

### Task C (MEDIUM, cautious / EXPLORATORY): Channel & receptor transcript direction priors
ONLY after A and B. For the specific genes mapping to model mechanisms, extract
**direction + CPS-timing** (not magnitude) from SEA-AD snRNA-seq and/or Zielonka Table S2/S8:

| Model mechanism | Gene(s) to check | Modifier |
|-----------------|------------------|----------|
| Kv3.1 (PV fast-spiking) | KCNC1 | `ad_pv_kv31_factor` |
| SK (PYR AHP) | KCNN1/2/3 | `ad_pyr_sk_factor` |
| Ih (HCN) | HCN1/2 | `ad_pyr_ih_factor` |
| NaP (persistent Na) | SCN8A (Nav1.6), SCN2A | `ad_late_nap_factor` |
| Tonic GABA-A (α5) | GABRA5, GABRB, gephyrin GPHN | `ad_sst_tonic_factor` |
| AMPA/NMDA | GRIA1-4, GRIN1/2A/2B | `ad_ampa_factor`, `ad_nmda_factor` |
| Excitatory synapse | SLC17A7 (VGLUT1), DLG4 (PSD95) | `ad_exc_weight_factor` |
| Glutamate clearance | SLC1A2 (EAAT2) | (future EAAT2 modifier) |

- Output: a table of (gene → cell type → CPS direction → confidence). Use it to **confirm or
  flag** the sign of each existing modifier. If a modifier's assumed direction disagrees with the
  transcriptomic trend, that's a finding to surface, not silently override.
- **Every entry stays EXPLORATORY tier.** Magnitudes still come from electrophysiology/proteomics
  (Poirel, Yao/Allen). Transcriptomics only corroborates direction.

### Task D (DOCUMENTATION): Update the mechanism-to-model translation table
For each mechanism where SEA-AD and/or Zielonka now provide support, update the **Evidence tier**
column of the AD-microcircuit Notion/CSV translation table. A mechanism with converging
proteomic (Poirel) + transcriptomic (SEA-AD/Zielonka) + ephys (Yao/Allen) support can move toward
HUMAN_DIRECT. This is how the transcriptomic data earns its place — by strengthening tiers, not by
inventing parameters.

---

## 3. Where the data lives (for extraction code)

- **SEA-AD annotated AnnData (h5ad):** AD Knowledge Portal, study `syn26223298`; also
  `syn64410371`. Use the **ABC Atlas Python API** (`abc_atlas_access`) for cell taxonomy + spatial
  tables, notebook reference:
  `alleninstitute.github.io/abc_atlas_access/notebooks/asap_pmdbs_seaad_taxonomy.html`
- **SEA-AD CPS:** described in Gabitto et al. (PMC11577961). CPS is the continuous axis to map
  model stages onto.
- **Zielonka data + code:** `github.com/mmzielonka/Zielonka_NeuronalCascade_2026`. Supplementary
  Table S2 (metacell DE along pseudotime) and Table S8 (kinase/phosphatase SNITCH classification)
  are the relevant per-gene-trend tables. Underlying cohorts: SEA-AD (AWS open data), Gazestani
  (braincelldata.org), Mathys (Synapse syn52293417), Lu (EGA, restricted).
- The project already has `SEA_AD.pdf` and the Zielonka PDF in `/mnt/project` for reference.

**Note for Claude Code:** these are large genomics datasets. Do NOT attempt to download full
h5ad atlases into the repo. Plan extraction as: (1) pull only the specific genes/cell-types/CPS
columns needed, (2) write small derived CSVs (e.g. `cell_loss_vs_CPS.csv`,
`channel_direction_priors.csv`) into a `data/transcriptomic/` folder, (3) keep raw atlases
out of git.

---

## 4. Hard constraints (do not violate)

1. **No conductance magnitudes from transcriptomics.** Direction + timing + cell-type only,
   unless explicitly flagged EXPLORATORY and routed to sensitivity analysis.
2. **Do not modify `ad_modifiers.py` factor magnitudes** based on this data without explicit
   sign-off — the magnitudes are anchored to ephys/proteomics. Transcriptomic work produces
   *priors and corroboration tables*, separate from the modifier values.
3. **This is downstream of the locked baseline and H01.** Per the 3-checkpoint plan, the AD
   modifier layer (where this data feeds) comes AFTER `ad-v0-healthy-locked` and after H01
   integration. This guide prepares the data/figures; it does not change the run order.
4. **Keep SEA-AD (parameters) and Zielonka (corroboration) roles distinct** in any output, so the
   provenance of each number is auditable.
5. **Cross-regional caveat applies** (thesis lines 67): SEA-AD is middle temporal gyrus, the model
   validates against precuneus TMS-EEG, BA9 for Poirel markers. Mechanisms are treated as conserved
   across human association cortex; region-specific magnitude differences go to sensitivity sweeps,
   not separate models.

---

## 5. Suggested first deliverable

Start with **Task A** (staging-axis alignment table + figure) because it is (a) fully defensible,
(b) directly supports thesis Fig. 3, (c) requires no large data download — only the stage
definitions already in `ad_modifiers.py` plus the published SEA-AD CPS bins and Zielonka Stage 1/2/3
boundaries. Produce:
- `data/transcriptomic/stage_axis_alignment.csv` — model stage ↔ CPS bin ↔ Zielonka stage ↔
  active modifiers ↔ supporting gene programs (direction-annotated).
- A figure rendering this alignment (the "why each modifier turns on when" panel).

Then proceed to Task B (cell-loss extraction) which needs the SEA-AD h5ad, then C/D.

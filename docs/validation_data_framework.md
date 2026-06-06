# Validation Data Framework — Multiscale Human AD/TMS Cortical Model

**Purpose:** Define what data validates the model at each scale and disease state, where it lives,
and — critically — keep *parameterization* data separate from *validation* data so validation is
never circular. This governs the incremental development: healthy baseline → H01 → AD layer →
TMS/EEG.

**Anchoring principle (read first):**
> **Parameterize ≠ validate.** Data used to SET a parameter cannot also VALIDATE the output that
> parameter drives — that's circular and proves nothing. For every scale/state, this doc names the
> parameter source AND a *separate* validation target. Where human data is too scarce to separate
> them, that overlap is flagged as a known weakness, not hidden.

---

## 1. The two-axis structure

Validation is organized by **scale** (single-cell → microcircuit → mesoscale/LFP → scalp/EEG) and
**disease state** (healthy baseline → AD continuum). Each cell of that grid needs: (a) a parameter
source, (b) an independent validation target, (c) an honest note on data availability.

A blunt reality that shapes everything below: **human single-unit and LFP electrophysiology from
AD cortex barely exists.** Human Neuropixels/single-unit data is from epilepsy/DBS/tumor surgical
patients (healthy-ish tissue), in small n; there is no human-AD-cortex single-unit dataset. AD
electrophysiology at cellular resolution is overwhelmingly rodent. This means **AD validation leans
on composition (SEA-AD), E:I markers (postmortem), and scalp-level TMS-EEG (Santarnecchi)** — NOT
on human AD single-unit recordings, which don't exist. State this limitation explicitly in the
thesis rather than implying cellular AD validation we can't do.

---

## 2. HEALTHY BASELINE — validation (Checkpoint 1, the locked `ad-v0-healthy-locked`)

The healthy baseline is the `s=0` anchor. It must be validated BEFORE AD perturbations, against
data NOT used to build it.

| Scale | Parameter source (what built it) | Independent validation target | Where / status | Data tier |
|-------|----------------------------------|-------------------------------|----------------|-----------|
| Single-cell intrinsic | Allen/Yao human L2/3 cell models, ion-channel fits | Human patch-seq / patch-clamp intrinsic properties (Rin, sag, AP shape, F-I) for L2/3 PYR + interneuron types | Allen Cell Types DB (celltypes.brain-map.org); Allen patch-seq human cortical neurons (Howard 2022, *verify ref*) | HUMAN_DIRECT |
| Cell-type firing rates | Yao 2022 LFPy reference rates | Independent human cortical firing-rate envelopes; human Neuropixels single-unit rates (as a sanity range, NOT AD) | Yao Table S2; human Neuropixels datasets (epilepsy/DBS surgical cortex — Paulk/Chung 2022 Neuron; openlists/ElectrophysiologyData) | HUMAN_DIRECT (rates), but note: surgical "healthy-ish", not normal |
| Microcircuit spectral | (emergent — not parameterized) | Human resting EEG/MEG spectral signatures (gamma, alpha/theta) in healthy adults | Published human EEG norms | HUMAN_DIRECT |
| Connectivity structure | Yao connection probabilities (current); H01 distances (Checkpoint 2) | H01 EM convergence (contacts/neuron) as a structural sanity check | H01 (Shapson-Coe 2024); the ~132k conn / ~656k contact counts captured in the lock pkls | HUMAN_DIRECT (anatomy) |

**Current healthy validation (what the locked baseline actually checks):**
- Primary: all-cells population firing rates ± seed SD vs Yao/LFPy reference (PYR ~0.89 / SST ~5.41
  / PV ~11.33 / VIP ~3.69 Hz). *Note: this is validation-against-the-replicated-model (Yao), which
  is reproduction, not independent validation.*
- **The honest gap:** rate-matching to Yao proves we reproduced Yao, not that the circuit matches
  *independent* human data. To make healthy validation non-circular, add at least one INDEPENDENT
  target Yao didn't define: e.g. human EEG spectral shape, or human single-unit rate ranges from the
  Neuropixels surgical datasets. **Action: identify one independent healthy target before claiming
  the baseline is "validated" rather than "reproduced."**

---

## 3. AD CONTINUUM — validation (Checkpoint 3 + Aim 1)

The hard case, because human AD cellular electrophysiology doesn't exist. Validation is necessarily
indirect and multi-modal.

| Scale | Parameter source (sets f(CPS)) | Independent validation target | Where / status | Data tier |
|-------|-------------------------------|-------------------------------|----------------|-----------|
| Cell loss / composition | SEA-AD cell-loss vs CPS (Task B: SST early, PYR/PV late, VIP spared) | **Independent cohort** composition: Zielonka 2026 (4-cohort L2/3 trajectory); the 1373-donor bulk study (β≈−0.48 SST) | SEA-AD (built); Zielonka bioRxiv 2026.04.14.718430; AMP-AD bulk | P0 HUMAN_DIRECT; cross-cohort = strong |
| E:I balance | Postmortem synaptic markers (Poirel BA9; VGLUT1/PSD95) | **Direct E:I electrophysiology** — Lauterborn/Scaduto human parietal cortex E:I measurements (independent of the markers used to set it) | Lauterborn 2021 / Scaduto — parietal human AD tissue | HUMAN_DIRECT |
| Channel/intrinsic (Kv3.1, SK, Ih, NaP) | Mechanistic literature (Aβ-Kv3.1, etc.); transcriptomic DIRECTION only (Zielonka GP_3) | **Weak point** — no human AD single-unit data to validate intrinsic changes. Falls back to: do the *network-level* consequences (gamma, E:I) match? | Indirect only — flag explicitly | EXPLORATORY → validated only via downstream network effects |
| Glutamate clearance (EAAT2) | EAAT2 70–90% reduction, temporal/parietal cortex | Same-modality replication across regions | Temporal cortex EAAT2 studies | HUMAN_DIRECT (temporal/parietal); equivocal BA9 |
| Mesoscale / LFP | (emergent) | Human AD LFP/iEEG if available; mostly EEG-band power shifts | Sparse — human AD depth-LFP is rare; lean on scalp EEG | LIMITED |
| Gamma oscillations | (emergent from PV/SST) | Human MCI/AD gamma reduction (EEG/MEG) | Casula 2022; Murty 2021; van Deursen | HUMAN_DIRECT (scalp) |

**The AD validation philosophy (state in thesis):** because human AD cellular ephys doesn't exist,
the model's AD-stage validity rests on a *chain*: composition (SEA-AD, strong) + E:I markers
(postmortem, strong) set/check the cellular layer, and the *network-level emergent outputs* (gamma,
spectral shifts, and ultimately TMS-EEG) are validated against human scalp data. Intrinsic-channel
perturbations are validated only *indirectly* — by whether their network consequences match. This
is a real limitation and the honest framing is: "cellular AD perturbations are constrained by
molecular/proteomic data and validated through their mesoscale consequences, not by direct human AD
single-unit recordings, which do not exist."

---

## 4. TMS-EEG — validation (Aim 2, the apex target)

This is where the model meets its strongest *independent* clinical validation: the Santarnecchi
precuneus TMS-EEG cohort. TMS-EEG biomarkers are NOT used to build the model — so they are a clean
validation target.

| Quantity | Validation target | Source / status | Notes |
|----------|-------------------|-----------------|-------|
| TEP components (P30, N45, P60, N100) | Subject-level TEPs, Santarnecchi precuneus cohort | Santarnecchi (access confirmed; **form to be audited**) | Sensor vs source space TBD |
| Evoked gamma | Subject-level evoked gamma, same cohort | Santarnecchi | PV/SST circuit readout |
| Local complexity (PCI-inspired) | PCI-ST values (exploratory comparison) | Santarnecchi + literature (Casali, Comolatti) | NOT promised as primary; single-column ≠ whole-cortex PCI |
| Stratification | Braak stage, APOE genotype | Santarnecchi metadata | Braak as CPS-equivalent proxy |

**CRITICAL — Santarnecchi data-form audit (proposal commits to month-1):** before building the
validation pipeline, audit: (a) sensor-space vs source-localized; (b) individualized MRI present
(for head model) or template needed; (c) sample size per Braak×APOE cell; (d) single vs multiple
timepoints. The pipeline must run at sensor AND source level in parallel (proposal mitigation).
**I do not currently have these specifics — treat Santarnecchi as "access confirmed, form
unaudited" until the audit is done.**

---

## 5. Where each dataset lives (quick reference)

- **SEA-AD atlas** (composition, CPS, cell loss): AWS `sea-ad-single-cell-profiling` (open); ABC
  Atlas API `abc_atlas_access`; AD Knowledge Portal `syn26223298`. *Built into Task B.*
- **Zielonka 2026** (cross-cohort L2/3 trajectory — AD composition validation): bioRxiv
  2026.04.14.718430; code `github.com/mmzielonka/Zielonka_NeuronalCascade_2026`.
- **Santarnecchi TMS-EEG** (precuneus, AD): via the project / collaborator; **form audit pending**.
- **Allen Cell Types / patch-seq** (healthy single-cell intrinsic): celltypes.brain-map.org;
  Howard 2022 (*verify ref*).
- **Human Neuropixels single-unit** (healthy-ish surgical cortex — rate sanity, NOT AD): Paulk/Chung
  et al. 2022 Neuron; Dryad `doi:10.5061/dryad.d2547d840`; `github.com/openlists/ElectrophysiologyData`.
- **Yao 2022** (the replicated healthy L2/3 model — reproduction reference, not independent):
  Yao et al. 2022; rates in Table S2.
- **Postmortem E:I / synaptic markers** (AD): Poirel BA9; Lauterborn/Scaduto parietal; EAAT2
  temporal/parietal — via Paperpile.
- **Human AD gamma (scalp)**: Casula 2022, Murty 2021, van Deursen — via Paperpile.

---

## 6. Standing limitations to state in the thesis (don't hide these)

1. **Healthy "validation" is currently reproduction of Yao.** Add ≥1 independent healthy target
   (EEG spectral shape or surgical-cortex single-unit rate range) to make it true validation.
2. **No human AD cellular electrophysiology exists.** AD intrinsic-channel perturbations are
   validated only via network-level consequences, not direct recordings. Cellular AD ephys is rodent.
3. **Cross-regional integration** (SEA-AD = MTG, Santarnecchi = precuneus, Poirel = BA9): mechanisms
   assumed conserved across human association cortex; region-specific magnitudes go to sensitivity
   analysis (proposal §"Reconciling cross-regional human evidence").
4. **Santarnecchi form unaudited** — pipeline must handle sensor/source in parallel; template head
   models where MRI absent.
5. **Aging vs AD confound** — g(age) and f(CPS) on partially overlapping mechanisms; validation
   stratified within age bins.

---

## 7. Action items (incremental dev order)

- [ ] **Healthy:** add one INDEPENDENT validation target beyond Yao reproduction (EEG spectral or
      surgical single-unit rate range). Required to call the baseline "validated."
- [ ] **H01 (Checkpoint 2):** validate connectivity convergence against the ~132k/~656k baseline
      captured in the lock; re-validate rates hold after position/distance changes.
- [ ] **AD (Checkpoint 3):** wire SEA-AD cell-loss curves (Task B) into `apply_ad_cell_loss`;
      validate composition trajectory against Zielonka + 1373-donor cohort (independent); validate
      E:I against Lauterborn/Scaduto (independent of the markers that set it).
- [ ] **TMS-EEG (Aim 2):** complete Santarnecchi data-form audit FIRST; build sensor+source
      pipeline; validate TEP/gamma against subject-level data.
- [ ] Keep a living table: for every parameter, record its source AND its (separate) validation
      target. Any row where they're the same = a circularity flag to resolve.

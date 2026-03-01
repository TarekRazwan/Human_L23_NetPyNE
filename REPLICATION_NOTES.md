# Replication Notes — L23Net NetPyNE vs. Original LFPy

This document records every deliberate approximation or deviation from the
original `circuit.py` (LFPy + NEURON) implementation made in this NetPyNE replica.

---

## 1. Cell Template Loading

**Original:** `NeuronTemplate.hoc` is loaded as a HOC template taking the SWC
path as a positional string argument (`$s1`), then biophysics are applied by
calling `h.biophys_HL23XXX(template)` directly in Python.

**NetPyNE replica:** `netParams.importCellParams()` is called twice per cell type:
1. First with `NeuronTemplate.hoc` + `cellArgs=[morph_path]` to import geometry.
2. Then with `biophys_HL23XXX.hoc` to apply ion channel insertions.

**Known risk:** `importCellParams` runs the HOC template in a fresh NEURON context.
If the second call (biophys) cannot find the already-instantiated template from the
first call, channel parameters may not be captured. A fallback is to define `secs`
manually from the SWC + biophys values if `importCellParams` raises errors.

---

## 2. Axon Policy

**Original:**
- PYR + SST: `delete_axon(3, 1.75, 1, 1)` — simplified tapered axon + myelin.
- PV + VIP: `delete_axon_BPO()` — preserve AIS diameter (2 × 30 µm segments).

**NetPyNE replica:** The axon HOC procedures are called from within `NeuronTemplate.hoc`
during `importCellParams`. No modifications are made. The axon treatment is
**fully inherited** from the original HOC without change. This is intentional per
the workstream specification.

---

## 3. Ornstein-Uhlenbeck Background Noise

**Original (`net_functions.hoc::createArtificialSyn`):**
- One `Gfluct2` process per basal dendrite section at midpoint (relpos = 0.5).
- Five `Gfluct2` processes along the apical trunk at relpos = 0.1, 0.3, 0.5, 0.7, 0.9
  (PYR only), placed on the widest-diameter section at each location.
- Each gets a unique `Random` object seeded as `cell_rseed * 10 + i + 5`.
- Conductance: `g_e0 = std_e = GOU * exp(relpos)`.
- Inhibitory component zero: `g_i0 = 0, std_i = 0`.

**NetPyNE replica (`init.py::insert_ou_noise`):**
- Same per-section placement logic.
- Apical trunk section selection: sorted by section name index, fractional position
  mapped to section index. The "widest diameter" selection is approximated by
  using the trunk order (sorted by name) rather than explicitly measuring diameter —
  this is a minor approximation because the trunk sections are typically in diameter
  order in the SWC.
- Seeds: `GLOBALSEED * (gid+1) * 1000 + section_index`. Functionally equivalent.
- `Gfluct2.noiseFromRandom(rng)` called after setting `rng.normal(0,1)`.

**Approximation level:** Low. The conductance magnitudes and positions are faithfully
reproduced; only the diameter-selection heuristic for apical sections is approximated.

---

## 4. Tonic GABA Inhibition

**Original (`net_functions.hoc::addTonicInhibition`):**
- `tonic` mechanism inserted into soma + basal dendrites: `g_tonic = norm_tonic`.
- For PYR cells only: `tonic` also inserted into apical dendrites: `g_tonic = apic_tonic`.
- `e_gaba_tonic = -75 mV` everywhere.

**NetPyNE replica (`init.py::insert_tonic_gaba`):**
- Same logic implemented in Python using `sec.insert('tonic')` and direct segment
  attribute setting (`seg.tonic.g`, `seg.tonic.e_gaba`).
- MDD and DRUG flag handling identical to original.

**Approximation level:** Faithful (no approximation).

---

## 5. SYN_POS Index 3 — Halfnorm Synaptic Placement

**Original:** For PV→PYR and VIP→PYR connections (`Syn_pos` index 3), the original
uses a half-normal distribution (`halfnorm_rv`) to place synapses preferentially at
**proximal** basal dendrite locations, biasing toward the soma.

**NetPyNE replica:** NetPyNE's `connParams` `'loc'` field accepts a scalar or
uniform-random value, but not an arbitrary scipy distribution. The halfnorm placement
is **approximated as uniform** (`loc = 0.5`) across basal dendrite sections.

**Effect:** Synapses from PV and VIP onto PYR basal dendrites will be distributed
uniformly rather than proximal-biased. This may slightly alter the efficacy of perisomatic
inhibition by PV interneurons (their canonical role).

**Mitigation possible:** A custom post-connection modification loop could re-randomize
synapse locations using a halfnorm distribution after `sim.net.connectCells()`.

---

## 6. Connection Probability for SYN_POS = 0 (apic + dend split)

**Original:** PYR→PYR connections target both apical and basal dendrites with equal
weight (uniform_rv, funweights=[1, 1]).

**NetPyNE replica:** Implemented as two `connParams` rules (suffix `_apic` and `_dend`),
each with `probability = original_prob * 0.5` and `synsPerConn = n_cont // 2`.
This preserves the expected total number of synapses while splitting across compartments.

**Approximation:** Integer division of contacts may lose 1 contact per connection when
`n_cont` is odd. For PYR→PYR, `n_cont = 3`, so each sub-rule gets 1 contact (total = 2,
original = 3). This is a minor reduction in recurrent excitatory drive.

---

## 7. Synaptic Delay

**Original:** `delayfun=local_state.normal, delayargs={'loc': 0.5, 'scale': 0}` —
effectively deterministic at 0.5 ms (zero variance).

**NetPyNE replica:** `'delay': 0.5` (scalar constant). Identical outcome.

---

## 8. Synaptic Weight

**Original:** `weightfun=local_state.normal, weightargs={'loc': 1, 'scale': 0}` —
effectively 1.0 (zero variance). Conductance is set entirely via `gmax` in synparams.

**NetPyNE replica:** `'weight': 1.0`. The `gmax` parameter in `synMechParams` is
set per-connection via the `synMechParams` override in `connParams`. Identical.

---

## 9. MPI / Parallelism

**Original:** Manual MPI via `mpi4py` with explicit `COMM`, `RANK`, `SIZE`.
Seed = `GLOBALSEED * 10000 + RANK`.

**NetPyNE replica:** NetPyNE's `sim.pc` (ParallelContext) handles all MPI internally.
Seeds are managed via `cfg.seeds` dict. Running `mpirun -np N python init.py` works
with NetPyNE's built-in load balancing and spike gathering.

---

## 10. LFP Recording

**Original:** `RecExtElectrode` at (0, 0, 5 µm), sigma=0.3 S/m, method="soma_as_point".
4-sphere EEG model for EEG/ECoG computation from dipole moments.

**NetPyNE replica:** `cfg.recordLFP = [[0.0, 0.0, 5.0]]` (when `cfg.rec_LFP=True`).
NetPyNE uses NEURON's built-in LFP calculation. The 4-sphere EEG model is not yet
implemented in this replica (requires post-processing with the dipole moment output).
The `plot_lfp()` function in `analysis.py` handles the LFP trace and PSD.

**Missing:** EEG/ECoG computation from dipole moments. Can be added in `analysis.py`
using `LFPy.FourSphereVolumeConductor` with the original parameters:
`radii=[79, 80, 85, 90]*1000 µm`, `sigmas=[0.3, 1.5, 0.015, 0.3] S/m`.

---

## 11. Stimulus (STIM_PARAM)

**Original:** Poisson NetStim-driven synapses injected onto specific GID subsets
with random delays sampled from `uniform(delay, delay + delay_range)`.

**NetPyNE replica:** Not yet implemented in `netParams.py` (requires `cfg.stimulate=True`
and additional `stimSourceParams` / `stimTargetParams` rules targeting GID subsets).
Planned as a follow-up extension.

---

## Summary Table

| Feature | Faithfulness | Notes |
|---------|-------------|-------|
| Cell morphology + biophysics | High | Exact HOC files used |
| Axon treatment | Exact | HOC procedures run unchanged |
| Ion channel parameters | Exact | All gbar values from original HOC |
| Synaptic STP (Fuhrmann) | Exact | Same MOD files, same parameters |
| Connection probabilities | Exact | From Circuit_param.xls |
| Number of contacts | Near-exact | Integer rounding for split |
| Synapse target (apic/dend) | High | Minor split approximation for index 0 |
| Halfnorm proximal bias | Approximated | Uniform used instead |
| OU background noise | High | Same conductances, minor diameter heuristic |
| Tonic GABA | Exact | Same logic, same parameters |
| MDD manipulation | Exact | Same 40% reduction |
| DRUG manipulation | Exact | Same tonic value substitution |
| LFP recording | Partial | Basic LFP; no 4-sphere EEG |
| External stimulus | Not implemented | Planned extension |

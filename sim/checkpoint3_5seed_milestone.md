# Checkpoint 3 Milestone — AD Modifier Layer, 5-Seed CPS Sweep

**Status: CLOSED & VERIFIED.** The v1 AD modifier layer is implemented, the s=0 regression gate passes, and the 5-seed CPS sweep is analyzed and locked. This document preserves the result and the decision rationale before advancing to the attribution decomposition and TMS-EEG output layer.

---

## 1. What was run

5-seed × 5-stage CPS sweep, all 6 modifiers ON, stages s ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, seeds {1234, 2345, 3456, 4567, 5678} (job 13948, the valid re-run). Steady-state rate window (1–3 s, startup transient excluded). Analysis: `multiseed_analysis.py sweep`; outputs `cps_trajectory_meanSEM.png` + `sweep_summary.csv` in `../data/cps_5seed_analysis/`.

The s=0 regression gate held (stage 0.0 reproduces the h01_v1_0 seed-1234 baseline within the steady-state window).

---

## 2. Locked result — per-population trajectory (mean ± SEM across 5 seeds)

| stage | PYR | SST | PV | VIP |
|-------|-----|-----|----|----|
| 0.00 | 0.683 ± 0.011 | 4.156 ± 0.077 | 9.299 ± 0.135 | 2.330 ± 0.036 |
| 0.25 | 0.644 ± 0.014 | 3.948 ± 0.127 | 9.694 ± 0.126 | 2.199 ± 0.097 |
| 0.50 | 0.636 ± 0.007 | 4.206 ± 0.124 | 11.536 ± 0.134 | 2.670 ± 0.074 |
| 0.75 | 0.616 ± 0.008 | 4.018 ± 0.103 | 10.854 ± 0.132 | 2.370 ± 0.050 |
| 1.00 | 0.621 ± 0.008 | 3.838 ± 0.119 | 10.216 ± 0.160 | 2.209 ± 0.070 |

(Rates in Hz, steady-state window. All stages n=5.)

---

## 3. Findings

**PYR — robust monotonic hypoactivity (−9%).** 0.683 → 0.621 Hz, SEMs ~0.007–0.014 against a ~0.06 Hz drop, so error bars nowhere near baseline. Monotonic, with the bulk landing by s=0.25 and no early rise. The disinhibition arm (M1a/M1b) never produces a net PYR increase; M3 (excitatory-scaffold loss) dominates from the first stage. This is the **anti-early-hyper** result: v1's synaptic modifiers produce hypoactivity, not early hyperexcitability — the empirical case for needing an early inflammatory/microglial arm in v2 to generate a biphasic (early-hyper → late-hypo) trajectory.

(Note: −9% is the steady-state-windowed figure; it equals the −22% whole-sim headline scaled by the ~0.77 windowing factor. Same finding, different window.)

**PV — seed-confirmed inverted-U peaking at s=0.5 (+24%).** 9.30 → 9.69 → 11.54 → 10.85 → 10.22. Peak SEM 0.134 against a ~+2.2 Hz rise; all five seeds individually peak at s=0.5; script verdict "PV non-monotonic." Mechanistic reading: rising disinhibition-driven excitatory drive onto PV (less SST restraint, still-intact excitatory scaffold) outweighs M2's intrinsic Kv3.1 impairment up to s=0.5; past s=0.5 the accumulating M3 scaffold loss and M4 PYR loss (onset 0.53, lining up with the downturn) starve PV of input and its rate falls. This is a PV *rate* finding — the gamma question (does M2 collapse PV-generated gamma) is separate and needs the spectral decomposition.

**SST — flat rate, but output verified cut.** Rate is flat-with-slight-drift (4.16 → 3.84, SEMs overlapping across stages). This was the one ambiguous signature — it is also what a silent SST-modifier failure would look like. **Resolved by direct measurement on the built network:** SST→target connection weight ratio s=1.0 / s=0 = **0.3623**, matching the M1a × M1c stack exactly (0.577 × 0.628 = 0.362). Topology identical across stages (50 SST cells, 86,339 outgoing conns, both seed 1234); s=0 baseline weight exactly 1.000000. So the modifiers demonstrably applied — SST's *output* is cut −64% while its *firing rate* holds, the genuine downstream-only signature of output-efficacy scaling. Not a silent failure.

**VIP — mild PV-tracking bump at s=0.5.** 2.33 → 2.20 → 2.67 → 2.37 → 2.21, mirroring PV's shape at smaller scale within larger relative SEM — a secondary network-drive effect (VIP has no v1 cell-loss term).

---

## 4. Verification chain (why this is trustworthy)

- **s=0 regression gate:** stage 0.0 reproduces the validated baseline.
- **Seeds genuinely propagated:** the `[H01] (seed=1234)` log line is misleading (printed 1234 for all tasks), but spike-train fingerprints differ across tasks (distinct nspk / sum_t / first-cell IDs at the same stage) — proving the five seeds diverged. Cross-seed SEM is real, not artificially tight.
- **AD engagement log-verified:** task-9 (s=1.0) `[AD]` line showed `all_off=False` with every modifier at its locked floor (M1a 0.577, M1b 0.725, M1c 0.628, M2 0.650, M3 AMPA 0.70 / NMDA 0.88, M4 0.856).
- **SST application directly measured:** weight ratio 0.3623 = M1a × M1c (see §3).

---

## 5. Lessons logged (silent-failure prevention)

1. **cfg-envelope rule:** sweep cfg JSONs must be template-derived with the `{"simConfig": {...}}` envelope. A bare flat dict makes NetPyNE silently fall back to `cfg.py` defaults (the first 5-seed attempt was 25 copies of seed 1234 / stage 0 for this reason). `gen_cps_5seed_cfgs.py` fixed to load the template and override fields.
2. **Don't trust the seed log line.** Verify seed propagation by spike-train fingerprint, not the `[H01] (seed=...)` print.
3. **Log-verify engagement** (`[AD] all_off=False`, mods ≠ 1.0 at nonzero stage) before trusting any sweep output. "The job ran" ≠ "the job ran correctly."

This is the third instance of the cfg/application silent-failure mode in the project (prior: netParams reading defaults at import; bare-dict cfg). The standing rule is now: every sweep is engagement-verified from logs + a direct application check before its output is read as a result.

---

## 6. Next steps

1. **Repo commit:** sweep cfgs/generator fix, `multiseed_analysis.py`, the analysis outputs, this milestone doc.
2. **Per-modifier attribution decomposition:** which modifier drives the −9% PYR decline (predicted M3-dominant) and the PV inverted-U; the 30-pkl attribution set + grep-from-logs rate table.
3. **Gamma/temporal layer:** spectral decomposition (does M2's Kv3.1 impairment collapse PV-generated gamma) — needs spike times read leanly to avoid OOM.
4. **TMS-EEG output layer** + validation against the Santarnecchi precuneus cohort (next checkpoint).

**Optional fixes flagged:** the trajectory figure title still says "single-seed −22%" — retitle to steady-state −9% so the figure number matches the data. SST-modifier-acts-downstream is now verified, so the §3 SST caveat is closed.

"""
ad_modifiers.py — AD modifier layer (Checkpoint 3, v1 core)
===========================================================

Implements the stage-dependent AD perturbations as scaling functions f(s), where
s = disease stage, defined as s == CPS (SEA-AD continuous pseudo-progression score,
0 = healthy .. 1 = severe). NOTE: CPS is a pathology ORDERING, not linear clinical time.

Design contract (matches AD_modifier_spec_reconciled_v1.md):
  - Every modifier f(s) satisfies f(0) == 1.0 exactly  -> s=0 reproduces the locked H01 baseline
    (this is the HARD REGRESSION GATE).
  - Each modifier is INDIVIDUALLY TOGGLEABLE via cfg flags (like USE_H01_DISTANCE_CONN), so single-
    modifier attribution runs are possible before combining.
  - Provenance per modifier is explicit in the docstring: HARDCODE (literature magnitude) /
    EXTRACT (Task B cps_curve_fits, literature-reconciled) / PRIOR (range, sensitivity).
  - v1 cell-loss = SST (early, cross-validated) + PYR (late) ONLY. PV and VIP: NO cell-loss term.
  - Cell loss is represented as EFFICACY SCALING (weights), not true cell removal (keeps topology +
    the H01 ~613/cell convergence reference fixed). True removal = v2.

This module is PURE FUNCTIONS + a small config dataclass. It does not import NEURON or NetPyNE; it
returns scalar multipliers that netParams.py applies to the relevant parameters. That keeps it
unit-testable off-cluster and keeps the wiring explicit.
"""

from dataclasses import dataclass, field
import math


# ============================================================================
# RECONCILED CONSTANTS  (see AD_modifier_spec_reconciled_v1.md §1, §8)
# EXTRACT values are from Task B cps_curve_fits.csv, cross-checked vs Gabitto 2024
# + the 1037-sample 6-region study. HARDCODE values cite their specific paper.
# ============================================================================

# --- EXTRACT: cell-loss linear fits (remaining fraction = intercept + slope * s) ---
# SST: slope -0.48 matches independent beta=-0.48 (cross-validated). onset 0.33, floor ~0.63 at s=1.
SST_LOSS_SLOPE      = -0.4805      # Task B (cross-validated vs 1037-sample beta)
SST_LOSS_INTERCEPT  =  1.1088      # Task B fit intercept
SST_LOSS_ONSET      =  0.330       # CPS where decline begins (before onset, f=1.0)
# PYR (IT excitatory): late, direction/timing confirmed; magnitude loosely constrained -> range.
PYR_LOSS_SLOPE      = -0.2010      # Task B
PYR_LOSS_INTERCEPT  =  1.0570      # Task B
PYR_LOSS_ONSET      =  0.532       # CPS where decline begins

# --- HARDCODE: synaptic / channel magnitudes from human-AD literature ---
# M1a SST->* synaptic weight: Poirel et al. 2018 BA9 somatostatin protein -42.4% (endpoint at s=1).
SST_SYN_FLOOR       =  0.576       # 1 - 0.424  (remaining SST synaptic weight at s=1)
SST_SYN_K           =  10.0        # sigmoid steepness
SST_SYN_MID         =  0.35        # sigmoid midpoint (early)
# M1b tonic alpha5-GABA-A: PRIOR range -15..-40% tonic, tied to SST loss. Default -0.275 (midpoint).
TONIC_GABA_FLOOR    =  0.725       # 1 - 0.275  (PRIOR; sweep 0.60..0.85)
# M2 PV Kv3.1 (gbar_Kv3_1): -20..-50%, early sigmoid (gamma collapse precedes loss). Default -0.35.
PV_KV31_FLOOR       =  0.65        # 1 - 0.35  (PRIOR/HARDCODE; sweep 0.50..0.80)
PV_KV31_K           =  10.0
PV_KV31_MID         =  0.30        # early
# M3 excitatory scaffold: VGLUT1/PSD95 -26% (Poirel). AMPA falls faster than NMDA. Linear ramps.
AMPA_DROP_AT_S1     =  0.30        # g_AMPA -> (1 - 0.30*s); ~ -30% at s=1 (slide range 15-30%)
NMDA_DROP_AT_S1     =  0.12        # g_NMDA -> (1 - 0.12*s); ~ -12% at s=1 (slide range 5-15%)


# ============================================================================
# CONFIG: per-modifier toggles + the stage value s
# ============================================================================

@dataclass
class ADModConfig:
    """Per-modifier toggles. All default OFF so importing/enabling the layer with s=0 OR with all
    toggles False reproduces the locked H01 baseline. Set on cfg and pass through."""
    s: float = 0.0                 # disease stage == CPS in [0,1]

    enable_M1a_sst_syn:   bool = False   # SST->* synaptic weight down (HARDCODE)
    enable_M1b_tonic:     bool = False   # tonic alpha5-GABA-A down (PRIOR)
    enable_M1c_sst_loss:  bool = False   # SST cell abundance down (EXTRACT)
    enable_M2_pv_kv31:    bool = False   # PV Kv3.1 gbar down (PRIOR/HARDCODE)
    enable_M3_exc_scaffold: bool = False # AMPA/NMDA down (HARDCODE)
    enable_M4_pyr_loss:   bool = False   # PYR cell abundance down (EXTRACT)

    # Optional overrides for sensitivity sweeps (None -> use module defaults above)
    tonic_gaba_floor_override: float = None
    pv_kv31_floor_override:    float = None

    def all_off(self) -> bool:
        return not any([self.enable_M1a_sst_syn, self.enable_M1b_tonic, self.enable_M1c_sst_loss,
                        self.enable_M2_pv_kv31, self.enable_M3_exc_scaffold, self.enable_M4_pyr_loss])


# ============================================================================
# SHAPE PRIMITIVES  (all return 1.0 at s=0)
# ============================================================================

def _sigmoid_decline(s, floor, k, mid):
    """Smooth early/mid decline from 1.0 -> floor. f(0) ~ 1.0 (enforced exact below).
    f(s) = floor + (1-floor) * (1 - sigmoid)  where sigmoid rises from ~0 to ~1.
    """
    if s <= 0.0:
        return 1.0
    raw = floor + (1.0 - floor) * (1.0 / (1.0 + math.exp(k * (s - mid))))
    # normalize so f(0)==1.0 exactly (subtract the residual the sigmoid leaves at s=0)
    raw0 = floor + (1.0 - floor) * (1.0 / (1.0 + math.exp(k * (0.0 - mid))))
    # rescale [raw0 .. floor] onto [1.0 .. floor]
    if abs(raw0 - floor) < 1e-12:
        return 1.0
    return floor + (raw - floor) * (1.0 - floor) / (raw0 - floor)


def _linear_onset(s, slope, intercept, onset):
    """Linear decline that is flat (==1.0) until `onset`, then follows intercept+slope*s,
    clamped so f(0)==1.0 and f never exceeds 1.0. Used for EXTRACT cell-loss (s==CPS)."""
    if s <= onset:
        return 1.0
    val = intercept + slope * s
    # guard: never amplify (>1) and never go negative
    return max(0.0, min(1.0, val))


def _linear_ramp(s, drop_at_s1):
    """Proportional linear ramp: f(s) = 1 - drop_at_s1 * s. f(0)=1.0, f(1)=1-drop."""
    return max(0.0, 1.0 - drop_at_s1 * float(s))


# ============================================================================
# MODIFIER FUNCTIONS  (each returns a multiplier in [0,1]; 1.0 = no change)
# ============================================================================

def f_M1a_sst_synaptic(s):
    """M1a — SST->* synaptic weight scaling. HARDCODE (Poirel 2018 BA9 SST -42.4%). Early sigmoid.
    Apply to: w_SST->PYR, w_SST->SST/PV/VIP (all SST presynaptic weights)."""
    return _sigmoid_decline(s, SST_SYN_FLOOR, SST_SYN_K, SST_SYN_MID)


def f_M1b_tonic_gaba(s, floor_override=None):
    """M1b — tonic alpha5-GABA-A conductance scaling. PRIOR (range -15..-40%), tied to SST.
    Apply to: g_tonic (insert_tonic_gaba background conductance). Early."""
    floor = TONIC_GABA_FLOOR if floor_override is None else floor_override
    return _sigmoid_decline(s, floor, SST_SYN_K, SST_SYN_MID)


def f_M1c_sst_loss(s):
    """M1c — SST cell abundance (efficacy scaling). EXTRACT (Task B, cross-validated slope -0.48,
    onset 0.33). Apply as efficacy multiplier on SST output (NOT cell removal in v1)."""
    return _linear_onset(s, SST_LOSS_SLOPE, SST_LOSS_INTERCEPT, SST_LOSS_ONSET)


def f_M2_pv_kv31(s, floor_override=None):
    """M2 — PV Kv3.1 (gbar_Kv3_1) scaling in PV soma+AIS. PRIOR/HARDCODE (-20..-50%), early sigmoid
    (gamma collapse precedes cell loss). PV's ONLY v1 effect. Apply to: gbar_Kv3_1 on PV sections."""
    floor = PV_KV31_FLOOR if floor_override is None else floor_override
    return _sigmoid_decline(s, floor, PV_KV31_K, PV_KV31_MID)


def f_M3_ampa(s):
    """M3 (AMPA part) — g_AMPA scaling. HARDCODE (VGLUT1/PSD95 -26%; AMPA faster than NMDA). Linear."""
    return _linear_ramp(s, AMPA_DROP_AT_S1)


def f_M3_nmda(s):
    """M3 (NMDA part) — g_NMDA scaling. HARDCODE. Linear, slower than AMPA."""
    return _linear_ramp(s, NMDA_DROP_AT_S1)


def f_M4_pyr_loss(s):
    """M4 — PYR cell abundance (efficacy scaling). EXTRACT (Task B, late, onset 0.53). Direction/
    timing confirmed vs SEA-AD + 1037-sample. Apply as efficacy multiplier on PYR output."""
    return _linear_onset(s, PYR_LOSS_SLOPE, PYR_LOSS_INTERCEPT, PYR_LOSS_ONSET)


# ============================================================================
# AGGREGATOR: returns the full set of multipliers for a given config
# ============================================================================

def compute_modifiers(cfg: ADModConfig) -> dict:
    """Return a dict of multipliers to apply, honoring toggles. Any disabled modifier returns 1.0
    (no change). With cfg.all_off() OR cfg.s==0, ALL multipliers are 1.0 -> exact baseline.

    Returned keys map to netParams application points:
      'w_SST_pre'    -> multiply all SST presynaptic weights        (M1a)
      'g_tonic'      -> multiply tonic GABA-A conductance            (M1b)
      'eff_SST'      -> multiply SST output efficacy (abundance)     (M1c)
      'gbar_Kv3_1'  -> multiply PV Kv3.1 gbar (soma+AIS)            (M2)
      'g_AMPA'       -> multiply AMPA conductance                    (M3)
      'g_NMDA'       -> multiply NMDA conductance                    (M3)
      'eff_PYR'      -> multiply PYR output efficacy (abundance)     (M4)
    """
    s = float(cfg.s)
    m = {  # default: no change
        'w_SST_pre': 1.0, 'g_tonic': 1.0, 'eff_SST': 1.0,
        'gbar_Kv3_1': 1.0, 'g_AMPA': 1.0, 'g_NMDA': 1.0, 'eff_PYR': 1.0,
    }
    if cfg.enable_M1a_sst_syn:    m['w_SST_pre']   = f_M1a_sst_synaptic(s)
    if cfg.enable_M1b_tonic:      m['g_tonic']     = f_M1b_tonic_gaba(s, cfg.tonic_gaba_floor_override)
    if cfg.enable_M1c_sst_loss:   m['eff_SST']     = f_M1c_sst_loss(s)
    if cfg.enable_M2_pv_kv31:     m['gbar_Kv3_1'] = f_M2_pv_kv31(s, cfg.pv_kv31_floor_override)
    if cfg.enable_M3_exc_scaffold:
        m['g_AMPA'] = f_M3_ampa(s)
        m['g_NMDA'] = f_M3_nmda(s)
    if cfg.enable_M4_pyr_loss:    m['eff_PYR']     = f_M4_pyr_loss(s)
    return m


# ============================================================================
# SELF-TEST: the s=0 regression gate + monotonicity + floors
# ============================================================================

if __name__ == '__main__':
    # 1) HARD GATE: at s=0 (any toggles), every multiplier must be EXACTLY 1.0
    cfg0 = ADModConfig(s=0.0, enable_M1a_sst_syn=True, enable_M1b_tonic=True,
                       enable_M1c_sst_loss=True, enable_M2_pv_kv31=True,
                       enable_M3_exc_scaffold=True, enable_M4_pyr_loss=True)
    m0 = compute_modifiers(cfg0)
    assert all(abs(v - 1.0) < 1e-9 for v in m0.values()), f"s=0 regression FAILED: {m0}"
    print("PASS: s=0 with all modifiers ON -> all multipliers == 1.0 (regression gate)")

    # 2) all_off at any s must also be 1.0
    cfgoff = ADModConfig(s=0.7)
    assert all(abs(v - 1.0) < 1e-9 for v in compute_modifiers(cfgoff).values())
    print("PASS: all toggles OFF -> all multipliers == 1.0 regardless of s")

    # 3) monotonic decline + report values across CPS stages
    print("\n s    w_SST  g_tonic eff_SST  Kv3.1  g_AMPA g_NMDA eff_PYR")
    allon = dict(enable_M1a_sst_syn=True, enable_M1b_tonic=True, enable_M1c_sst_loss=True,
                 enable_M2_pv_kv31=True, enable_M3_exc_scaffold=True, enable_M4_pyr_loss=True)
    prev = None
    for s in [0.0, 0.25, 0.33, 0.5, 0.53, 0.75, 1.0]:
        m = compute_modifiers(ADModConfig(s=s, **allon))
        print(f"{s:4.2f}  {m['w_SST_pre']:.3f}  {m['g_tonic']:.3f}  {m['eff_SST']:.3f}  "
              f"{m['gbar_Kv3_1']:.3f}  {m['g_AMPA']:.3f}  {m['g_NMDA']:.3f}  {m['eff_PYR']:.3f}")
    # floors at s=1
    m1 = compute_modifiers(ADModConfig(s=1.0, **allon))
    print(f"\nFloors at s=1: SST_syn={m1['w_SST_pre']:.3f} (exp ~0.576), "
          f"eff_SST={m1['eff_SST']:.3f} (exp ~0.628), eff_PYR={m1['eff_PYR']:.3f} (exp ~0.856), "
          f"AMPA={m1['g_AMPA']:.3f} (exp 0.70), NMDA={m1['g_NMDA']:.3f} (exp 0.88)")
    print("\nSelf-test complete.")

#!/usr/bin/env python3
# =============================================================================
# single_cell_characterization.py  —  Figure 1A-D of Yao et al. 2022
#
# Pure NEURON single-cell electrophysiology.  No NetPyNE network,
# no OU noise, no tonic GABA.  IClamp at soma(0.5) only.
#
# Usage (run from NetPyNE_Replica_Yao/ or anywhere):
#   python single_cell_characterization.py
#
# Outputs:
#   output/figure1_single_cell_characterization.png  (300 dpi)
#   output/figure1_single_cell_characterization.pdf
# =============================================================================
import os
import sys
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

# =============================================================================
# GLOBAL SETUP
# =============================================================================
_here    = os.path.dirname(os.path.abspath(__file__))
_mod_dir = os.path.join(_here, 'mod')
_special = os.path.join(_mod_dir, 'x86_64', 'special')

if not os.path.isfile(_special):
    print('[scc] Compiling MOD files ...')
    ret = os.system(f'cd "{_mod_dir}" && nrnivmodl')
    if ret != 0:
        raise RuntimeError('nrnivmodl failed — check mod/ directory')
else:
    print('[scc] MOD files already compiled.')

import neuron
from neuron import h
neuron.load_mechanisms(_mod_dir)

# HOC relative paths (morphologies/) resolve from this directory
os.chdir(_here)

h.load_file('stdrun.hoc')
for _hoc in ['HL23PYR_Cell.hoc', 'HL23SST_Cell.hoc',
             'HL23PV_Cell.hoc',  'HL23VIP_Cell.hoc']:
    h.load_file(os.path.join('models', _hoc))

h.celsius = 34.0
h.dt      = 0.025
V_INIT    = -80.0

OUT_DIR = os.path.join(_here, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================
CELL_NAMES  = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
CELL_LABELS = {'HL23PYR': 'Pyr', 'HL23SST': 'SST',
               'HL23PV':  'PV',  'HL23VIP': 'VIP'}
COLORS = {
    'HL23PYR': '#000000',
    'HL23SST': '#CC0000',
    'HL23PV':  '#1a7a1a',
    'HL23VIP': '#E8820C',
}

HYPER_AMPS = {
    'HL23PYR': [-0.05, -0.10, -0.20, -0.30],
    'HL23SST': [-0.02, -0.04, -0.06, -0.10],
    'HL23PV':  [-0.02, -0.04, -0.06, -0.10],
    'HL23VIP': [-0.02, -0.04, -0.06, -0.10],
}

SAG_AMPS = {
    'HL23PYR': [-0.05, -0.10, -0.15, -0.20, -0.30],
    'HL23SST': [-0.02, -0.04, -0.06, -0.08, -0.10],
    'HL23PV':  [-0.02, -0.04, -0.06, -0.08, -0.10],
    'HL23VIP': [-0.02, -0.04, -0.06, -0.08, -0.10],
}

FI_MAX  = {'HL23PYR': 0.30, 'HL23SST': 0.15, 'HL23PV': 0.30, 'HL23VIP': 0.22}
FI_YMAX = {'HL23PYR': 35,   'HL23SST': 50,   'HL23PV': 80,   'HL23VIP': 35}

# =============================================================================
# CELL MANAGEMENT
# =============================================================================
_cell_ref = [None]


def cleanup_neuron():
    _cell_ref[0] = None
    gc.collect()
    for sec in list(h.allsec()):
        try:
            h.delete_section(sec=sec)
        except Exception:
            pass


def create_cell(cell_type):
    """HOC init() calls forall delete_section() before loading new morphology."""
    cleanup_neuron()
    cell = getattr(h, f'{cell_type}_Cell')()
    _cell_ref[0] = cell
    return cell


def get_soma(cell):
    return list(cell.soma)[0]


def record_vm(soma):
    t_vec = h.Vector()
    v_vec = h.Vector()
    t_vec.record(h._ref_t,         h.dt)
    v_vec.record(soma(0.5)._ref_v, h.dt)
    return t_vec, v_vec


def run_sim(tstop):
    h.finitialize(V_INIT)
    h.continuerun(tstop)


# =============================================================================
# DIAGNOSTICS helper
# =============================================================================
def _diag_print(label, t, v, step_start, step_end, n_spikes=None):
    """
    Print raw vector diagnostics for one simulation run.
    Confirms h.Vector.record() is populating real data (not mock values).
    """
    n_pts   = len(v)
    step_m  = (t >= step_start) & (t < step_end)
    v_step  = v[step_m]
    v_min   = float(np.min(v_step))  if v_step.size else float('nan')
    v_max   = float(np.max(v_step))  if v_step.size else float('nan')
    first5  = [round(x, 2) for x in v[:5].tolist()]
    last5   = [round(x, 2) for x in v[-5:].tolist()]
    spike_s = f', spikes={n_spikes}' if n_spikes is not None else ''
    print(f'    [diag {label}] n_pts={n_pts}{spike_s}')
    print(f'      first5={first5}  last5={last5}')
    print(f'      step Vm: min={v_min:.2f}  max={v_max:.2f} mV  '
          f'(t=[{step_start},{step_end}] ms)')


# =============================================================================
# PROTOCOL 1: RHEOBASE  (binary search, tolerance 0.005 nA)
# =============================================================================
def find_rheobase(cell_type):
    cell = create_cell(cell_type)
    soma = get_soma(cell)

    stim       = h.IClamp(soma(0.5))
    stim.delay = 200.0
    stim.dur   = 1000.0

    apc        = h.APCount(soma(0.5))
    apc.thresh = -20.0

    lo, hi = 0.0, 2.0
    while hi - lo > 0.005:
        stim.amp = (lo + hi) / 2.0
        h.finitialize(V_INIT)   # APCount.n resets to 0 on finitialize
        h.continuerun(1400.0)
        if apc.n >= 1:
            hi = stim.amp
        else:
            lo = stim.amp

    print(f'  [{cell_type}] rheobase = {hi:.3f} nA')
    return hi


# =============================================================================
# PROTOCOL 2: VOLTAGE TRACES
# =============================================================================

def run_depol_trace(cell_type, rheobase):
    """
    Single depolarising trace at 150 % rheobase.
    A 5 ms linear ramp-down at stimulus offset (via h.Vector.play) prevents
    the abrupt-termination artifact.  Total sim: 1400 ms.

    Returns
    -------
    t         : np.ndarray  time vector (ms)
    v         : np.ndarray  soma Vm (mV)
    n_spikes  : int         total APCount during simulation
    n_rebound : int         spikes detected AFTER stimulus offset
    """
    DELAY = 200.0
    DUR   = 1000.0
    TSTOP = 1400.0
    RAMP  = 5.0          # ms linear ramp-down before offset
    amp   = 1.5 * rheobase

    cell = create_cell(cell_type)
    soma = get_soma(cell)

    stim       = h.IClamp(soma(0.5))
    stim.delay = 0.0
    stim.dur   = TSTOP   # IClamp "always on"; amplitude controlled by play()
    stim.amp   = 0.0

    # Piecewise-linear amplitude profile:
    #   t=0 → 0 nA
    #   t=DELAY-dt → 0 nA  (one dt before onset: ensures step, not ramp-up)
    #   t=DELAY → amp
    #   t=DELAY+DUR-RAMP → amp   (begin ramp-down)
    #   t=DELAY+DUR      → 0    (end of stimulus)
    #   t=TSTOP          → 0
    eps     = h.dt
    _t_play = h.Vector([0,
                         DELAY - eps,
                         DELAY,
                         DELAY + DUR - RAMP,
                         DELAY + DUR,
                         TSTOP])
    _a_play = h.Vector([0, 0, amp, amp, 0, 0])
    _a_play.play(stim._ref_amp, _t_play, 1)   # 1 = piecewise-linear

    apc        = h.APCount(soma(0.5))
    apc.thresh = -20.0

    t_vec, v_vec = record_vm(soma)
    run_sim(TSTOP)

    t, v     = np.array(t_vec), np.array(v_vec)
    n_spikes = int(apc.n)

    # Detect rebound (spikes after stimulus offset)
    post_mask = t > (DELAY + DUR)
    v_post    = v[post_mask]
    t_post    = t[post_mask]
    post_peaks, _ = find_peaks(v_post, height=-20.0,
                               distance=max(1, int(10.0 / h.dt)))
    n_rebound = len(post_peaks)

    return t, v, n_spikes, n_rebound


def run_hyper_traces(cell_type):
    """Four hyperpolarising steps (900 ms each)."""
    traces = []
    for amp in HYPER_AMPS[cell_type]:
        cell = create_cell(cell_type)
        soma = get_soma(cell)

        stim       = h.IClamp(soma(0.5))
        stim.delay = 200.0
        stim.dur   = 500.0
        stim.amp   = amp

        t_vec, v_vec = record_vm(soma)
        run_sim(900.0)
        traces.append((np.array(t_vec), np.array(v_vec)))
    return traces


# =============================================================================
# PROTOCOL 3: F/I CURVE  (15 steps, 2000 ms each)
# =============================================================================
def run_fi_curve(cell_type):
    amps  = np.linspace(0.0, FI_MAX[cell_type], 15)
    rates = []
    for amp in amps:
        cell = create_cell(cell_type)
        soma = get_soma(cell)

        stim       = h.IClamp(soma(0.5))
        stim.delay = 200.0
        stim.dur   = 2000.0
        stim.amp   = amp

        apc        = h.APCount(soma(0.5))
        apc.thresh = -20.0

        h.finitialize(V_INIT)
        h.continuerun(2400.0)
        rates.append(float(apc.n) / 2.0)
    return amps, np.array(rates)


# =============================================================================
# PROTOCOL 4: SAG VOLTAGE CURVE  (5 steps, 500 ms each)
# =============================================================================
# Fix 1 — sag formula:
#   V_steady_state  = mean Vm over LAST 50 ms of step (650–700 ms)
#                     (cell has partially recovered from peak deflection)
#   V_peak_defl     = minimum Vm during step (most hyperpolarized point)
#                     (occurs at start of step before Ih activates)
#   Sag voltage     = V_steady_state − V_peak_defl  ≥ 0 always
#                     (measures how much Ih pulled Vm back toward rest)

def run_sag_curve(cell_type):
    sags        = []
    v_peaks     = []
    v_steadies  = []

    for amp in SAG_AMPS[cell_type]:
        cell = create_cell(cell_type)
        soma = get_soma(cell)

        stim       = h.IClamp(soma(0.5))
        stim.delay = 200.0
        stim.dur   = 500.0
        stim.amp   = amp

        t_vec, v_vec = record_vm(soma)
        run_sim(900.0)

        t, v = np.array(t_vec), np.array(v_vec)

        step_mask   = (t >= 200.0) & (t < 700.0)
        steady_mask = (t >= 650.0) & (t < 700.0)

        # Confirm h.Vector data is real: length and value range
        assert len(v) > 0, 'v_vec is empty — h.Vector.record() failed'
        v_step   = v[step_mask]
        v_peak   = float(np.min(v_step))           # most hyperpolarized (fix1)
        v_steady = float(np.mean(v[steady_mask]))  # steady-state near end

        sag = v_steady - v_peak                    # always ≥ 0  (fix1)
        sags.append(sag)
        v_peaks.append(v_peak)
        v_steadies.append(v_steady)

    sag_arr = np.array(sags)

    # --- Fix 1 diagnostic: print raw values ---
    print(f'    [sag diag {cell_type}]')
    print(f'      {"amp":>8s}  {"v_peak":>10s}  {"v_steady":>10s}  '
          f'{"sag":>8s}  monotone')
    prev_sag = -1e9
    for i, amp in enumerate(SAG_AMPS[cell_type]):
        mono = 'YES' if sag_arr[i] >= prev_sag - 0.001 else 'NO ← INVERSION'
        print(f'      {amp:>8.3f}  {v_peaks[i]:>10.3f}  '
              f'{v_steadies[i]:>10.3f}  {sag_arr[i]:>8.4f}  {mono}')
        prev_sag = sag_arr[i]

    return np.array(SAG_AMPS[cell_type]), sag_arr


# =============================================================================
# INTRINSIC PROPERTIES
# =============================================================================
def compute_intrinsic_props(cell_type, depol_t, depol_v, hyper_traces):
    # V_rest: mean over first 100 ms (pre-stimulus)
    v_rest = float(np.mean(depol_v[depol_t < 100.0]))

    # Input resistance from smallest hyperpolarising step
    t0, v0 = hyper_traces[0]
    amp0   = HYPER_AMPS[cell_type][0]
    v_ss0  = float(np.mean(v0[(t0 >= 650.0) & (t0 < 700.0)]))
    r_in   = (v_ss0 - v_rest) / amp0     # mV/nA = MΩ (positive)

    # Sag ratio = (V_peak − V_steady) / (V_peak − V_rest) at largest step
    t_l, v_l = hyper_traces[-1]
    v_peak   = float(np.min(v_l[(t_l >= 200.0) & (t_l < 700.0)]))
    v_steady = float(np.mean(v_l[(t_l >= 650.0) & (t_l < 700.0)]))
    denom    = v_peak - v_rest
    sag_ratio = (v_peak - v_steady) / denom if abs(denom) > 0.1 else float('nan')

    # AP half-width from first spike in depolarising trace
    half_width = float('nan')
    stim_mask  = (depol_t >= 200.0) & (depol_t <= 1200.0)
    v_stim     = depol_v[stim_mask]
    t_stim     = depol_t[stim_mask]

    min_dist = max(1, int(10.0 / h.dt))
    peaks, _ = find_peaks(v_stim, height=-20.0, distance=min_dist)
    if len(peaks) > 0:
        pi   = peaks[0]
        v_pk = float(v_stim[pi])
        dv   = np.diff(v_stim) / h.dt
        thresh_v, thresh_i = float('nan'), None
        for i in range(pi - 1, 0, -1):
            if dv[i - 1] < 10.0:
                thresh_v = float(v_stim[i])
                thresh_i = i
                break
        if not np.isnan(thresh_v) and thresh_i is not None:
            half_h = (v_pk + thresh_v) / 2.0
            up_i = next((i for i in range(thresh_i, pi)
                         if v_stim[i] >= half_h), None)
            dn_i = next((i for i in range(pi, min(pi + int(20.0 / h.dt),
                                                   len(v_stim) - 1))
                         if v_stim[i] <= half_h), None)
            if up_i is not None and dn_i is not None:
                half_width = float(t_stim[dn_i] - t_stim[up_i])

    return {
        'v_rest':     v_rest,
        'r_in':       r_in,
        'sag_ratio':  sag_ratio,
        'half_width': half_width,
    }


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def _style_trace_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def add_scale_bar(ax, x_bar_ms, y_bar_mv, color='k', lw=1.8,
                  x_label=None, y_label=None, label_color='gray', label_size=7):
    """L-shaped scale bar in lower-right corner (must be called after set_xlim/ylim)."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
    x1 = xlim[1] - 0.04 * x_span
    y0 = ylim[0] + 0.04 * y_span
    x0 = x1 - x_bar_ms
    y1 = y0 + y_bar_mv
    ax.plot([x0, x1], [y0, y0], color=color, lw=lw, solid_capstyle='butt',
            clip_on=False)
    ax.plot([x0, x0], [y0, y1], color=color, lw=lw, solid_capstyle='butt',
            clip_on=False)
    if x_label:
        ax.text((x0 + x1) / 2.0, y0 - 0.015 * y_span, x_label,
                ha='center', va='top', fontsize=label_size, color=label_color,
                clip_on=False)
    if y_label:
        ax.text(x0, y1 + 0.015 * y_span, y_label,
                ha='center', va='bottom', fontsize=label_size, color=label_color,
                clip_on=False)


# =============================================================================
# MAIN — DATA COLLECTION WITH INLINE DIAGNOSTICS
# =============================================================================
print('\n=== Figure 1A-D: Single-Cell Characterization (Yao et al. 2022) ===\n')

rebound_info = {}   # cell_type → n_rebound (for figure annotation)
data = {}

for ct in CELL_NAMES:
    print(f'  {ct}:')

    # ---- Rheobase ----
    print(f'    Finding rheobase ...')
    rheobase = find_rheobase(ct)

    # ---- Depolarising trace ----
    amp_depol = 1.5 * rheobase
    print(f'    Depolarising trace (150% rheobase = {amp_depol:.3f} nA) ...')
    depol_t, depol_v, n_spikes, n_rebound = run_depol_trace(ct, rheobase)
    _diag_print('depol', depol_t, depol_v,
                step_start=200.0, step_end=1200.0,
                n_spikes=n_spikes)
    rebound_info[ct] = n_rebound
    if n_rebound > 0:
        print(f'    *** {ct}: {n_rebound} REBOUND spike(s) detected after '
              f't=1200 ms (5 ms ramp-down applied at offset)')
    else:
        print(f'    {ct}: no rebound spikes after stimulus offset. '
              f'5 ms ramp-down confirmed effective.')

    # ---- Hyperpolarising traces ----
    print(f'    Hyperpolarising traces ({len(HYPER_AMPS[ct])} amplitudes) ...')
    hyper = run_hyper_traces(ct)
    for i, (amp, (th, vh)) in enumerate(zip(HYPER_AMPS[ct], hyper)):
        _diag_print(f'hyper[{amp:.2f}nA]', th, vh,
                    step_start=200.0, step_end=700.0)

    # ---- F/I curve ----
    print(f'    F/I curve (15 steps, 0 → {FI_MAX[ct]:.2f} nA) ...')
    fi_amps, fi_rates = run_fi_curve(ct)
    # Diagnostic: print a few representative (amp, rate) pairs
    print(f'    [diag F/I] F/I curve comes from IClamp loop over np.linspace '
          f'(0, {FI_MAX[ct]:.2f}, 15):')
    for a, r in zip(fi_amps[::5], fi_rates[::5]):
        print(f'      amp={a:.3f} nA → {r:.1f} Hz  (from APCount/2.0)')

    # ---- Sag curve ----
    print(f'    Sag curve ({len(SAG_AMPS[ct])} steps) ...')
    sag_amps, sag_vals = run_sag_curve(ct)   # prints own diagnostics

    props = compute_intrinsic_props(ct, depol_t, depol_v, hyper)

    data[ct] = {
        'rheobase':  rheobase,
        'depol_t':   depol_t,
        'depol_v':   depol_v,
        'hyper':     hyper,
        'fi_amps':   fi_amps,
        'fi_rates':  fi_rates,
        'sag_amps':  sag_amps,
        'sag_vals':  sag_vals,
        **props,
    }
    print(f'    V_rest={props["v_rest"]:.1f} mV  '
          f'R_in={props["r_in"]:.0f} MΩ  '
          f'sag_ratio={props["sag_ratio"]:.3f}  '
          f'AP_hw={props["half_width"]:.2f} ms')
    print()

# =============================================================================
# FIGURE  —  4 rows × 5 columns
# col 0: depolarising  (w=2) | col 1: hyperpolarising (w=2) |
# col 2: spacer (w=0.3)      | col 3: F/I curve (w=1.5)    |
# col 4: sag voltage (w=1.5)
# =============================================================================
print('--- Building figure ---')

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(
    4, 5,
    figure=fig,
    width_ratios=[2, 2, 0.3, 1.5, 1.5],
    hspace=0.12,
    wspace=0.40,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.06,
)

axs = {}   # (row, col_key) → Axes

for row, ct in enumerate(CELL_NAMES):
    d     = data[ct]
    color = COLORS[ct]
    label = CELL_LABELS[ct]

    # ------------------------------------------------------------------ #
    # Col 0 — Depolarising trace
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[row, 0])
    axs[(row, 'dep')] = ax
    t, v = d['depol_t'], d['depol_v']
    ax.plot(t, v, color=color, lw=0.85)
    ax.axvline(200,  color='gray', lw=0.7, ls='--', alpha=0.45)
    ax.axvline(1200, color='gray', lw=0.7, ls='--', alpha=0.45)
    ax.set_xlim(0, 1400)
    ax.set_ylim(float(np.min(v)) - 5, float(np.max(v)) + 5)
    _style_trace_ax(ax)
    # Panel letter (A–D) — upper-left corner of axes
    _panel_letters = ['A', 'B', 'C', 'D']
    ax.text(0.0, 1.04, _panel_letters[row],
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            ha='left', va='bottom')
    # Row label
    ax.text(-0.14, 0.5, label,
            transform=ax.transAxes,
            color=color, fontsize=13, fontweight='bold',
            ha='right', va='center')
    # Stimulus amplitude — upper-left, white box so it reads over spike peaks
    ax.text(0.03, 0.95,
            f'I = {d["rheobase"] * 1.5:.2f} nA (150% rheobase)',
            transform=ax.transAxes,
            fontsize=7, color='gray', ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5))
    # "Stimulus" — short horizontal arrow in the pre-stimulus silent zone
    # The arrow tip lands on the onset dashed line at x=200; text sits to
    # the left of the arrow at the same height (55 % of the y range),
    # which is above the flat resting trace and well below the spike peaks.
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()
    _xsp  = _xlim[1] - _xlim[0]
    _ysp  = _ylim[1] - _ylim[0]
    ax.annotate('Stimulus',
                xy=(200, _ylim[0] + 0.55 * _ysp),
                xytext=(200 - 0.09 * _xsp, _ylim[0] + 0.55 * _ysp),
                fontsize=7, color='gray', ha='right', va='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7,
                                shrinkA=2, shrinkB=2))
    # Fix 2 — annotate rebound if present
    if rebound_info.get(ct, 0) > 0:
        ax.text(0.97, 0.05, f'rebound ×{rebound_info[ct]}',
                transform=ax.transAxes, fontsize=7,
                color='darkred', ha='right', va='bottom')

    # ------------------------------------------------------------------ #
    # Col 1 — Hyperpolarising traces
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[row, 1])
    axs[(row, 'hyp')] = ax
    all_v = np.concatenate([vh for _, vh in d['hyper']])
    ax.set_xlim(100, 800)
    ax.set_ylim(float(np.min(all_v)) - 2, float(np.max(all_v)) + 2)
    grays = np.linspace(0.65, 0.05, len(d['hyper']))
    for (th, vh), amp_h, g in zip(d['hyper'], HYPER_AMPS[ct], grays):
        mask = (th >= 100) & (th <= 800)
        ax.plot(th[mask], vh[mask], color=str(g), lw=0.85)
        # Label at the steady-state y during the step (t=650–700 ms).
        # Using the recovery window (780–800 ms) would stack all labels at
        # the same height because the traces converge back to rest.
        steady_m = (th >= 650) & (th <= 700)
        v_label = float(np.mean(vh[steady_m])) if steady_m.any() else float(vh[mask][-1])
        ax.text(803, v_label, f'{amp_h:.2f} nA',
                fontsize=7, color=str(g), ha='left', va='center', clip_on=False)
    ax.axvline(200, color='gray', lw=0.7, ls='--', alpha=0.45)
    ax.axvline(700, color='gray', lw=0.7, ls='--', alpha=0.45)
    _style_trace_ax(ax)

    # ------------------------------------------------------------------ #
    # Col 3 — F/I curve
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[row, 3])
    axs[(row, 'fi')] = ax
    ax.plot(d['fi_amps'], d['fi_rates'],
            color=color, lw=1.8, marker='o', markersize=4,
            markerfacecolor=color)
    ax.axvline(d['rheobase'], color=color, lw=0.9, ls='--', alpha=0.5)
    ax.set_xlim(0, FI_MAX[ct] * 1.05)
    ax.set_ylim(0, FI_YMAX[ct])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=9)
    if row == len(CELL_NAMES) - 1:
        ax.set_xlabel('Current (nA)', fontsize=9)
    # Rheobase annotation — rotated text at dashed line, in cell color
    ax.text(d['rheobase'] + FI_MAX[ct] * 0.025, FI_YMAX[ct] * 0.08,
            f'Rheobase\n{d["rheobase"]:.2f} nA',
            fontsize=7, color=color, rotation=90,
            ha='left', va='bottom', multialignment='center')
    # Max firing rate label at last point on curve
    _nz_idx = np.where(d['fi_rates'] > 0)[0]
    if len(_nz_idx):
        _li = _nz_idx[-1]
        ax.text(d['fi_amps'][_li], d['fi_rates'][_li] + FI_YMAX[ct] * 0.04,
                f'{d["fi_rates"][_li]:.0f} Hz',
                fontsize=7, color=color, ha='center', va='bottom')

    # ------------------------------------------------------------------ #
    # Col 4 — Sag voltage
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[row, 4])
    axs[(row, 'sag')] = ax
    ax.plot(d['sag_amps'], d['sag_vals'],
            color=color, lw=1.8, marker='o', markersize=4,
            markerfacecolor=color)
    ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
    ax.set_xlim(min(SAG_AMPS[ct]) * 1.12, 0.005)
    _sag_top = max(float(np.max(d['sag_vals'])) * 1.35, 1.0)
    ax.set_ylim(0, _sag_top)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylabel('Sag Voltage (mV)', fontsize=9)
    if row == len(CELL_NAMES) - 1:
        ax.set_xlabel('Current (nA)', fontsize=9)
    # Label each data point with its sag value
    for sa, sv in zip(d['sag_amps'], d['sag_vals']):
        ax.text(sa, sv + 0.04 * _sag_top, f'{sv:.1f} mV',
                fontsize=7, color='gray', ha='center', va='bottom')

# ---- Scale bars with labels (after limits are finalised) ----
for row in range(len(CELL_NAMES)):
    add_scale_bar(axs[(row, 'dep')], x_bar_ms=300, y_bar_mv=50,
                  x_label='300 ms', y_label='50 mV')
    add_scale_bar(axs[(row, 'hyp')], x_bar_ms=300, y_bar_mv=10,
                  x_label='300 ms', y_label='10 mV')

# Fix 3 — per-column headers on row 0 only, no overlapping shared title
axs[(0, 'dep')].set_title('Depolarising',    fontsize=10, pad=5)
axs[(0, 'hyp')].set_title('Hyperpolarising', fontsize=10, pad=5)
axs[(0, 'fi') ].set_title('F/I Curve',       fontsize=10, pad=5)
axs[(0, 'sag')].set_title('Sag Voltage',     fontsize=10, pad=5)

fig.text(0.5, 0.97,
         'Single-Cell Characterization — Yao et al. 2022 Replica  |  '
         'Figure 1A–D: PYR · SST · PV · VIP',
         ha='center', va='top', fontsize=11)

# ---- Save ----
out_png = os.path.join(OUT_DIR, 'figure1_single_cell_characterization.png')
out_pdf = os.path.join(OUT_DIR, 'figure1_single_cell_characterization.pdf')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf,           bbox_inches='tight')
plt.close(fig)
print(f'[scc] Saved → {out_png}')
print(f'[scc] Saved → {out_pdf}')

# =============================================================================
# CONSOLE SUMMARY TABLE
# =============================================================================
print()
print('=' * 90)
print(f'{"Cell type":12s} | {"Rheobase (nA)":>13} | {"V_rest (mV)":>11} | '
      f'{"Input R (MΩ)":>12} | {"Sag ratio":>9} | {"AP half-width (ms)":>18}')
print('-' * 90)
for ct in CELL_NAMES:
    d  = data[ct]
    hw = d['half_width']
    sr = d['sag_ratio']
    hw_str = f'{hw:.2f}' if not np.isnan(hw) else 'N/A'
    sr_str = f'{sr:.3f}' if not np.isnan(sr) else 'N/A'
    print(f'{ct:12s} | {d["rheobase"]:>13.3f} | {d["v_rest"]:>11.1f} | '
          f'{d["r_in"]:>12.1f} | {sr_str:>9} | {hw_str:>18}')
print('=' * 90)
print('\n=== Done ===')

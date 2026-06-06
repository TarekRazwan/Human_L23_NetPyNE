#!/usr/bin/env python3
# =============================================================================
# validate_healthy.py  —  Baseline validation for Yao et al. 2022 replica
#
# Reads NetPyNE .pkl output directly from the simulation save folder.
# Produces:
#   fig2C_raster.png         — baseline raster (single seed)
#   fig2D_baseline_rates.png — mean firing rates (all available seeds)
#   fig2F_voltage_traces.png — example somatic Vm traces
#   fig2G_spike_PSD.png      — PYR population spike PSD (1/f)
#   validation_report.txt    — text summary with dual rate definitions
#
# Usage:
#   cd sim/output && python validate_healthy.py
#   python validate_healthy.py --data-dir ../../data/v1_batch5_lock
# =============================================================================
import os
import sys
import glob
import pickle
import argparse
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as ss
from scipy import stats as st

# ---------------------------------------------------------------------------
# Constants — must match cfg.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'data', 'v1_batch5_lock'))
FIG_DIR    = SCRIPT_DIR   # figures and report go next to this script

TSTOP     = 3000.0                           # ms  (cfg.duration)
TRANSIENT = 0.0                              # ms  (cfg.transient — full window)
DT        = 0.025                            # ms
FS        = 1000.0 / DT                      # Hz  (40 000)

CELL_NAMES = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
POP_LABELS  = {'HL23PYR': 'PYR', 'HL23SST': 'SST',
               'HL23PV':  'PV',  'HL23VIP': 'VIP'}
POP_COLORS  = {'HL23PYR': '#808080', 'HL23SST': '#CC0000',
               'HL23PV':  '#008000', 'HL23VIP': '#FF8C00'}

NON_SILENT_THRESHOLD = 0.2   # Hz — for secondary "non-silent" metric

# Paper targets (mean +/- SD from Yao 2022 Table S2, non-silent definition)
PAPER_TARGETS = {
    'HL23PYR': (1.2,  0.2),
    'HL23SST': (5.62, 0.27),
    'HL23PV':  (10.19, 0.51),
    'HL23VIP': (3.52, 0.37),
}

# Original LFPy reference rates (seed 1234, all-cells, full window)
LFPY_REFERENCE = {
    'HL23PYR': 0.784,
    'HL23SST': 5.544,
    'HL23PV':  9.944,
    'HL23VIP': 3.266,
}


# ---------------------------------------------------------------------------
# Helpers — load pkl and extract spike data
# ---------------------------------------------------------------------------
def _reconstruct_pop_gids(sim_config):
    """Reconstruct {pop: [gid0, gid1, ...]} from simConfig."""
    cell_number  = sim_config.get('cellNumber',  {})
    cell_number0 = sim_config.get('cellNumber0', {})
    pop_gids = {}
    for name in CELL_NAMES:
        start = cell_number0.get(name, 0)
        count = cell_number.get(name, 0)
        pop_gids[name] = list(range(int(start), int(start) + int(count)))
    return pop_gids


def _pop_gids_from_net(net_dict):
    """Extract {pop: [gids]} from data['net']['pops'] if available."""
    pops = net_dict.get('pops', {})
    pop_gids = {}
    for name in CELL_NAMES:
        pop_entry = pops.get(name, {})
        gids = pop_entry.get('cellGids', [])
        if gids:
            pop_gids[name] = list(gids)
    return pop_gids if all(name in pop_gids for name in CELL_NAMES) else None


def _extract_conn_counts(net_dict):
    """Extract total connections and synaptic contacts from net['cells'].
    Returns dict with 'totalConnections', 'totalSynapses', 'numCells', or None if
    the data is unavailable (net dict missing or cells not saved)."""
    cells = net_dict.get('cells', None)
    if cells is None:
        return None
    if not isinstance(cells, (list, tuple)) or len(cells) == 0:
        return None

    num_cells = len(cells)
    total_synapses = 0
    pre_gid_sets = []   # one set of unique preGids per cell, for connection count
    for cell in cells:
        conns = cell.get('conns', [])
        total_synapses += len(conns)
        if conns:
            pre_gids = set()
            for c in conns:
                if isinstance(c, dict):
                    pre_gids.add(c.get('preGid', -1))
                elif isinstance(c, (list, tuple)):
                    pre_gids.add(c[0])
            pre_gid_sets.append(pre_gids)
        else:
            pre_gid_sets.append(set())

    total_connections = sum(len(s) for s in pre_gid_sets)

    return {
        'totalConnections': total_connections,
        'totalSynapses':    total_synapses,
        'numCells':         num_cells,
        'connsPerCell':     total_connections / num_cells if num_cells else 0,
        'synsPerCell':      total_synapses / num_cells if num_cells else 0,
    }


def load_pkl(pkl_path):
    """Load a NetPyNE pkl file, return standardised dict with spkt/spkid/popGids/simConfig."""
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    sim_data   = raw.get('simData', {})
    sim_config = raw.get('simConfig', {})

    spkt  = np.array(sim_data.get('spkt',  []), dtype=float)
    spkid = np.array(sim_data.get('spkid', []), dtype=float)

    # Try to get popGids from net dict first, fall back to reconstruction
    net_dict = raw.get('net', {})
    pop_gids = _pop_gids_from_net(net_dict)
    if pop_gids is None:
        pop_gids = _reconstruct_pop_gids(sim_config)

    # Extract connection / synaptic contact counts from net['cells']
    conn_counts = _extract_conn_counts(net_dict)

    return {
        'spkt':       spkt,
        'spkid':      spkid,
        'popGids':    pop_gids,
        'simConfig':  sim_config,
        'popRates':   sim_data.get('popRates', {}),   # NetPyNE's own rates if present
        'connCounts': conn_counts,
    }


def discover_pkls(data_dir):
    """Find all *_data.pkl files in data_dir, return sorted list of paths."""
    pattern = os.path.join(data_dir, '*_data.pkl')
    return sorted(glob.glob(pattern))


def seed_from_pkl_path(pkl_path):
    """Extract seed label from cfg stored in the pkl (seeds.stim), or from filename."""
    # Use filename index as fallback label
    base = os.path.basename(pkl_path)
    return base.replace('_data.pkl', '')


# ---------------------------------------------------------------------------
# Rate computation — two definitions
# ---------------------------------------------------------------------------
def compute_rates_allcells(data, t_start=None, t_stop=None):
    """PRIMARY: total spikes / total cells / duration.
    Matches NetPyNE printPopAvgRates (all cells, full window)."""
    if t_start is None:
        t_start = TRANSIENT
    if t_stop is None:
        t_stop = TSTOP
    spkt    = data['spkt']
    spkid   = data['spkid']
    pop_gids = data['popGids']
    dur_s   = (t_stop - t_start) / 1000.0
    mask    = (spkt >= t_start) & (spkt < t_stop)
    rates   = {}
    for name in CELL_NAMES:
        gids = pop_gids.get(name, [])
        n_cells = len(gids)
        if n_cells == 0 or dur_s <= 0:
            rates[name] = 0.0
            continue
        gid_set = set(gids)
        n_spikes = np.sum(mask & np.isin(spkid, list(gid_set)))
        rates[name] = float(n_spikes) / n_cells / dur_s
    return rates


def compute_rates_nonsilent(data, t_start=None, t_stop=None):
    """SECONDARY: mean over cells firing > 0.2 Hz.
    Matches Yao 2022 Table S2 reporting convention."""
    if t_start is None:
        t_start = TRANSIENT
    if t_stop is None:
        t_stop = TSTOP
    spkt    = data['spkt']
    spkid   = data['spkid']
    pop_gids = data['popGids']
    dur_s   = (t_stop - t_start) / 1000.0
    mask    = (spkt >= t_start) & (spkt < t_stop)
    rates   = {}
    for name in CELL_NAMES:
        gids = pop_gids.get(name, [])
        if not gids or dur_s <= 0:
            rates[name] = 0.0
            continue
        cell_r = np.array([np.sum((spkid == g) & mask) / dur_s for g in gids])
        active = cell_r[cell_r > NON_SILENT_THRESHOLD]
        rates[name] = float(np.mean(active)) if len(active) else 0.0
    return rates


# ===========================================================================
# PLOT 1 — Figure 2C: Baseline raster (first pkl, 700 ms window)
# ===========================================================================
def plot_fig2C(data, seed_label):
    spkt     = data['spkt']
    spkid    = data['spkid']
    pop_gids = data['popGids']

    T0, T1 = TRANSIENT, TRANSIENT + 700.0

    fig, ax = plt.subplots(figsize=(14, 5))
    y_offset = 0
    yticks, ytick_labels, dividers = [], [], []
    for name in CELL_NAMES:
        gids    = sorted(pop_gids.get(name, []))
        if not gids:
            continue
        gid_min = gids[0]
        mask    = np.isin(spkid, gids) & (spkt >= T0) & (spkt < T1)
        t_sel   = spkt[mask] - T0
        id_sel  = spkid[mask]
        y_vals  = y_offset + (id_sel - gid_min)
        ax.plot(t_sel, y_vals, '|', color=POP_COLORS[name],
                markersize=1.5, markeredgewidth=0.5, label=POP_LABELS[name])
        pop_size = len(gids)
        yticks.append(y_offset + pop_size / 2)
        ytick_labels.append(POP_LABELS[name])
        if y_offset > 0:
            dividers.append(y_offset)
        y_offset += pop_size
    for d in dividers:
        ax.axhline(d, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlim(0, 700)
    ax.set_ylim(-5, y_offset + 5)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron index', fontsize=12)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_title(f'Figure 2C: Baseline Raster (Healthy, {seed_label})', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, markerscale=4)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2C_raster.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2C] Saved -> {path}')


# ===========================================================================
# PLOT 2 — Figure 2D: Baseline firing rates
# ===========================================================================
def plot_fig2D(all_rates_allcells, n_seeds):
    if not all_rates_allcells:
        print('[fig2D] No data — skipping.')
        return

    means = {name: np.mean(all_rates_allcells[name]) for name in CELL_NAMES}
    sds   = {name: (np.std(all_rates_allcells[name], ddof=1)
                     if len(all_rates_allcells[name]) > 1 else 0.0)
             for name in CELL_NAMES}

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(CELL_NAMES))
    ax.bar(x,
           [means[n] for n in CELL_NAMES],
           yerr=[sds[n] for n in CELL_NAMES],
           color=[POP_COLORS[n] for n in CELL_NAMES],
           edgecolor='k', linewidth=1.0,
           error_kw={'elinewidth': 1.5, 'capsize': 4})
    # Paper target markers
    for i, name in enumerate(CELL_NAMES):
        tgt, _ = PAPER_TARGETS[name]
        ax.plot([i - 0.4, i + 0.4], [tgt, tgt], 'k--', lw=1.5, alpha=0.7)
    # LFPy reference markers
    for i, name in enumerate(CELL_NAMES):
        ref = LFPY_REFERENCE[name]
        ax.plot([i - 0.3, i + 0.3], [ref, ref], 'b:', lw=1.2, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([POP_LABELS[n] for n in CELL_NAMES], fontsize=12)
    ax.set_ylabel('Firing Rate (Hz)\n[all cells, full window]', fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(f'Figure 2D: Baseline Firing Rates (n={n_seeds} seed{"s" if n_seeds > 1 else ""})',
                 fontsize=12)
    ax.plot([], [], 'k--', lw=1.5, label='Yao 2022 target')
    ax.plot([], [], 'b:', lw=1.2, label='LFPy reference')
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2D_baseline_rates.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2D] Saved -> {path}')


# ===========================================================================
# PLOT 3 — Figure 2F: Voltage traces from pkl (500 ms window)
# ===========================================================================
def plot_fig2F(data, seed_label):
    sim_config = data['simConfig']
    cell_number0 = sim_config.get('cellNumber0', {})

    # NetPyNE stores traces as simData[traceName][f'cell_{gid}']
    # We want one representative cell per pop (the first recorded one)
    pop_keys = []
    for name in CELL_NAMES:
        gid0 = int(cell_number0.get(name, -1))
        trace_key = f'cell_{gid0}'
        pop_keys.append((trace_key, POP_LABELS[name], name, gid0))

    t0_ms = 500.0    # show 500-1000 ms window
    t1_ms = 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    has_data = False
    for ax, (trace_key, label, pop, gid) in zip(axes, pop_keys):
        v_soma = data.get('simConfig', {}).get('recordTraces', {})
        # The trace is stored under the trace name (e.g. 'V_soma') → cell key
        # In pkl: simData['V_soma']['cell_0'] = list of voltages
        # We need to go back to the raw pkl for this — data dict doesn't carry it
        # Skip if no trace data available
        ax.text(0.5, 0.5, f'{label} (GID {gid}): traces require recordCells',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_ylabel(label, fontsize=12, color=POP_COLORS[pop])
        ax.set_ylim(-90, 55)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=12)
    axes[-1].tick_params(axis='x', labelsize=10)
    axes[0].set_title(f'Figure 2F: Voltage Traces (Healthy, {seed_label})', fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2F_voltage_traces.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2F] Saved -> {path}')


def plot_fig2F_from_pkl(pkl_path, seed_label):
    """Load full pkl to extract voltage traces (separate from spike-only load)."""
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    sim_data   = raw.get('simData', {})
    sim_config = raw.get('simConfig', {})
    cell_number0 = sim_config.get('cellNumber0', {})

    t_vec = sim_data.get('t', [])
    if not t_vec:
        print('[fig2F] No time vector in pkl — skipping.')
        return
    t = np.array(t_vec)

    t0_ms, t1_ms = 500.0, 1000.0
    tmask = (t >= t0_ms) & (t < t1_ms)
    t_plot = t[tmask] - t0_ms

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for ax, name in zip(axes, CELL_NAMES):
        gid0 = int(cell_number0.get(name, -1))
        trace_key = f'cell_{gid0}'
        v_soma_dict = sim_data.get('V_soma', {})
        v_data = v_soma_dict.get(trace_key, None)
        if v_data is not None:
            v = np.array(v_data)
            if len(v) >= len(tmask):
                ax.plot(t_plot, v[tmask], color=POP_COLORS[name], lw=0.8)
            else:
                ax.text(0.5, 0.5, f'{POP_LABELS[name]}: trace length mismatch',
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
        else:
            ax.text(0.5, 0.5, f'{POP_LABELS[name]} (GID {gid0}): not recorded',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_ylabel(POP_LABELS[name], fontsize=12, color=POP_COLORS[name])
        ax.set_ylim(-90, 55)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=12)
    axes[-1].tick_params(axis='x', labelsize=10)
    axes[0].set_title(f'Figure 2F: Voltage Traces (Healthy, {seed_label})', fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2F_voltage_traces.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2F] Saved -> {path}')


# ===========================================================================
# PLOT 4 — Figure 2G: PYR population spike PSD
# ===========================================================================
def plot_fig2G(all_data):
    if not all_data:
        print('[fig2G] No data — skipping.')
        return

    dur_ms  = TSTOP - TRANSIENT
    n_samp  = int(dur_ms / DT)
    nperseg = n_samp

    all_psds = []
    freqs_ref = None

    for data in all_data:
        spkt  = data['spkt']
        spkid = data['spkid']
        gids  = data['popGids'].get('HL23PYR', [])
        mask_t   = (spkt >= TRANSIENT) & (spkt < TSTOP)
        mask_pop = np.isin(spkid, gids)
        pyr_t    = spkt[mask_t & mask_pop] - TRANSIENT
        pop_train = np.zeros(n_samp, dtype=float)
        for st_ms in pyr_t:
            idx = int(st_ms / DT)
            if 0 <= idx < n_samp:
                pop_train[idx] += 1.0
        freqs, psd = ss.welch(pop_train, fs=FS, nperseg=nperseg,
                              window='hann', scaling='density')
        all_psds.append(psd)
        if freqs_ref is None:
            freqs_ref = freqs

    all_psds = np.array(all_psds)
    mean_psd = all_psds.mean(axis=0)

    fmask  = (freqs_ref >= 1.0) & (freqs_ref <= 100.0)
    f_plot = freqs_ref[fmask]
    m_plot = mean_psd[fmask]

    fig = plt.figure(figsize=(8, 6))
    ax_main = fig.add_subplot(111)
    ax_main.semilogy(f_plot, m_plot, color='#404040', lw=1.5, label='Mean PYR PSD')
    if len(all_psds) > 1:
        n_boot = 500
        rng_bs = np.random.default_rng(42)
        boot   = rng_bs.choice(all_psds, size=(n_boot, len(all_psds)), replace=True, axis=0)
        boot_mean = boot.mean(axis=1)
        ci_lo = np.percentile(boot_mean, 2.5, axis=0)[fmask]
        ci_hi = np.percentile(boot_mean, 97.5, axis=0)[fmask]
        ax_main.fill_between(f_plot, ci_lo, ci_hi, color='#404040', alpha=0.25, label='95% CI')
    ax_main.set_xlim(0, 100)
    ax_main.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_main.set_ylabel('PSD (spike^2/Hz)', fontsize=12)
    ax_main.tick_params(labelsize=10)
    ax_main.set_title(f'Figure 2G: PYR Population Spike PSD (n={len(all_psds)} seed{"s" if len(all_psds)>1 else ""})',
                      fontsize=12)
    ax_main.legend(fontsize=10)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # Inset: log-log with 1/f reference
    ax_ins = ax_main.inset_axes([0.55, 0.55, 0.40, 0.38])
    ax_ins.loglog(f_plot, m_plot, color='#404040', lw=1.2)
    f_ref = np.array([2.0, 80.0])
    idx5  = np.argmin(np.abs(f_plot - 5.0))
    psd_ref = m_plot[idx5] * (f_ref[0] / f_ref) ** 1.0
    ax_ins.loglog(f_ref, psd_ref, 'r--', lw=1.2, label='1/f')
    ax_ins.set_xlim(1, 100)
    ax_ins.set_xlabel('Hz', fontsize=9)
    ax_ins.set_ylabel('PSD', fontsize=9)
    ax_ins.tick_params(labelsize=8)
    ax_ins.legend(fontsize=8, loc='upper right')
    ax_ins.set_title('log-log', fontsize=8)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2G_spike_PSD.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2G] Saved -> {path}')


# ===========================================================================
# Validation report — dual rate definitions
# ===========================================================================
def write_report(all_data, seed_labels):
    n = len(all_data)
    lines = []
    lines.append('=== Yao 2022 Healthy Model Validation Report ===')
    lines.append(f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'Seeds evaluated: {seed_labels}')
    lines.append(f'Duration: {TSTOP} ms, transient: {TRANSIENT} ms, dt: {DT} ms')
    lines.append('')

    if not all_data:
        lines.append('No pkl files found.')
        _write_and_print(lines)
        return

    # Collect rates under both definitions
    all_ac = {name: [] for name in CELL_NAMES}   # all-cells
    all_ns = {name: [] for name in CELL_NAMES}   # non-silent
    for data in all_data:
        r_ac = compute_rates_allcells(data)
        r_ns = compute_rates_nonsilent(data)
        for name in CELL_NAMES:
            all_ac[name].append(r_ac[name])
            all_ns[name].append(r_ns[name])

    # --- PRIMARY: all-cells, full window (matches NetPyNE printPopAvgRates) ---
    lines.append('PRIMARY RATES — all cells, full window [0, {:.0f}] ms'.format(TSTOP))
    lines.append('  (Definition: total_spikes / num_cells / duration_s)')
    lines.append('  (This matches NetPyNE printPopAvgRates output)')
    lines.append('')
    lines.append(f'{"Population":12s} | {"Our value":14s} | '
                 f'{"LFPy ref":10s} | {"Yao target":14s} | {"% err vs Yao":12s}')
    lines.append('-' * 72)
    for name in CELL_NAMES:
        m  = np.mean(all_ac[name])
        sd = np.std(all_ac[name], ddof=1) if n > 1 else 0.0
        ref = LFPY_REFERENCE[name]
        tm, ts = PAPER_TARGETS[name]
        pct = (m - tm) / tm * 100
        lines.append(
            f'{POP_LABELS[name]:12s} | {m:.3f} +/- {sd:.3f} Hz | '
            f'{ref:.3f} Hz   | {tm:.2f} +/- {ts:.2f} Hz  | {pct:+.1f}%')

    lines.append('')
    lines.append('PER-SEED PRIMARY RATES:')
    header = f'{"Seed":20s}  ' + '  '.join(f'{POP_LABELS[n]:10s}' for n in CELL_NAMES)
    lines.append(header)
    for label, data in zip(seed_labels, all_data):
        r = compute_rates_allcells(data)
        row = f'{label:20s}  ' + '  '.join(f'{r[n]:>8.3f} Hz' for n in CELL_NAMES)
        lines.append(row)

    # Also show popRates from NetPyNE if present (cross-check)
    for label, data in zip(seed_labels, all_data):
        pr = data.get('popRates', {})
        if pr:
            lines.append('')
            lines.append(f'NetPyNE popRates (from pkl, {label}):')
            for name in CELL_NAMES:
                lines.append(f'  {POP_LABELS[name]:6s}: {pr.get(name, "N/A")} Hz')

    # --- SECONDARY: non-silent cells (matches Yao 2022 Table S2 convention) ---
    lines.append('')
    lines.append('SECONDARY RATES — non-silent cells only (>{:.1f} Hz), '
                 'window [{:.0f}, {:.0f}] ms'.format(NON_SILENT_THRESHOLD, TRANSIENT, TSTOP))
    lines.append('  (Definition: mean of per-cell rates for cells firing >{:.1f} Hz)'.format(
        NON_SILENT_THRESHOLD))
    lines.append('  (This matches the Yao 2022 Table S2 reporting convention)')
    lines.append('')
    lines.append(f'{"Population":12s} | {"Our value":14s} | '
                 f'{"Yao target":14s} | {"% err vs Yao":12s}')
    lines.append('-' * 60)
    for name in CELL_NAMES:
        m  = np.mean(all_ns[name])
        sd = np.std(all_ns[name], ddof=1) if n > 1 else 0.0
        tm, ts = PAPER_TARGETS[name]
        pct = (m - tm) / tm * 100
        lines.append(
            f'{POP_LABELS[name]:12s} | {m:.3f} +/- {sd:.3f} Hz | '
            f'{tm:.2f} +/- {ts:.2f} Hz  | {pct:+.1f}%')

    # --- NETWORK STRUCTURE: connection and synaptic contact counts ---
    lines.append('')
    lines.append('NETWORK STRUCTURE (baseline convergence reference):')
    any_counts = False
    for label, data in zip(seed_labels, all_data):
        cc = data.get('connCounts', None)
        if cc is None:
            lines.append(f'  {label}: ERROR — data["net"]["cells"] missing from pkl.')
            lines.append(f'           Ensure cfg.saveDataInclude includes "net" and')
            lines.append(f'           cfg.saveCellConns = True. Cannot verify convergence.')
        else:
            any_counts = True
            lines.append(f'  {label}:')
            lines.append(f'    Cells:              {cc["numCells"]}')
            lines.append(f'    Connections:         {cc["totalConnections"]} '
                         f'({cc["connsPerCell"]:.2f} per cell)')
            lines.append(f'    Synaptic contacts:   {cc["totalSynapses"]} '
                         f'({cc["synsPerCell"]:.2f} per cell)')
    if not any_counts:
        lines.append('  WARNING: no connection counts available for any seed.')
        lines.append('  Post-H01 convergence checks will have no baseline reference.')

    lines.append('')
    lines.append('GENERATED PLOTS:')
    for fname in ['fig2C_raster.png', 'fig2D_baseline_rates.png',
                  'fig2F_voltage_traces.png', 'fig2G_spike_PSD.png']:
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath) // 1024
            lines.append(f'  {fname}  ({sz} KB)')
        else:
            lines.append(f'  {fname}  -- NOT GENERATED')

    _write_and_print(lines)


def _write_and_print(lines):
    report_path = os.path.join(FIG_DIR, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n[report] Saved -> {report_path}')
    for line in lines:
        print(line)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate healthy baseline against Yao 2022')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help=f'Path to simulation output folder (default: {DEFAULT_DATA_DIR})')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print(f'=== validate_healthy.py  {datetime.datetime.now()} ===')
    print(f'Data dir:   {data_dir}')
    print(f'Figure dir: {FIG_DIR}')

    pkl_files = discover_pkls(data_dir)
    print(f'Found {len(pkl_files)} pkl file(s): {[os.path.basename(p) for p in pkl_files]}')

    if not pkl_files:
        print('ERROR: No *_data.pkl files found. Run the simulation first.')
        sys.exit(1)

    all_data = []
    seed_labels = []
    for pf in pkl_files:
        label = seed_from_pkl_path(pf)
        print(f'Loading {os.path.basename(pf)} ...')
        d = load_pkl(pf)
        all_data.append(d)
        seed_labels.append(label)

    # Plots
    plot_fig2C(all_data[0], seed_labels[0])
    all_rates_ac = {name: [] for name in CELL_NAMES}
    for data in all_data:
        r = compute_rates_allcells(data)
        for name in CELL_NAMES:
            all_rates_ac[name].append(r[name])
    plot_fig2D(all_rates_ac, len(all_data))
    plot_fig2F_from_pkl(pkl_files[0], seed_labels[0])
    plot_fig2G(all_data)

    # Report
    write_report(all_data, seed_labels)

    print('\n=== All done ===')

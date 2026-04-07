#!/usr/bin/env python3
# =============================================================================
# validate_healthy.py  —  Figure 2 validation panels for Yao et al. 2022
#
# Loads pre-saved .npy spike/trace files from output/ and produces:
#   fig2C_raster.png         — baseline raster (single seed)
#   fig2D_baseline_rates.png — mean ± SD firing rates (n=10 seeds)
#   fig2E_SST_silencing.png  — SST silencing effect on PYR rate
#   fig2F_voltage_traces.png — example somatic Vm traces
#   fig2G_spike_PSD.png      — PYR population spike PSD (1/f)
#
# Usage:
#   python validate_healthy.py
# =============================================================================
import os
import glob
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as ss
from scipy import stats as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
SEEDS     = list(range(1234, 1244))          # 1234 … 1243  (10 seeds)
TSTOP     = 4500.0                           # ms
TRANSIENT = 2000.0                           # ms to discard
DT        = 0.025                            # ms
FS        = 1000.0 / DT                      # Hz  (40 000)

CELL_NAMES = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
POP_LABELS  = {'HL23PYR': 'PYR', 'HL23SST': 'SST',
               'HL23PV':  'PV',  'HL23VIP': 'VIP'}
POP_COLORS  = {'HL23PYR': '#808080', 'HL23SST': '#CC0000',
               'HL23PV':  '#008000', 'HL23VIP': '#FF8C00'}

# Paper targets (mean ± SD from Yao 2022 Table S2)
PAPER_TARGETS = {
    'HL23PYR': (1.2,  0.2),
    'HL23SST': (5.62, 0.27),
    'HL23PV':  (10.19, 0.51),
    'HL23VIP': (3.52, 0.37),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_spikes(seed_label):
    """Load spike file. seed_label is e.g. '1234' or '1234_SSTsilenced'."""
    path = os.path.join(OUT_DIR, f'spikes_seed{seed_label}.npy')
    if not os.path.isfile(path):
        return None
    return np.load(path, allow_pickle=True).item()


def compute_rates(data):
    """Return dict {pop: mean_hz} using non-silent filter (>0.2 Hz)."""
    spkt    = data['spkt']
    spkid   = data['spkid']
    popGids = data['popGids']
    dur_s   = (TSTOP - TRANSIENT) / 1000.0
    mask    = spkt >= TRANSIENT
    rates   = {}
    for name in CELL_NAMES:
        gids = popGids.get(name, [])
        if not gids:
            rates[name] = 0.0
            continue
        cell_r = np.array([np.sum((spkid == g) & mask) / dur_s
                           for g in gids])
        active = cell_r[cell_r > 0.2]
        rates[name] = float(np.mean(active)) if len(active) else 0.0
    return rates


def available_seeds(silenced=False):
    found = []
    for s in SEEDS:
        lbl = f'{s}_SSTsilenced' if silenced else str(s)
        if os.path.isfile(os.path.join(OUT_DIR, f'spikes_seed{lbl}.npy')):
            found.append(s)
    return found


# ===========================================================================
# PLOT 1 — Figure 2C: Baseline raster (seed 1234, 700 ms window)
# ===========================================================================
def plot_fig2C():
    data = load_spikes('1234')
    if data is None:
        print('[fig2C] spikes_seed1234.npy not found — skipping.')
        return

    spkt    = data['spkt']
    spkid   = data['spkid']
    popGids = data['popGids']

    T0, T1 = TRANSIENT, TRANSIENT + 700.0   # 700 ms window after transient

    fig, ax = plt.subplots(figsize=(14, 5))

    y_offset = 0
    yticks, ytick_labels, dividers = [], [], []
    for name in CELL_NAMES:
        gids    = sorted(popGids.get(name, []))
        if not gids:
            continue
        gid_min = gids[0]
        mask    = np.isin(spkid, gids) & (spkt >= T0) & (spkt < T1)
        t_sel   = spkt[mask] - T0           # shift to 0-based
        id_sel  = spkid[mask]
        y_vals  = y_offset + (id_sel - gid_min)

        ax.plot(t_sel, y_vals, '|', color=POP_COLORS[name],
                markersize=1.5, markeredgewidth=0.5,
                label=POP_LABELS[name])

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
    ax.set_title('Figure 2C: Baseline Raster (Healthy, seed=1234)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, markerscale=4)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2C_raster.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2C] Saved → {path}')


# ===========================================================================
# PLOT 2 — Figure 2D: Baseline firing rates (n=10 seeds, mean ± SD)
# ===========================================================================
def plot_fig2D():
    ok_seeds = available_seeds(silenced=False)
    if not ok_seeds:
        print('[fig2D] No healthy spike files found — skipping.')
        return

    all_rates = {name: [] for name in CELL_NAMES}
    for s in ok_seeds:
        d = load_spikes(str(s))
        r = compute_rates(d)
        for name in CELL_NAMES:
            all_rates[name].append(r[name])

    n = len(ok_seeds)
    means = {name: np.mean(all_rates[name]) for name in CELL_NAMES}
    sds   = {name: np.std(all_rates[name],  ddof=1) for name in CELL_NAMES}

    print(f'\n[fig2D] Firing rates (n={n} seeds, mean ± SD):')
    for name in CELL_NAMES:
        tgt_m, tgt_s = PAPER_TARGETS[name]
        pct = (means[name] - tgt_m) / tgt_m * 100
        print(f'  {name:10s}: {means[name]:.2f} ± {sds[name]:.2f} Hz  '
              f'(target {tgt_m:.2f} ± {tgt_s:.2f}, {pct:+.0f}%)')

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(CELL_NAMES))
    bars = ax.bar(x,
                  [means[n] for n in CELL_NAMES],
                  yerr=[sds[n]  for n in CELL_NAMES],
                  color=[POP_COLORS[n] for n in CELL_NAMES],
                  edgecolor='k', linewidth=1.0,
                  error_kw={'elinewidth': 1.5, 'capsize': 4})

    # Paper target markers
    for i, name in enumerate(CELL_NAMES):
        tgt, _ = PAPER_TARGETS[name]
        ax.plot([i - 0.4, i + 0.4], [tgt, tgt], 'k--', lw=1.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([POP_LABELS[n] for n in CELL_NAMES], fontsize=12)
    ax.set_ylabel('Mean Firing Rate (Hz),\nnon-silent cells', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(f'Figure 2D: Baseline Firing Rates (n={n} seeds)', fontsize=12)
    # Dashed line legend entry
    ax.plot([], [], 'k--', lw=1.5, label='Paper target')
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2D_baseline_rates.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2D] Saved → {path}')

    return all_rates


# ===========================================================================
# PLOT 3 — Figure 2E: SST silencing doubles PYR rate
# ===========================================================================
def plot_fig2E(healthy_rates=None):
    ok_healthy  = available_seeds(silenced=False)
    ok_silenced = available_seeds(silenced=True)

    if not ok_healthy or not ok_silenced:
        print(f'[fig2E] Need healthy ({len(ok_healthy)}) and silenced '
              f'({len(ok_silenced)}) files — skipping.')
        return

    # Use seeds present in both conditions
    shared = sorted(set(ok_healthy) & set(ok_silenced))
    if len(shared) < 2:
        print(f'[fig2E] Only {len(shared)} shared seeds — need ≥2. Skipping.')
        return

    pyr_healthy  = []
    pyr_silenced = []
    for s in shared:
        dh = load_spikes(str(s))
        ds = load_spikes(f'{s}_SSTsilenced')
        pyr_healthy.append(compute_rates(dh)['HL23PYR'])
        pyr_silenced.append(compute_rates(ds)['HL23PYR'])

    pyr_healthy  = np.array(pyr_healthy)
    pyr_silenced = np.array(pyr_silenced)

    mean_h, sd_h = pyr_healthy.mean(),  pyr_healthy.std(ddof=1)
    mean_s, sd_s = pyr_silenced.mean(), pyr_silenced.std(ddof=1)
    ratio = mean_s / mean_h if mean_h > 0 else float('nan')

    tstat, pval = st.ttest_rel(pyr_silenced, pyr_healthy)
    cohens_d = (mean_s - mean_h) / np.sqrt(
        ((len(shared) - 1) * sd_s**2 + (len(shared) - 1) * sd_h**2) /
        (2 * len(shared) - 2)) if (sd_s + sd_h) > 0 else 0.0

    print(f'\n[fig2E] SST silencing (n={len(shared)} shared seeds):')
    print(f'  Healthy PYR:  {mean_h:.3f} ± {sd_h:.3f} Hz')
    print(f'  Silenced PYR: {mean_s:.3f} ± {sd_s:.3f} Hz')
    print(f'  Ratio:        {ratio:.2f}  (target ~2.0)')
    print(f'  p-value:      {pval:.4f}  (paired t-test)')
    print(f'  Cohen\'s d:    {cohens_d:.2f}')

    fig, ax = plt.subplots(figsize=(5, 5))
    conditions = ['Healthy', 'SST\nSilenced']
    means_plot = [mean_h, mean_s]
    sds_plot   = [sd_h,   sd_s]
    colors     = ['#555555', '#AAAAAA']

    ax.bar([0, 1], means_plot, yerr=sds_plot,
           color=colors, edgecolor='k', linewidth=1.0,
           error_kw={'elinewidth': 1.5, 'capsize': 4})

    # Individual seed dots + lines
    for ph, ps in zip(pyr_healthy, pyr_silenced):
        ax.plot([0, 1], [ph, ps], 'o-', color='k', alpha=0.4,
                markersize=4, lw=0.8)

    # Significance bracket
    y_top = max(means_plot) + max(sds_plot) + 0.3
    ax.plot([0, 0, 1, 1], [y_top, y_top + 0.1, y_top + 0.1, y_top],
            'k-', lw=1.2)
    sig_str = ('***' if pval < 0.001 else '**' if pval < 0.01
               else '*' if pval < 0.05 else 'ns')
    ax.text(0.5, y_top + 0.15, sig_str, ha='center', fontsize=13)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylabel('PYR Mean Firing Rate (Hz)', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(f'Figure 2E: SST Silencing\n'
                 f'ratio={ratio:.2f}×, p={pval:.4f}', fontsize=12)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2E_SST_silencing.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2E] Saved → {path}')


# ===========================================================================
# PLOT 4 — Figure 2F: Example voltage traces (500 ms window, seed 1234)
# ===========================================================================
def plot_fig2F():
    # Use the first available trace file (healthy seeds, in order)
    trace_file = None
    trace_seed = None
    for _s in SEEDS:
        _p = os.path.join(OUT_DIR, f'traces_seed{_s}.npy')
        if os.path.isfile(_p):
            trace_file = _p
            trace_seed = _s
            break
    if trace_file is None:
        print('[fig2F] No trace files found (traces_seed*.npy). '
              'Will be available from seed 1235+ with the direct-recording fix.')
        # Produce placeholder figure
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.text(0.5, 0.5,
                'Voltage traces not yet recorded.\n'
                'Run seeds 1235+ to generate traces automatically.',
                ha='center', va='center', fontsize=13,
                transform=ax.transAxes)
        ax.axis('off')
        path = os.path.join(OUT_DIR, 'fig2F_voltage_traces.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'[fig2F] Placeholder saved → {path}')
        return

    td = np.load(trace_file, allow_pickle=True).item()
    t  = td.get('t', np.array([]))
    pop_keys = [('V_PYR', 'PYR', 'HL23PYR'),
                ('V_SST', 'SST', 'HL23SST'),
                ('V_PV',  'PV',  'HL23PV'),
                ('V_VIP', 'VIP', 'HL23VIP')]

    # Extract 500 ms window starting at TRANSIENT
    t0_ms = TRANSIENT
    t1_ms = TRANSIENT + 500.0
    if len(t) > 0:
        mask = (t >= t0_ms) & (t < t1_ms)
        t_plot = t[mask] - t0_ms
    else:
        mask = slice(None)
        t_plot = np.array([])

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    for ax, (key, label, pop) in zip(axes, pop_keys):
        if key in td and len(td[key]) > 0:
            v = np.array(td[key])
            if len(v) > len(t_plot) and len(t_plot) > 0:
                v = v[mask]
            ax.plot(t_plot if len(t_plot) > 0 else np.arange(len(v)) * DT,
                    v, color=POP_COLORS[pop], lw=0.8)
        else:
            ax.text(0.5, 0.5, f'{label}: no data',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11)

        ax.set_ylabel(label, fontsize=12, color=POP_COLORS[pop])
        ax.set_ylim(-90, 55)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=12)
    axes[-1].tick_params(axis='x', labelsize=10)
    axes[0].set_title(f'Figure 2F: Example Voltage Traces (Healthy, seed={trace_seed})',
                       fontsize=12)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2F_voltage_traces.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2F] Saved → {path}')


# ===========================================================================
# PLOT 5 — Figure 2G: PYR population spike PSD (1/f, n=10 seeds)
# ===========================================================================
def plot_fig2G():
    ok_seeds = available_seeds(silenced=False)
    if not ok_seeds:
        print('[fig2G] No healthy spike files — skipping.')
        return

    dur_ms   = TSTOP - TRANSIENT          # 2500 ms analysis window
    n_samp   = int(dur_ms / DT)           # 100 000 samples at 40 kHz
    nperseg  = n_samp                     # full window → Δf = 0.4 Hz

    all_psds = []
    freqs_ref = None

    for s in ok_seeds:
        d     = load_spikes(str(s))
        spkt  = d['spkt']
        spkid = d['spkid']
        gids  = d['popGids'].get('HL23PYR', [])

        # Build population spike train vector
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

    all_psds = np.array(all_psds)   # shape (n_seeds, n_freqs)

    # Bootstrap 95% CI across seeds
    n_boot = 500
    rng_bs = np.random.default_rng(42)
    boot   = rng_bs.choice(all_psds, size=(n_boot, len(ok_seeds)),
                           replace=True, axis=0)
    boot_mean = boot.mean(axis=1)           # (n_boot, n_freqs)
    mean_psd  = all_psds.mean(axis=0)
    ci_lo     = np.percentile(boot_mean, 2.5,  axis=0)
    ci_hi     = np.percentile(boot_mean, 97.5, axis=0)

    # Frequency mask: 1–100 Hz
    fmask  = (freqs_ref >= 1.0) & (freqs_ref <= 100.0)
    f_plot = freqs_ref[fmask]
    m_plot = mean_psd[fmask]
    lo_p   = ci_lo[fmask]
    hi_p   = ci_hi[fmask]

    fig = plt.figure(figsize=(8, 6))
    gs  = gridspec.GridSpec(1, 1)
    ax_main = fig.add_subplot(gs[0])

    # Main panel: log-scale y, linear x
    ax_main.semilogy(f_plot, m_plot, color='#404040', lw=1.5,
                     label='Mean PYR PSD')
    ax_main.fill_between(f_plot, lo_p, hi_p,
                         color='#404040', alpha=0.25, label='95% CI')
    ax_main.set_xlim(0, 100)
    ax_main.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_main.set_ylabel('PSD (spike²/Hz)', fontsize=12)
    ax_main.tick_params(labelsize=10)
    ax_main.set_title(f'Figure 2G: PYR Population Spike PSD '
                      f'(n={len(ok_seeds)} seeds)', fontsize=12)
    ax_main.legend(fontsize=10)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # Inset: log-log with 1/f reference
    ax_ins = ax_main.inset_axes([0.55, 0.55, 0.40, 0.38])
    ax_ins.loglog(f_plot, m_plot, color='#404040', lw=1.2)
    ax_ins.fill_between(f_plot, lo_p, hi_p,
                        color='#404040', alpha=0.2)
    # 1/f reference line (slope -1 on log-log)
    f_ref  = np.array([2.0, 80.0])
    psd_ref_start = m_plot[np.argmin(np.abs(f_plot - 5.0))]
    psd_ref = psd_ref_start * (f_ref[0] / f_ref) ** 1.0
    ax_ins.loglog(f_ref, psd_ref, 'r--', lw=1.2, label='1/f')
    ax_ins.set_xlim(1, 100)
    ax_ins.set_xlabel('Hz', fontsize=9)
    ax_ins.set_ylabel('PSD', fontsize=9)
    ax_ins.tick_params(labelsize=8)
    ax_ins.legend(fontsize=8, loc='upper right')
    ax_ins.set_title('log-log', fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2G_spike_PSD.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[fig2G] Saved → {path}')


# ===========================================================================
# Validation report
# ===========================================================================
def write_report():
    ok_seeds  = available_seeds(silenced=False)
    ok_sil    = available_seeds(silenced=True)

    lines = []
    lines.append('=== Yao 2022 Healthy Model Validation Report ===')
    lines.append(f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'Seeds evaluated (healthy): {ok_seeds}')
    lines.append(f'Seeds evaluated (SST-silenced): {ok_sil}')
    lines.append('')

    if ok_seeds:
        all_rates = {name: [] for name in CELL_NAMES}
        for s in ok_seeds:
            d = load_spikes(str(s))
            r = compute_rates(d)
            for name in CELL_NAMES:
                all_rates[name].append(r[name])

        lines.append('FIRING RATES (mean ± SD, non-silent >0.2 Hz, t>2000ms):')
        lines.append(f'{"Population":12s} | {"Our value":14s} | '
                     f'{"Paper target":14s} | {"% error":8s}')
        lines.append('-' * 58)
        for name in CELL_NAMES:
            m   = np.mean(all_rates[name])
            s_  = np.std(all_rates[name], ddof=1) if len(all_rates[name]) > 1 else 0.0
            tm, ts = PAPER_TARGETS[name]
            pct = (m - tm) / tm * 100
            lines.append(
                f'{POP_LABELS[name]:12s} | {m:.2f} ± {s_:.2f} Hz     | '
                f'{tm:.2f} ± {ts:.2f} Hz    | {pct:+.1f}%')

        lines.append('')
        lines.append('PER-SEED RATES:')
        header = f'{"Seed":6s}  ' + '  '.join(f'{POP_LABELS[n]:8s}' for n in CELL_NAMES)
        lines.append(header)
        for s in ok_seeds:
            d = load_spikes(str(s))
            r = compute_rates(d)
            row = f'{s:6d}  ' + '  '.join(f'{r[n]:.2f} Hz' for n in CELL_NAMES)
            lines.append(row)
    else:
        lines.append('No healthy spike files found.')

    lines.append('')
    shared = sorted(set(ok_seeds) & set(ok_sil))
    if len(shared) >= 2:
        pyr_h, pyr_s = [], []
        for s in shared:
            pyr_h.append(compute_rates(load_spikes(str(s)))['HL23PYR'])
            pyr_s.append(compute_rates(load_spikes(f'{s}_SSTsilenced'))['HL23PYR'])
        pyr_h, pyr_s = np.array(pyr_h), np.array(pyr_s)
        _, pval = st.ttest_rel(pyr_s, pyr_h)
        ratio   = pyr_s.mean() / pyr_h.mean() if pyr_h.mean() > 0 else float('nan')
        lines.append('SST SILENCING TEST:')
        lines.append(f'  Healthy PYR:  {pyr_h.mean():.3f} ± {pyr_h.std(ddof=1):.3f} Hz')
        lines.append(f'  Silenced PYR: {pyr_s.mean():.3f} ± {pyr_s.std(ddof=1):.3f} Hz')
        lines.append(f'  Ratio:        {ratio:.2f}  (target ~2.0)')
        lines.append(f'  p-value:      {pval:.6f}')
    else:
        lines.append(f'SST silencing: insufficient shared seeds '
                     f'({len(shared)}) — need ≥2.')

    lines.append('')
    lines.append('GENERATED PLOTS:')
    for fname in ['fig2C_raster.png', 'fig2D_baseline_rates.png',
                  'fig2E_SST_silencing.png', 'fig2F_voltage_traces.png',
                  'fig2G_spike_PSD.png']:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath) // 1024
            lines.append(f'  {fname}  ({sz} KB)')
        else:
            lines.append(f'  {fname}  — NOT GENERATED')

    report_path = os.path.join(OUT_DIR, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n[report] Saved → {report_path}')
    for line in lines:
        print(line)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(f'=== validate_healthy.py  {datetime.datetime.now()} ===')
    print(f'Output dir: {OUT_DIR}')

    healthy_seeds  = available_seeds(silenced=False)
    silenced_seeds = available_seeds(silenced=True)
    print(f'Healthy spike files:  {len(healthy_seeds)} / {len(SEEDS)} seeds')
    print(f'Silenced spike files: {len(silenced_seeds)} / {len(SEEDS)} seeds')

    plot_fig2C()
    all_rates = plot_fig2D()
    plot_fig2E(healthy_rates=all_rates)
    plot_fig2F()
    plot_fig2G()
    write_report()

    print('\n=== All done ===')

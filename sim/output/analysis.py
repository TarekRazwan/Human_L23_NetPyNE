# =============================================================================
# analysis.py  —  Post-simulation analysis for L23Net NetPyNE Replica
# =============================================================================
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as ss


# Pop display colours match original circuit_functions.py
POP_COLORS = {
    'HL23PYR': 'dimgray',
    'HL23SST': 'crimson',
    'HL23PV':  'green',
    'HL23VIP': 'darkorange',
}
POP_LABELS = {
    'HL23PYR': 'PYR',
    'HL23SST': 'SST',
    'HL23PV':  'PV',
    'HL23VIP': 'VIP',
}


def compute_firing_rates(sim_data, cfg, cell_names):
    """
    Compute mean firing rates (Hz) for each population.
    Discards spikes before cfg.transient (ms) in full runs.
    Only non-silent cells (individual rate > 0.2 Hz) are included in the
    population mean, matching the paper's reported statistics.
    Returns {cell_name: mean_hz}.
    """
    spkt  = np.array(sim_data.get('spkt',  []))
    spkid = np.array(sim_data.get('spkid', []))

    t_start = cfg.transient if not cfg.testing else 0.0
    t_stop  = cfg.duration
    dur_s   = max((t_stop - t_start) / 1000.0, 1e-9)   # seconds

    pop_gids = sim_data.get('popGids', {})
    mask_t   = spkt >= t_start

    rates = {}
    for name in cell_names:
        gids = pop_gids.get(name, [])
        if not gids:
            rates[name] = 0.0
            continue

        gids_arr = np.array(sorted(gids))

        # Per-cell spike counts (vectorised over the spike arrays)
        cell_rates = np.array([
            np.sum((spkid == gid) & mask_t) / dur_s
            for gid in gids_arr
        ])

        # Non-silent filter: paper reports mean over active cells only
        active = cell_rates[cell_rates > 0.2]
        rates[name] = float(np.mean(active)) if len(active) > 0 else 0.0

    return rates


def plot_raster(sim_data, cfg, cell_names, out_dir, seed=None):
    """
    Spike raster + mean firing-rate bar chart.
    Saves PNG files to out_dir.
    """
    spkt  = np.array(sim_data.get('spkt',  []))
    spkid = np.array(sim_data.get('spkid', []))
    pop_gids = sim_data.get('popGids', {})

    label = seed if seed is not None else 'run'
    t_start = cfg.transient if not cfg.testing else 0.0
    t_stop  = cfg.duration

    # ------------------------------------------------------------------
    # Raster
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Assign a y-offset per population so all GIDs are plotted
    y_offset = 0
    yticks, ytick_labels = [], []
    for name in cell_names:
        gids = pop_gids.get(name, [])
        if not gids:
            continue
        gid_arr = np.array(sorted(gids))
        gid_min = gid_arr.min()

        mask = np.isin(spkid, gid_arr) & (spkt >= t_start)
        t_sel  = spkt[mask]
        id_sel = spkid[mask]

        # Local y = offset + (gid - gid_min)
        y_vals = y_offset + (id_sel - gid_min)
        ax.plot(t_sel, y_vals, '|', color=POP_COLORS[name], markersize=2,
                label=POP_LABELS[name])

        pop_size = len(gids)
        yticks.append(y_offset + pop_size / 2)
        ytick_labels.append(POP_LABELS[name])
        y_offset += pop_size + 5   # small gap between populations

    ax.set_xlabel('Time (ms)', fontsize=13)
    ax.set_ylabel('Neuron index', fontsize=13)
    ax.set_xlim(t_start, t_stop)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title(f'L23Net NetPyNE — Spike Raster  (seed={label})', fontsize=13)
    fig.tight_layout()

    raster_path = os.path.join(out_dir, f'raster_{label}.png')
    fig.savefig(raster_path, dpi=150)
    plt.close(fig)
    print(f'[analysis] Raster saved → {raster_path}')

    # ------------------------------------------------------------------
    # Firing rate bar chart
    # ------------------------------------------------------------------
    rates = compute_firing_rates(sim_data, cfg, cell_names)
    names  = [n for n in cell_names if n in rates]
    hz_vals = [rates[n] for n in names]
    colors  = [POP_COLORS[n] for n in names]
    labels  = [POP_LABELS[n] for n in names]

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.bar(range(len(names)), hz_vals, color=colors, edgecolor='k', linewidth=1.2)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(labels, fontsize=13)
    ax2.set_ylabel('Mean Firing Rate (Hz)', fontsize=13)
    ax2.set_title(f'Population Firing Rates  (seed={label})', fontsize=13)
    fig2.tight_layout()

    rates_path = os.path.join(out_dir, f'rates_{label}.png')
    fig2.savefig(rates_path, dpi=150)
    plt.close(fig2)
    print(f'[analysis] Rates chart saved → {rates_path}')

    return rates


def bandpass_filter(signal_arr, low=0.1, high=100.0, fs=40000.0):
    """2nd-order Butterworth bandpass filter (mirrors original circuit_functions.py)."""
    b, a = ss.butter(2, [low, high], btype='bandpass', fs=fs)
    return ss.filtfilt(b, a, signal_arr)


def plot_lfp(sim_data, cfg, out_dir, seed=None):
    """Plot LFP trace and PSD if cfg.rec_LFP=True (optional)."""
    lfp = sim_data.get('LFP', None)
    if lfp is None:
        print('[analysis] No LFP data found — skipping LFP plots.')
        return

    label = seed if seed is not None else 'run'
    fs    = 1000.0 / cfg.dt   # Hz  (40 000 Hz)
    t1    = int(cfg.transient / cfg.dt)

    lfp_arr = np.array(lfp)
    if lfp_arr.ndim == 2:
        lfp_arr = lfp_arr[:, 0]   # first electrode

    tvec = np.arange(len(lfp_arr)) * cfg.dt
    lfp_filt = bandpass_filter(lfp_arr[t1:], fs=fs)
    freq, psd = ss.welch(lfp_filt, fs=fs, nperseg=int(fs / 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(tvec[t1:], lfp_filt, 'k', lw=0.6)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('LFP (mV)')
    ax1.set_title('LFP trace (filtered 0.1–100 Hz)')

    ax2.semilogy(freq, psd, 'k', lw=1)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD')
    ax2.set_title('LFP Power Spectrum')

    fig.tight_layout()
    path = os.path.join(out_dir, f'lfp_{label}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[analysis] LFP plot saved → {path}')

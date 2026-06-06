#!/usr/bin/env python3
# =============================================================================
# validate_healthy.py  —  Validation report for L23 NetPyNE replica (Yao 2022)
#
# Reads NetPyNE *_data.pkl files directly — no .npy extraction step.
# Auto-discovers all *_data.pkl in --data-dir, so it works whether 1 or 5
# seeds have finished.
#
# Usage:
#   python validate_healthy.py --data-dir ../../data/v1_batch5_lock
#   python validate_healthy.py   # uses default path
#
# Outputs:
#   validation_report.txt   — dual-rate report (all-cells + non-silent)
#   fig2C_raster.png        — baseline raster from first seed
#   fig2D_baseline_rates.png — mean ± SD firing rates across seeds
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Analysis window: full 3 s simulation, no transient discard.
# cfg.duration = 3000 ms, and we want rates over the entire run to match
# the NetPyNE printPopAvgRates convention (which uses the full window).
TSTOP     = 3000.0   # ms  — must match cfg.duration
TRANSIENT = 0.0      # ms  — no warm-up discard for primary rate

CELL_NAMES = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
POP_LABELS = {'HL23PYR': 'PYR', 'HL23SST': 'SST',
              'HL23PV':  'PV',  'HL23VIP': 'VIP'}
POP_COLORS = {'HL23PYR': '#808080', 'HL23SST': '#CC0000',
              'HL23PV':  '#008000', 'HL23VIP': '#FF8C00'}

# Paper targets (mean ± SD from Yao 2022 Table S2, non-silent convention)
PAPER_TARGETS = {
    'HL23PYR': (1.2,  0.2),
    'HL23SST': (5.62, 0.27),
    'HL23PV':  (10.19, 0.51),
    'HL23VIP': (3.52, 0.37),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def discover_pkls(data_dir):
    """Find all *_data.pkl files in data_dir, sorted by name.

    Returns a list of absolute paths.  Sorting by name gives us
    index-order (v1_batch5_lock_0, _1, _2, …) because the batch
    label is a common prefix and the index is the varying suffix.
    """
    pattern = os.path.join(data_dir, '*_data.pkl')
    found   = sorted(glob.glob(pattern))
    return found


def load_pkl(path):
    """Load a single NetPyNE *_data.pkl and return the raw dict.

    The pkl is written by sim.saveData() and contains top-level keys
    controlled by cfg.saveDataInclude.  At minimum we need:
      - 'simData'   with 'spkt' and 'spkid'
      - 'simConfig' with 'cellNumber' and 'cellNumber0'

    Raises RuntimeError if the minimum keys are missing.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # --- Validate minimum required structure ---
    if 'simData' not in data:
        raise RuntimeError(f'{path}: missing "simData" key — corrupt or non-NetPyNE pkl')
    sd = data['simData']
    for k in ('spkt', 'spkid'):
        if k not in sd:
            raise RuntimeError(f'{path}: simData missing "{k}"')

    if 'simConfig' not in data:
        raise RuntimeError(f'{path}: missing "simConfig" — cannot reconstruct GID ranges')

    return data


# ---------------------------------------------------------------------------
# GID reconstruction
# ---------------------------------------------------------------------------
def reconstruct_pop_gids(sim_config):
    """Build {pop_name: array_of_gids} from simConfig.cellNumber / cellNumber0.

    The simConfig is a NetPyNE SimConfig object (attribute access), but we
    handle plain dict too for robustness.  cfg.py sets:
        cfg.cellNumber  = {'HL23PYR': 800, 'HL23SST': 50, ...}
        cfg.cellNumber0 = {'HL23PYR': 0,   'HL23SST': 800, ...}
    so GID ranges are  [cellNumber0[pop], cellNumber0[pop]+cellNumber[pop]).
    """
    def _get(obj, attr):
        """Attribute-or-key accessor — works for SimConfig objects and dicts."""
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj[attr]
        raise RuntimeError(f'simConfig has no "{attr}" — cannot reconstruct GID ranges. '
                           f'Available: {list(obj.__dict__.keys()) if hasattr(obj, "__dict__") else "?"}')

    cell_number  = _get(sim_config, 'cellNumber')   # {pop: count}
    cell_number0 = _get(sim_config, 'cellNumber0')   # {pop: first_gid}

    pop_gids = {}
    for name in CELL_NAMES:
        start = int(cell_number0[name])
        count = int(cell_number[name])
        pop_gids[name] = np.arange(start, start + count)

    return pop_gids


# ---------------------------------------------------------------------------
# Rate computation — DUAL definitions
# ---------------------------------------------------------------------------
def compute_rates(spkt, spkid, pop_gids, tstop, transient):
    """Compute per-population firing rates under two conventions.

    Returns (all_rates, nonsilent_rates):
      all_rates     — mean Hz across ALL cells in each pop (matches
                      NetPyNE printPopAvgRates, the primary reference).
      nonsilent_rates — mean Hz across cells firing > 0.2 Hz (matches
                        Yao 2022 Table S2 convention).
    """
    spkt  = np.asarray(spkt,  dtype=float)
    spkid = np.asarray(spkid, dtype=float)

    dur_s = (tstop - transient) / 1000.0
    mask  = spkt >= transient          # keep spikes after transient

    all_rates      = {}
    nonsilent_rates = {}

    for name in CELL_NAMES:
        gids = pop_gids[name]
        if len(gids) == 0:
            all_rates[name] = 0.0
            nonsilent_rates[name] = 0.0
            continue

        # Per-cell spike count in the analysis window
        cell_counts = np.array([
            np.sum((spkid == g) & mask) for g in gids
        ])
        cell_hz = cell_counts / dur_s

        # PRIMARY: all-cells mean (NetPyNE convention)
        all_rates[name] = float(np.mean(cell_hz))

        # SECONDARY: non-silent only (Yao Table S2 convention)
        active = cell_hz[cell_hz > 0.2]
        nonsilent_rates[name] = float(np.mean(active)) if len(active) > 0 else 0.0

    return all_rates, nonsilent_rates


# ---------------------------------------------------------------------------
# Connection / synapse counting
# ---------------------------------------------------------------------------
def count_connections(data, pkl_path):
    """Extract total connections and synaptic contacts from the saved network.

    NetPyNE stores per-cell connection lists under data['net']['cells'][i]['conns'].
    Each conn dict represents one connection; the number of synaptic contacts per
    connection is stored in 'syns' or as len of the synMech list.

    This data is ONLY present if 'netCells' is in cfg.saveDataInclude.
    If missing, we error loudly rather than silently reporting zero.
    """
    # --- Check if net data was saved ---
    if 'net' not in data:
        raise RuntimeError(
            f'{pkl_path}: "net" key missing from pkl.\n'
            f'  cfg.saveDataInclude does not include "netCells".\n'
            f'  To get connection stats, add "netCells" to the list in cfg.py:\n'
            f'    cfg.saveDataInclude = [\'simData\', \'simConfig\', \'netParams\', \'netCells\']\n'
            f'  and re-run the simulation.'
        )

    net = data['net']
    if 'cells' not in net or len(net['cells']) == 0:
        raise RuntimeError(
            f'{pkl_path}: net["cells"] is empty or missing.\n'
            f'  Ensure cfg.saveCellConns = True and "netCells" is in saveDataInclude.'
        )

    cells = net['cells']
    total_conns = 0
    total_syns  = 0
    for cell in cells:
        conns = cell.get('conns', [])
        total_conns += len(conns)
        # Each connection may have multiple synaptic contacts.
        # NetPyNE stores the synapse count in different ways depending on version;
        # the most common is that each entry in 'conns' is one synaptic contact,
        # and the higher-level "connection" count comes from unique (pre, sec, loc) tuples.
        # For now, count raw entries as synaptic contacts (matches sim analysis output).
        total_syns += len(conns)

    return total_conns, total_syns, len(cells)


# ---------------------------------------------------------------------------
# Raster plot (first seed)
# ---------------------------------------------------------------------------
def plot_raster(spkt, spkid, pop_gids, seed_label, out_dir):
    """Raster of first 700 ms (after transient) — mirrors original Fig 2C."""
    spkt  = np.asarray(spkt,  dtype=float)
    spkid = np.asarray(spkid, dtype=float)

    T0 = TRANSIENT
    T1 = TRANSIENT + 700.0

    fig, ax = plt.subplots(figsize=(14, 5))
    y_offset = 0
    yticks, ytick_labels, dividers = [], [], []

    for name in CELL_NAMES:
        gids    = pop_gids[name]
        if len(gids) == 0:
            continue
        gid_min = gids[0]
        mask    = np.isin(spkid, gids) & (spkt >= T0) & (spkt < T1)
        t_sel   = spkt[mask] - T0
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
    ax.set_title(f'Baseline Raster ({seed_label})', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, markerscale=4)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig2C_raster.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  [raster] Saved -> {path}')


# ---------------------------------------------------------------------------
# Bar chart — mean ± SD firing rates across seeds
# ---------------------------------------------------------------------------
def plot_rate_bars(all_seeds_rates, rate_label, filename, out_dir):
    """Bar chart of mean ± SD rates, with paper target overlay."""
    n = len(list(all_seeds_rates.values())[0])
    means = {name: np.mean(all_seeds_rates[name]) for name in CELL_NAMES}
    sds   = {name: np.std(all_seeds_rates[name], ddof=1)
             if n > 1 else 0.0
             for name in CELL_NAMES}

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(CELL_NAMES))
    ax.bar(x,
           [means[n] for n in CELL_NAMES],
           yerr=[sds[n] for n in CELL_NAMES],
           color=[POP_COLORS[n] for n in CELL_NAMES],
           edgecolor='k', linewidth=1.0,
           error_kw={'elinewidth': 1.5, 'capsize': 4})

    for i, name in enumerate(CELL_NAMES):
        tgt, _ = PAPER_TARGETS[name]
        ax.plot([i - 0.4, i + 0.4], [tgt, tgt], 'k--', lw=1.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([POP_LABELS[n] for n in CELL_NAMES], fontsize=12)
    ax.set_ylabel('Mean Firing Rate (Hz)', fontsize=12)
    ax.set_title(f'{rate_label} (n={n} seeds)', fontsize=12)
    ax.plot([], [], 'k--', lw=1.5, label='Paper target')
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  [rates]  Saved -> {path}')


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
def write_report(out_dir, seed_results, conn_results):
    """Write validation_report.txt with dual-rate tables and connectivity."""

    lines = []
    lines.append('=' * 65)
    lines.append('  Yao 2022 L23 Healthy Model — Validation Report')
    lines.append('=' * 65)
    lines.append(f'Date:  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'Seeds: {len(seed_results)} pkls loaded')
    lines.append('')

    # --- List each seed's source file and GLOBALSEED ---
    lines.append('PKL FILES:')
    for sr in seed_results:
        lines.append(f'  {sr["label"]:30s}  GLOBALSEED={sr["globalseed"]}')
    lines.append('')

    # --- PRIMARY: all-cells rates (NetPyNE convention) ---
    all_cells_rates = {name: [] for name in CELL_NAMES}
    for sr in seed_results:
        for name in CELL_NAMES:
            all_cells_rates[name].append(sr['all_rates'][name])

    n = len(seed_results)
    lines.append('-' * 65)
    lines.append('PRIMARY RATES — all cells, full window (NetPyNE printPopAvgRates)')
    lines.append(f'  Analysis window: {TRANSIENT:.0f} – {TSTOP:.0f} ms')
    lines.append('-' * 65)
    lines.append(f'{"Population":12s} | {"Our value":16s} | {"Paper target":16s} | {"% err":7s}')
    lines.append('-' * 58)
    for name in CELL_NAMES:
        m  = np.mean(all_cells_rates[name])
        sd = np.std(all_cells_rates[name], ddof=1) if n > 1 else 0.0
        tm, ts = PAPER_TARGETS[name]
        pct = (m - tm) / tm * 100
        lines.append(
            f'{POP_LABELS[name]:12s} | {m:5.2f} +/- {sd:5.2f} Hz | '
            f'{tm:5.2f} +/- {ts:4.2f} Hz  | {pct:+.1f}%')

    lines.append('')
    lines.append('PER-SEED (all-cells):')
    hdr = f'  {"Label":30s}  ' + '  '.join(f'{POP_LABELS[n]:>8s}' for n in CELL_NAMES)
    lines.append(hdr)
    for sr in seed_results:
        row = f'  {sr["label"]:30s}  '
        row += '  '.join(f'{sr["all_rates"][n]:8.2f}' for n in CELL_NAMES)
        lines.append(row)

    # --- SECONDARY: non-silent rates (Yao Table S2 convention) ---
    ns_rates = {name: [] for name in CELL_NAMES}
    for sr in seed_results:
        for name in CELL_NAMES:
            ns_rates[name].append(sr['nonsilent_rates'][name])

    lines.append('')
    lines.append('-' * 65)
    lines.append('SECONDARY RATES — non-silent cells only (>0.2 Hz, Yao Table S2)')
    lines.append('-' * 65)
    lines.append(f'{"Population":12s} | {"Our value":16s} | {"Paper target":16s} | {"% err":7s}')
    lines.append('-' * 58)
    for name in CELL_NAMES:
        m  = np.mean(ns_rates[name])
        sd = np.std(ns_rates[name], ddof=1) if n > 1 else 0.0
        tm, ts = PAPER_TARGETS[name]
        pct = (m - tm) / tm * 100
        lines.append(
            f'{POP_LABELS[name]:12s} | {m:5.2f} +/- {sd:5.2f} Hz | '
            f'{tm:5.2f} +/- {ts:4.2f} Hz  | {pct:+.1f}%')

    lines.append('')
    lines.append('PER-SEED (non-silent):')
    lines.append(hdr)
    for sr in seed_results:
        row = f'  {sr["label"]:30s}  '
        row += '  '.join(f'{sr["nonsilent_rates"][n]:8.2f}' for n in CELL_NAMES)
        lines.append(row)

    # --- Connectivity ---
    lines.append('')
    lines.append('-' * 65)
    lines.append('CONNECTIVITY')
    lines.append('-' * 65)
    if conn_results:
        for cr in conn_results:
            lines.append(
                f'  {cr["label"]:30s}  cells={cr["n_cells"]}  '
                f'conns={cr["total_conns"]}  '
                f'({cr["total_conns"]/cr["n_cells"]:.2f}/cell)  '
                f'syn_contacts={cr["total_syns"]}  '
                f'({cr["total_syns"]/cr["n_cells"]:.2f}/cell)')
    else:
        lines.append('  ** NOT AVAILABLE **')
        lines.append('  cfg.saveDataInclude does not include "netCells".')
        lines.append('  To get connection stats, add "netCells" to the list in cfg.py:')
        lines.append('    cfg.saveDataInclude = [\'simData\', \'simConfig\', \'netParams\', \'netCells\']')
        lines.append('  and re-run the simulation.')

    lines.append('')
    lines.append('=' * 65)

    report_path = os.path.join(out_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  [report] Saved -> {report_path}')

    # Also print to stdout
    for line in lines:
        print(line)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Validate L23 NetPyNE healthy model against Yao 2022 targets.'
    )
    parser.add_argument(
        '--data-dir',
        default=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'v1_batch5_lock'),
        help='Directory containing *_data.pkl files (default: ../../data/v1_batch5_lock)'
    )
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    print(f'=== validate_healthy.py  {datetime.datetime.now()} ===')
    print(f'Data dir: {data_dir}')

    # --- Discover pkls ---
    pkl_paths = discover_pkls(data_dir)
    print(f'Found {len(pkl_paths)} pkl files.')
    if not pkl_paths:
        print(f'No *_data.pkl files in {data_dir} — nothing to validate.')
        print('If the array job is still running, re-run after more seeds finish.')
        sys.exit(0)

    for p in pkl_paths:
        print(f'  {os.path.basename(p)}')

    # --- Process each pkl ---
    seed_results = []     # one entry per seed: rates, label, globalseed
    conn_results = []     # connection stats (may be empty if netCells not saved)
    conn_error_shown = False

    for pkl_path in pkl_paths:
        label = os.path.basename(pkl_path).replace('_data.pkl', '')
        print(f'\nProcessing {label} ...')

        try:
            data = load_pkl(pkl_path)
        except RuntimeError as e:
            print(f'  ERROR: {e}')
            continue

        sim_config = data['simConfig']
        sim_data   = data['simData']

        # --- Reconstruct GID -> population mapping ---
        try:
            pop_gids = reconstruct_pop_gids(sim_config)
        except RuntimeError as e:
            print(f'  ERROR: {e}')
            continue

        total_cells = sum(len(g) for g in pop_gids.values())
        print(f'  GID reconstruction: {total_cells} cells '
              f'({", ".join(f"{POP_LABELS[n]}={len(pop_gids[n])}" for n in CELL_NAMES)})')

        # --- Extract GLOBALSEED ---
        def _get(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return default

        globalseed = _get(sim_config, 'GLOBALSEED', '?')

        # --- Compute dual rates ---
        spkt  = sim_data['spkt']
        spkid = sim_data['spkid']
        all_rates, nonsilent_rates = compute_rates(spkt, spkid, pop_gids, TSTOP, TRANSIENT)

        print(f'  All-cells rates:    '
              + '  '.join(f'{POP_LABELS[n]}={all_rates[n]:.2f}' for n in CELL_NAMES))
        print(f'  Non-silent rates:   '
              + '  '.join(f'{POP_LABELS[n]}={nonsilent_rates[n]:.2f}' for n in CELL_NAMES))

        seed_results.append({
            'label':           label,
            'globalseed':      globalseed,
            'all_rates':       all_rates,
            'nonsilent_rates': nonsilent_rates,
            'spkt':            spkt,
            'spkid':           spkid,
            'pop_gids':        pop_gids,
        })

        # --- Connection stats (try, error loudly if missing) ---
        if not conn_error_shown:
            try:
                total_conns, total_syns, n_cells = count_connections(data, pkl_path)
                conn_results.append({
                    'label':       label,
                    'total_conns': total_conns,
                    'total_syns':  total_syns,
                    'n_cells':     n_cells,
                })
                print(f'  Connections: {total_conns} ({total_conns/n_cells:.1f}/cell)  '
                      f'Syn contacts: {total_syns} ({total_syns/n_cells:.1f}/cell)')
            except RuntimeError as e:
                # Print the error once — it will be the same for all pkls
                # since they all share the same cfg.saveDataInclude
                print(f'  WARNING: {e}')
                conn_error_shown = True

    if not seed_results:
        print('\nNo pkls could be loaded successfully. Nothing to report.')
        sys.exit(1)

    # --- Note partial results ---
    if len(seed_results) < len(pkl_paths):
        print(f'\nNOTE: {len(pkl_paths) - len(seed_results)} pkls failed to load — '
              f'reporting on {len(seed_results)} seed(s).')

    # --- Raster from first seed ---
    first = seed_results[0]
    plot_raster(first['spkt'], first['spkid'], first['pop_gids'],
                first['label'], data_dir)

    # --- Bar charts (both definitions) ---
    if len(seed_results) >= 1:
        all_cells_agg = {name: [sr['all_rates'][name] for sr in seed_results]
                         for name in CELL_NAMES}
        nonsilent_agg = {name: [sr['nonsilent_rates'][name] for sr in seed_results]
                         for name in CELL_NAMES}

        plot_rate_bars(all_cells_agg, 'All-Cells Rates',
                       'fig2D_baseline_rates_allcells.png', data_dir)
        plot_rate_bars(nonsilent_agg, 'Non-Silent Rates (Yao Table S2)',
                       'fig2D_baseline_rates_nonsilent.png', data_dir)

    # --- Write report ---
    write_report(data_dir, seed_results, conn_results)

    print(f'\n=== Done ({len(seed_results)} seeds validated) ===')


if __name__ == '__main__':
    main()

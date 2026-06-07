#!/usr/bin/env python3
"""
epileptiform_metrics.py — early-stage hyperexcitability / epileptiform validation across CPS
=============================================================================================
Computes spike-based epileptiform / synchrony metrics from the CPS sweep, to test whether the
model produces an EARLY-MID hyperexcitable regime (the early-AD epileptiform expectation) or stays
hypoactive (v1's predicted failure). This is the EPILEPSY-AS-VALIDATION axis (Tier A Part 5):
an independent, mechanistically-direct, spike-only validation target — NO TMS, NO new architecture.

Run on HPC (netpyne env), from sim/, modest memory (lean loader frees each pkl):
    sbatch run_epileptiform.sh        # --mem=32G --time=01:00:00, python -u
  (script: python -u epileptiform_metrics.py --sweep ../data/cps_sweep --out ../data/epileptiform)

Reads the CPS-sweep pkls + their stage mapping. Spikes only (spkt/spkid + pop gids); the heavy
net/V_soma are freed immediately to avoid OOM (same pattern as attr_gamma_excit.py).

METRICS (all from spikes, vs CPS):
  - pop_burst_rate     : # synchronous network bursts / sec (candidate epileptiform events)
  - hypersynchrony     : peak co-active fraction in tight windows, normalized to chance
  - ei_excursion_rate  : # transient runaway-excitation events / sec (E-rate spikes above thresh)
  - fano               : Fano factor of the population spike count (burstiness)
  - pyr_rate, ei_ratio : context (mean PYR rate, E:I ratio)

INTERPRETATION (Tier A Part 5.2): if pop_burst_rate / hypersynchrony RISE at early-mid CPS →
the model produces the early-hyper regime (epilepsy-validation PASSES). If they stay flat/fall
while the model is monotonically hypoactive → v1 FAILS the early-hyper bar — the diagnostic
signal that the early-hyper mechanism (microglial/NMDA arm) is missing. Either result is useful.
"""
import argparse, glob, os, pickle, csv, gc, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POPS = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
EXC = ['HL23PYR']; INH = ['HL23SST', 'HL23PV', 'HL23VIP']
T0 = 1000.0   # discard transient


def extract_spikes_and_gids(path):
    """Load pkl, pull ONLY spkt/spkid + pop->gids + duration, free the rest (avoid OOM)."""
    with open(path, 'rb') as f:
        D = pickle.load(f)
    sd = D['simData']
    spkt = np.asarray(sd['spkt'], dtype=float)
    spkid = np.asarray(sd['spkid'], dtype=int)
    net = D['net']
    cells = net['cells'] if isinstance(net, dict) and 'cells' in net else net.cells
    gids = {p: [] for p in POPS}
    for c in cells:
        c = c if isinstance(c, dict) else c.__dict__
        p = c.get('tags', {}).get('pop'); g = c.get('gid')
        if p in gids: gids[p].append(g)
    gids = {p: np.array(sorted(v)) for p, v in gids.items()}
    sc = D['simConfig']; sc = sc if isinstance(sc, dict) else sc.__dict__
    dur = float(sc.get('duration', 3000.0))
    stage = float(sc.get('ad_stage', sc.get('AD_STAGE', np.nan)))
    del D, sd, net, cells; gc.collect()
    return spkt, spkid, gids, dur, stage


def pop_rate_ts(spkt, spkid, gids, t0, t1, bin_ms):
    sel = np.isin(spkid, gids) & (spkt >= t0) & (spkt < t1)
    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    h, _ = np.histogram(spkt[sel], bins=edges)
    return h, edges


def epileptiform_metrics(spkt, spkid, gids, t0, t1):
    n_all = sum(len(gids[p]) for p in POPS)
    all_g = np.concatenate([gids[p] for p in POPS])
    dur_s = (t1 - t0) / 1000.0

    # --- population burst detection: 5ms-binned total spike count; a "burst" = bin exceeding
    #     mean + 3*std of the binned count (synchronous network event) ---
    counts, edges = pop_rate_ts(spkt, spkid, all_g, t0, t1, 5.0)
    mu, sd_ = counts.mean(), counts.std()
    thresh = mu + 3.0 * sd_
    # count contiguous supra-threshold runs as single bursts
    supra = counts > thresh
    n_bursts = int(np.sum(supra[1:] & ~supra[:-1]) + (1 if supra[0] else 0))
    pop_burst_rate = n_bursts / dur_s

    # --- hypersynchrony: in each 5ms bin, fraction of ALL cells that spiked; peak vs chance.
    #     chance co-active fraction ~ mean per-bin fraction; report peak/mean ratio + raw peak ---
    # per-bin unique-cell count
    sel = (spkt >= t0) & (spkt < t1)
    st, si = spkt[sel], spkid[sel]
    bin_idx = ((st - t0) / 5.0).astype(int)
    nbin = len(counts)
    coactive = np.zeros(nbin)
    # unique cells per bin
    order = np.argsort(bin_idx)
    bi_s, si_s = bin_idx[order], si[order]
    start = 0
    for b in range(nbin):
        # slice cells in bin b
        end = start
        while end < len(bi_s) and bi_s[end] == b:
            end += 1
        if end > start:
            coactive[b] = len(np.unique(si_s[start:end])) / max(n_all, 1)
        start = end if end > start else start
        # (note: bins with no spikes stay 0; loop robust to gaps)
    # recompute robustly (the above loop assumes contiguous bins; do it cleanly):
    coactive = np.zeros(nbin)
    for b in np.unique(bin_idx):
        if 0 <= b < nbin:
            coactive[b] = len(np.unique(si[bin_idx == b])) / max(n_all, 1)
    peak_coactive = float(coactive.max()) if nbin else 0.0
    mean_coactive = float(coactive[coactive > 0].mean()) if np.any(coactive > 0) else 0.0
    hypersync = float(peak_coactive / mean_coactive) if mean_coactive > 0 else 0.0

    # --- E/I excursion rate: E-population rate timeseries; count transient excursions where
    #     E-rate exceeds mean + 2*std (runaway-excitation events) ---
    e_g = np.concatenate([gids[p] for p in EXC])
    e_counts, _ = pop_rate_ts(spkt, spkid, e_g, t0, t1, 5.0)
    e_mu, e_sd = e_counts.mean(), e_counts.std()
    e_thr = e_mu + 2.0 * e_sd
    e_supra = e_counts > e_thr
    n_exc = int(np.sum(e_supra[1:] & ~e_supra[:-1]) + (1 if e_supra[0] else 0))
    ei_excursion_rate = n_exc / dur_s

    # --- Fano factor of population spike count (burstiness) ---
    fano = float(sd_**2 / mu) if mu > 0 else 0.0

    # --- context: PYR rate, E:I ratio ---
    pyr_counts, _ = pop_rate_ts(spkt, spkid, gids['HL23PYR'], t0, t1, 5.0)
    pyr_rate = float(pyr_counts.sum() / (max(len(gids['HL23PYR']), 1) * dur_s))
    i_g = np.concatenate([gids[p] for p in INH])
    i_counts, _ = pop_rate_ts(spkt, spkid, i_g, t0, t1, 5.0)
    re = e_counts.sum() / (max(len(e_g), 1) * dur_s)
    ri = i_counts.sum() / (max(len(i_g), 1) * dur_s)
    ei_ratio = float(re / ri) if ri > 0 else np.nan

    return dict(pop_burst_rate=pop_burst_rate, hypersynchrony=hypersync,
                ei_excursion_rate=ei_excursion_rate, fano=fano,
                pyr_rate=pyr_rate, ei_ratio=ei_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True, help='dir with CPS-sweep pkls')
    ap.add_argument('--out', default='../data/epileptiform')
    ap.add_argument('--glob', default='*_data.pkl', help='pkl filename glob')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pkls = sorted(glob.glob(os.path.join(args.sweep, args.glob)))
    if not pkls:
        print(f'No pkls matching {args.glob} in {args.sweep}'); return

    results = {}  # stage -> metrics
    for pf in pkls:
        spkt, spkid, gids, dur, stage = extract_spikes_and_gids(pf)
        m = epileptiform_metrics(spkt, spkid, gids, T0, dur)
        results[stage] = m
        print(f'  stage={stage:.3f}  burst={m["pop_burst_rate"]:.2f}/s  '
              f'hypersync={m["hypersynchrony"]:.2f}  ei_exc={m["ei_excursion_rate"]:.2f}/s  '
              f'fano={m["fano"]:.2f}  PYR={m["pyr_rate"]:.3f}  E/I={m["ei_ratio"]:.3f}',
              flush=True)
        del spkt, spkid, gids; gc.collect()

    stages = sorted(results)
    keys = ['pop_burst_rate', 'hypersynchrony', 'ei_excursion_rate', 'fano', 'pyr_rate', 'ei_ratio']

    # CSV
    with open(os.path.join(args.out, 'epileptiform_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['stage'] + keys)
        for s in stages:
            w.writerow([s] + [round(results[s][k], 5) for k in keys])

    # FIG: each metric vs CPS — the early-hyper test (do burst/hypersync RISE early?)
    titles = {'pop_burst_rate': 'Population burst rate (events/s)',
              'hypersynchrony': 'Hypersynchrony (peak/mean co-active)',
              'ei_excursion_rate': 'E/I excursion rate (events/s)',
              'fano': 'Fano factor (burstiness)',
              'pyr_rate': 'PYR mean rate (Hz)', 'ei_ratio': 'E:I ratio'}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, k in zip(axes.ravel(), keys):
        ax.plot(stages, [results[s][k] for s in stages], 'o-', lw=2, color='#b3402f')
        ax.set_title(titles[k]); ax.set_xlabel('CPS stage'); ax.set_ylabel(titles[k]); ax.grid(alpha=0.3)
    fig.suptitle('Epileptiform / synchrony metrics vs CPS (early-stage validation axis)\n'
                 'EARLY-HYPER expectation: burst rate + hypersynchrony RISE at early-mid CPS.\n'
                 'If they stay flat/fall while PYR declines monotonically → v1 FAILS the early-hyper bar.',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(args.out, 'epileptiform_vs_cps.png'), dpi=150)
    plt.close(fig)

    # verdict line
    s0, s_early = stages[0], stages[min(1, len(stages)-1)]
    rose = (results[s_early]['pop_burst_rate'] > results[s0]['pop_burst_rate'] or
            results[s_early]['hypersynchrony'] > results[s0]['hypersynchrony'])
    print('\n=== EARLY-HYPER VERDICT ===')
    print(f'burst rate  s0={results[s0]["pop_burst_rate"]:.2f} -> early={results[s_early]["pop_burst_rate"]:.2f}')
    print(f'hypersync   s0={results[s0]["hypersynchrony"]:.2f} -> early={results[s_early]["hypersynchrony"]:.2f}')
    print(f'PYR rate    s0={results[s0]["pyr_rate"]:.3f} -> early={results[s_early]["pyr_rate"]:.3f}')
    print('EARLY-HYPER SIGNAL PRESENT' if rose else
          'NO early-hyper signal -> v1 FAILS epileptiform validation (expected; motivates microglial arm)')
    print(f'\nFigure + epileptiform_summary.csv -> {args.out}')


if __name__ == '__main__':
    main()

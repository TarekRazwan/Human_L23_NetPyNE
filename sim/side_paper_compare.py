#!/usr/bin/env python3
"""
side_paper_compare.py
=====================
Side-paper characterization: H01 real spatial structure vs flat probabilistic
baseline. Single controlled variable = USE_H01_DISTANCE_CONN (real soma
positions + distance-dependent connectivity) vs flat. Both healthy, same seed,
same convergence target.

Run on the HPC (netpyne conda env), from sim/:
    python side_paper_compare.py \
        --h01  ../data/h01_sanity \
        --flat ../data/v1_batch5_lock \
        --out  ../data/side_paper_analysis

Reads *_data.pkl from each dir. Produces labeled PNGs + a stats CSV in --out.
Only the small PNGs/CSV need copying to the Mac (NOT the GB pkls).

Computable from saved data (spkt, spkid, V_soma, net positions):
  1. Per-population firing-rate distributions (CDF overlay + KS test)
  2. E/I balance (E vs I activity ratio + per-pop mean rate bars)
  3. CV-ISI distributions per population
  4. Distance-resolved pairwise spike correlation (THE spatial signature)
  5. Population-rate power spectrum (gamma band; from spikes, since no LFP)
"""
import argparse, glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless on HPC
import matplotlib.pyplot as plt
from scipy import stats, signal

POPS = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
EXC  = ['HL23PYR']
INH  = ['HL23SST', 'HL23PV', 'HL23VIP']
COL  = {'HL23PYR':'#2c5f8a', 'HL23SST':'#c0563f', 'HL23PV':'#3f8a52', 'HL23VIP':'#8a6d3f'}
COND_COL = {'H01':'#1f4e79', 'Flat':'#a02020'}   # H01 = blue, Flat = red
T_WARMUP = 1000.0               # ms; discard transient (analysis window 1000-3000)


def load_run(d):
    pf = sorted(glob.glob(os.path.join(d, '*_data.pkl')))
    if not pf:
        raise FileNotFoundError(f'no *_data.pkl in {d}')
    with open(pf[0], 'rb') as f:
        D = pickle.load(f)
    return D, pf[0]


def pop_gids(net):
    """Map population -> sorted list of global cell ids, from net structure."""
    cells = net['cells'] if isinstance(net, dict) and 'cells' in net else net.cells
    out = {p: [] for p in POPS}
    for c in cells:
        c = c if isinstance(c, dict) else c.__dict__
        pop = c.get('tags', {}).get('pop')
        gid = c.get('gid')
        if pop in out:
            out[pop].append(gid)
    return {p: sorted(v) for p, v in out.items()}


def cell_positions(net):
    """gid -> (x,y,z) soma position, for distance-resolved correlation."""
    cells = net['cells'] if isinstance(net, dict) and 'cells' in net else net.cells
    pos = {}
    for c in cells:
        c = c if isinstance(c, dict) else c.__dict__
        t = c.get('tags', {})
        pos[c.get('gid')] = np.array([t.get('x', 0.0), t.get('y', 0.0), t.get('z', 0.0)])
    return pos


def spikes_by_gid(sd, t0=T_WARMUP):
    """gid -> np.array of spike times (ms), after warmup."""
    spkt = np.asarray(sd['spkt']); spkid = np.asarray(sd['spkid'])
    keep = spkt >= t0
    spkt, spkid = spkt[keep], spkid[keep]
    out = {}
    for g in np.unique(spkid):
        out[int(g)] = np.sort(spkt[spkid == g])
    return out


def rates_per_cell(spk, gids, dur_s):
    """mean firing rate (Hz) per cell in a population (0 for silent cells)."""
    return np.array([len(spk.get(g, [])) / dur_s for g in gids])


def cv_isi_per_cell(spk, gids):
    """CV of inter-spike-intervals per cell (needs >=3 spikes)."""
    out = []
    for g in gids:
        s = spk.get(g, [])
        if len(s) >= 3:
            isi = np.diff(s)
            if isi.mean() > 0:
                out.append(isi.std() / isi.mean())
    return np.array(out)


def binned_rate(sd, gids, t0, t1, bin_ms=2.0):
    """population spike-count histogram (for synchrony / spectrum)."""
    spkt = np.asarray(sd['spkt']); spkid = np.asarray(sd['spkid'])
    sel = np.isin(spkid, gids) & (spkt >= t0) & (spkt < t1)
    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    h, _ = np.histogram(spkt[sel], bins=edges)
    return h.astype(float), edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h01', required=True)
    ap.add_argument('--flat', required=True)
    ap.add_argument('--out', default='../data/side_paper_analysis')
    ap.add_argument('--corr-pairs', type=int, default=4000,
                    help='random cell pairs sampled for distance-correlation')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {}
    for label, d in [('H01', args.h01), ('Flat', args.flat)]:
        D, pf = load_run(d)
        sc = D['simConfig']
        sc = sc if isinstance(sc, dict) else sc.__dict__
        runs[label] = dict(D=D, sd=D['simData'], net=D['net'], sc=sc, path=pf)
        print(f'[{label}] {pf}')

    # ---- self-check: confirm SINGLE controlled variable ----------------------
    print('\n=== controlled-variable check (should differ ONLY in USE_H01_DISTANCE_CONN) ===')
    keys = ['USE_H01_DISTANCE_CONN', 'GLOBALSEED', 'duration', 'ad_stage',
            'DRUG', 'LOAD_MATRIX_LFPy', 'scale']
    for k in keys:
        a = runs['H01']['sc'].get(k); b = runs['Flat']['sc'].get(k)
        flag = '' if (a == b) else '  <-- differs'
        if k == 'USE_H01_DISTANCE_CONN':
            flag = '  <-- EXPECTED difference (the controlled variable)'
        print(f'  {k:24s} H01={a!s:8s} Flat={b!s:8s}{flag}')
    print('  (any OTHER differing line means the comparison is NOT single-variable)\n')

    # duration in seconds for rate calc (analysis window after warmup)
    for L in runs:
        dur_ms = float(runs[L]['sc'].get('duration', 3000.0))
        runs[L]['t0'], runs[L]['t1'] = T_WARMUP, dur_ms
        runs[L]['dur_s'] = (dur_ms - T_WARMUP) / 1000.0
        runs[L]['gids'] = pop_gids(runs[L]['net'])
        runs[L]['spk'] = spikes_by_gid(runs[L]['sd'], T_WARMUP)
        runs[L]['pos'] = cell_positions(runs[L]['net'])

    summary = []   # rows for CSV

    # ========================================================================
    # FIG 1 — per-population firing-rate CDFs + KS test
    # ========================================================================
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, pop in zip(axes, POPS):
        for L in ['H01', 'Flat']:
            r = rates_per_cell(runs[L]['spk'], runs[L]['gids'][pop], runs[L]['dur_s'])
            xs = np.sort(r); ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=COND_COL[L], lw=2,
                    label=f'{L} (mean {r.mean():.2f} Hz)')
            runs[L][f'rate_{pop}'] = r
        rH = runs['H01'][f'rate_{pop}']; rF = runs['Flat'][f'rate_{pop}']
        ks, p = stats.ks_2samp(rH, rF)
        ax.set_title(f'{pop}\nKS={ks:.3f}, p={p:.1e}', fontsize=11)
        ax.set_xlabel('firing rate (Hz)'); ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)
        summary.append(dict(metric='mean_rate_Hz', pop=pop,
                            H01=round(rH.mean(), 4), Flat=round(rF.mean(), 4),
                            test='KS_2samp', stat=round(ks, 4), p=p))
    axes[0].set_ylabel('cumulative fraction of cells')
    fig.suptitle('Firing-rate distributions per population — H01 spatial vs Flat baseline '
                 '(2-sample KS test)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.out, 'fig1_rate_CDFs.png'), dpi=160)
    plt.close(fig)

    # ========================================================================
    # FIG 2 — E/I balance: per-pop mean rate bars + E:I activity ratio
    # ========================================================================
    fig, (axb, axr) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={'width_ratios': [2.2, 1]})
    x = np.arange(len(POPS)); w = 0.38
    for i, L in enumerate(['H01', 'Flat']):
        means = [runs[L][f'rate_{p}'].mean() for p in POPS]
        sems  = [runs[L][f'rate_{p}'].std() / np.sqrt(len(runs[L][f'rate_{p}'])) for p in POPS]
        axb.bar(x + (i - 0.5) * w, means, w, yerr=sems, capsize=3,
                color=COND_COL[L], label=L, alpha=0.85)
    axb.set_xticks(x); axb.set_xticklabels(POPS, rotation=15)
    axb.set_ylabel('mean firing rate (Hz)')
    axb.set_title('Per-population mean rate (±SEM)'); axb.legend(); axb.grid(alpha=0.3, axis='y')

    # E:I activity ratio = total E spikes/s / total I spikes/s (population-weighted)
    ratios = {}
    for L in ['H01', 'Flat']:
        e = sum(runs[L][f'rate_{p}'].sum() for p in EXC)   # total exc spikes/s
        ii = sum(runs[L][f'rate_{p}'].sum() for p in INH)  # total inh spikes/s
        ratios[L] = e / ii if ii > 0 else np.nan
        summary.append(dict(metric='EI_activity_ratio', pop='E_over_I',
                            H01=round(ratios['H01'], 4) if L == 'Flat' else None,
                            Flat=round(ratios['Flat'], 4) if L == 'Flat' else None,
                            test='', stat='', p=''))
    axr.bar(list(ratios.keys()), list(ratios.values()),
            color=[COND_COL[k] for k in ratios], alpha=0.85)
    axr.set_ylabel('E : I activity ratio\n(total exc rate / total inh rate)')
    axr.set_title('E/I balance'); axr.grid(alpha=0.3, axis='y')
    for k, v in ratios.items():
        axr.text(k, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    fig.suptitle('Excitation/Inhibition balance — H01 vs Flat', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(args.out, 'fig2_EI_balance.png'), dpi=160)
    plt.close(fig)

    # ========================================================================
    # FIG 3 — CV-ISI distributions per population
    # ========================================================================
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, pop in zip(axes, POPS):
        for L in ['H01', 'Flat']:
            cv = cv_isi_per_cell(runs[L]['spk'], runs[L]['gids'][pop])
            if len(cv):
                ax.hist(cv, bins=20, density=True, histtype='step', lw=2,
                        color=COND_COL[L], label=f'{L} (med {np.median(cv):.2f})')
            runs[L][f'cv_{pop}'] = cv
        ax.set_title(pop, fontsize=11); ax.set_xlabel('CV of ISI')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        cH, cF = runs['H01'][f'cv_{pop}'], runs['Flat'][f'cv_{pop}']
        if len(cH) and len(cF):
            ks, p = stats.ks_2samp(cH, cF)
            summary.append(dict(metric='median_CV_ISI', pop=pop,
                                H01=round(np.median(cH), 4), Flat=round(np.median(cF), 4),
                                test='KS_2samp', stat=round(ks, 4), p=p))
    axes[0].set_ylabel('density')
    fig.suptitle('ISI irregularity (CV) per population — H01 vs Flat', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.out, 'fig3_CV_ISI.png'), dpi=160)
    plt.close(fig)

    # ========================================================================
    # FIG 4 — distance-resolved pairwise spike correlation (SPATIAL SIGNATURE)
    # ========================================================================
    # bin spike trains, sample random pairs, correlate, bin by 3D soma distance
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    dist_bins = np.arange(0, 400, 40.0)
    centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
    for L in ['H01', 'Flat']:
        sd = runs[L]['sd']; pos = runs[L]['pos']
        all_gids = [g for p in POPS for g in runs[L]['gids'][p]]
        # binned trains for all cells
        bin_ms = 5.0
        edges = np.arange(runs[L]['t0'], runs[L]['t1'] + bin_ms, bin_ms)
        spkt = np.asarray(sd['spkt']); spkid = np.asarray(sd['spkid'])
        trains = {}
        for g in all_gids:
            s = spkt[(spkid == g) & (spkt >= runs[L]['t0'])]
            h, _ = np.histogram(s, bins=edges)
            trains[g] = h.astype(float)
        # sample pairs
        npair = min(args.corr_pairs, len(all_gids) * (len(all_gids) - 1) // 2)
        dd, cc = [], []
        for _ in range(npair):
            a, b = rng.choice(all_gids, 2, replace=False)
            ta, tb = trains[a], trains[b]
            if ta.std() > 0 and tb.std() > 0:
                r = np.corrcoef(ta, tb)[0, 1]
                d = np.linalg.norm(pos[a] - pos[b])
                dd.append(d); cc.append(r)
        dd, cc = np.array(dd), np.array(cc)
        # mean correlation per distance bin
        my, se = [], []
        for lo, hi in zip(dist_bins[:-1], dist_bins[1:]):
            m = (dd >= lo) & (dd < hi)
            my.append(cc[m].mean() if m.sum() else np.nan)
            se.append(cc[m].std() / np.sqrt(m.sum()) if m.sum() > 1 else 0)
        my, se = np.array(my), np.array(se)
        ax.errorbar(centers, my, yerr=se, color=COND_COL[L], lw=2,
                    marker='o', capsize=3, label=f'{L}')
        # slope of corr vs distance (the signature: H01 should be non-flat)
        ok = ~np.isnan(my)
        if ok.sum() > 2:
            sl = np.polyfit(centers[ok], my[ok], 1)[0]
            summary.append(dict(metric='corr_vs_distance_slope', pop='all',
                                H01=None, Flat=None, test='linfit',
                                stat=f'{L}:{sl:.2e}', p=''))
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('pairwise 3D soma distance (µm)')
    ax.set_ylabel('mean pairwise spike-count correlation')
    ax.set_title('Distance-resolved spike correlation — the spatial-structure signature\n'
                 '(H01 real positions+distance connectivity vs flat probabilistic)', fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4_distance_correlation.png'), dpi=160)
    plt.close(fig)

    # ========================================================================
    # FIG 5 — population-rate power spectrum (gamma band; from spikes, no LFP)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in ['H01', 'Flat']:
        all_gids = [g for p in POPS for g in runs[L]['gids'][p]]
        h, edges = binned_rate(runs[L]['sd'], all_gids, runs[L]['t0'], runs[L]['t1'], bin_ms=1.0)
        fs = 1000.0  # 1 ms bins -> 1000 Hz
        f, Pxx = signal.welch(h - h.mean(), fs=fs, nperseg=min(1024, len(h)))
        band = (f >= 1) & (f <= 100)
        ax.semilogy(f[band], Pxx[band], color=COND_COL[L], lw=2, label=L)
        g = (f >= 30) & (f <= 80)
        summary.append(dict(metric='gamma_power_30_80Hz', pop='popRate',
                            H01=None, Flat=None, test='welch',
                            stat=f'{L}:{Pxx[g].sum():.3e}', p=''))
    ax.axvspan(30, 80, color='gold', alpha=0.15, label='gamma 30–80 Hz')
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('population-rate power (a.u.)')
    ax.set_title('Population-rate power spectrum — H01 vs Flat\n'
                 '(from binned population spikes; no LFP recorded)', fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig5_power_spectrum.png'), dpi=160)
    plt.close(fig)

    # ---- write summary table -------------------------------------------------
    import csv
    csv_path = os.path.join(args.out, 'side_paper_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'pop', 'H01', 'Flat', 'test', 'stat', 'p'])
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print('\n=== summary ===')
    for row in summary:
        print('  ', row)
    print(f'\nFigures + {os.path.basename(csv_path)} written to {args.out}')
    print('Copy ONLY the PNGs + CSV to the Mac (small); leave the pkls on the HPC.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
attr_gamma_excit.py — temporal/gamma + excitability decomposition (memory-light)
================================================================================
Per-modifier gamma-band power and excitability metrics across CPS, from the
30 attribution runs. Loads each pkl, extracts ONLY spike arrays + gids, frees
it immediately (gc) so peak memory ~= one pkl (avoids the 16GB OOM accumulation).

Run on HPC, modest memory (32G is plenty since we free each pkl):
    srun --nodes=1 --ntasks=1 --mem=32G --time=00:20:00 \
        python attr_gamma_excit.py --attr ../data/attribution --out ../data/attribution_dynamics

Metrics per run (within-run temporal, spikes only):
  - gamma_power: 30-80 Hz power of 1ms-binned total population rate (Welch)
  - rate_cv:     CV of the population-rate time series (temporal variability)
  - ei:          E:I activity ratio (mean exc rate / mean inh rate)
  - synchrony:   fraction of variance in pop-rate (a coarse synchrony proxy)
  - per-pop mean rates

Outputs (small): fig_gamma_vs_cps.png, fig_excitability_vs_cps.png,
fig_gamma_heatmap.png, gamma_excit_summary.csv. SPONTANEOUS dynamics (no TMS).
"""
import argparse, glob, os, pickle, csv, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

POPS = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
EXC = ['HL23PYR']; INH = ['HL23SST', 'HL23PV', 'HL23VIP']
T0 = 1000.0
MOD_LABEL = {
    'enable_M1a_sst_syn':'M1a SST-syn', 'enable_M1b_tonic':'M1b tonic-GABA',
    'enable_M1c_sst_loss':'M1c SST-loss', 'enable_M2_pv_kv31':'M2 PV-Kv3.1',
    'enable_M3_exc_scaffold':'M3 AMPA/NMDA', 'enable_M4_pyr_loss':'M4 PYR-loss',
}
MOD_ORDER = list(MOD_LABEL.keys())
MOD_COL = dict(zip(MOD_ORDER, plt.cm.tab10(np.linspace(0, 0.9, len(MOD_ORDER)))))


def extract_spikes_and_gids(path):
    """Load pkl, pull ONLY spkt/spkid + pop->gids + duration, free the rest."""
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
    # free the big object before returning
    del D, sd, net, cells
    gc.collect()
    return spkt, spkid, gids, dur


def rate_ts(spkt, spkid, gids, t0, t1, bin_ms):
    sel = np.isin(spkid, gids) & (spkt >= t0) & (spkt < t1)
    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    h, _ = np.histogram(spkt[sel], bins=edges)
    return h / (max(len(gids), 1) * bin_ms / 1000.0)


def metrics(spkt, spkid, gids, t0, t1):
    all_g = np.concatenate([gids[p] for p in POPS])
    r5 = rate_ts(spkt, spkid, all_g, t0, t1, 5.0)
    r1 = rate_ts(spkt, spkid, all_g, t0, t1, 1.0)
    f, Pxx = signal.welch(r1 - r1.mean(), fs=1000.0, nperseg=min(1024, len(r1)))
    gamma = float(Pxx[(f >= 30) & (f <= 80)].sum())
    rate_cv = float(r5.std() / r5.mean()) if r5.mean() > 0 else 0.0
    re = rate_ts(spkt, spkid, np.concatenate([gids[p] for p in EXC]), t0, t1, 5.0).mean()
    ri = rate_ts(spkt, spkid, np.concatenate([gids[p] for p in INH]), t0, t1, 5.0).mean()
    ei = float(re / ri) if ri > 0 else np.nan
    popmean = {p: float(rate_ts(spkt, spkid, gids[p], t0, t1, 5.0).mean()) for p in POPS}
    return dict(gamma=gamma, rate_cv=rate_cv, ei=ei,
                **{f'rate_{p}': popmean[p] for p in POPS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attr', required=True)
    ap.add_argument('--out', default='../data/attribution_dynamics')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    man = {}
    with open(os.path.join(args.attr, 'manifest.csv')) as f:
        for row in csv.DictReader(f):
            man[int(row['task'])] = (row['modifier'], float(row['stage']))

    results = {m: {} for m in MOD_ORDER}
    stages_seen = set()
    for task, (mod, s) in sorted(man.items()):
        pf = glob.glob(os.path.join(args.attr, f'attribution_{task}_data.pkl'))
        if not pf:
            print(f'  WARNING task {task} no pkl'); continue
        spkt, spkid, gids, dur = extract_spikes_and_gids(pf[0])
        results[mod][s] = metrics(spkt, spkid, gids, T0, dur)
        stages_seen.add(s)
        del spkt, spkid, gids; gc.collect()
        print(f'  task {task:2d} {MOD_LABEL[mod]:16s} s={s:.2f}  '
              f"gamma={results[mod][s]['gamma']:.3e} PYR={results[mod][s]['rate_HL23PYR']:.3f}")
    stages = sorted(stages_seen)

    # FIG 1 — gamma + excitability metrics vs CPS, per modifier
    panels = [('gamma', 'gamma 30-80 Hz power (a.u.)', 'Gamma power vs CPS'),
              ('rate_cv', 'pop-rate CV (temporal variability)', 'Rate variability vs CPS'),
              ('ei', 'E:I activity ratio', 'E/I (excitability) vs CPS'),
              ('rate_HL23PYR', 'PYR rate (Hz)', 'PYR excitability vs CPS')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, ylab, title) in zip(axes.ravel(), panels):
        for m in MOD_ORDER:
            if not results[m]: continue
            xs = [s for s in stages if s in results[m]]
            ys = [results[m][s][key] for s in xs]
            ax.plot(xs, ys, 'o-', color=MOD_COL[m], lw=1.8, label=MOD_LABEL[m])
        ax.set_title(title); ax.set_xlabel('CPS stage'); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle('Per-modifier gamma + excitability across CPS (spontaneous; no TMS)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(args.out, 'fig_gamma_excit_vs_cps.png'), dpi=150)
    plt.close(fig)

    # FIG 2 — gamma heatmap (modifier x stage, relative to s=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    mat = np.full((len(MOD_ORDER), len(stages)), np.nan)
    for i, m in enumerate(MOD_ORDER):
        for j, s in enumerate(stages):
            if s in results[m]: mat[i, j] = results[m][s]['gamma']
    rel = mat / np.where(mat[:, 0:1] == 0, np.nan, mat[:, 0:1])
    im = ax.imshow(rel, aspect='auto', cmap='RdBu_r', vmin=0, vmax=2)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels([f'{s:.2f}' for s in stages])
    ax.set_yticks(range(len(MOD_ORDER))); ax.set_yticklabels([MOD_LABEL[m] for m in MOD_ORDER])
    ax.set_xlabel('CPS stage')
    ax.set_title('Gamma power relative to s=0 per modifier\n(blue<1 reduced, red>1 increased)')
    plt.colorbar(im, ax=ax, label='gamma / gamma(s=0)')
    for i in range(len(MOD_ORDER)):
        for j in range(len(stages)):
            if not np.isnan(rel[i, j]):
                ax.text(j, i, f'{rel[i,j]:.2f}', ha='center', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_gamma_heatmap.png'), dpi=150)
    plt.close(fig)

    # CSV
    keys = ['gamma', 'rate_cv', 'ei'] + [f'rate_{p}' for p in POPS]
    with open(os.path.join(args.out, 'gamma_excit_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['modifier', 'stage'] + keys)
        for m in MOD_ORDER:
            for s in sorted(results[m]):
                w.writerow([MOD_LABEL[m], s] + [round(results[m][s][k], 6) for k in keys])

    print('\n=== s=1.0 gamma + excitability per modifier ===')
    print(f"{'modifier':16s} {'gamma':>11s} {'gamma/s0':>9s} {'rate_cv':>8s} {'E/I':>7s} {'PYR':>7s}")
    for m in MOD_ORDER:
        if 1.0 in results[m] and 0.0 in results[m]:
            r = results[m][1.0]; g0 = results[m][0.0]['gamma']
            ratio = r['gamma'] / g0 if g0 > 0 else float('nan')
            print(f"{MOD_LABEL[m]:16s} {r['gamma']:11.3e} {ratio:9.2f} "
                  f"{r['rate_cv']:8.3f} {r['ei']:7.3f} {r['rate_HL23PYR']:7.3f}")
    print(f"\nFigures + gamma_excit_summary.csv -> {args.out}")


if __name__ == '__main__':
    main()

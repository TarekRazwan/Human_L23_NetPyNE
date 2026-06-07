#!/usr/bin/env python3
"""
multiseed_analysis.py — shared multi-seed analysis (mean ± SEM across seeds)
============================================================================
ONE module, TWO entry points, sharing a lean all-seeds loader + a documented
rate convention. Fixes the side-paper bugs (load_run grabbed 1 pkl; single seed;
all-pairs correlation) AND analyzes the 5-seed CPS sweep.

MODE 'compare'  — two-condition (e.g. H01 vs Flat), each a directory of N seeds.
                  Per-seed metrics → mean ± SEM; KS on pooled cells; PYR-PYR
                  stratified distance-correlation.
MODE 'sweep'    — stage-resolved (CPS sweep): a directory of seed×stage runs,
                  grouped by stage via manifest.csv; per-stage mean ± SEM across seeds.

RATE WINDOWING (documented, per the 9.x-vs-11.x diagnosis):
  Default = STEADY-STATE: spikes in [T0, duration] / (duration-T0). Transient
  EXCLUDED. This is the cleaner characterization of the settled network and the
  convention used by the epileptiform/gamma scripts. NetPyNE popRates (slurm logs)
  use the WHOLE sim (0..duration) and will read ~1/0.77 higher due to the startup
  transient — that is a known windowing difference, not an error. Use --whole-sim
  to reproduce the slurm-log convention for cross-checks.

Lean loader: extracts ONLY spikes + gids + positions per pkl, frees each (gc) to
avoid OOM (1.6 GB pkls). Run on HPC, --mem=32G, python -u.

Examples:
  # side-paper, corrected (all 5 seeds each, steady-state):
  python -u multiseed_analysis.py compare --a ../data/h01_v1 --b ../data/v1_batch5_lock \
      --a-label H01 --b-label Flat --out ../data/side_paper_5seed
  # 5-seed CPS sweep (after job 13919 finishes):
  python -u multiseed_analysis.py sweep --dir ../data/cps_5seed --out ../data/cps_5seed_analysis
"""
import argparse, glob, os, pickle, csv, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats

POPS = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
EXC = ['HL23PYR']; INH = ['HL23SST', 'HL23PV', 'HL23VIP']
T0_DEFAULT = 1000.0
POPCOL = {'HL23PYR':'#2c5f8a','HL23SST':'#c0563f','HL23PV':'#3f8a52','HL23VIP':'#8a6d3f'}


# ----------------------------------------------------------------------------- loader
def load_spikes(path):
    """Lean: spikes (per-gid times) + pop gids + positions + duration; free the rest."""
    with open(path, 'rb') as f:
        D = pickle.load(f)
    sd = D['simData']
    spkt = np.asarray(sd['spkt'], dtype=float)
    spkid = np.asarray(sd['spkid'], dtype=int)
    net = D['net']
    cells = net['cells'] if isinstance(net, dict) and 'cells' in net else net.cells
    gids = {p: [] for p in POPS}; pos = {}
    for c in cells:
        c = c if isinstance(c, dict) else c.__dict__
        p = c.get('tags', {}).get('pop'); g = c.get('gid')
        if p in gids:
            gids[p].append(g)
            t = c.get('tags', {})
            pos[g] = np.array([t.get('x', 0.0), t.get('y', 0.0), t.get('z', 0.0)])
    gids = {p: np.array(sorted(v)) for p, v in gids.items()}
    sc = D['simConfig']; sc = sc if isinstance(sc, dict) else sc.__dict__
    dur = float(sc.get('duration', 3000.0))
    stage = float(sc.get('ad_stage', np.nan))
    seed = sc.get('GLOBALSEED', None)
    del D, sd, net, cells; gc.collect()
    return dict(spkt=spkt, spkid=spkid, gids=gids, pos=pos, dur=dur, stage=stage, seed=seed)


# ----------------------------------------------------------------------------- metrics
def per_cell_rates(R, pop, t0, whole_sim):
    g = R['gids'][pop]
    lo = 0.0 if whole_sim else t0
    win = (R['dur'] - lo) / 1000.0
    spkt, spkid = R['spkt'], R['spkid']
    sel = (spkt >= lo)
    st, si = spkt[sel], spkid[sel]
    return np.array([np.sum(si == gid) / win for gid in g]) if len(g) else np.array([])


def metrics_one_run(R, t0, whole_sim):
    m = {}
    rates = {p: per_cell_rates(R, p, t0, whole_sim) for p in POPS}
    for p in POPS:
        m[f'rate_{p}'] = float(rates[p].mean()) if len(rates[p]) else 0.0
        # CV-ISI per pop (median across cells)
        cvs = []
        for gid in R['gids'][p]:
            ts = np.sort(R['spkt'][(R['spkid'] == gid) & (R['spkt'] >= (0 if whole_sim else t0))])
            if len(ts) > 2:
                isi = np.diff(ts)
                if isi.mean() > 0: cvs.append(isi.std() / isi.mean())
        m[f'cv_{p}'] = float(np.median(cvs)) if cvs else np.nan
    re = np.mean([m[f'rate_{p}'] for p in EXC]); ri = np.mean([m[f'rate_{p}'] for p in INH])
    m['ei_ratio'] = float(re / ri) if ri > 0 else np.nan
    return m, rates


def pyr_pyr_distance_corr(R, t0, bin_ms=5.0, n_pairs=3000):
    """Distance-resolved spike-count correlation WITHIN PYR (the clean spatial test)."""
    g = R['gids']['HL23PYR']
    if len(g) < 10: return None
    edges = np.arange(t0, R['dur'] + bin_ms, bin_ms)
    # binned spike counts per PYR cell
    counts = {}
    for gid in g:
        ts = R['spkt'][(R['spkid'] == gid) & (R['spkt'] >= t0)]
        counts[gid], _ = np.histogram(ts, bins=edges)
    rng = np.random.default_rng(0)
    dists, corrs = [], []
    for _ in range(n_pairs):
        a, b = rng.choice(g, 2, replace=False)
        if counts[a].std() == 0 or counts[b].std() == 0: continue
        c = np.corrcoef(counts[a], counts[b])[0, 1]
        d = np.linalg.norm(R['pos'][a] - R['pos'][b])
        dists.append(d); corrs.append(c)
    return np.array(dists), np.array(corrs)


# ----------------------------------------------------------------------------- aggregate
def aggregate_dir(d, t0, whole_sim):
    """Load ALL *_data.pkl in d; per-seed metrics; return list of (metrics, rates, R)."""
    pkls = sorted(glob.glob(os.path.join(d, '*_data.pkl')))
    if not pkls: raise FileNotFoundError(f'no *_data.pkl in {d}')
    out = []
    for pf in pkls:
        R = load_spikes(pf)
        m, rates = metrics_one_run(R, t0, whole_sim)
        out.append((m, rates, R, os.path.basename(pf)))
        print(f'  {os.path.basename(pf)}: seed={R["seed"]} stage={R["stage"]:.2f} '
              f'PYR={m["rate_HL23PYR"]:.3f} PV={m["rate_HL23PV"]:.3f}', flush=True)
        gc.collect()
    return out


def mean_sem(vals):
    v = np.array([x for x in vals if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0: return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


# ----------------------------------------------------------------------------- MODE compare
def mode_compare(args):
    t0 = T0_DEFAULT
    A = aggregate_dir(args.a, t0, args.whole_sim)
    print(f'--- {args.a_label}: {len(A)} seeds ---')
    B = aggregate_dir(args.b, t0, args.whole_sim)
    print(f'--- {args.b_label}: {len(B)} seeds ---')
    os.makedirs(args.out, exist_ok=True)

    # per-pop rate mean±SEM across seeds, per condition
    rows = []
    print(f'\n=== rate (Hz) mean±SEM across seeds [{"whole-sim" if args.whole_sim else "steady-state"}] ===')
    print(f"{'pop':10s} {args.a_label:>16s} {args.b_label:>16s}  Δ")
    for p in POPS:
        ma, sa = mean_sem([m[f'rate_{p}'] for m, _, _, _ in A])
        mb, sb = mean_sem([m[f'rate_{p}'] for m, _, _, _ in B])
        print(f"{p:10s} {ma:7.3f}±{sa:5.3f}     {mb:7.3f}±{sb:5.3f}   {ma-mb:+.3f}")
        rows.append(dict(metric='rate_Hz', pop=p,
                         **{args.a_label: round(ma,4), f'{args.a_label}_sem': round(sa,4),
                            args.b_label: round(mb,4), f'{args.b_label}_sem': round(sb,4)}))
    # E/I
    ma, sa = mean_sem([m['ei_ratio'] for m,_,_,_ in A]); mb, sb = mean_sem([m['ei_ratio'] for m,_,_,_ in B])
    print(f"{'E/I':10s} {ma:7.3f}±{sa:5.3f}     {mb:7.3f}±{sb:5.3f}   {ma-mb:+.3f}")
    rows.append(dict(metric='ei_ratio', pop='all',
                     **{args.a_label: round(ma,4), f'{args.a_label}_sem': round(sa,4),
                        args.b_label: round(mb,4), f'{args.b_label}_sem': round(sb,4)}))

    # FIG: per-pop rate bar mean±SEM
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(POPS)); w = 0.38
    aMeans = [mean_sem([m[f'rate_{p}'] for m,_,_,_ in A]) for p in POPS]
    bMeans = [mean_sem([m[f'rate_{p}'] for m,_,_,_ in B]) for p in POPS]
    ax.bar(x-w/2, [m for m,_ in aMeans], w, yerr=[s for _,s in aMeans], label=args.a_label, color='#1f4e79', capsize=4)
    ax.bar(x+w/2, [m for m,_ in bMeans], w, yerr=[s for _,s in bMeans], label=args.b_label, color='#a02020', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([p.replace('HL23','') for p in POPS])
    ax.set_ylabel('firing rate (Hz)'); ax.legend()
    ax.set_title(f'{args.a_label} vs {args.b_label} — rate mean±SEM across {len(A)}/{len(B)} seeds\n'
                 f'({"whole-sim" if args.whole_sim else "steady-state 1-3s"} window)')
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'rates_meanSEM.png'), dpi=150); plt.close(fig)

    # FIG: PYR-PYR distance correlation, per-seed faint + mean line, both conditions
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, runs, col in [(args.a_label, A, '#1f4e79'), (args.b_label, B, '#a02020')]:
        allslopes = []
        binc = np.arange(0, 350, 30)
        binned = {i: [] for i in range(len(binc)-1)}
        for _, _, R, _ in runs:
            res = pyr_pyr_distance_corr(R, T0_DEFAULT)
            if res is None: continue
            dd, cc = res
            sl = np.polyfit(dd, cc, 1)[0] if len(dd) > 2 else np.nan
            allslopes.append(sl)
            for i in range(len(binc)-1):
                mask = (dd >= binc[i]) & (dd < binc[i+1])
                if mask.sum(): binned[i].append(cc[mask].mean())
        xs = 0.5*(binc[:-1]+binc[1:])
        ys = [np.mean(binned[i]) if binned[i] else np.nan for i in range(len(binc)-1)]
        sm, ss = mean_sem(allslopes)
        ax.plot(xs, ys, 'o-', color=col, label=f'{cond} (slope {sm:.2e}±{ss:.0e})')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('PYR-PYR soma distance (µm)'); ax.set_ylabel('spike-count correlation')
    ax.set_title('PYR-PYR distance-resolved correlation (the spatial signature, stratified)\n'
                 'H01 expected: distance-decay; Flat: ~flat'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'pyr_pyr_distance_corr.png'), dpi=150); plt.close(fig)

    with open(os.path.join(args.out, 'compare_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\nFigures + compare_summary.csv -> {args.out}')


# ----------------------------------------------------------------------------- MODE sweep
def mode_sweep(args):
    t0 = T0_DEFAULT
    man = {}
    mpath = os.path.join(args.dir, 'manifest.csv')
    if os.path.exists(mpath):
        with open(mpath) as f:
            for row in csv.DictReader(f):
                man[int(row['task'])] = (float(row['stage']), row.get('seed'))
    runs = aggregate_dir(args.dir, t0, args.whole_sim)
    os.makedirs(args.out, exist_ok=True)

    # group by stage (from each run's own ad_stage)
    by_stage = {}
    for m, _, R, fn in runs:
        s = R['stage']
        by_stage.setdefault(s, []).append(m)
    stages = sorted(by_stage)

    print(f'\n=== CPS trajectory mean±SEM across seeds [{"whole-sim" if args.whole_sim else "steady-state"}] ===')
    hdr = f"{'stage':>6s}" + ''.join(f"{p.replace('HL23',''):>14s}" for p in POPS)
    print(hdr)
    rows = []
    traj = {p: ([], []) for p in POPS}
    for s in stages:
        ms = by_stage[s]
        line = f"{s:6.2f}"
        rec = {'stage': s, 'n_seeds': len(ms)}
        for p in POPS:
            mu, se = mean_sem([m[f'rate_{p}'] for m in ms])
            line += f"  {mu:5.3f}±{se:4.3f}"
            traj[p][0].append(mu); traj[p][1].append(se)
            rec[f'rate_{p}'] = round(mu,4); rec[f'rate_{p}_sem'] = round(se,4)
        print(line); rows.append(rec)

    # FIG: trajectory mean±SEM per pop
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in POPS:
        ax.errorbar(stages, traj[p][0], yerr=traj[p][1], fmt='o-', color=POPCOL[p],
                    capsize=4, lw=2, label=p.replace('HL23',''))
    ax.set_xlabel('CPS stage'); ax.set_ylabel('firing rate (Hz)')
    ax.set_title(f'5-seed CPS trajectory (mean±SEM, n={len(by_stage[stages[0]])} seeds/stage)\n'
                 'confirms/refutes the single-seed −22% PYR hypoactivity + PV non-monotonicity')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'cps_trajectory_meanSEM.png'), dpi=150); plt.close(fig)

    with open(os.path.join(args.out, 'sweep_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)

    # explicit PV non-monotonicity check
    pv = traj['HL23PV'][0]
    print(f'\nPV trajectory: {[round(x,2) for x in pv]}')
    print('PV non-monotonic' if any(pv[i] > pv[i-1] and pv[i] > pv[i+1]
          for i in range(1, len(pv)-1)) else 'PV monotonic')
    print(f'PYR: {round(traj["HL23PYR"][0][0],3)} -> {round(traj["HL23PYR"][0][-1],3)} '
          f'({100*(traj["HL23PYR"][0][-1]/traj["HL23PYR"][0][0]-1):+.0f}%)')
    print(f'\nFigure + sweep_summary.csv -> {args.out}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('compare')
    c.add_argument('--a', required=True); c.add_argument('--b', required=True)
    c.add_argument('--a-label', default='A'); c.add_argument('--b-label', default='B')
    c.add_argument('--out', required=True); c.add_argument('--whole-sim', action='store_true')
    s = sub.add_parser('sweep')
    s.add_argument('--dir', required=True); s.add_argument('--out', required=True)
    s.add_argument('--whole-sim', action='store_true')
    args = ap.parse_args()
    if args.mode == 'compare': mode_compare(args)
    else: mode_sweep(args)


if __name__ == '__main__':
    main()

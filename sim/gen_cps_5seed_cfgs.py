#!/usr/bin/env python3
"""
gen_cps_5seed_cfgs.py — 5-seed × 5-stage combined CPS sweep config generator
=============================================================================
Generates 25 cfg JSONs = 5 seeds × 5 CPS stages, all 6 AD modifiers ON (the
combined sweep). Confirms whether the single-seed −22% PYR hypoactivity
trajectory + PV non-monotonicity hold across seeds (the robustness check that
unblocks locking today's disease-side findings).

IMPORTANT — this is a TEMPLATE that must be reconciled with your existing
gen_cps_cfgs.py before running. The single-seed sweep already worked, so the
AD-flag names + structure there are CORRECT. This script's job is ONLY to add
the seed loop. Verify the two MATCHing-flagged blocks below against your
working gen_cps_cfgs.py, fix any flag-name mismatch, THEN run.

Seed handling (CRITICAL): to actually re-randomize across seeds, BOTH
cfg.GLOBALSEED AND the four cfg.seeds streams (conn/stim/loc/cell) must change
per seed — otherwise connectivity/placement/stim won't differ. cfg.py line 47-48
shows: cfg.seeds = {'conn':1234,'stim':1234,'loc':1234,'cell':1234}; GLOBALSEED=1234.

Usage:
    python gen_cps_5seed_cfgs.py --out ../data/cps_5seed
    # then submit the array (run_cps_5seed_array.sh), 25 tasks
"""
import argparse, json, os, itertools

# ===================== MATCH-1: stages + seeds =====================
STAGES = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS  = [1234, 2345, 3456, 4567, 5678]   # seed 1234 = the existing single-seed run (reproduces it)

# ===================== MATCH-2: AD flags (VERIFY against gen_cps_cfgs.py) =====================
# These flag names MUST match what your working single-seed gen_cps_cfgs.py used.
# The combined sweep has ALL SIX modifiers ON. If your working script used different
# key names, fix them HERE before running. (Common names from this session:)
AD_FLAGS_ALL_ON = {
    'enable_M1a_sst_syn':   True,
    'enable_M1b_tonic':     True,
    'enable_M1c_sst_loss':  True,
    'enable_M2_pv_kv31':    True,
    'enable_M3_exc_scaffold': True,
    'enable_M4_pyr_loss':   True,
}
AD_STAGE_KEY = 'ad_stage'   # VERIFY: the key the model reads for CPS severity


def make_cfg(seed, stage, base_h01=True):
    """One cfg dict. Mirrors the single-seed sweep + per-seed seed streams.
    base_h01: H01 distance connectivity ON (the production model)."""
    cfg = {
        'GLOBALSEED': seed,
        'seeds': {'conn': seed, 'stim': seed, 'loc': seed, 'cell': seed},
        'USE_H01_DISTANCE_CONN': base_h01,
        AD_STAGE_KEY: stage,
    }
    cfg.update(AD_FLAGS_ALL_ON)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='../data/cps_5seed')
    ap.add_argument('--label', default='cps5')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = []
    task = 0
    for seed, stage in itertools.product(SEEDS, STAGES):
        cfg = make_cfg(seed, stage)
        cfg['simLabel'] = f'{args.label}_{task}'
        cfg['saveFolder'] = args.out
        path = os.path.join(args.out, f'{args.label}_{task}_cfg.json')
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
        manifest.append({'task': task, 'seed': seed, 'stage': stage})
        task += 1

    # manifest for the analysis step (task -> seed,stage)
    import csv
    with open(os.path.join(args.out, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'seed', 'stage'])
        w.writeheader()
        for row in manifest:
            w.writerow(row)

    print(f'Wrote {task} cfgs (5 seeds × 5 stages) to {args.out}')
    print(f'Seeds: {SEEDS}')
    print(f'Stages: {STAGES}')
    print(f'Manifest: {os.path.join(args.out, "manifest.csv")}')
    print('\nBEFORE RUNNING: diff the AD flag names + ad_stage key against your working')
    print('gen_cps_cfgs.py. If the single-seed sweep used different names, fix MATCH-2 above.')
    print('Sanity check: task 0 should be seed=1234 stage=0.0 (reproduces the s=0 gate).')


if __name__ == '__main__':
    main()

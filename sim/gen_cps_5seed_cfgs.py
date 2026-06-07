import json, os, csv, itertools
# Corrected 5-seed CPS sweep generator.
# Based EXACTLY on the working gen_cps_cfgs.py pattern: LOAD the template cfg
# (which has the full simConfig envelope + all fields), override only what changes,
# dump. The previous version wrote a bare flat dict from scratch -> NetPyNE could
# not find simConfig -> fell back to cfg.py defaults (every run seed 1234, stage 0).

STAGES = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS  = [1234, 2345, 3456, 4567, 5678]   # seed 1234 reproduces the existing single-seed sweep
TEMPLATE = '../data/v1_batch5_lock/v1_batch5_lock_0_cfg.json'   # same template the working script used
OUTDIR = '../data/cps_5seed'
LABEL = 'cps5'
os.makedirs(OUTDIR, exist_ok=True)

manifest = []
task = 0
for seed, s in itertools.product(SEEDS, STAGES):
    with open(TEMPLATE) as f:
        cfg = json.load(f)
    sc = cfg['simConfig']                      # <-- the envelope the bare-dict version was missing
    # --- seed (BOTH GLOBALSEED and the 4 streams, per seed) ---
    sc['GLOBALSEED'] = seed
    sc['seeds'] = {'conn': seed, 'stim': seed, 'loc': seed, 'cell': seed}
    # --- H01 + AD (identical to the working single-seed script) ---
    sc['USE_H01_DISTANCE_CONN'] = True
    sc['H01_CSV_PATH'] = '../data/cell_positions_h01_rotated_full.csv'
    sc['saveDataInclude'] = ['simData', 'simConfig', 'netParams', 'netCells']
    sc['ad_stage'] = s
    for k in ['enable_M1a_sst_syn', 'enable_M1b_tonic', 'enable_M1c_sst_loss',
              'enable_M2_pv_kv31', 'enable_M3_exc_scaffold', 'enable_M4_pyr_loss']:
        sc[k] = True
    # --- labels/paths ---
    sc['simLabel'] = f'{LABEL}_{task}'
    sc['saveFolder'] = OUTDIR
    with open(f'{OUTDIR}/{LABEL}_{task}_cfg.json', 'w') as f:
        json.dump(cfg, f, indent=4)          # dumps the FULL cfg (with simConfig envelope)
    manifest.append({'task': task, 'seed': seed, 'stage': s})
    print(f'  [{task}] seed={seed} ad_stage={s} -> {LABEL}_{task}_cfg.json')
    task += 1

with open(f'{OUTDIR}/manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task', 'seed', 'stage'])
    w.writeheader()
    for row in manifest:
        w.writerow(row)

print(f'\n{task} cfgs written to {OUTDIR} (5 seeds x 5 stages)')
print('VERIFY before running: head -3 of a cfg should show {"simConfig": {... and a NONZERO')
print('seed in task 5 (2345), task 10 (3456). Task 0 = seed 1234 stage 0.0 (reproduces s=0 gate).')

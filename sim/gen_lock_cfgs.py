#!/usr/bin/env python3
"""
gen_lock_cfgs.py  --  Generate per-seed SimConfig JSONs for the v1_batch5_lock array.

Run once from sim/ before submitting the SLURM array:
    python gen_lock_cfgs.py

Creates  ../data/v1_batch5_lock/v1_batch5_lock_{0..4}_cfg.json
using the verified v1_batch3 cfg as a template.
"""
import os
import json
import copy
import shutil

BATCH_LABEL = 'v1_batch5_lock'
SAVE_DIR    = os.path.join('..', 'data', BATCH_LABEL)
SEEDS       = [1234, 1235, 1236, 1237, 1238]

# --- Load known-good template from the last successful single-seed run ---
TEMPLATE_PATH = os.path.join('..', 'data', 'v1_batch3', 'v1_batch3_0_cfg.json')

if not os.path.isfile(TEMPLATE_PATH):
    raise FileNotFoundError(
        f'Template not found: {TEMPLATE_PATH}\n'
        'Copy v1_batch3_0_cfg.json to the cluster or regenerate with batch.py.'
    )

with open(TEMPLATE_PATH) as f:
    base = json.load(f)

os.makedirs(SAVE_DIR, exist_ok=True)

for i, seed in enumerate(SEEDS):
    cfg = copy.deepcopy(base)
    sc  = cfg['simConfig']

    # Override seed fields -- this is what batch.py's grid sweep would do
    sc['GLOBALSEED'] = seed
    sc['seeds']      = {'conn': seed, 'stim': seed, 'loc': seed, 'cell': seed}

    # Output identity
    sc['simLabel']   = f'{BATCH_LABEL}_{i}'
    sc['saveFolder'] = SAVE_DIR

    # Fix saveDataInclude: template had duplicate 'netParams' and no 'netCells'.
    # We need netCells so the pkl captures per-cell connectivity (the ~132k/~656k
    # convergence baseline for H01 comparison).
    sc['saveDataInclude'] = ['simData', 'simConfig', 'netParams', 'netCells']

    out_path = os.path.join(SAVE_DIR, f'{BATCH_LABEL}_{i}_cfg.json')
    with open(out_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f'  [{i}] GLOBALSEED={seed}  ->  {out_path}')

# Copy netParams.py alongside the cfgs (mirrors what batch.py does)
np_dst = os.path.join(SAVE_DIR, f'{BATCH_LABEL}_netParams.py')
shutil.copy2('netParams.py', np_dst)
print(f'  netParams  ->  {np_dst}')

print(f'\n{len(SEEDS)} cfg files written to {SAVE_DIR}/')
print('Submit with:  sbatch run_array.sh')

#!/usr/bin/env python3
"""
gen_h01_cfgs.py  --  Generate per-seed SimConfig JSONs for the H01 5-seed array.

Run once from sim/ before submitting the SLURM array:
    python gen_h01_cfgs.py

Creates  ../data/h01_v1/h01_v1_{0..4}_cfg.json
using the verified v1_batch3 cfg as a template, with H01 integration enabled.
"""
import os
import json
import copy
import shutil

BATCH_LABEL = 'h01_v1'
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

    # Override seed fields
    sc['GLOBALSEED'] = seed
    sc['seeds']      = {'conn': seed, 'stim': seed, 'loc': seed, 'cell': seed}

    # Output identity
    sc['simLabel']   = f'{BATCH_LABEL}_{i}'
    sc['saveFolder'] = SAVE_DIR

    # Save netCells for convergence verification
    sc['saveDataInclude'] = ['simData', 'simConfig', 'netParams', 'netCells']

    # H01 integration flags
    sc['USE_H01_DISTANCE_CONN'] = True
    sc['H01_CSV_PATH'] = '../data/cell_positions_h01_rotated_full.csv'

    out_path = os.path.join(SAVE_DIR, f'{BATCH_LABEL}_{i}_cfg.json')
    with open(out_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f'  [{i}] GLOBALSEED={seed}  ->  {out_path}')

# Copy netParams.py alongside the cfgs (mirrors what batch.py does)
np_dst = os.path.join(SAVE_DIR, f'{BATCH_LABEL}_netParams.py')
shutil.copy2('netParams.py', np_dst)
print(f'  netParams  ->  {np_dst}')

print(f'\n{len(SEEDS)} cfg files written to {SAVE_DIR}/')
print('Submit with:  sbatch run_h01_array.sh')

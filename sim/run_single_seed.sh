#!/bin/bash
# =============================================================================
# run_single_seed.sh — Single-seed sanity check for H01 integration
#
# Runs ONE seed (GLOBALSEED=1234) on a compute node to verify:
#   1. Expected convergence ~656 contacts/cell (printed by netParams.py)
#   2. dist_3D values on µm scale (printed by init.py diagnostic)
#   3. Population rates near locked baseline
#
# Submit from sim/:
#   sbatch run_single_seed.sh
# =============================================================================

#SBATCH --job-name=h01_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=../data/h01_sanity/slurm_%j.out
#SBATCH --error=../data/h01_sanity/slurm_%j.err

# ---------------------------------------------------------------------------
# Environment  (uncomment / adjust for your cluster)
# ---------------------------------------------------------------------------
# module load python/3.x
# module load neuron/8.x
# module load openmpi/4.x
# source /path/to/venv/bin/activate

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p ../data/h01_sanity

echo "========================================================"
echo "  H01 integration — single-seed sanity check"
echo "  Node       : $(hostname)"
echo "  Cores      : ${SLURM_NTASKS}"
echo "  Time       : $(date)"
echo "========================================================"

# Use default cfg.py settings (GLOBALSEED=1234, USE_H01_DISTANCE_CONN=True)
# Override simLabel and saveFolder for this sanity run
python3 -c "
import json, copy, os
template = '../data/v1_batch5_lock/v1_batch5_lock_0_cfg.json'
with open(template) as f:
    cfg = json.load(f)
sc = cfg['simConfig']
sc['GLOBALSEED'] = 1234
sc['seeds'] = {'conn': 1234, 'stim': 1234, 'loc': 1234, 'cell': 1234}
sc['simLabel'] = 'h01_sanity_0'
sc['saveFolder'] = '../data/h01_sanity'
sc['saveDataInclude'] = ['simData', 'simConfig', 'netParams', 'netCells']
sc['USE_H01_DISTANCE_CONN'] = True
sc['H01_CSV_PATH'] = '../data/cell_positions_h01_rotated_full.csv'
os.makedirs('../data/h01_sanity', exist_ok=True)
with open('../data/h01_sanity/h01_sanity_0_cfg.json', 'w') as f:
    json.dump(cfg, f, indent=4)
print('Generated h01_sanity_0_cfg.json')
"

mpiexec -n ${SLURM_NTASKS} nrniv -python -mpi init.py \
    simConfig=../data/h01_sanity/h01_sanity_0_cfg.json \
    netParams=netParams.py

echo ""
echo "=== Sanity check finished at $(date) ==="

#!/bin/bash
# =============================================================================
# run_h01_array.sh  --  SLURM job array for h01_v1 (5 seeds, 1 node each)
#
# Each array task gets its OWN node with 20 MPI ranks — no core contention.
# Pre-requisite:  python gen_h01_cfgs.py   (generates the per-seed cfg JSONs)
#
# Submit from sim/:
#   sbatch run_h01_array.sh
# =============================================================================

#SBATCH --job-name=h01_v1
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=../data/h01_v1/slurm_%A_%a.out
#SBATCH --error=../data/h01_v1/slurm_%A_%a.err

# ---------------------------------------------------------------------------
# Seed map:  array index  ->  GLOBALSEED
# ---------------------------------------------------------------------------
SEEDS=(1234 1235 1236 1237 1238)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BATCH_LABEL="h01_v1"
SAVE_DIR="../data/${BATCH_LABEL}"
CFG_JSON="${SAVE_DIR}/${BATCH_LABEL}_${SLURM_ARRAY_TASK_ID}_cfg.json"

# ---------------------------------------------------------------------------
# Environment  (uncomment / adjust for your cluster)
# ---------------------------------------------------------------------------
# module load python/3.x
# module load neuron/8.x
# module load openmpi/4.x
# source /path/to/venv/bin/activate

# ---------------------------------------------------------------------------
# Run from the directory where sbatch was invoked (should be sim/)
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "========================================================"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  GLOBALSEED : ${SEED}"
echo "  Node       : $(hostname)"
echo "  Cores      : ${SLURM_NTASKS}"
echo "  CFG        : ${CFG_JSON}"
echo "  Time       : $(date)"
echo "========================================================"

if [ ! -f "${CFG_JSON}" ]; then
    echo "ERROR: cfg JSON not found: ${CFG_JSON}"
    echo "Run  python gen_h01_cfgs.py  first."
    exit 1
fi

mpiexec -n ${SLURM_NTASKS} nrniv -python -mpi init.py \
    simConfig="${CFG_JSON}" \
    netParams=netParams.py

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} (seed ${SEED}) finished at $(date) ==="

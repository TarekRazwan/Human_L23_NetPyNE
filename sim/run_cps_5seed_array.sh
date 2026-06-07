#!/bin/bash
#SBATCH --job-name=cps5seed
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --array=0-24
#SBATCH --output=../data/cps_5seed/slurm_%A_%a.out
#SBATCH --error=../data/cps_5seed/slurm_%A_%a.err
# 5-seed × 5-stage combined CPS sweep = 25 tasks (array 0-24).
# Mirrors the single-seed run_cps_array.sh; only the array range changed (0-4 -> 0-24).
# VERIFY this matches your working run_cps_array.sh (mpiexec line, init.py invocation).

cd "$SLURM_SUBMIT_DIR"

# Matches working run_cps_array.sh exactly (no netParams= arg; uses SLURM_NTASKS):
mpiexec -n ${SLURM_NTASKS} nrniv -python -mpi init.py \
    simConfig=../data/cps_5seed/cps5_${SLURM_ARRAY_TASK_ID}_cfg.json

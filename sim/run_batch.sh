#!/bin/bash
#SBATCH --job-name=fsb_l23_5seed
#SBATCH --nodes=1
#SBATCH --ntasks=20                  # EDIT FOR YOUR CLUSTER: cores per seed run
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G                    # EDIT FOR YOUR CLUSTER: ~50G for 1000 cells with saveCellSecs=True
#SBATCH --time=06:00:00              # 5 seeds x ~40 min/seed sequential + margin
#SBATCH --output=/ddn/%u/logs/%x_%j.out   # EDIT FOR YOUR CLUSTER: log path
#SBATCH --error=/ddn/%u/logs/%x_%j.err    # EDIT FOR YOUR CLUSTER: log path
#SBATCH --export=ALL
# #SBATCH --partition=EDIT_PARTITION  # EDIT FOR YOUR CLUSTER: uncomment and set partition
# #SBATCH --account=EDIT_ALLOCATION   # EDIT FOR YOUR CLUSTER: uncomment and set allocation

# ---------------------------------------------------------------------------
# Environment setup — EDIT FOR YOUR CLUSTER
# ---------------------------------------------------------------------------
source ~/.bashrc

# Option A: conda environment
# conda activate EDIT_ENV_NAME

# Option B: module loads (comment out conda, uncomment these)
# module load neuron/8.2
# module load openmpi/4.1

# ---------------------------------------------------------------------------
# Working directory — EDIT FOR YOUR CLUSTER
# ---------------------------------------------------------------------------
mkdir -p /ddn/$USER/logs
cd /ddn/$USER/FSB_Human_L23_Netpyne/sim

# Compile mechanisms if not already done
if [ ! -d "arm64" ] && [ ! -d "x86_64" ]; then
    nrnivmodl mod
fi

ulimit -l unlimited

# ---------------------------------------------------------------------------
# Run 5-seed batch (sequential via NetPyNE batch manager)
# batch.py sweeps GLOBALSEED = [1234, 1235, 1236, 1237, 1238]
# Each seed launches as a separate mpiexec subprocess.
# Output: ../data/v1_batch5_lock/v1_batch5_lock_{0..4}_data.pkl
# ---------------------------------------------------------------------------
python batch.py

# ---------------------------------------------------------------------------
# Post-run: generate validation report
# ---------------------------------------------------------------------------
echo ""
echo "=== Running validation ==="
cd output
python validate_healthy.py --data-dir ../../data/v1_batch5_lock
echo "=== Done ==="

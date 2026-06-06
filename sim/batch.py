"""
batch.py

Batch simulation for L23Net NetPyNE replica (Yao et al. 2022)

Sweeps GLOBALSEED to produce genuinely different network realizations
(connectivity, cell placement, rotation, OU noise all vary per seed).
Single sweep param -> 5 runs, no Cartesian product.

Run mode: mpi_direct — all seeds run sequentially within a single job.
Cores are read from SGE's NSLOTS environment variable at runtime.

Contributors: salvadordura@gmail.com, fernandodasilvaborges@gmail.com
"""
import os
from netpyne.batch import Batch
from netpyne import specs
import numpy as np

# ----------------------------------------------------------------------------------------------
# Custom — 5-seed baseline lock
# ----------------------------------------------------------------------------------------------
def custom():
    params = specs.ODict()

    # Sweep GLOBALSEED: each value produces a distinct network realization.
    # cfg.seeds (conn/stim/loc/cell) and OU noise are all derived from
    # GLOBALSEED in init.py, so a single param controls all randomness.
    params['GLOBALSEED'] = [1234, 1235, 1236, 1237, 1238]

    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py')

    return b

# ----------------------------------------------------------------------------------------------
# Run configuration — mpi_direct (sequential seeds within one job)
# ----------------------------------------------------------------------------------------------
def setRunCfg(b):
    """Configure for mpi_direct execution.
    Cores are read from NSLOTS (SGE) at runtime, with fallback to 20."""
    b.runCfg = {
        'type': 'mpi_direct',
        'script': 'init.py',
        'mpiCommand': 'mpiexec',
        'cores': int(os.environ.get('NSLOTS', os.environ.get('SLURM_NTASKS', 20))),
        'skip': False,
    }

# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    b = custom()

    b.batchLabel = 'v1_batch5_lock'
    b.saveFolder = '../data/' + b.batchLabel
    b.method = 'grid'
    setRunCfg(b)
    b.run()

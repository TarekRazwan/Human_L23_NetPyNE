"""
batch.py 

Batch simulation for S1 model using NetPyNE

Contributors: salvadordura@gmail.com, fernandodasilvaborges@gmail.com
"""
from netpyne.batch import Batch
from netpyne import specs
import numpy as np

# ----------------------------------------------------------------------------------------------
# Custom
# ----------------------------------------------------------------------------------------------
def custom():
    params = specs.ODict()
    
    params[('seeds', 'stim')] =  [1234]    
    # params[('gExc')] = [1.0]

    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py')

    return b

# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_bulletin'):
    if type=='mpi_bulletin' or type=='mpi':
        b.runCfg = {'type': 'mpi_bulletin', 
            'script': 'init.py', 
            'skip': True}

    elif type=='mpi_direct':
        b.runCfg = {'type': 'mpi_direct',
            'cores': 20,
            'script': 'init.py',
            'mpiCommand': 'mpiexec', # --use-hwthread-cpus
            'skip': True}

    elif type == 'hpc_slurm_largeExpanse':
        b.runCfg = {'type': 'hpc_slurm',
                    'allocation': 'TG-IBN140002',
                    'partition': 'large-shared',
                    'walltime': '2:00:00',
                    'nodes': 1,
                    'coresPerNode': 128,
                    'email': 'fernandodasilvaborges@gmail.com',
                    'folder': '/home/fborges/Human_L23_NetPyNE/sim/',
                    'script': 'init.py',
                    'mpiCommand': 'mpirun',
                    'custom': '#SBATCH --mem=1024G\n#SBATCH --export=ALL\n#SBATCH --partition=large-shared',
                    'skip': True}
        
    elif type == 'hpc_slurm_Expanse':
        b.runCfg = {'type': 'hpc_slurm',
                    'allocation': 'TG-IBN140002',
                    'partition': 'compute',
                    'walltime': '2:00:00',
                    'nodes': 2,
                    'coresPerNode': 128,
                    'email': 'fernandodasilvaborges@gmail.com',
                    'folder': '/home/fborges/Human_L23_NetPyNE/sim/',
                    'script': 'init.py',
                    'mpiCommand': 'mpirun',
                    'custom': '#SBATCH --mem=240G\n#SBATCH --export=ALL\n#SBATCH --partition=compute',
                    'skip': True}
                
# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------
if __name__ == '__main__': 
    b = custom() #

    b.batchLabel = 'v1_batch3'  
    b.saveFolder = '../data/'+b.batchLabel
    b.method = 'grid'
    setRunCfg(b, 'mpi_direct')  #  setRunCfg(b, 'hpc_slurm_Expanse')
    b.run() # run batch

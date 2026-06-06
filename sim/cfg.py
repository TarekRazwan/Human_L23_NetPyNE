# =============================================================================
# cfg.py  —  SimConfig for L23Net NetPyNE Replica (Yao et al. 2022)
# =============================================================================
import sys
import pandas as pd
from netpyne import specs

cfg = specs.SimConfig()

# ---------------------------------------------------------------------------
# Identity / I/O
# ---------------------------------------------------------------------------
cfg.simLabel = 'v1_batch5_lock'
cfg.saveFolder = '../data/'+cfg.simLabel
cfg.savePickle  = True
cfg.saveJson    = False   # disabled for speed — spikes saved as .npy per seed

cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'netCells']
cfg.backupCfgFile = None 		##
cfg.gatherOnlySimData = False	##
cfg.saveCellSecs = True			## include morphology for LFP/EEG analysis
cfg.saveCellConns = True		##  

cfg.verbose = False
cfg.hParams = {'celsius': 34, 'v_init': -80}  
cfg.verbose = False
cfg.createNEURONObj = True
cfg.createPyStruct = True
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.printRunTime = 0.5

cfg.includeParamsLabel = False
cfg.printPopAvgRates = True
cfg.checkErrors = False

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------
cfg.duration = 3000.0    # ms
cfg.dt       = 0.025     # ms  (40 kHz)
cfg.transient = 0.0      # ms — analysis window starts here (0 = full run, matches NetPyNE popAvgRates)
cfg.testing   = False    # when True, analysis skips transient cutoff

# ---------------------------------------------------------------------------
# Random seeds
# GLOBALSEED is the single master seed swept by batch.py.
# All four NetPyNE seed channels are derived from it so that a single
# batch parameter controls connectivity, placement, rotation, and stimulus.
# OU background noise also keys off GLOBALSEED (via init.py).
# To later fix anatomy and vary only noise (e.g. TMS on a fixed H01
# substrate), override cfg.seeds['conn'/'loc'/'cell'] to constants
# after this block while still sweeping GLOBALSEED for the OU path.
# ---------------------------------------------------------------------------
cfg.GLOBALSEED = 1234
cfg.seeds = {
    'conn': cfg.GLOBALSEED,   # connectivity wiring
    'stim': cfg.GLOBALSEED,   # external stimulus (not yet implemented)
    'loc':  cfg.GLOBALSEED,   # cell spatial placement
    'cell': cfg.GLOBALSEED,   # cell rotation
}
cfg.DRUG = False

# ---------------------------------------------------------------------------
# H01 integration — real EM positions + distance-dependent connectivity
# Set USE_H01_DISTANCE_CONN = False to recover the flat-probability baseline.
# ---------------------------------------------------------------------------
cfg.USE_H01_DISTANCE_CONN = True
cfg.H01_CSV_PATH = '../data/cell_positions_h01_rotated_full.csv'

# ---------------------------------------------------------------------------
# LFP recording electrode (only used when cfg.rec_LFP = True)
# ---------------------------------------------------------------------------
# cfg.recordLFP = [[0.0, 0.0, 5.0]]

# ---------------------------------------------------------------------------
# Recording — one soma trace per population
# GID layout (full run): PYR 0-799, SST 800-849, PV 850-919, VIP 920-999
# ---------------------------------------------------------------------------


circuit_params = pd.read_excel('Circuit_param.xls', sheet_name = None, index_col = 0)
N_HL23PYR = int(circuit_params['SING_CELL_PARAM'].at['cell_num','HL23PYR'])
N_HL23SST = int(circuit_params['SING_CELL_PARAM'].at['cell_num','HL23SST'])
N_HL23PV = int(circuit_params['SING_CELL_PARAM'].at['cell_num','HL23PV'])
N_HL23VIP = int(circuit_params['SING_CELL_PARAM'].at['cell_num','HL23VIP'])

N_cells = N_HL23PYR + N_HL23SST + N_HL23PV + N_HL23VIP

cell_indices_to_plot = [0, 1, 
                        N_HL23PYR, N_HL23PYR + 1, 
                        N_HL23PYR + N_HL23SST, N_HL23PYR + N_HL23SST + 1, 
                        N_HL23PYR + N_HL23SST + N_HL23PV, N_HL23PYR + N_HL23SST + N_HL23PV + 1]
# cell_indices_to_plot = []
# for j in range(10):
#    cell_indices_to_plot = cell_indices_to_plot + [20 + j, 
#                         N_HL23PYR + j, 
#                         N_HL23PYR + N_HL23SST + j, 
#                         N_HL23PYR + N_HL23SST + N_HL23PV + j]
    
# cell_indices_to_plot = range(0,N_cells)

cfg.recordCells  = cell_indices_to_plot
cfg.recordTraces = {'V_soma': {'sec': 'soma_0', 'loc': 0.5, 'var': 'v'}}

cfg.cellNumber = {}
cfg.cellNumber['HL23PYR'] = N_HL23PYR
cfg.cellNumber['HL23SST'] = N_HL23SST
cfg.cellNumber['HL23PV'] = N_HL23PV
cfg.cellNumber['HL23VIP'] = N_HL23VIP

cfg.cellNumber0 = {}
cfg.cellNumber0['HL23PYR'] = 0
cfg.cellNumber0['HL23SST'] = N_HL23PYR
cfg.cellNumber0['HL23PV'] = N_HL23PYR + N_HL23SST
cfg.cellNumber0['HL23VIP'] = N_HL23PYR + N_HL23SST + N_HL23PV

# LFPy conn distribution comparation
cfg.gExc = 0.945 # The factor ~0.945 compensates for the main conn difference if cfg.LOAD_MATRIX_LFPy = False
cfg.LOAD_MATRIX_LFPy = False # need run LFPy in ../data/L23Net_LFPy
cfg.invertedYCoord = False
cfg.ROTATE_Y = True
cfg.Change_axon_names = True

#------------------------------------------------------------------------------
# Analysis and plotting 
# ------------------------------------------------------------------------------

Ipops = ['HL23SST', 'HL23PV', 'HL23VIP']
Epops = ['HL23PYR']

cfg.allpops = Epops + Ipops

cfg.recordStep = cfg.dt    # record every time step

cfg.analysis['plotRaster'] = {'include':  cfg.allpops, 'saveFig': True, 'showFig': False,'orderInverse': True, 'timeRange': [1000,cfg.duration], 'figSize': (12,6), 'fontSize':4, 'markerSize':4, 'marker': 'o', 'dpi': 300} 
cfg.analysis['plot2Dnet']   = {'include':  cfg.allpops, 'saveFig': True, 'showConns': False, 'figSize': (15,15), 'view': 'xy', 'fontSize':16}   # Plot 2D cells xy
cfg.analysis['plotTraces'] = {'include': cfg.recordCells, 'oneFigPer': 'trace', 'axis': False, 'overlay': False, 'timeRange': [1000,cfg.duration], 'saveFig': True, 'subtitles': None, 'legend': None, 'showFig': False, 'figSize':(18,18)}
# , 'ylim': [-85,40]
# cfg.analysis['plotTraces'] = {'oneFigPer': 'trace', 'overlay': True, 'timeRange': [0,cfg.duration], 'saveFig': True, 'showFig': False, 'figSize':(12,4)} # , 'ylim': [-90,30] Plot recorded traces for this list of cells
cfg.analysis['plotShape'] = {'includePre':  cfg.allpops,'includePost': cfg.allpops, #  [0, 200, 600, 800, 850, 920]
                             'includeAxon': False, 'showSyns': False, 'showElectrodes': False,
                                'cvar': 'voltage', 'dist': 0.65, 'elev': 95, 'azim':-90, 
                                'axisLabels':True, 'synStyle':'o', 
                                'clim': [-70, -40.], 'showFig': False, 'synSize': 2,                             
                                'saveFig': True, 'figSize':(24,24)}

cfg.analysis['plotConn'] = {'includePre': cfg.allpops, 'includePost': cfg.allpops, 'feature': 'numConns', 'groupBy': 'pop', 'figSize': (24,24), 
                            'saveFig': True, 'orderBy': 'gid', 'graphType': 'matrix', 'saveData': None, 'fontSize': 18}


#------------------------------------------------------------------------------
# General network parameters
#------------------------------------------------------------------------------
cfg.scale = 1.0 # reduce size
cfg.sizeY = 3300.0
cfg.sizeX = 500.0 # r = 250 um 
cfg.sizeZ = 500.0
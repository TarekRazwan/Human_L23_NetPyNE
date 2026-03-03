# =============================================================================
# cfg.py  —  SimConfig for L23Net NetPyNE Replica (Yao et al. 2022)
# =============================================================================
import sys
from netpyne import specs

cfg = specs.SimConfig()

# ---------------------------------------------------------------------------
# Identity / I/O
# ---------------------------------------------------------------------------
try:
    cfg.GLOBALSEED = int(sys.argv[1])
except (IndexError, ValueError):
    cfg.GLOBALSEED = 1234

cfg.saveFolder  = 'output'
cfg.simLabel    = f'L23Net_seed{cfg.GLOBALSEED}'
cfg.savePickle  = False
cfg.saveJson    = True   # disabled for speed — spikes saved as .npy per seed
cfg.saveDataInclude = ['simData'] 
cfg.verbose     = False

# ---------------------------------------------------------------------------
# Control flags  (mirror original circuit.py)
# ---------------------------------------------------------------------------
cfg.testing         = False   # False → full 1000-cell run, 4500 ms
cfg.no_connectivity = False   # disable all synaptic connections
cfg.DRUG            = False   # use drug_tonic values instead of norm_tonic
cfg.stimulate       = False   # add Poisson-based external stimulus
cfg.rec_LFP         = False   # record extracellular LFP at z=5 µm
cfg.rec_DIPOLES     = False   # record population dipole moments
cfg.silence_SST     = False   # zero all SST→* gmax (set by --silence-sst CLI flag)

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------
cfg.duration = 1000.0    # ms
cfg.dt       = 0.025     # ms  (40 kHz)
cfg.celsius  = 34.0
cfg.v_init   = -80.0

# ---------------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------------
cfg.seeds = {
    'conn':  cfg.GLOBALSEED,
    'stim':  cfg.GLOBALSEED + 1,
    'loc':   cfg.GLOBALSEED + 2,
    'cell':  cfg.GLOBALSEED + 3,
}

# ---------------------------------------------------------------------------
# Analysis / plotting
# ---------------------------------------------------------------------------
cfg.transient     = 20.0   # ms to discard in analysis (full run only)
cfg.analysis_tmin = 20.0  # ms to start analysis (full run only)
cfg.analysis_tmax = cfg.duration

cfg.recordStep = cfg.dt    # record every time step

# ---------------------------------------------------------------------------
# LFP recording electrode (only used when cfg.rec_LFP = True)
# ---------------------------------------------------------------------------
if cfg.rec_LFP:
    cfg.recordLFP = [[0.0, 0.0, 5.0]]

# ---------------------------------------------------------------------------
# Recording — one soma trace per population
# GID layout (full run): PYR 0-799, SST 800-849, PV 850-919, VIP 920-999
# ---------------------------------------------------------------------------
cfg.recordCells  = [0, 800, 850, 920]
cfg.recordTraces = {'V_soma': {'sec': 'soma_0', 'loc': 0.5, 'var': 'v'}}

#------------------------------------------------------------------------------
# Analysis and plotting 
# ------------------------------------------------------------------------------
# cfg.analysis['plotRaster'] = {'include': cfg.S1cells, 'saveFig': True, 'showFig': False,'orderInverse': True, 'timeRange': [0,cfg.duration], 'figSize': (24,8), 'fontSize':4, 'markerSize':4, 'marker': 'o', 'dpi': 300} 
cfg.analysis['plot2Dnet']   = {'saveFig': True, 'showConns': False, 'figSize': (24,24), 'view': 'xy', 'fontSize':16}   # Plot 2D cells xy
cfg.analysis['plotTraces'] = {'include': cfg.recordCells, 'oneFigPer': 'cell', 'overlay': True, 'timeRange': [0,cfg.duration], 'saveFig': True, 'showFig': False, 'figSize':(12,4)}

# cfg.analysis['plotTraces'] = {'oneFigPer': 'trace', 'overlay': True, 'timeRange': [0,cfg.duration], 'saveFig': True, 'showFig': False, 'figSize':(12,4)} # , 'ylim': [-90,30] Plot recorded traces for this list of cells
# cfg.analysis['plotShape'] = {'includePre':  [ii for ii in range(10)],'includePre':  [ii for ii in range(10)], 'saveFig': True, 'showFig': True, 'figSize':(12,12)}

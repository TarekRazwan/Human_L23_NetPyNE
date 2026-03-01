# =============================================================================
# params/circuit_params.py
# Static translation of Circuit_param.xls for the L23Net NetPyNE replica.
# Source: Yao et al. 2022 human L2/3 cortical microcircuit.
# All values read from Circuit_param.xls via pandas and hard-coded here
# so no runtime Excel dependency exists.
# Column/row order matches the original spreadsheet axes.
# =============================================================================

CELL_NAMES = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']

# ---------------------------------------------------------------------------
# Connection probability matrix  [pre][post]
# From sheet: conn_probs
# ---------------------------------------------------------------------------
CONN_PROBS = {
    'HL23PYR': {'HL23PYR': 0.150, 'HL23SST': 0.190, 'HL23PV': 0.090, 'HL23VIP': 0.090},
    'HL23SST': {'HL23PYR': 0.190, 'HL23SST': 0.040, 'HL23PV': 0.200, 'HL23VIP': 0.060},
    'HL23PV':  {'HL23PYR': 0.094, 'HL23SST': 0.050, 'HL23PV': 0.370, 'HL23VIP': 0.030},
    'HL23VIP': {'HL23PYR': 0.000, 'HL23SST': 0.350, 'HL23PV': 0.100, 'HL23VIP': 0.050},
}

# ---------------------------------------------------------------------------
# Peak synaptic conductance gmax (µS)  [pre][post]
# From sheet: syn_cond
# ---------------------------------------------------------------------------
SYN_COND = {
    'HL23PYR': {'HL23PYR': 0.000248, 'HL23SST': 0.000380, 'HL23PV': 0.000337, 'HL23VIP': 0.000310},
    'HL23SST': {'HL23PYR': 0.001240, 'HL23SST': 0.000340, 'HL23PV': 0.000330, 'HL23VIP': 0.000460},
    'HL23PV':  {'HL23PYR': 0.002910, 'HL23SST': 0.000330, 'HL23PV': 0.000330, 'HL23VIP': 0.000340},
    'HL23VIP': {'HL23PYR': 0.000000, 'HL23SST': 0.000360, 'HL23PV': 0.000340, 'HL23VIP': 0.000340},
}

# ---------------------------------------------------------------------------
# Number of synaptic contacts per connection (multapses)  [pre][post]
# From sheet: n_cont
# ---------------------------------------------------------------------------
N_CONT = {
    'HL23PYR': {'HL23PYR':  3, 'HL23SST':  8, 'HL23PV':  8, 'HL23VIP':  4},
    'HL23SST': {'HL23PYR': 12, 'HL23SST': 12, 'HL23PV': 13, 'HL23VIP':  5},
    'HL23PV':  {'HL23PYR': 17, 'HL23SST': 16, 'HL23PV': 15, 'HL23VIP':  7},
    'HL23VIP': {'HL23PYR':  0, 'HL23SST':  9, 'HL23PV': 11, 'HL23VIP':  7},
}

# ---------------------------------------------------------------------------
# Short-term depression time constant Dep (ms)  [pre][post]
# From sheet: Depression
# ---------------------------------------------------------------------------
DEPRESSION = {
    'HL23PYR': {'HL23PYR':  670, 'HL23SST':  140, 'HL23PV':  510, 'HL23VIP':  670},
    'HL23SST': {'HL23PYR': 1300, 'HL23SST':  720, 'HL23PV':  710, 'HL23VIP':  890},
    'HL23PV':  {'HL23PYR':  710, 'HL23SST':  700, 'HL23PV':  710, 'HL23VIP':  720},
    'HL23VIP': {'HL23PYR':  300, 'HL23SST':  760, 'HL23PV':  720, 'HL23VIP':  720},
}

# ---------------------------------------------------------------------------
# Short-term facilitation time constant Fac (ms)  [pre][post]
# From sheet: Facilitation
# ---------------------------------------------------------------------------
FACILITATION = {
    'HL23PYR': {'HL23PYR':  17, 'HL23SST': 670, 'HL23PV': 180, 'HL23VIP':  17},
    'HL23SST': {'HL23PYR':   2, 'HL23SST':  21, 'HL23PV':  21, 'HL23VIP':  25},
    'HL23PV':  {'HL23PYR':  23, 'HL23SST':  21, 'HL23PV':  21, 'HL23VIP':  21},
    'HL23VIP': {'HL23PYR': 160, 'HL23SST':  22, 'HL23PV':  21, 'HL23VIP':  21},
}

# ---------------------------------------------------------------------------
# Initial release probability Use  [pre][post]
# From sheet: Use
# ---------------------------------------------------------------------------
USE_PROB = {
    'HL23PYR': {'HL23PYR': 0.46, 'HL23SST': 0.09, 'HL23PV': 0.88, 'HL23VIP': 0.50},
    'HL23SST': {'HL23PYR': 0.30, 'HL23SST': 0.25, 'HL23PV': 0.25, 'HL23VIP': 0.31},
    'HL23PV':  {'HL23PYR': 0.08, 'HL23SST': 0.25, 'HL23PV': 0.25, 'HL23VIP': 0.26},
    'HL23VIP': {'HL23PYR': 0.23, 'HL23SST': 0.27, 'HL23PV': 0.25, 'HL23VIP': 0.26},
}

# ---------------------------------------------------------------------------
# Synaptic compartment target index  [pre][post]
# From sheet: Syn_pos
# Maps to syn_pos[] in original circuit.py:
#   0 → apic + dend (both, equal weight)
#   1 → apic only
#   2 → dend only
#   3 → dend only (halfnorm biased proximal — approximated as dend in NetPyNE)
# ---------------------------------------------------------------------------
SYN_POS = {
    'HL23PYR': {'HL23PYR': 0, 'HL23SST': 2, 'HL23PV': 2, 'HL23VIP': 2},
    'HL23SST': {'HL23PYR': 1, 'HL23SST': 2, 'HL23PV': 2, 'HL23VIP': 2},
    'HL23PV':  {'HL23PYR': 3, 'HL23SST': 2, 'HL23PV': 2, 'HL23VIP': 2},
    'HL23VIP': {'HL23PYR': 3, 'HL23SST': 2, 'HL23PV': 2, 'HL23VIP': 2},
}

# SYN_POS index → NetPyNE section target string(s)
# Index 0: both apic and dend  → handled with two connParams entries (see netParams.py)
# Index 1: apic only
# Index 2: dend only
# Index 3: dend only (halfnorm proximal — approximated; see REPLICATION_NOTES.md)
SYN_POS_SECTIONS = {
    0: ['apic', 'dend'],   # split across both compartments
    1: ['apic'],
    2: ['dend'],
    3: ['dend'],           # APPROX: halfnorm → uniform dend (see notes)
}

# ---------------------------------------------------------------------------
# Single-cell parameters  [param][cell_type]
# From sheet: SING_CELL_PARAM
# ---------------------------------------------------------------------------
SING_CELL_PARAM = {
    'HL23PYR': {
        'cell_num':        800,    # Full-run cell count
        'cell_num_test':    80,    # Test-run cell count (from circuit.py)
        'rotate_x':       1.57,   # radians
        'rotate_y':       2.62,
        'GOU':            0.000028,  # OU noise conductance scaling (µS)
        'norm_tonic':     0.000938,  # Somatic tonic GABA (S/cm²)
        'apic_tonic':     0.000938,  # Apical tonic GABA (S/cm²)
        'drug_tonic':     0.001352,
        'drug_apic_tonic':0.001352,
    },
    'HL23SST': {
        'cell_num':         50,
        'cell_num_test':     5,
        'rotate_x':        1.77,
        'rotate_y':        2.77,
        'GOU':             0.000030,
        'norm_tonic':      0.000938,
        'apic_tonic':      0.000938,
        'drug_tonic':      0.001030,
        'drug_apic_tonic': 0.001030,
    },
    'HL23PV': {
        'cell_num':         70,
        'cell_num_test':     7,
        'rotate_x':        1.26,
        'rotate_y':        2.57,
        'GOU':             0.000280,
        'norm_tonic':      0.000938,
        'apic_tonic':      0.000938,
        'drug_tonic':      0.001091,
        'drug_apic_tonic': 0.001091,
    },
    'HL23VIP': {
        'cell_num':         80,
        'cell_num_test':     8,
        'rotate_x':       -1.57,
        'rotate_y':        3.57,
        'GOU':             0.000066,
        'norm_tonic':      0.000938,
        'apic_tonic':      0.000938,
        'drug_tonic':      0.000938,
        'drug_apic_tonic': 0.000938,
    },
}

# ---------------------------------------------------------------------------
# Stimulus parameters
# From sheet: STIM_PARAM  (4 stimuli defined in original)
# Only used when cfg.stimulate = True
# ---------------------------------------------------------------------------
STIM_PARAMS = [
    {
        'cell_name':   'HL23PYR',
        'num_cells':   55,
        'start_index': 0,
        'num_stim':    1,
        'interval':    7.0,   # ms between spikes
        'start_time':  4000,  # ms
        'delay':       2.0,
        'delay_range': 2.0,
        'loc_num':     5,
        'loc':         'dend',
        'gmax':        0.0040,
        'stim_type':   'ProbAMPANMDA',
        'syn_params':  'HL23PYRHL23PYR',
    },
    {
        'cell_name':   'HL23PV',
        'num_cells':   35,
        'start_index': 0,
        'num_stim':    1,
        'interval':    10.0,
        'start_time':  4000,
        'delay':       2.0,
        'delay_range': 0.5,
        'loc_num':     8,
        'loc':         'dend',
        'gmax':        0.0020,
        'stim_type':   'ProbAMPANMDA',
        'syn_params':  'HL23PYRHL23PV',
    },
    {
        'cell_name':   'HL23VIP',
        'num_cells':   65,
        'start_index': 0,
        'num_stim':    1,
        'interval':    1.0,
        'start_time':  4000,
        'delay':       7.0,
        'delay_range': 5.0,
        'loc_num':     4,
        'loc':         'dend',
        'gmax':        0.0022,
        'stim_type':   'ProbAMPANMDA',
        'syn_params':  'HL23PYRHL23VIP',
    },
    {
        'cell_name':   'HL23VIP',
        'num_cells':   80,
        'start_index': 0,
        'num_stim':    1,
        'interval':    1.0,
        'start_time':  4000,
        'delay':       2.0,
        'delay_range': 0.5,
        'loc_num':     4,
        'loc':         'dend',
        'gmax':        0.0028,
        'stim_type':   'ProbAMPANMDA',
        'syn_params':  'HL23PYRHL23VIP',
    },
]

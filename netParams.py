# =============================================================================
# netParams.py  —  Network Parameters for L23Net NetPyNE Replica
# Yao et al. 2022 human Layer 2/3 cortical microcircuit
# =============================================================================
import os
import sys
import numpy as np
from netpyne import specs

# Ensure params package is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfg import cfg
from params.circuit_params import (
    CELL_NAMES, CONN_PROBS, SYN_COND, N_CONT, DEPRESSION,
    FACILITATION, USE_PROB, SYN_POS, SING_CELL_PARAM
)

netParams = specs.NetParams()

# ---------------------------------------------------------------------------
# Paths (relative to NetPyNE_Replica_Yao/ working directory)
# ---------------------------------------------------------------------------
MODELS_DIR = 'models'

# =============================================================================
# SECTION 1: Cell Parameters (cellParams)
#
# Each cell type has a standalone wrapper HOC template (HL23XXX_Cell.hoc) that:
#   1. Loads the cell-specific SWC morphology (path hardcoded in the template)
#   2. Runs geom_nseg()
#   3. Applies the correct axon policy (delete_axon or delete_axon_BPO)
#   4. Calls biophys() — all channel insertions are done inside init()
#
# A single importCellParams call per cell type is sufficient.
# importCellParams instantiates the template, then scans h.allsec() to capture
# all section geometry + mechanisms into the cellParams secs dict.
# =============================================================================

for cell_name in CELL_NAMES:
    wrapper_hoc   = os.path.join(MODELS_DIR, f'{cell_name}_Cell.hoc')
    template_name = f'{cell_name}_Cell'

    cellRule = netParams.importCellParams(
        label          = cell_name,
        fileName       = wrapper_hoc,
        cellName       = template_name,
        importSynMechs = False,
        somaAtOrigin   = False,
    )

    # Tag the rule so popParams can reference it by condition
    netParams.cellParams[cell_name]['conds'] = {
        'cellType':  cell_name,
        'cellModel': 'HH_full',
    }

    # Define named section lists for connParams compartment targeting.
    # These mirror the SWC section types used in the original HOC model.
    secs = netParams.cellParams[cell_name]['secs']
    apic_secs  = sorted([s for s in secs if s.startswith('apic')])
    dend_secs  = sorted([s for s in secs if s.startswith('dend')])
    soma_secs  = sorted([s for s in secs if s.startswith('soma')])
    axon_secs  = sorted([s for s in secs if s.startswith('axon') or s.startswith('myelin')])

    netParams.cellParams[cell_name]['secLists'] = {
        'all':     soma_secs + dend_secs + apic_secs + axon_secs,
        'somatic': soma_secs,
        'basal':   dend_secs,
        'apical':  apic_secs,
        'axonal':  axon_secs,
        # 'alldend' used internally; SYN_POS=0 is split into two rules below
    }

# =============================================================================
# SECTION 2: Synaptic Mechanism Parameters (synMechParams)
# Two main point-process synapses (with STP) + background noise + tonic GABA.
# STP parameters (Dep, Fac, Use, gmax) vary per connection and are overridden
# in connParams using a per-connection synMechParams list.
# =============================================================================

# Excitatory: ProbAMPANMDA (AMPA + NMDA dual-exponential with Fuhrmann STP)
netParams.synMechParams['Exc'] = {
    'mod':        'ProbAMPANMDA',
    'tau_r_AMPA': 0.3,
    'tau_d_AMPA': 3.0,
    'tau_r_NMDA': 2.0,
    'tau_d_NMDA': 65.0,
    'e':          0.0,
    'u0':         0.0,
    'Dep':        670.0,   # overridden per connection in connParams
    'Fac':        17.0,
    'Use':        0.46,
    'gmax':       0.000248,
}

# Inhibitory: ProbUDFsyn (GABA-A dual-exponential with Fuhrmann STP)
netParams.synMechParams['Inh'] = {
    'mod':   'ProbUDFsyn',
    'tau_r': 1.0,
    'tau_d': 10.0,
    'e':     -80.0,
    'u0':    0.0,
    'Dep':   710.0,   # overridden per connection in connParams
    'Fac':   23.0,
    'Use':   0.08,
    'gmax':  0.002910,
}

# Ornstein-Uhlenbeck background noise: Gfluct2
# Inserted per-section via custom function in init.py; listed here for reference.
netParams.synMechParams['Gfluct2'] = {
    'mod':   'Gfluct2',
    'E_e':   0.0,
    'E_i':   -80.0,
    'g_e0':  0.0,
    'g_i0':  0.0,
    'std_e': 0.0,
    'std_i': 0.0,
    'tau_e': 65.0,
    'tau_i': 20.0,
}

# Tonic GABA: tonic.mod  (steady-state shunting conductance)
# Inserted as distributed mechanism in init.py; listed here for reference.
netParams.synMechParams['TonicGABA'] = {
    'mod':    'tonic',
    'e_gaba': -75.0,
}

# =============================================================================
# SECTION 3: Population Parameters (popParams)
# L2/3 spatial geometry from original circuit.py:
#   upper = -250 µm, lower = -1200 µm  (z-axis, soma depth)
#   cylinder radius = 250 µm in x-y
# =============================================================================

L23_UPPER  = -250   # µm
L23_LOWER  = -1200  # µm
L23_RADIUS =  250   # µm

for cell_name in CELL_NAMES:
    p = SING_CELL_PARAM[cell_name]
    num_cells = p['cell_num_test'] if cfg.testing else p['cell_num']

    netParams.popParams[cell_name] = {
        'cellType':  cell_name,
        'cellModel': 'HH_full',
        'numCells':  num_cells,
        'xRange': [-L23_RADIUS, L23_RADIUS],
        'yRange': [-L23_RADIUS, L23_RADIUS],
        'zRange': [L23_LOWER,   L23_UPPER],
    }

# =============================================================================
# SECTION 4: Connectivity Parameters (connParams)
# One rule per (pre, post) pair = up to 16 rules (skip zero-prob pairs).
#
# Synaptic target section determined by SYN_POS index:
#   0 → apic + dend  (split into two sub-rules, each at prob * 0.5)
#   1 → apical
#   2 → basal (dend)
#   3 → basal (dend, proximal-biased in original; approximated as uniform here)
#
# Per-connection STP parameters override the base synMechParams defaults.
# =============================================================================

for pre in CELL_NAMES:
    for post in CELL_NAMES:
        conn_prob = CONN_PROBS[pre][post]
        if conn_prob == 0.0:
            continue   # VIP→PYR probability is 0; skip

        # Synapse type
        is_exc = ('PYR' in pre)
        synmech = 'Exc' if is_exc else 'Inh'

        gmax = SYN_COND[pre][post]

        # Per-connection STP overrides
        stp_params = {
            'Dep':  DEPRESSION[pre][post],
            'Fac':  FACILITATION[pre][post],
            'Use':  USE_PROB[pre][post],
            'gmax': gmax,
        }

        # Number of multaptic contacts
        n_contacts = max(1, int(N_CONT[pre][post]))

        # Target compartment
        syn_pos_idx = SYN_POS[pre][post]
        rule_base   = f'{pre}->{post}'

        if syn_pos_idx == 0:
            # SYN_POS 0: target both apic AND dend — split into two equal rules.
            # Each sub-rule gets half the probability and half the contacts.
            # (See REPLICATION_NOTES.md §6 for the integer-rounding note.)
            contacts_each = max(1, n_contacts // 2)
            for sec_name, suffix in [('apical', '_apic'), ('basal', '_dend')]:
                netParams.connParams[rule_base + suffix] = {
                    'preConds':      {'pop': pre},
                    'postConds':     {'pop': post},
                    'probability':   conn_prob * 0.5,
                    'synMech':       synmech,
                    'synMechParams': stp_params,
                    'weight':        1.0,
                    'delay':         0.5,
                    'synsPerConn':   contacts_each,
                    'sec':           sec_name,
                    'loc':           0.5,
                }
        else:
            # SYN_POS 1 → apical; SYN_POS 2/3 → basal
            if syn_pos_idx == 1:
                sec_name = 'apical'
            else:
                sec_name = 'basal'

            # SYN_POS 3 is halfnorm-proximal in original; approximated as uniform.
            # See REPLICATION_NOTES.md §5.
            netParams.connParams[rule_base] = {
                'preConds':      {'pop': pre},
                'postConds':     {'pop': post},
                'probability':   conn_prob,
                'synMech':       synmech,
                'synMechParams': stp_params,
                'weight':        1.0,
                'delay':         0.5,
                'synsPerConn':   n_contacts,
                'sec':           sec_name,
                'loc':           0.5,
            }

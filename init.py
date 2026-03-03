"""
init.py  ## see https://github.com/suny-downstate-medical-center/thalamus_netpyne/blob/main/sim/init.py for OU noise implementation 

Starting script to run NetPyNE-based 

Usage:
    python init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 8 nrniv -python -mpi init.py

"""

# import matplotlib; matplotlib.use('agg')  # to avoid graphics error in servers

from netpyne import sim
from neuron import h
import numpy as np

from params.circuit_params import CELL_NAMES, SING_CELL_PARAM


cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')
# sim.create(netParams, cfg)
# sim.createSimulateAnalyze(netParams, cfg)

sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations

# Also load net_functions.hoc so HOC helpers are available at cell-build time.
h.load_file('net_functions.hoc')

_OU_REFS = []   # keeps HocObjects alive for entire simulation lifetime


def _neuron_distance_setup(hobj):
    hobj.push()
    h.distance(0, 0.0)
    h.pop_section()


def _neuron_distance(hobj, x):
    hobj.push()
    d = h.distance(x)
    h.pop_section()
    return d


def _get_longest_branch(hobjs):
    max_L = 0.0
    for hobj in hobjs:
        sref = h.SectionRef(sec=hobj)
        if sref.nchild() == 0:
            d = _neuron_distance(hobj, 1)
            if d > max_L:
                max_L = d
    if max_L == 0.0 and hobjs:
        max_L = _neuron_distance(hobjs[0], 1)
    return max_L


def _locate_sites(hobjs, site):
    hits = []
    for hobj in hobjs:
        d0 = _neuron_distance(hobj, 0)
        d1 = _neuron_distance(hobj, 1)
        if d0 > d1:
            d0, d1 = d1, d0
        if d1 <= d0:
            continue
        if d0 <= site <= d1:
            x = (site - d0) / (d1 - d0)
            x = max(0.01, min(0.99, x))
            hits.append((hobj, x))
    return hits


def insert_ou_noise(sim_obj, cfg_obj, sing_cell_param):
    """Insert Ornstein-Uhlenbeck background noise into each cell."""
    np.random.seed(cfg_obj.GLOBALSEED)
    n_total      = 0
    type_basal   = {}   # {cell_type: total basal Gfluct2}
    type_apical  = {}   # {cell_type: total apical Gfluct2}
    type_n_cells = {}   # {cell_type: cell count}

    for cell in sim_obj.net.cells:
        cell_type = cell.tags.get('cellType', '')
        gou       = sing_cell_param[cell_type]['GOU']
        base_seed = cfg_obj.GLOBALSEED * (cell.gid + 1)
        idx       = 0
        n_basal   = 0
        n_apical  = 0

        def _place_ou(hobj, x_pos, relpos_for_g, seed_offset):
            g_val = gou * np.exp(relpos_for_g)
            ou = h.Gfluct2(x_pos, sec=hobj)
            ou.E_e   = 0.0;    ou.E_i   = -80.0
            ou.g_e0  = g_val;  ou.g_i0  = 0.0
            ou.std_e = g_val;  ou.std_i = 0.0
            ou.tau_e = 65.0;   ou.tau_i = 20.0
            rng = h.Random(base_seed * 1000 + seed_offset)
            rng.normal(0, 1)
            ou.noiseFromRandom(rng)
            _OU_REFS.append((ou, rng))

        # ----------------------------------------------------------------
        # Basal: locateSites("dend", 0.5 * getLongestBranch)
        # FIX: relpos_for_g = 0.5 (CONSTANT), matching HOC line 50.
        #      Do NOT pass x (section-local position) — that was a bug.
        # ----------------------------------------------------------------
        dend_secs = [s for s in cell.secs if s.startswith('dend')]
        if dend_secs:
            dend_hobjs = [cell.secs[s]['hObj'] for s in dend_secs]
            _neuron_distance_setup(dend_hobjs[0])
            max_L = _get_longest_branch(dend_hobjs)
            for hobj, x in _locate_sites(dend_hobjs, 0.5 * max_L):
                _place_ou(hobj, x, 0.5, idx + 5)   # relpos_for_g = 0.5 FIXED
                idx += 1
                n_basal += 1

        # ----------------------------------------------------------------
        # Apical (PYR only): 5 proportional distances, widest-diam section
        # relpos_for_g = relpos (varies 0.1→0.9) — correct, matches HOC
        # ----------------------------------------------------------------
        if 'PYR' in cell_type:
            apic_secs = [s for s in cell.secs if s.startswith('apic')]
            if apic_secs:
                apic_hobjs = [cell.secs[s]['hObj'] for s in apic_secs]
                _neuron_distance_setup(apic_hobjs[0])
                max_L_apic = _get_longest_branch(apic_hobjs)
                for relpos in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    hits = _locate_sites(apic_hobjs, relpos * max_L_apic)
                    if not hits:
                        hits = [(apic_hobjs[0], min(relpos, 0.9))]
                    best_hobj, best_x = max(hits, key=lambda t: t[0](t[1]).diam)
                    _place_ou(best_hobj, best_x, relpos, idx)
                    idx += 1
                    n_apical += 1

        n_total += idx
        type_basal[cell_type]   = type_basal.get(cell_type, 0)   + n_basal
        type_apical[cell_type]  = type_apical.get(cell_type, 0)  + n_apical
        type_n_cells[cell_type] = type_n_cells.get(cell_type, 0) + 1

    print(f'[init] Inserted OU noise into {len(sim_obj.net.cells)} cells '
          f'({n_total} Gfluct2 total):')
    for _ct in sorted(type_basal):
        _nc = type_n_cells[_ct]
        _nb = type_basal[_ct]   // _nc
        _na = type_apical.get(_ct, 0) // _nc
        _g_basal = sing_cell_param[_ct]['GOU'] * np.exp(0.5) * 1e6
        print(f'  [OU] {_ct}: {_nb} basal/cell, {_na} apical/cell  '
              f'(g_e0_basal={_g_basal:.1f} pS = GOU*exp(0.5))')


# ---------------------------------------------------------------------------
# 3.  Helper: insert tonic GABA inhibition per cell
# ---------------------------------------------------------------------------
def insert_tonic_gaba(sim_obj, cfg_obj, sing_cell_param):
    """Insert tonic GABA (tonic.mod) into soma/basal of all cells,
    and into apical of PYR cells."""
    for cell in sim_obj.net.cells:
        cell_type = cell.tags.get('cellType', '')
        p = sing_cell_param[cell_type]

        if cfg_obj.DRUG:
            g_soma = p['drug_tonic']
            g_apic = p['drug_apic_tonic']
        else:
            g_soma = p['norm_tonic']
            g_apic = p['apic_tonic']

        for sname in [s for s in cell.secs if s.startswith('soma')]:
            sec = cell.secs[sname]['hObj']
            sec.insert('tonic')
            for seg in sec:
                seg.tonic.g      = g_soma
                seg.tonic.e_gaba = -75.0

        for sname in [s for s in cell.secs if s.startswith('dend')]:
            sec = cell.secs[sname]['hObj']
            sec.insert('tonic')
            for seg in sec:
                seg.tonic.g      = g_soma
                seg.tonic.e_gaba = -75.0

        if 'PYR' in cell_type:
            for sname in [s for s in cell.secs if s.startswith('apic')]:
                sec = cell.secs[sname]['hObj']
                sec.insert('tonic')
                for seg in sec:
                    seg.tonic.g      = g_apic
                    seg.tonic.e_gaba = -75.0

    print(f'[init] Inserted tonic GABA into {len(sim_obj.net.cells)} cells.')

# ---------------------------------------------------------------------------
# 5.  Build and run simulation
# ---------------------------------------------------------------------------


print('[init] Inserting background noise (Gfluct2) ...')
insert_ou_noise(sim, cfg, SING_CELL_PARAM)

print('[init] Inserting tonic GABA inhibition ...')
insert_tonic_gaba(sim, cfg, SING_CELL_PARAM)

# print('[init] Creating connections ...')
sim.net.connectCells()

# print('[init] Adding external stimuli ...')
sim.net.addStims()

print('[init] Setting up recording ...')
sim.setupRecording()

print('[init] Running simulation ...')
sim.runSim()

print('[init] Gathering data ...')
sim.gatherData()

print('[init] Saving data ...')
sim.saveData()

print('[init] Plotting data ...')
sim.analysis.plotData()           		# plot spikes, V traces, rasters, etc.
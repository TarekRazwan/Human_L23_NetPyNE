#!/usr/bin/env python3
# =============================================================================
# init.py  —  Entry point for L23Net NetPyNE Replica (Yao et al. 2022)
#
# Usage:
#   python init.py [SEED] [--silence-sst]
#
# Examples:
#   python init.py 1234               # healthy, seed 1234
#   python init.py 1234 --silence-sst # SST-silenced, seed 1234
#   python init.py                    # healthy, default seed 1234
#
# MOD files must be compiled before running:
#   cd mod && nrnivmodl && cd ..
# =============================================================================
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# 0.  Compile MOD files if needed
# ---------------------------------------------------------------------------
_mod_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mod')
_special = os.path.join(_mod_dir, 'x86_64', 'special')

if not os.path.isfile(_special):
    print('[init] Compiling MOD files ...')
    ret = os.system(f'cd "{_mod_dir}" && nrnivmodl')
    if ret != 0:
        raise RuntimeError('nrnivmodl failed — check mod/ directory')
    print('[init] Compilation done.')
else:
    print('[init] MOD files already compiled.')

# ---------------------------------------------------------------------------
# 1.  NetPyNE / NEURON imports (AFTER mod compilation)
# ---------------------------------------------------------------------------
import neuron
from netpyne import sim
from neuron import h

# NOTE: MPI via `mpirun` does NOT work on macOS with Open MPI 5 —
# MPI_Init_thread crashes on the shmem transport (PML add procs, -13).
# NOTE: NEURON thread parallelism (pc.nthread) cannot be used because
# Gfluct2 noiseFromRandom() uses non-thread-safe Random objects.
# Simulation runs single-threaded.

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

neuron.load_mechanisms(_mod_dir)

from cfg import cfg

# ---------------------------------------------------------------------------
# CLI overrides — BEFORE netParams imports cfg at module load.
#   python init.py [SEED] [--silence-sst] [--full]
# ---------------------------------------------------------------------------
import argparse as _ap
_cli = _ap.ArgumentParser(add_help=False)
_cli.add_argument('--silence-sst', action='store_true',
                  help='Zero all SST→* gmax (SST silencing condition)')
_cli.add_argument('--full', action='store_true',
                  help='Force full run (testing=False)')
_args, _ = _cli.parse_known_args()

if _args.silence_sst:
    cfg.silence_SST = True
    cfg.simLabel    = f'L23Net_seed{cfg.GLOBALSEED}_SSTsilenced'

if _args.full:
    cfg.testing       = False
    cfg.duration      = 4500.0
    cfg.analysis_tmin = 2000.0
    cfg.analysis_tmax = cfg.duration

print(f'[init] Seed={cfg.GLOBALSEED}  silence_SST={cfg.silence_SST}  '
      f'duration={cfg.duration} ms  label={cfg.simLabel}')

from netParams import netParams
from params.circuit_params import CELL_NAMES, SING_CELL_PARAM

# SST silencing: zero all SST→* connection gmax post-import
if cfg.silence_SST:
    silenced = []
    for key in list(netParams.connParams.keys()):
        if key.startswith('HL23SST'):
            netParams.connParams[key]['synMechParams']['gmax'] = 0.0
            silenced.append(key)
    print(f'[init] SST silenced: zeroed gmax for {len(silenced)} rule(s): '
          f'{silenced}')

# Also load net_functions.hoc so HOC helpers are available at cell-build time.
_net_func_hoc = os.path.join(_here, 'net_functions.hoc')
h.load_file(_net_func_hoc)

# ---------------------------------------------------------------------------
# 2.  Helper: insert Gfluct2 OU noise per cell
#
# Mirrors net_functions.hoc::createArtificialSyn() exactly:
#
#   Basal:  locateSites("dend", 0.5 * getLongestBranch("dend"))
#           → one Gfluct2 per dend section whose path-distance range spans
#             50% of the longest basal branch.
#           g_e0 = GOU * exp(0.5)  [relpos=0.5, FIXED — matches HOC line 50]
#
#   Apical (PYR only): 5 distances (0.1,0.3,0.5,0.7,0.9 × maxL_apic).
#           At each distance, pick the widest-diameter section spanning that
#           point.  g_e0 = GOU * exp(relpos)  [relpos varies 0.1→0.9]
#
# Inhibitory OU component is zero (g_i0=0, std_i=0) — matches original.
# ---------------------------------------------------------------------------
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
# 4.  Create output directory
# ---------------------------------------------------------------------------
out_dir = os.path.join(_here, cfg.saveFolder)
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 5.  Build and run simulation
# ---------------------------------------------------------------------------
import datetime
print(f'[init] Start time: {datetime.datetime.now()}')

print('[init] Initializing simulation ...')
sim.initialize(netParams=netParams, simConfig=cfg)

print('[init] Creating populations ...')
sim.net.createPops()

print('[init] Creating cells ...')
sim.net.createCells()

print('[init] Inserting background noise (Gfluct2) ...')
insert_ou_noise(sim, cfg, SING_CELL_PARAM)

print('[init] Inserting tonic GABA inhibition ...')
insert_tonic_gaba(sim, cfg, SING_CELL_PARAM)

print('[init] Creating connections ...')
sim.net.connectCells()

print('[init] Adding external stimuli ...')
sim.net.addStims()

print('[init] Setting up recording ...')
sim.setupRecording()

# ---------------------------------------------------------------------------
# Direct NEURON vector recording (bypasses broken cfg.recordCells in v1.0.6)
# Records soma Vm from first cell of each population.
# ---------------------------------------------------------------------------
_trace_gids = {'V_PYR': 0, 'V_SST': 800, 'V_PV': 850, 'V_VIP': 920}
_trace_vecs = {}
_trace_tvec = h.Vector()
_trace_tvec.record(h._ref_t, cfg.recordStep)
for _lbl, _gid in _trace_gids.items():
    _cell_objs = [c for c in sim.net.cells if c.gid == _gid]
    if _cell_objs:
        _hobj = _cell_objs[0].secs['soma']['hObj']
        _v = h.Vector()
        _v.record(_hobj(0.5)._ref_v, cfg.recordStep)
        _trace_vecs[_lbl] = _v
        print(f'[init] Direct recording: {_lbl} from cell GID {_gid}')
    else:
        print(f'[init] WARNING: cell GID {_gid} not found for {_lbl}')

print('[init] Running simulation ...')
sim.runSim()

print('[init] Gathering data ...')
sim.gatherData()

print('[init] Saving data ...')
sim.saveData()

# ---------------------------------------------------------------------------
# 6.  Post-sim: save .npy spike file + rates + traces (rank 0 only)
# ---------------------------------------------------------------------------
if sim.rank == 0:
    # --- Build popGids map ---
    popGids = {name: sim.net.pops[name].cellGids for name in CELL_NAMES}
    sim.allSimData['popGids'] = popGids

    # --- Spike arrays ---
    spkt  = np.array(sim.allSimData.get('spkt',  []))
    spkid = np.array(sim.allSimData.get('spkid', []))

    spike_label = (f'{cfg.GLOBALSEED}_SSTsilenced' if cfg.silence_SST
                   else str(cfg.GLOBALSEED))
    spk_file = os.path.join(out_dir, f'spikes_seed{spike_label}.npy')
    np.save(spk_file, {'spkt': spkt, 'spkid': spkid, 'popGids': popGids,
                       'duration': cfg.duration, 'transient': cfg.transient,
                       'silence_SST': cfg.silence_SST,
                       'GLOBALSEED': cfg.GLOBALSEED})
    print(f'[init] Spike data saved → {spk_file}  '
          f'({len(spkt)} spikes)')

    # --- Voltage traces (from direct h.Vector recording) ---
    if _trace_vecs:
        trace_dict = {'t': np.array(_trace_tvec.to_python())}
        for _lbl, _v in _trace_vecs.items():
            trace_dict[_lbl] = np.array(_v.to_python())
        trace_file = os.path.join(out_dir, f'traces_seed{spike_label}.npy')
        np.save(trace_file, trace_dict)
        print(f'[init] Voltage traces saved → {trace_file}  '
              f'(keys: {list(trace_dict.keys())})')
    else:
        print('[init] No voltage traces recorded (no cells found).')

    # --- Firing rates ---
    from analysis import compute_firing_rates, plot_raster
    rates = compute_firing_rates(sim.allSimData, cfg, CELL_NAMES)

    rate_file = os.path.join(out_dir, f'rates_seed{spike_label}.txt')
    with open(rate_file, 'w') as _f:
        for pop, hz in rates.items():
            _f.write(f'{pop}: {hz:.4f} Hz\n')

    print(f'\n--- Mean Firing Rates (seed={cfg.GLOBALSEED}) ---')
    targets = {'HL23PYR': 1.2, 'HL23SST': 5.6, 'HL23PV': 10.2, 'HL23VIP': 3.5}
    for name, hz in rates.items():
        tgt = targets.get(name, 0)
        pct = (hz - tgt) / tgt * 100 if tgt else 0
        print(f'  {name:10s}: {hz:.2f} Hz  (target {tgt:.1f} Hz, '
              f'{pct:+.0f}%)')

    plot_raster(sim.allSimData, cfg, CELL_NAMES, out_dir,
                seed=spike_label)

print(f'[init] Done.  End time: {datetime.datetime.now()}')

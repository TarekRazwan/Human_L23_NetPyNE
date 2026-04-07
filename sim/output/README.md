# L23Net — NetPyNE Replica (Yao et al. 2022)

NetPyNE translation of the human Layer 2/3 cortical microcircuit model originally
implemented in LFPy + NEURON by Yao et al. 2022.

The original LFPy implementation lives in the parent directory (`../`).
This replica lives in `NetPyNE_Replica_Yao/` and does **not** modify any original files.

---

## Quick Start

### 1. Install dependencies

```bash
pip install netpyne neuron mpi4py numpy scipy matplotlib pandas
```

### 2. Compile NMODL mechanisms

```bash
cd NetPyNE_Replica_Yao/mod
nrnivmodl
cd ..
```

Verify that `mod/x86_64/special` was created.

### 3. Run test simulation (100 cells, 1 second)

```bash
cd NetPyNE_Replica_Yao
python init.py
```

With MPI (N parallel processes):

```bash
mpirun -np 4 python init.py
```

Outputs are written to `output/`.

### 4. Run full simulation (1000 cells, 4.5 seconds)

Edit `cfg.py`:

```python
cfg.testing = False
```

Then run as above (MPI recommended for full run).

---

## Directory Structure

```
NetPyNE_Replica_Yao/
├── init.py               # Entry point: build → OU noise → tonic GABA → run
├── cfg.py                # SimConfig (all control flags here)
├── netParams.py          # cellParams, synMechParams, popParams, connParams
├── analysis.py           # Firing rates, raster, LFP plots
├── net_functions.hoc     # HOC: createArtificialSyn + addTonicInhibition (ref copy)
├── Circuit_param.xls     # Original Excel parameter file (reference)
├── models/               # HOC biophysics + NeuronTemplate (unmodified copies)
├── morphologies/         # SWC morphology files (unmodified copies)
├── mod/                  # NMODL mechanisms (unmodified copies)
│   └── x86_64/           # Compiled after running nrnivmodl
├── params/
│   ├── __init__.py
│   └── circuit_params.py # Hard-coded Python translation of Circuit_param.xls
└── output/               # Simulation output (auto-created)
```

---

## Control Flags (`cfg.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `cfg.testing` | `True` | 100-cell test (80/7/5/8), 1 s run |
| `cfg.no_connectivity` | `False` | Disable all synaptic connections |
| `cfg.MDD` | `False` | Reduce SST output by 40% (Major Depressive Disorder) |
| `cfg.DRUG` | `False` | Use pharmacological tonic conductance values |
| `cfg.stimulate` | `False` | Enable Poisson external stimulus |
| `cfg.rec_LFP` | `False` | Record extracellular LFP |
| `cfg.rec_DIPOLES` | `False` | Record population dipole moments |

---

## Cell Types

| Population | Type | Count (test) | Count (full) |
|------------|------|-------------|-------------|
| HL23PYR | Pyramidal (excitatory) | 80 | 800 |
| HL23SST | Somatostatin interneuron | 5 | 50 |
| HL23PV | Parvalbumin interneuron | 7 | 70 |
| HL23VIP | VIP interneuron | 8 | 80 |

---

## Expected Validation Targets (test run, 1000 ms)

From Yao 2022 Fig. 2D (approximate):

| Population | Mean Rate (Hz) |
|------------|---------------|
| HL23PYR | 0.3 – 0.8 |
| HL23PV | 10 – 20 |
| HL23SST | 4 – 8 |
| HL23VIP | 2 – 5 |

---

## See Also

- `REPLICATION_NOTES.md` — full list of approximations vs. original
- Original model: `../circuit.py`
- Reference: Yao et al. 2022, *PLOS Computational Biology*

# Hybrid Classical-Quantum Spectra

Code accompanying the paper on hybrid classical-quantum algorithms for computing molecular spectra.

## Overview

This repository contains the implementation of a hybrid classical-quantum algorithm for computing excitation spectra of molecular systems. The method combines:
- Classical sampling from perturbed wavefunctions
- Quantum/tensor network evolution of sampled basis states
- Classical reconstruction of dynamical correlation functions

## Repository Structure

```
.
├── hybridqc/
│   ├── src/
│   │   ├── python/
│   │   │   ├── generate_hamiltonian/    # Hamiltonian generation from PySCF
│   │   │   ├── subspace_evolution/      # Main algorithm implementation
│   │   │   │   ├── RunAlgorithm.py      # Single molecule evolution
│   │   │   │   └── Samples_RunAlgorithm.py  # Scaling analysis
│   │   │   └── tensor_network/          # Tensor network methods
│   │   │       ├── evolution.py         # TN-based time evolution
│   │   │       └── retrieve_data.py     # Data retrieval utilities
│   │   └── julia/
│   │       ├── DMRG_groundstate.jl      # DMRG ground state solver
│   │       ├── QuantumEvolution.jl      # TDVP time evolution
│   │       ├── sample_perturbed_state.jl # State sampling
│   │       └── MPSutils.jl              # MPS utility functions
│   ├── plots/
│   │   ├── plot_energy_levels.py        # Energy level comparison plots
│   │   ├── plot_fourier_spectra.py      # Fourier spectra plots
│   │   └── plot_scaling_samples.py      # Scaling analysis plots
│   ├── data/
│   │   ├── plot_data/                   # Extracted data for plots (JSON)
│   │   ├── results/                     # Algorithm output (generated)
│   │   ├── ground_states/               # DMRG ground states (generated)
│   │   ├── sampled/                     # Sampled states (generated)
│   │   └── hamiltonians_json/           # Hamiltonians in JSON format for Julia
│   ├── hamiltonians/                    # Molecular Hamiltonian files (.dat FCIDUMP format)
│   └── figures/                         # Generated figures (PDF)
├── pyproject.toml                       # Python dependencies (uv)
├── uv.lock                              # Python dependency lock (uv)
├── Project.toml                         # Julia dependencies
└── Manifest.toml                        # Julia dependency lock
```

## Installation

### Python

Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
uv sync
```

### Julia

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

### Generating Plots

All plots can be regenerated from the extracted data:

```bash
# Using uv
uv run python hybridqc/plots/plot_energy_levels.py
uv run python hybridqc/plots/plot_fourier_spectra.py
uv run python hybridqc/plots/plot_scaling_samples.py
```

### Molecules Studied

| Molecule | Basis Set | Active Space (orbitals, electrons) |
|----------|-----------|-----------------------------------|
| LiH      | 6-31g     | (10, 4)                          |
| LiH      | cc-pvdz   | (16, 4)                          |
| N₂       | cc-pvdz   | (8, 10)                          |
| HCl      | sto-6g    | (10, 18)                         |
| CO       | cc-pvdz   | (8, 10)                          |

## Workflow

### Running the Full Algorithm (Python - exact diagonalization)

The main algorithm can be run from `hybridqc/src/python/subspace_evolution/`:

```bash
cd hybridqc/src/python/subspace_evolution
uv run python RunAlgorithm.py <molecule_index>
```

Where `molecule_index` is 0-7 corresponding to:
- 0: LiH (6-31g, 10 orbitals)
- 1: LiH (cc-pvdz, 16 orbitals)
- 2: N2 (cc-pvdz, 8 orbitals)
- 3: HCl (sto-6g, 10 orbitals)
- 4-7: Alternative perturbations for the same molecules

Results are saved to `hybridqc/data/results/`.

### Running with Tensor Networks (Julia - for larger systems)

The tensor network approach uses DMRG for ground states and TDVP for time evolution, enabling simulations of larger systems.

**Step 1: Convert Hamiltonians to JSON format**
```bash
cd hybridqc/src/python/generate_hamiltonian
uv run python convert_fcidump_to_json.py
```
This converts all `.dat` files in `hybridqc/hamiltonians/` to JSON format in `hybridqc/data/hamiltonians_json/`.

**Step 2: Compute the ground state with DMRG**
```bash
cd hybridqc/src/julia
julia --project=../../.. DMRG_groundstate.jl <molecule> <basis> <n_electrons> <n_orbitals> <R>
```
Output: `hybridqc/data/ground_states/<molecule>_<basis>_<no>o<ne>e_R<R>.jld2`

**Step 3: Sample the perturbed state**
```bash
julia --project=../../.. sample_perturbed_state.jl <molecule> <basis> <n_electrons> <n_orbitals> <R> <n_samples>
```
Output: `hybridqc/data/sampled/<molecule>_.../perturbed_state_nsamples<n>.jld2`

**Step 4: Run TDVP time evolution**
```bash
julia --project=../../.. QuantumEvolution.jl <molecule> <basis> <n_electrons> <n_orbitals> <R> <start_idx> <end_idx>
```
This evolves sampled states from index `start_idx` to `end_idx` and saves the evolved basis states.

**Example (LiH with 6-31g basis):**
```bash
cd hybridqc/src/julia
julia --project=../../.. DMRG_groundstate.jl LiH 6-31g 4 10 1.5
julia --project=../../.. sample_perturbed_state.jl LiH 6-31g 4 10 1.5 1000000
julia --project=../../.. QuantumEvolution.jl LiH 6-31g 4 10 1.5 1 100
```

## Methods

### Hamiltonian Generation
The `generate_hamiltonian/` module uses PySCF to compute molecular integrals and construct Hamiltonians in the FCIDUMP format.

### Subspace Evolution Algorithm
1. Sample basis states from the perturbed ground state
2. Evolve each basis state independently (quantum simulation)
3. Measure evolved states to construct a local basis
4. Project the Hamiltonian onto the subspace
5. Classically evolve and reconstruct the Green's function
6. Extract excitation energies from Fourier transform peaks

### Tensor Network Methods
Julia implementations using ITensors for:
- DMRG ground state calculations
- TDVP time evolution

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this work for any purpose, provided you give appropriate credit.

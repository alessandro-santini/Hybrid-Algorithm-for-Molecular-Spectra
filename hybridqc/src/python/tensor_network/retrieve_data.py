import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

import netket.experimental as nkx

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, "..", "..", "..")
sys.path.append(os.path.join(script_dir, "..", "generate_hamiltonian"))

from pyscf_to_nk import *

basis_and_weights = []
mol = "N2"
basis_name = "6-31g"
no = 16
ne = 10
R = 1.10

def convert_to_netket_bitstring(no, state_16):
    up = np.zeros(no, dtype=np.int8)
    down = np.zeros(no, dtype=np.int8)

    for i, val in enumerate(state_16):
        if val == 2:
            up[i] = 1
        elif val == 3:
            down[i] = 1
        elif val == 4:
            up[i] = 1
            down[i] = 1
        # val == 1 → do nothing (empty)

    return np.concatenate([up, down])  # length 2*no
def generate_netket_bitstrings(states, no):
    for s in states:
        yield convert_to_netket_bitstring(no, s)

data_dir = os.path.join(repo_root, "data", "sampled")
for x in range(1,563):
    path = os.path.join(data_dir, f"{mol}_{basis_name}_{no}o{ne}e_R{R}", "sampled_basis", f"evolved_states_bases_{x}.jld2")
    
    with h5py.File(path, "r") as f:
        compound_scalar = f["state_basis"][()]
        ref = compound_scalar["kvvec"]
        main_array = f[ref][()]  # Array of (object_ref, scalar)
    
        for subref, weight in main_array:
            basis_vector = list(f[subref][()])  # Convert to list of ints
            basis_and_weights.append((basis_vector, weight))

unique_basis_weights = {}
#threshold = 1e-7  # or whatever makes sense for your data
threshold = 0.
for basis_vector, weight in basis_and_weights:
    if abs(weight) < threshold:
        continue  # skip small weights

    key = tuple(basis_vector)
    if key not in unique_basis_weights:
        unique_basis_weights[key] = weight
unique_basis = list(unique_basis_weights.keys())
weights = np.array([unique_basis_weights[state] for state in unique_basis])

print( np.log2(len(unique_basis)))

name_molfile = os.path.join(repo_root, "hamiltonians", f'{mol}_{basis_name}_{no}o{ne}e_R{R}_hami.dat')
E_hf, nuclear_repulsion_energy, h1, h2_reordered = hf_and_integrals_from_fcidump(molfile=name_molfile)
h_list, h_coeffs, e_nuc, num_orbitals = generate_hami_from_fcidump(name_molfile)

def replace_signs(data):
    return [[(num, 1 if sign == '+' else 0) for num, sign in sublist] for sublist in data]

hi = nkx.hilbert.SpinOrbitalFermions(no, s=0.5,  n_fermions_per_spin=(ne//2,ne//2))
H  = nkx.operator.FermionOperator2nd(hi, replace_signs(h_list), h_coeffs)

from scipy.sparse import lil_matrix

# Generate NetKet-compatible basis
netket_basis = [bitstring for bitstring in generate_netket_bitstrings(unique_basis, no)]

N = len(netket_basis)
H_proj = lil_matrix((N, N), dtype=np.complex128)  # Use complex128 if needed

# Map basis states to indices
index_of = {tuple(s): i for i, s in enumerate(netket_basis)}

# Project the Hamiltonian
for i, s in tqdm(
    enumerate(netket_basis),
    total=N,
    desc="Projecting H",
    dynamic_ncols=False,  # avoid terminal width detection
    mininterval=5.0,       # update at most once every 5 seconds
    file=sys.stdout        # ensure it writes to stdout
):
    connected, coeffs = H.get_conn(s)
    for s_p, c in zip(connected, coeffs):
        j = index_of.get(tuple(s_p))
        if j is not None:
            H_proj[i, j] += c

# (Optional) convert to CSR format for efficient arithmetic
H_proj = H_proj.tocsr()

from scipy.sparse import save_npz
from scipy.sparse.linalg import eigsh

H_proj = H_proj.tocsr()  # make sure it's in CSR format
save_npz("H_proj.npz", H_proj)

k = 1  # number of eigenvalues (1 = ground state)
which = 'SA'  # 'SA' = Smallest Algebraic (for ground state)

evals, evecs = eigsh(H_proj, k=k, which=which)

E0 = evals[0]       # ground state energy
psi0 = evecs[:, 0]  # ground state wavefunction

print('ground state energy',E0)
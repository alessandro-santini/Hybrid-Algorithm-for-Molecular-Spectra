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

from scipy.sparse import load_npz

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
H_proj = load_npz("H_proj.npz")

from scipy.sparse.linalg import eigsh

# H_proj must be Hermitian (real symmetric or complex Hermitian) and in CSR format
evals, evecs = eigsh(H_proj, k=1, which='SA')  # SA = smallest algebraic
print("Ground state energy:", evals[0])

from scipy.sparse.linalg import expm_multiply

t0, tf, nsteps = -1000, 1000, 4097
t_space = np.linspace(t0, tf, nsteps)
dt = t_space[1] - t_space[0]

weights = weights / np.linalg.norm(weights)

psi_t = expm_multiply(-1j * H_proj * t0, weights)

G = np.empty(nsteps, dtype=np.complex128)
psi = psi_t.copy()

for i in tqdm(range(nsteps), desc="Time stepping"):
    G[i] = np.vdot(weights, psi)  # G(t) = ⟨ψ₀|ψ(t)⟩
    psi = expm_multiply(-1j * H_proj * dt, psi)

G *= np.exp(1j * evals[0] * t_space)

np.save("G.npy", G)
import sys
import os

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "generate_hamiltonian"))

from pyscf_to_nk import *
import netket as nk
import netket.experimental as nkx
import seaborn as sns

import pickle

import psutil
import os 

from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

import numpy as np
import random

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

p = psutil.Process(os.getpid())
freq = psutil.cpu_freq()

print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Physical CPUs:", psutil.cpu_count(logical=False))
print(f"CPU Frequency: current = {freq.current / 1000:.2f} GHz, min = {freq.min / 1000:.2f} GHz, max = {freq.max / 1000:.2f} GHz")#print("CPU Stats:", psutil.cpu_stats())
print("CPU Usage (total):", psutil.cpu_percent(interval=1), "%")
print("Threads used by this notebook:", p.num_threads(),"\n")

mem = psutil.virtual_memory()
total_gb = mem.total / (1024 ** 3)
used_gb = mem.used / (1024 ** 3)
available_gb = mem.available / (1024 ** 3)
print(f"Total RAM:     {total_gb:.2f} GB")
print(f"Used RAM:      {used_gb:.2f} GB")
print(f"Available RAM: {available_gb:.2f} GB")
print(f"Usage percent: {mem.percent:.1f}%")

########################################################################################################################
########################################################################################################################

# List of molecule configurations
molecule_list = [
    {'molecule': 'LiH', 'no': 10,  'ne': 4,  'R': 1.5, 'base': '6-31g'},
    {'molecule': 'LiH', 'no': 16,  'ne': 4,  'R': 1.5, 'base': 'cc-pvdz'},
    {'molecule': 'N2',  'no': 8,  'ne': 10, 'R': 1.1, 'base': 'cc-pvdz'},
    {'molecule': 'HCl', 'no': 10, 'ne': 18, 'R': 1.2, 'base': 'sto-6g'},
    {'molecule': 'LiH1', 'no': 10,  'ne': 4,  'R': 1.5, 'base': '6-31g'},
    {'molecule': 'LiH1', 'no': 16,  'ne': 4,  'R': 1.5, 'base': 'cc-pvdz'},
    {'molecule': 'HCl1', 'no': 10, 'ne': 18, 'R': 1.2, 'base': 'sto-6g'},
    {'molecule': 'CO', 'no':8, 'ne':10, 'R':1.2, 'base':'cc-pvdz'}
]

# Choose one configuration from the list
selected_molecule = molecule_list[int(sys.argv[1])] 

# Set variables from the selected configuration
no = selected_molecule['no']
ne = selected_molecule['ne']
R = selected_molecule['R']
base_mol = selected_molecule['base']
molecule = selected_molecule['molecule']

if molecule.endswith('1'):
    molecule_file_name = molecule[:-1]
else:
    molecule_file_name = molecule

# Path to hamiltonians directory (relative to repository root)
repo_root = os.path.join(script_dir, "..", "..", "..")
name_molfile = os.path.join(repo_root, "hamiltonians", f'{molecule_file_name}_{base_mol}_{no}o{ne}e_R{R}_hami.dat')

# Optional: print to confirm
print(molecule, no, ne, R, base_mol)
print(name_molfile)

E_hf, nuclear_repulsion_energy, h1, h2_reordered = hf_and_integrals_from_fcidump(molfile=name_molfile)
h_list, h_coeffs, e_nuc, num_orbitals = generate_hami_from_fcidump(name_molfile)

def replace_signs(data):
    return [[(num, 1 if sign == '+' else 0) for num, sign in sublist] for sublist in data]

hi = nkx.hilbert.SpinOrbitalFermions(no, s=0.5,  n_fermions_per_spin=(ne//2,ne//2))
H  = nkx.operator.FermionOperator2nd(hi, replace_signs(h_list), h_coeffs)

print('Relative size in qubits symmetric subspace', np.log2(hi.n_states))

# Perturbation dict
d_dict = {
    'N2':   lambda hi: nk.operator.fermion.create(hi,6,1) @ nk.operator.fermion.create(hi,5,1) @ nk.operator.fermion.destroy(hi,4,1) @ nk.operator.fermion.destroy(hi,3,1),
    'LiH':  lambda hi: nk.operator.fermion.create(hi,2,1) @ nk.operator.fermion.destroy(hi,1,1),
    'LiH1': lambda hi: nk.operator.fermion.create(hi,3,1) @ nk.operator.fermion.destroy(hi,1,1),
    'HCl':  lambda hi: nk.operator.fermion.create(hi,9,1) @ nk.operator.fermion.destroy(hi,7,1),
    'HCl1': lambda hi: nk.operator.fermion.create(hi,9,1) @ nk.operator.fermion.destroy(hi,8,1),
    'CO': lambda hi: nk.operator.fermion.create(hi,6,1) @ nk.operator.fermion.destroy(hi,4,1),
}

eigval, eigvec = nk.exact.lanczos_ed(H, k=4, compute_eigenvectors=True)
print('E0:', eigval[0])
psi_0 = eigvec[:,0]
E0 = eigval[0]

d = d_dict[molecule](hi).to_sparse()

psi_1 = d@psi_0
avg_d = np.vdot(psi_1,psi_1)

psi_1 = psi_1/np.linalg.norm(psi_1)

perturbed_states_dict = {'psi_1':psi_1, 'avg_d_psi0':avg_d}

########################################################################################################################
########################################################################################################################

# We are computing G(t) = <psi_0|exp(+iHt) d^\dagger exp(-iHt) d|psi_0>
# which is equal to e^(+i*E0*t)*<psi_0| d^\dagger exp(-iHt) d|psi_0> = e^(+i*E0*t)*<psi_1| exp(-iHt) |psi_1>
# Therefore we can use expm_multiply to compute the time evolution of the state |psi_1>

Hsp = H.to_sparse()
t0,tf,nsteps = -1000, 1000, 4097
t_space = np.linspace(t0, tf, nsteps)
dt = t_space[1] - t_space[0]

G = np.zeros(nsteps, dtype=complex)

psi_t = expm_multiply(-1j * Hsp * t0, psi_1)
G[0] = np.vdot(psi_1, psi_t) * np.exp(1j * E0 * t_space[0])

for t in tqdm(range(1, nsteps)):
    psi_t = expm_multiply(-1j*Hsp*dt, psi_t)
    G[t] = np.vdot(psi_1, psi_t)*np.exp(1j*E0*t_space[t])

########################################################################################################################
########################################################################################################################

basis_dict = {}

# HYBRID ALGORITHM #

# psi_1: complex array representing the wavefunction
p1 = np.abs(psi_1)**2
p1 /= p1.sum()  # Normalize to 1 just in case

# Number of samples you want
n_samples = 1000000

# Sample indices according to the probability distribution
samples = np.random.choice(len(p1), size=n_samples, p=p1)

# If you want the unique basis states among samples:
unique_samples = np.unique(samples)
x_best = unique_samples
print(f"Number of unique basis states in {n_samples} samples: {len(unique_samples)}")

basis_dict['sampled_from_perturbed_state'] = x_best

# Sample the perturbed wavefunction
base_yofx = []

# Time evolution parameters
t_evo_qc = np.arange(0, 10.26, 1.0)
dt_evo_qc = t_evo_qc[1] - t_evo_qc[0]

# Number of samples you want per time step
samples_per_step = 4096

psi_dict = {}  # maps x → array of shape (n_time, N_basis)
for x in tqdm(x_best):
    base = set()
    psi_list = []
    # Initialize basis state: delta at position x
    psi = np.zeros_like(psi_1, dtype=complex)
    psi[x] = 1.0
    
    for t in range(t_evo_qc.size):
        psi_list.append(psi.copy())
        # Sample from |psi|^2 at this time
        prob = np.abs(psi)**2
        prob /= prob.sum()  # normalize
        sampled_indices = np.random.choice(len(prob), size=samples_per_step, p=prob)
        base.update(map(int, sampled_indices))

        # Evolve the wavefunction to next step
        if t < t_evo_qc.size - 1:
            psi = expm_multiply(-1j * Hsp * dt_evo_qc, psi)
    psi_dict[x] = np.stack(psi_list)
    base_yofx.append(list(base))

basis_dict['sampled_from_evolved'] = base_yofx

# Output directory for results
output_dir = os.path.join(repo_root, "data", "results")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, f"psi_dict_{molecule}_{base_mol}_{no}o{ne}e_R{R}.pkl"), "wb") as f:
    data = {'psi_dict':psi_dict,'basis_dict': basis_dict, 'perturbed_states_dict':perturbed_states_dict}
    pickle.dump(data, f)

def count_unique_lists(list_of_lists):
    unique_lists = set(tuple(sorted(lst)) for lst in list_of_lists)
    return len(unique_lists), [list(x) for x in unique_lists]

num_unique_lists, bases_unique = count_unique_lists(base_yofx)
print(f'There are {num_unique_lists} different bases.')
print(max(list(map(len,base_yofx))))

########################################################################################################################
########################################################################################################################


# Project the Hamiltonian onto the subspace defined by the sampled basis and compute the Green's function

check_herm = False
G_proj = np.zeros(nsteps, dtype=complex)
for ix, projection_basis in tqdm(enumerate(base_yofx), total=len(base_yofx)):
    
    projection_indices = np.ix_(projection_basis, projection_basis)
    # Create the projected Hamiltonian
    Hsp_proj = Hsp[projection_indices]

    # Check if Hsp_proj is hermitian
    if check_herm:
        print('Is hermitean?', np.allclose(Hsp_proj.todense(), Hsp_proj.todense().conj().T))

    eigvals, eigvecs = np.linalg.eigh(Hsp_proj.todense())
    #print(np.where( x_best[ix] == np.array(projection_basis))[0])
    index_x_best_in_projection_basis = np.where(projection_basis == x_best[ix])[0].item()
    eigvecs = np.array(eigvecs)
    
    x_pos = np.where( x_best[ix] == np.array(projection_basis))[0].item()
    psi_evo = np.zeros_like(projection_basis,dtype=complex)
    psi_evo[x_pos] = psi_1[x_best[ix]]

    for it, t in enumerate(t_space):
        G_proj[it] += np.vdot(psi_1[projection_basis], eigvecs @ np.diag(np.exp(-1j * eigvals * t)) @ eigvecs.T @ psi_evo) * np.exp(1j * E0 * t)

G_dict = {
    'full': G,
    'proj': G_proj,
    'num_basis': num_unique_lists,
    'basis':bases_unique,
    'Hsp':Hsp
}

with open(os.path.join(output_dir, f'G_dict_{molecule}_{base_mol}_{no}o{ne}e_R{R}.pkl'), 'wb') as f:
    pickle.dump(G_dict, f)

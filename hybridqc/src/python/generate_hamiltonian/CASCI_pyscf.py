import numpy as np

import pyscf
from pyscf import gto,ao2mo, scf, tools, fci, mcscf



# Create the molecule
R = 1.1
mol_string_N2 = f'N 0 0 0; N 0 0 {R}'

mol_name = 'N2'

#basis = 'cc-pvdz'
basis = '6-31g'
#basis = 'sto-6g'
mol = gto.M(atom = mol_string_N2, basis =basis,verbose=4,spin=0,symmetry="Coov")
mol.max_memory = 64000

# Calculate HF energies
rhf = scf.RHF(mol)
hf_e = rhf.kernel()

print("HF energy: ", hf_e)

no = 16
ne = 10

ncas, nelecas = (no, ne) # n orbitals, n electrons in the active space    
cas_mol = mcscf.CASCI(rhf, ncas, nelecas)

h1eff, ecore = cas_mol.get_h1eff()
h2eff = cas_mol.get_h2eff()

output_filename = f"../hamiltonians/{mol_name}_{basis}_{no}o{ne}e_R{R}_hami.dat"

# Write the integrals to the FCIDUMP file
tools.fcidump.from_integrals(output_filename, h1eff, h2eff, ncas, nelecas, nuc=ecore)
print(output_filename)
exit()


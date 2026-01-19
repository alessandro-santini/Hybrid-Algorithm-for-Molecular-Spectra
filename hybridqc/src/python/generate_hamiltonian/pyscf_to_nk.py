import numpy as np
import pyscf
from pyscf import tools, ao2mo

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
### Here we put all the function to create an Hamiltonian operator 


def hf_and_integrals_from_fcidump(molfile):

    mf_as = tools.fcidump.to_scf(molfile)
    hcore = mf_as.get_hcore()
    num_orbitals = hcore.shape[0]
    eri = ao2mo.restore(1, mf_as._eri, num_orbitals)
    nuclear_repulsion_energy = mf_as.mol.energy_nuc()

    # Impose to not re-do the HF calculation
    mf_as.max_cycle = 0
    dm0 = np.zeros((num_orbitals,num_orbitals))
    for i in range(mf_as.mol.nelectron//2): dm0[i,i]= 2.0
    E_hf = mf_as.kernel(dm0=dm0)
    mf_as.mo_coeff = np.eye(mf_as.mo_coeff.shape[1])

    
    h1 = mf_as.mo_coeff.T.dot(hcore).dot(mf_as.mo_coeff)
    h2 = pyscf.lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri,
                              mf_as.mo_coeff, mf_as.mo_coeff, mf_as.mo_coeff, mf_as.mo_coeff)

    # explicit symmetrization of integrals
    h1 = (h1+h1.T)/2.


    # reorder for physicist notation
    h2_reordered = np.asarray(h2.transpose(0, 2, 3, 1), order='C')

    return E_hf, nuclear_repulsion_energy, h1, h2_reordered

def get_tensors_from_integrals_nk_order(one_body_integrals, two_body_integrals, EQ_TOLERANCE):
    '''Converts one and two-body integrals into tensor form
    Arguments:
        one_body_integrals [numpy array] -- the one-body integrals
            of the given Hamiltonian
        two_body_integrals [numpy array] -- the two-body integrals
            of the given Hamiltonian
    '''

    n_qubits = 2 * one_body_integrals.shape[0]

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits)) #*0j
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits)) #*0j
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[p, q] = one_body_integrals[p, q]
            one_body_coefficients[p + n_qubits // 2, q + n_qubits // 2] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):

                    # Mixed spin
                    two_body_coefficients[p, q + n_qubits // 2, r + n_qubits // 2, s] = (two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[p + n_qubits // 2, q, r, s + n_qubits // 2] = (two_body_integrals[p, q, r, s] /2.)

                    # Same spin
                    two_body_coefficients[p, q, r, s] = (two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[p + n_qubits // 2, q + n_qubits // 2, r + n_qubits // 2, s + n_qubits // 2] = (two_body_integrals[p, q, r, s] / 2.)

    # Truncate.
    one_body_coefficients[
        np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    return one_body_coefficients, two_body_coefficients


def tensor_to_nk_hami(N_modes, one_body_term,two_body_term ): 
    terms = []
    weights = []
    for i in range(N_modes):
        for j in range(N_modes):
            if np.abs(one_body_term[i, j])>10e-50:
                terms.append([(i, "+"), (j, "-")]) # c^\dag_i c_j
                weights = weights + [one_body_term[i, j], ]


    for i in range(N_modes):
        for j in range(N_modes):
            for k in range(N_modes):
                for l in range(N_modes):
                    if np.abs(two_body_term[i, j, k, l])>10e-15:
                        if i!=j and k!=l:
                            terms.append([(i, "+"), (j, "+"), (k, "-"), (l, "-")]) # c^\dag_i c^\dag_j c_k c_k
                            weights = weights + [np.around(two_body_term[i, j, k, l], decimals = 10), ]
    return terms, weights

def generate_hami_from_fcidump(molfile,eq_tolerance=1e-50):

    _ , e_nuc, h1_int, h2_int  = hf_and_integrals_from_fcidump(molfile)
    num_orbitals        = h1_int.shape[0]

    h1_coeffs, h2_coeffs   = get_tensors_from_integrals_nk_order(h1_int,h2_int,EQ_TOLERANCE=eq_tolerance)
    h_list, h_coeffs       = tensor_to_nk_hami(2*num_orbitals,h1_coeffs,h2_coeffs)

    return h_list, h_coeffs, e_nuc, num_orbitals
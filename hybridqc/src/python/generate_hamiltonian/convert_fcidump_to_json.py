#!/usr/bin/env python3
"""
Convert FCIDUMP Hamiltonian files (.dat) to JSON format for Julia.

Usage:
    python convert_fcidump_to_json.py

This script processes all *_hami.dat files in the hamiltonians/ directory
and outputs JSON files to data/hamiltonians_json/.
"""
import os
import json
import re

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, "..", "..", "..")

from pyscf_to_nk import *
import netket as nk
import netket.experimental as nkx

# Paths
input_dir = os.path.join(repo_root, "hamiltonians")
output_dir = os.path.join(repo_root, "data", "hamiltonians_json")
os.makedirs(output_dir, exist_ok=True)

# Regex to extract info from filename
filename_re = re.compile(r"(?P<mol>[A-Za-z0-9]+)_(?P<basis>[a-zA-Z0-9\-]+)_(?P<no>\d+)o(?P<ne>\d+)e_R(?P<R>[\d.]+)_hami\.dat")

def replace_signs(data):
    return [[(num, 1 if sign == '+' else 0) for num, sign in sublist] for sublist in data]

# Loop through all files
for fname in os.listdir(input_dir):
    if not fname.endswith("_hami.dat"):
        continue

    match = filename_re.match(fname)
    if not match:
        print(f"Filename format not recognized: {fname}")
        continue

    mol = match.group("mol")
    basis = match.group("basis")
    no = int(match.group("no"))
    ne = int(match.group("ne"))
    R = match.group("R")

    name_molfile = os.path.join(input_dir, fname)
    print(f"Processing {fname}...")

    try:
        # Read integrals and generate Hamiltonian
        E_hf, nuclear_repulsion_energy, h1, h2_reordered = hf_and_integrals_from_fcidump(molfile=name_molfile)
        h_list, h_coeffs, e_nuc, num_spatial = generate_hami_from_fcidump(name_molfile)

        hi = nkx.hilbert.SpinOrbitalFermions(num_spatial, s=0.5, n_fermions_per_spin=(ne // 2, ne // 2))
        H = nkx.operator.FermionOperator2nd(hi, replace_signs(h_list), h_coeffs)

        def convert_term(term, spatial_orbitals):
            new_term = []
            for i, op_type in term:
                spin = 1 if i < spatial_orbitals else 0
                spatial_index = i if i < spatial_orbitals else i - spatial_orbitals
                new_term.append([spatial_index, spin, op_type])
            return new_term

        terms_list = [convert_term(term, num_spatial) for term in H.terms]

        data = {
            "terms": terms_list,
            "weights": list(H.weights),
            "nuclear_repulsion_energy": nuclear_repulsion_energy
        }

        out_fname = f"{mol}_{basis}_{no}o{ne}e_R{R}.json"
        with open(os.path.join(output_dir, out_fname), "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Done: {out_fname}")

    except Exception as e:
        print(f"❌ Failed to process {fname}: {e}")

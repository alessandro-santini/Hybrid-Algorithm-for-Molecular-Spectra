using JSON
using ITensorMPS
using ITensors
using JLD2

# Get the script directory and repository root
script_dir = @__DIR__
repo_root = joinpath(script_dir, "..", "..")

# Load the data from the JSON file
molecule    = ARGS[1]
base        = ARGS[2]
n_electrons = parse(Int, ARGS[3])
n_orbitals  = parse(Int, ARGS[4])
R = parse(Float64, ARGS[5])
doubly_occupied_orbitals = n_electrons ÷ 2
unoccupied_orbitals = n_orbitals - doubly_occupied_orbitals
n_sites = n_orbitals

# Data directories
data_dir = joinpath(repo_root, "data")
hamiltonians_dir = joinpath(data_dir, "hamiltonians_json")
ground_states_dir = joinpath(data_dir, "ground_states")
mkpath(ground_states_dir)
mkpath(joinpath(ground_states_dir, "out_dmrg"))

filename_input  = joinpath(hamiltonians_dir, "$(molecule)_$(base)_$(n_orbitals)o$(n_electrons)e_R$(R).json")
filename_output = joinpath(ground_states_dir, "$(molecule)_$(base)_$(n_orbitals)o$(n_electrons)e_R$(R).jld2")

data = JSON.parsefile(filename_input)
terms = data["terms"]
weights = data["weights"]

sites = siteinds("Electron", n_orbitals; conserve_qns=true) # ITensor's fermionic sites
        
ampo = AutoMPO()

for (term, w) in zip(terms, weights)
    # Here we assume that each term's operator entries also include spin info
    # For instance, an operator entry might be [i, spin, op_type]
    # where `spin` might be 0 for down and 1 for up.
    if length(term) == 2
        # One-body term: e.g., term = [[i, spin1, op1], [j, spin2, op2]]
        i, spin1, op1 = term[1]  
        j, spin2, op2 = term[2]
        # Convert indices from 0-indexing (NetKet) to 1-indexing (Julia)
        i += 1; j += 1
        # Map to appropriate operator names
        op_name1 = (op1 == 1) ? (spin1 == 1 ? "Cdagup" : "Cdagdn") : (spin1 == 1 ? "Cup" : "Cdn")
        op_name2 = (op2 == 1) ? (spin2 == 1 ? "Cdagup" : "Cdagdn") : (spin2 == 1 ? "Cup" : "Cdn")
        add!(ampo, w, op_name1, i, op_name2, j)
    elseif length(term) == 4
        # Two-body term: e.g., term = [[i, s1, op1], [j, s2, op2], [k, s3, op3], [l, s4, op4]]
        i, s1, op1 = term[1]
        j, s2, op2 = term[2]
        k, s3, op3 = term[3]
        l, s4, op4 = term[4]
        i += 1; j += 1; k += 1; l += 1
        op_name1 = (op1 == 1) ? (s1 == 1 ? "Cdagup" : "Cdagdn") : (s1 == 1 ? "Cup" : "Cdn")
        op_name2 = (op2 == 1) ? (s2 == 1 ? "Cdagup" : "Cdagdn") : (s2 == 1 ? "Cup" : "Cdn")
        op_name3 = (op3 == 1) ? (s3 == 1 ? "Cdagup" : "Cdagdn") : (s3 == 1 ? "Cup" : "Cdn")
        op_name4 = (op4 == 1) ? (s4 == 1 ? "Cdagup" : "Cdagdn") : (s4 == 1 ? "Cup" : "Cdn")
        add!(ampo, w, op_name1, i, op_name2, j, op_name3, k, op_name4, l)
    else
        error("Unexpected term length: ", length(term))
    end
end

state_labels_occupied = repeat(["UpDn"], doubly_occupied_orbitals)
state_labels_unoccupied = repeat(["Emp"], unoccupied_orbitals)
state_labels = vcat(state_labels_occupied, state_labels_unoccupied)

# 3. Build the product state MPS.
println("Building the initial state")
psi0 = productMPS(sites, state_labels)

# Build the MPO from the AutoMPO using the electron sites
println("Building the MPO")
H_mpo = MPO(ampo, sites)

sweeps = Sweeps(20)
setmaxdim!(sweeps,16,16,32,32,32,64,64,64,128,128,256,256,256,512)
setcutoff!(sweeps,1E-10)
setmindim!(sweeps,16)
setnoise!(sweeps,0.05,0.01,0.01,0.)

println("Starting the DMRG calculation")

open(joinpath(ground_states_dir, "out_dmrg", "$(molecule)_$(base)_$(n_orbitals)o$(n_electrons)e_R$(R).txt"), "w") do io
    redirect_stdout(io) do
        energy, psi = dmrg(H_mpo, psi0, sweeps; outputlevel=3)

        println("energy:"   ,energy)
        println("Orbital   <Nup>    <Ndn>    <Sz>")
        
        for i in 1:n_sites
            # Compute the expectation values at site i.
            # Here, "inner" calculates the expectation value ⟨psi|O|psi⟩.
            nup_val = expect(psi, "Nup"; sites=i)
            ndn_val = expect(psi, "Ndn"; sites=i)
            
            # Optionally, you can compute the local spin polarization <Sz> = 0.5*(<Nup> - <Ndn>)
            sz_val = 0.5 * (nup_val - ndn_val)
            
            println("    $(i)    $(nup_val)   $(ndn_val)   $(sz_val)")
        end
        
        @save filename_output psi H_mpo sites energy
     end
end

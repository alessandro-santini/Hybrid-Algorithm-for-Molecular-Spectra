using ITensorMPS
using ITensors
using JLD2

include("MPSutils.jl")
using .MPSutils

# Get the script directory and repository root
script_dir = @__DIR__
repo_root = joinpath(script_dir, "..", "..")

const MOLECULE_PERTURBATIONS = Dict(
    "LiH" => [
        ["Cdagup", 3, "Cdagdn", 3, "Cdn", 2, "Cup", 2],
        ["Cdagup", 4, "Cdagdn", 4, "Cdn", 2, "Cup", 2]
    ],
    "N2" => [
        ["Cdagup", 7, "Cdagdn", 6, "Cdn", 5, "Cup", 4]
    ],
    "HCl" => [
        ["Cdagup", 10, "Cup", 8],
        ["Cdagup", 10, "Cup", 9]
    ],
    "2Fe2S" => [
        ["Cdagup", 9, "Cdagdn", 12, "Cdn", 10, "Cup", 11]
    ]
)

molecule    = ARGS[1]
base        = ARGS[2]
n_electrons = parse(Int, ARGS[3])
n_orbitals  = parse(Int, ARGS[4])
R = parse(Float64, ARGS[5])
nsamples = parse(Int,ARGS[6]) #Number of samples to take from the perturbed state

# Data directories
data_dir = joinpath(repo_root, "data")
ground_states_dir = joinpath(data_dir, "ground_states")
sampled_dir = joinpath(data_dir, "sampled")

# Load the data
@load joinpath(ground_states_dir, "$(molecule)_$(base)_$(n_orbitals)o$(n_electrons)e_R$(R).jld2") psi sites
save_folder = joinpath(sampled_dir, "$(molecule)_$(base)_$(n_orbitals)o$(n_electrons)e_R$(R)")
if !isdir(save_folder)
    mkpath(save_folder)
end

doubly_occupied_orbitals = n_electrons ÷ 2

ampo = AutoMPO()
i = doubly_occupied_orbitals
j = doubly_occupied_orbitals + 1
# C†_{j↑} C†_{j↓} C_{i↓} C_{i↑}
add!(ampo, 1.0, "Cdagup", j, "Cdagdn", j, "Cdn", i, "Cup", i)
perturbation = MPO(ampo, sites)


# Create the perturbed state
psiA = apply(perturbation, psi)
overlap_psiA = inner(psiA, psiA)
orthogonalize!(psiA, 1)
normalize!(psiA)

# Measure the perturbed state
threshold = 0.   # Threshold for accepting a state
sampled_states, probability = measure_states(psiA, sites, nsamples, threshold)

# Print the number of states to evolve
println("Number of states to evolve: ", length(sampled_states))
number_states = length(sampled_states)

# Save the sampled states and their probabilities
@save joinpath(save_folder, "perturbed_state_nsamples$(nsamples).jld2") psiA overlap_psiA sampled_states probability number_states

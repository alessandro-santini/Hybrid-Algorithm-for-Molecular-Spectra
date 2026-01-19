using ITensorMPS
using ITensors
using JLD2

include("MPSutils.jl")
using .MPSutils

# Get the script directory and repository root
script_dir = @__DIR__
repo_root = joinpath(script_dir, "..", "..")

# --------------------------
# Parse command-line arguments
# --------------------------
molecule    = ARGS[1]
base        = ARGS[2]
n_electrons = parse(Int, ARGS[3])
n_orbitals  = parse(Int, ARGS[4])
R = parse(Float64, ARGS[5])

index_state_init = parse(Int, ARGS[6])
index_state_end  = parse(Int, ARGS[7])

nsamples = 1000000

# Data directories
data_dir = joinpath(repo_root, "data")
ground_states_dir = joinpath(data_dir, "ground_states")
sampled_dir = joinpath(data_dir, "sampled")

save_folder = joinpath(sampled_dir, "$(molecule)_$(base)_$(n_orbitals)o$(n_electrons)e_R$(R)", "sampled_basis")

# Create the directory if it doesn't exist
if !isdir(save_folder)
    mkpath(save_folder)
end

# --------------------------
# Load saved data
# --------------------------
@load joinpath(ground_states_dir, "$(molecule)_$(base)_$(n_orbitals)o$(n_electrons)e_R$(R).jld2") sites H_mpo
@load joinpath(sampled_dir, "$(molecule)_$(base)_$(n_orbitals)o$(n_electrons)e_R$(R)", "perturbed_state_nsamples$(nsamples).jld2") sampled_states psiA

# Read in the indices from command-line arguments.
n_sites = length(sites)

measurement_times = [0.,1.,2.,3.]

for index_state in index_state_init:index_state_end
    # For each initial state, we accumulate outcomes in state_basis.
    state_basis = Dict{NTuple{n_sites, Int}, Float64}()
    
    state_start = sampled_states[index_state]
    # Create the initial MPS from the product state.
    phi = productMPS(sites, state_start)
    orthogonalize!(phi, 1) 
    
    # Initialize the history of states.
    phi_history = Dict{String, Dict{Symbol, Any}}()
    counter = 0

    nsamples_evo  = 1000000
    threshold = 0.
    measured_states, probabilities = measure_states(phi, sites, nsamples_evo, threshold)
    
    phi_history["state_$(counter)"] = Dict(
        :time => 0.0,
        :mps => copy(phi),
        :samples => measured_states,
        :probabilities => probabilities,
        :nsamples => nsamples_evo
    )
    
    prev_time = 0.0
    for t in measurement_times
        dt = t - prev_time  # incremental time evolution
        
        # Perform time evolution for the time increment dt.
        phi = tdvp(H_mpo, -dt*1im, phi; reverse_step=true, time_step=-0.05im, maxdim=128,cutoff=1e-8, mindim=4,outputlevel=3,normalize=true, updater_kwargs=(;krylovdim=10,tol=1e-7,maxiter=4, verbosity=0))
        counter += 1
        
        normalize!(phi)
        orthogonalize!(phi, 1)
        
        measured_states, probabilities = measure_states(phi, sites, nsamples_evo, threshold)
        
        phi_history["state_$(counter)"] = Dict(
            :time => t,
            :mps => copy(phi),
            :samples => measured_states,
            :probabilities => probabilities,
            :nsamples => nsamples_evo
        )
        
        # For each measured outcome, add it to the state_basis if not already present.
        for (state, prob) in zip(measured_states, probabilities)
            state_tuple = Tuple(state)
            if !haskey(state_basis, state_tuple)
                phi_basis = productMPS(sites, state)
                state_basis[state_tuple] = inner(psiA, phi_basis)
            end
        end
        
        prev_time = t
        @info "Index state = $index_state; t = $t; cumulative basis size = $(length(state_basis))"
    end
    
    @save joinpath(save_folder, "evolved_states_bases_$index_state.jld2") state_basis phi_history
end

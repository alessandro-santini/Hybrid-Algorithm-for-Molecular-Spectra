module MPSutils

using ITensorMPS 
using ITensors

export measure_states, compute_von_neumann_entropy, compute_entanglement_entropy_chain

"""
    xlogy(x, y)

Compute x * log(y) with the special case x = 0.
"""
function xlogy(x, y)
    return x == 0.0 ? 0.0 : x * log(y)
end

"""
    measure_states(psi, sites, nsamples, threshold)

Measure a quantum state `psi` (represented as an MPS) a total of `nsamples` times,
and return the unique measurement outcomes (as a list of state vectors) whose probability
is greater than `threshold`. The function uses the basis states defined by `sites`.
"""
function measure_states(psi, sites, nsamples::Int, threshold::Real)
    n_sites = length(sites)
    unique_outcomes = Set{NTuple{n_sites, Int}}()
    for i in 1:nsamples
        outcome = Tuple(sample(psi))
        push!(unique_outcomes, outcome)
    end
    unique_states = [collect(state) for state in unique_outcomes]
    measured_states = Vector{Vector{Int}}()
    probabilities = Float64[]
    for state in unique_states
        phi_basis = productMPS(sites, state)
        amp = inner(phi_basis, psi)
        prob = abs(amp)^2
        if prob > threshold
            push!(measured_states, state)
            push!(probabilities, prob)
        end
    end
    return measured_states, probabilities
end

"""
    compute_von_neumann_entropy(psi, b)

Orthogonalizes `psi` at site `b`, performs an SVD, and computes the von Neumann entropy.
"""
function compute_von_neumann_entropy(psi, b)
    phi = copy(psi)
    phi = orthogonalize(phi, b)
    U, S, V = svd(phi[b], (linkinds(phi, b-1)..., siteinds(phi, b)...))
    SvN = 0.0
    for n in 1:dim(S, 1)
      p = S[n, n]^2
      SvN -= xlogy(p, p)
    end
    return SvN
end

"""
    compute_entanglement_entropy_chain(psi)

Compute the entanglement (von Neumann) entropy for every bipartition along the MPS `psi`.
It returns a vector of entropies corresponding to the cuts between sites 1–2, 2–3, …, (N–1)–N.
"""
function compute_entanglement_entropy_chain(psi)
    n_sites = length(psi)
    entropies = zeros(Float64, n_sites - 1)
    phi = copy(psi)
    for b in 2:n_sites
        phi = orthogonalize(phi, b)
        U, S, V = svd(phi[b], (linkinds(phi, b-1)..., siteinds(phi, b)...))
        SvN = 0.0
        for n in 1:dim(S, 1)
          p = S[n, n]^2
          SvN -= xlogy(p, p)
        end
        entropies[b-1] = SvN
    end
    return entropies
end

end # module
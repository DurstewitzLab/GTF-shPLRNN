using ..PLRNNs
using ..ObservationModels

"""
    generate(model, z₁, T, [S,])

Generate a trajectory of length `T` using model `ℳ` given initial condition `z₁`.

Returns a `T × M` matrix of generated orbits in latent (model) space. If `S` is provided,
then `S` must be a `T' × N` matrix of external inputs with `T' ≥ T`.
"""
function generate(ℳ, z₁::AbstractVector, T::Int)
    # trajectory placeholder
    Z = similar(z₁, T, length(z₁))

    # initial condition for model
    Z[1, :] .= z₁

    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :] .= ℳ(Z[t-1, :])
    end
    return Z
end

function generate(ℳ, z₁::AbstractVector, T::Int, S::AbstractMatrix)
    # trajectory placeholder
    Z = similar(z₁, T, length(z₁))

    # initial condition for model
    Z[1, :] .= z₁

    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :] .= ℳ(Z[t-1, :], S[t, :])
    end
    return Z
end

"""
    generate(ℳ, 𝒪, x₁, T, [S,])

Generate a trajectory of length `T` using latent model `ℳ` and observation model
`𝒪` given initial condition `x₁` in observation space. Estimates latent state by inversion
of `𝒪` and evolves latent state using `ℳ`.

Returns a `T × M` matrix of generated orbits in observation space. If `S` is provided,
then `S` must be a `T' × N` matrix of external inputs with `T' ≥ T`.
"""
function generate(ℳ, 𝒪::ObservationModel, x₁::AbstractVector, T::Int)
    z₁ = init_state(𝒪, x₁)
    Z = generate(ℳ, z₁, T)
    return permutedims(𝒪(Z'), (2, 1))
end

function generate(ℳ, 𝒪::ObservationModel, x₁::AbstractVector, T::Int, S::AbstractMatrix)
    z₁ = init_state(𝒪, x₁)
    Z = generate(ℳ, z₁, T, S)
    return permutedims(𝒪(Z'), (2, 1))
end

keep_connectivity_offdiagonal!(m, g) = nothing
keep_connectivity_offdiagonal!(m::Union{AbstractVanillaPLRNN, AbstractDendriticPLRNN}, g) =
    offdiagonal!(g[m.W])
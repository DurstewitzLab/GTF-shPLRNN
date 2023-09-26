using Flux: @functor
using Tullio

# abstract type
abstract type AbstractDendriticPLRNN <: AbstractPLRNN end
step(m::AbstractDendriticPLRNN, z::AbstractVecOrMat) = m.A .* z .+ m.W * Φ(m, z) .+ m.h

"""
    dendPLRNN

Implementation of the dendritic PLRNN introduced in 
https://proceedings.mlr.press/v162/brenner22a.html.
"""
struct dendPLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractDendriticPLRNN
    A::V
    W::M
    h::V
    α::V
    H::M
    C::Maybe{M}
end
@functor dendPLRNN

# initialization/constructor
function dendPLRNN(M::Int, B::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = randn(Float32, M, B)
    return dendPLRNN(A, W, h, α, H, nothing)
end

function dendPLRNN(M::Int, B::Int, X::AbstractMatrix)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = uniform_threshold_init((M, B), X)
    return dendPLRNN(A, W, h, α, H, nothing)
end

function dendPLRNN(M::Int, B::Int, K::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = randn(Float32, M, B)
    C = Flux.glorot_uniform(M, K)
    return dendPLRNN(A, W, h, α, H, C)
end

function dendPLRNN(M::Int, B::Int, X::AbstractMatrix, K::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = uniform_threshold_init((M, B), X)
    C = Flux.glorot_uniform(M, K)
    return dendPLRNN(A, W, h, α, H, C)
end

Φ(m::dendPLRNN, z::AbstractVecOrMat) = basis_expansion(z, m.α, m.H)

basis_expansion(z::Matrix{T}, α::Vector{T}, H::Matrix{T}) where {T} =
    @tullio z̃[m, s] := α[b] * relu(z[m, s] - H[m, b])

basis_expansion(z::Vector{T}, α::Vector{T}, H::Matrix{T}) where {T} =
    @tullio z̃[m] := α[b] * relu(z[m] - H[m, b])

function basis_expansion(
    z::AbstractMatrix{T},
    α::AbstractVector{T},
    H::AbstractMatrix{T},
) where {T}
    M = size(z, 1)
    α_ = reshape(α, 1, 1, :)
    H_ = reshape(H, M, 1, :)
    z_ = reshape(z, M, :, 1)
    z̃ = sum(α_ .* relu.(z_ .- H_), dims = 3)
    return reshape(z̃, M, :)
end

function basis_expansion(
    z::AbstractVector{T},
    α::AbstractVector{T},
    H::AbstractMatrix{T},
) where {T}
    α_ = reshape(α, 1, :)
    z_ = reshape(z, :, 1)
    z̃ = sum(α_ .* relu.(z_ .- H), dims = 2)
    return reshape(z̃, :)
end

function jacobian(m::dendPLRNN, z::AbstractVector{T}) where {T}
    α_ = reshape(m.α, 1, :)
    z_ = reshape(z, :, 1)
    ∂Φ∂z = Diagonal(vec(sum(α_ .* (z_ .> m.H), dims = 2)))
    return Diagonal(m.A) + m.W * ∂Φ∂z
end

"""
    clippedDendPLRNN

State clipping formulation the the `dendPLRNN`, which guarantees bounded orbits if
||A||₂ < 1, where ||⋅||₂ is the spectral norm.
"""
struct clippedDendPLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractDendriticPLRNN
    A::V
    W::M
    h::V
    α::V
    H::M
    C::Maybe{M}
end
@functor clippedDendPLRNN

# initialization/constructor
function clippedDendPLRNN(M::Int, B::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = randn(Float32, M, B)
    return clippedDendPLRNN(A, W, h, α, H, nothing)
end

function clippedDendPLRNN(M::Int, B::Int, X::AbstractMatrix)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = uniform_threshold_init((M, B), X)
    return clippedDendPLRNN(A, W, h, α, H, nothing)
end

function clippedDendPLRNN(M::Int, B::Int, K::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = randn(Float32, M, B)
    C = Flux.glorot_uniform(M, K)
    return clippedDendPLRNN(A, W, h, α, H, C)
end

function clippedDendPLRNN(M::Int, B::Int, X::AbstractMatrix, K::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = uniform_threshold_init((M, B), X)
    C = Flux.glorot_uniform(M, K)
    return clippedDendPLRNN(A, W, h, α, H, C)
end

Φ(m::clippedDendPLRNN, z::AbstractVecOrMat) = clipping_basis_expansion(z, m.α, m.H)

clipping_basis_expansion(z::Matrix{T}, α::Vector{T}, H::Matrix{T}) where {T} =
    @tullio z̃[m, s] := α[b] * (relu(z[m, s] - H[m, b]) - relu(z[m, s]))

clipping_basis_expansion(z::Vector{T}, α::Vector{T}, H::Matrix{T}) where {T} =
    @tullio z̃[m] := α[b] * (relu(z[m] - H[m, b]) - relu(z[m]))

function clipping_basis_expansion(
    z::AbstractMatrix{T},
    α::AbstractVector{T},
    H::AbstractMatrix{T},
) where {T}
    M = size(z, 1)
    α_ = reshape(α, 1, 1, :)
    H_ = reshape(H, M, 1, :)
    z_ = reshape(z, M, :, 1)
    z̃ = sum(α_ .* (relu.(z_ .- H_) .- relu.(z_)), dims = 3)
    return reshape(z̃, M, :)
end

function clipping_basis_expansion(
    z::AbstractVector{T},
    α::AbstractVector{T},
    H::AbstractMatrix{T},
) where {T}
    α_ = reshape(α, 1, :)
    z_ = reshape(z, :, 1)
    z̃ = sum(α_ .* (relu.(z_ .- H) .- relu.(z_)), dims = 2)
    return reshape(z̃, :)
end

function jacobian(m::clippedDendPLRNN, z::AbstractVector{T}) where {T}
    α_ = reshape(m.α, 1, :)
    z_ = reshape(z, :, 1)
    ∂Φ∂z = Diagonal(vec(sum(α_ .* ((z_ .> m.H) .- (z_ .> 0)), dims = 2)))
    return Diagonal(m.A) + m.W * ∂Φ∂z
end
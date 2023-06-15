using Flux: @functor

abstract type AbstractShallowPLRNN <: AbstractPLRNN end

"""
    shallowPLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractShallowPLRNN

`PLRNN` formulation with a linear autoregressive term and a 1-hidden-layer nonlinear
network (shPLRNN).
"""
struct shallowPLRNN{V <: AbstractVector, M <: AbstractMatrix} <:
               AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    C::Maybe{M}
end
@functor shallowPLRNN

# initialization/constructor
function shallowPLRNN(M::Int, hidden_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    return shallowPLRNN(A, W₁, W₂, h₁, h₂, nothing)
end

function shallowPLRNN(M::Int, hidden_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    C = Flux.glorot_uniform(M, K)
    return shallowPLRNN(A, W₁, W₂, h₁, h₂, C)
end

function initialize_Ws(M, hidden_dim)
    W₁ = uniform_init((M, hidden_dim))
    W₂ = uniform_init((hidden_dim, M))
    return W₁, W₂
end

step(m::shallowPLRNN, z::AbstractVecOrMat) =
    m.A .* z .+ m.W₁ * relu.(m.W₂ * z .+ m.h₂) .+ m.h₁

jacobian(m::shallowPLRNN, z::AbstractVector) =
    Diagonal(m.A) + m.W₁ * Diagonal(m.W₂ * z .> -m.h₂) * m.W₂

"""
    clippedShallowPLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractShallowPLRNN

Clipped formulation of the `shPLRNN` with a clipped piecewise linear nonlinearity,
which guarantees bounded orbits if ||A||₂ < 1, where ||⋅||₂ is the spectral
norm.
"""
struct clippedShallowPLRNN{V <: AbstractVector, M <: AbstractMatrix} <:
               AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    C::Maybe{M}
end
@functor clippedShallowPLRNN

# initialization/constructor
function clippedShallowPLRNN(M::Int, hidden_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    return clippedShallowPLRNN(A, W₁, W₂, h₁, h₂, nothing)
end

function clippedShallowPLRNN(M::Int, hidden_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    C = Flux.glorot_uniform(M, K)
    return clippedShallowPLRNN(A, W₁, W₂, h₁, h₂, C)
end

function step(m::clippedShallowPLRNN, z::AbstractVecOrMat)
    W₂z = m.W₂ * z
    return m.A .* z .+ m.W₁ * (relu.(W₂z .+ m.h₂) .- relu.(W₂z)) .+ m.h₁
end

function jacobian(m::clippedShallowPLRNN, z::AbstractVector)
    W₂z = m.W₂ * z
    return Diagonal(m.A) + m.W₁ * Diagonal((W₂z .> -m.h₂) .- (W₂z .> 0)) * m.W₂
end

norm_upper_bound(m::Union{shallowPLRNN, clippedShallowPLRNN}) =
    maximum(abs, m.A) + opnorm(m.W₁) * opnorm(m.W₂)
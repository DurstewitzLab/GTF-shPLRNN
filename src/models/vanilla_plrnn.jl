using Flux: @functor
using Statistics: mean
using ThreadsX
using ForwardDiff

using ..ObservationModels: ObservationModel, init_state

# abstract type
abstract type AbstractPLRNN end
(m::AbstractPLRNN)(z::AbstractVecOrMat) = step(m, z)
(m::AbstractPLRNN)(z::AbstractVecOrMat, s::AbstractVecOrMat) = step(m, z, s)
step(m::AbstractPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat) = step(m, z) + m.C * s
jacobian(m::AbstractPLRNN, z::AbstractVector) = ForwardDiff.jacobian(m, z)
jacobian(m::AbstractPLRNN, z::AbstractMatrix) = jacobian.([m], eachcol(z))

abstract type AbstractVanillaPLRNN <: AbstractPLRNN end

struct PLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractVanillaPLRNN
    A::V
    W::M
    h::V
    C::Maybe{M}
end
@functor PLRNN

# initialization/constructor
PLRNN(M::Int) = PLRNN(initialize_A_W_h(M)..., nothing)
PLRNN(M::Int, K::Int) = PLRNN(initialize_A_W_h(M)..., Flux.glorot_uniform(M, K))

"""
    step(model, z)

Evolve `z` in time for one step according to the model `m` (equation).

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
step(m::PLRNN, z::AbstractVecOrMat) = m.A .* z .+ m.W * relu.(z) .+ m.h
jacobian(m::PLRNN, z::AbstractVector) = Diagonal(m.A) + m.W * Diagonal(z .> 0)

# mean-centered PLRNN
struct mcPLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractVanillaPLRNN
    A::V
    W::M
    h::V
    C::Maybe{M}
end
@functor mcPLRNN

# initialization/constructor
mcPLRNN(M::Int) = mcPLRNN(initialize_A_W_h(M)..., nothing)
mcPLRNN(M::Int, K::Int) = mcPLRNN(initialize_A_W_h(M)..., Flux.glorot_uniform(M, K))

mean_center(z::AbstractVecOrMat) = z .- mean(z, dims = 1)
step(m::mcPLRNN, z::AbstractVecOrMat) = m.A .* z .+ m.W * relu.(mean_center(z)) .+ m.h
function jacobian(m::mcPLRNN, z::AbstractVector)
    M, type = length(z), eltype(z)
    ℳ = type(1 / M) * (M * I - ones(type, M, M))
    return Diagonal(m.A) + m.W * Diagonal(ℳ * z .> 0) * ℳ
end
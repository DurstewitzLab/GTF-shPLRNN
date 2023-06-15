using LinearAlgebra
using CUDA

using ..PLRNNs
using ..ObservationModels

"""
    mar_loss()

Compute the Manifold Attractor Regularization (MAR) loss.
"""
function mar_loss(m::AbstractPLRNN, κ::Float32 = 0.5f0, λ::Float32 = 1.0f-3; p::Real = 2)
    M = length(m.A)
    # number of regularized states
    Mᵣ = Int(floor(κ * M))

    Lᴬ = norm(1 .- m.A[M-Mᵣ+1:end], p)
    Lᵂ = norm(m.W[M-Mᵣ+1:end, :], p)
    Lʰ = norm(m.h[M-Mᵣ+1:end], p)

    return λ * (Lᴬ + Lᵂ + Lʰ)
end

"""
    AR_convergence_loss(m::AbstractPLRNN, λ::Float32, p::Real = 2)

Pushes entries of `A` to stay below 1, avoiding divergent dynamics.
"""
function AR_convergence_loss(m::AbstractPLRNN, λ::Float32, p::Real = 2, ϵ::Float32 = 1.0f-4)
    @assert ϵ > 0.0f0
    loss = norm(relu.(m.A .- (1 - ϵ)), p)
    return λ * loss
end

"""
    regularize(model, λ; penalty)

Weight regularization. Defaults to L2 penalization.
"""
regularize(O::ObservationModel, λ::Float32; penalty = l2_penalty) =
    λ * sum(penalty, Flux.params(O))

regularize(O::Identity, λ::Float32; penalty = l2_penalty) = λ * penalty(O.L)

"""
Condition number regularization for the `Affine` observation model. Pulls the condition number
of `B` towards 1, ensuring invertibility which is needed for proper estimation of forcing signals.
"""
function regularize(O::Affine, λ::Float32; ϵ::Float32 = 1.0f-8)
    S = svd(O.B).S
    return λ * (1.0f0 - S[1] / (S[end] + ϵ))^2
end

function regularize(m::AbstractShallowPLRNN, λ::Float32; penalty = l2_penalty)
    A_reg = penalty(1 .- m.A)
    W₁_reg = penalty(m.W₁)
    W₂_reg = penalty(m.W₂)
    h₁_reg = penalty(m.h₁)
    h₂_reg = penalty(m.h₂)
    C_reg = penalty(m.C)
    return λ * (A_reg + W₁_reg + W₂_reg + h₁_reg + h₂_reg + C_reg)
end

function regularize(m::AbstractDendriticPLRNN, λ::Float32; penalty = l2_penalty)
    A_reg = penalty(1 .- m.A)
    W_reg = penalty(m.W)
    h_reg = penalty(m.h)
    α_reg = penalty(m.α)
    C_reg = penalty(m.C)
    return λ * (A_reg + W_reg + h_reg + α_reg + C_reg)
end

function regularize(m::AbstractVanillaPLRNN, λ::Float32; penalty = l2_penalty)
    A_reg = penalty(1 .- m.A)
    W_reg = penalty(m.W)
    h_reg = penalty(m.h)
    return λ * (A_reg + W_reg + h_reg)
end

regularize(m, args...; kwargs...) =
    throw("Regularization for model type $(typeof(m)) not implemented yet!")

l2_penalty(θ) = isnothing(θ) ? 0 : sum(abs2, θ)
l1_penalty(θ) = isnothing(θ) ? 0 : sum(abs, θ)

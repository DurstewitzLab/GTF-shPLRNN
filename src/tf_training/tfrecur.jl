using Flux

using ..PLRNNs
using ..Utilities
using ..ObservationModels

abstract type AbstractTFRecur end
Flux.trainable(tfrec::AbstractTFRecur) = (tfrec.model, tfrec.O)

(tfrec::AbstractTFRecur)(X::AbstractArray{T, 3}, Ẑ::AbstractArray{T, 3}) where {T} =
    forward(tfrec, X, Ẑ)

(tfrec::AbstractTFRecur)(
    X::AbstractArray{T, 3},
    Ẑ::AbstractArray{T, 3},
    S::AbstractArray{T, 3},
) where {T} = forward(tfrec, X, Ẑ, S)

# compute forcing signals using observation model
estimate_forcing_signals(tfrec::AbstractTFRecur, X::AbstractArray{T, 3}) where {T} =
    apply_inverse(tfrec.O, X)

"""
Inspired by `Flux.Recur` struct, which by default has no way
of incorporating teacher forcing.

This is just a convenience wrapper around stateful models,
to be used during training.
"""
mutable struct TFRecur{ℳ <: AbstractPLRNN, M <: AbstractMatrix, 𝒪 <: ObservationModel} <:
               AbstractTFRecur
    # stateful model, e.g. PLRNN
    model::ℳ
    # observation model
    O::𝒪
    # state of the model
    z::M
    # forcing interval
    const τ::Int
end
Flux.@functor TFRecur

function forward(tfrec::TFRecur, X::AbstractArray{T, 3}, Ẑ::AbstractArray{T, 3}) where {T}
    T_ = size(X, 3)

    # initialize latent state
    tfrec.z = init_state(tfrec.O, @view(X[:, :, 1]))

    # process sequence of forcing signals Ẑ
    Z = [tfrec(ẑ, t) for (ẑ, t) ∈ zip(Flux.eachlastdim(Ẑ), 2:T_)]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

function forward(
    tfrec::TFRecur,
    X::AbstractArray{T, 3},
    Ẑ::AbstractArray{T, 3},
    S::AbstractArray{T, 3},
) where {T}
    T_ = size(X, 3)

    # initialize latent state
    tfrec.z = init_state(tfrec.O, @view(X[:, :, 1]))

    # process sequence of forcing signals Ẑ
    Z = [
        tfrec(ẑ, s, t) for (ẑ, s, t) ∈
        zip(Flux.eachlastdim(Ẑ), Flux.eachlastdim(@view(S[:, :, 2:end])), 2:T_)
    ]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

function (tfrec::TFRecur)(x::AbstractMatrix, t::Int)
    # determine if it is time to force the model
    z = tfrec.z

    # perform one step using the model, update model state
    z = tfrec.model(z)

    # force
    z̃ = (t - 1) % tfrec.τ == 0 ? force(z, x) : z
    tfrec.z = z̃
    return z
end

function (tfrec::TFRecur)(x::AbstractMatrix, s::AbstractMatrix, t::Int)
    # determine if it is time to force the model
    z = tfrec.z

    # perform one step using the model, update model state
    z = tfrec.model(z, s)

    # force
    z̃ = (t - 1) % tfrec.τ == 0 ? force(z, x) : z
    tfrec.z = z̃
    return z
end

"""
    GTFRecur(model, O, z, α)

Generalized teacher forcing wrapper.
"""
mutable struct GTFRecur{
    ℳ <: AbstractPLRNN,
    𝒪 <: ObservationModel,
    M <: AbstractMatrix,
    T <: AbstractFloat,
} <: AbstractTFRecur
    model::ℳ
    # ObservationModel
    O::𝒪
    # state of the model
    z::M
    # forcing α
    α::T
end
Flux.@functor GTFRecur

"""
    forward(GTFRecur, X, Ẑ, [S,])

Forward pass using Generalized Teacher Forcing (GTF). If the latent dimension `M` of
the RNN is larger than the dimension of the provided forcing signals `Ẑ`, 
partial teacher forcing of the first `size(Ẑ, 1)` neurons is
used. Initializing latent state `z₁` is performed by inversion of the observation model applied to data `X`.
Optionally, the forward pass incorporates external inputs `S`.

Returns a `M × S × T` array of latent states (predictions).
"""
function forward(tfrec::GTFRecur, X::AbstractArray{T, 3}, Ẑ::AbstractArray{T, 3}) where {T}
    # initialize latent state
    tfrec.z = init_state(tfrec.O, @view(X[:, :, 1]))

    # process sequence of forcing signals Ẑ
    Z = [tfrec(z) for z ∈ Flux.eachlastdim(Ẑ)]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

function forward(
    tfrec::GTFRecur,
    X::AbstractArray{T, 3},
    Ẑ::AbstractArray{T, 3},
    S::AbstractArray{T, 3},
) where {T}
    # initialize latent state
    tfrec.z = init_state(tfrec.O, @view(X[:, :, 1]))

    # process sequence of forcing signals Ẑ
    Z = [
        tfrec(ẑ, s) for
        (ẑ, s) ∈ zip(Flux.eachlastdim(Ẑ), Flux.eachlastdim(@view(S[:, :, 2:end])))
    ]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

function (tfrec::GTFRecur)(ẑ::AbstractMatrix)
    z = tfrec.z
    D, M = size(ẑ, 1), size(z, 1)
    z = tfrec.model(z)
    # gtf
    z̃ = force(@view(z[1:D, :]), ẑ, tfrec.α)
    z̃ = (D == M) ? z̃ : force(z, z̃)

    tfrec.z = z̃
    return z
end

function (tfrec::GTFRecur)(ẑ::AbstractMatrix, s::AbstractMatrix)
    z = tfrec.z
    D, M = size(ẑ, 1), size(z, 1)
    z = tfrec.model(z, s)
    # gtf
    z̃ = force(@view(z[1:D, :]), ẑ, tfrec.α)
    z̃ = (D == M) ? z̃ : force(z, z̃)

    tfrec.z = z̃
    return z
end
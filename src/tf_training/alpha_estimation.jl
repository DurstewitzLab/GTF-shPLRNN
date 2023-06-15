using StatsBase: geomean
using ThreadsX

# available α estimation methods
const α_METHODS = [
    "constant",
    "jacobian_upper_bound",
    "batch_maximum",
    "product_upper_bound",
    "explog_approx",
    "arithmetic_mean",
]

"""
    compute_α(tfrec, Ẑ, est_method)

Compute `α` given recursion wrapper `tfrec`, forcing signals `Ẑ` and
an `α` estimation method `est_method`. `Ẑ` is an `M × S × T` array where `M`
denotes the latent dimension, `S` is the batch dimension and `T` is the time dimension.
Throws an error if `est_method` is not one of the methods in `α_METHODS`.

Returns `α` and spectral norm `𝒩` used to compute it.
"""
function compute_α(tfrec, Ẑ::AbstractArray{T, 3}, est_method::String) where {T}
    # model
    ℳ = tfrec.model

    # select method
    if est_method == "jacobian_upper_bound"
        α, 𝒩 = compute_α_jacobian_upper_bound(ℳ)
    elseif est_method == "batch_maximum"
        α, 𝒩 = compute_α_batch_maximum(ℳ, Ẑ)
    elseif est_method == "product_upper_bound"
        α, 𝒩 = compute_α_product_upper_bound(ℳ, Ẑ)
    elseif est_method == "explog_approx"
        α, 𝒩 = compute_α_explog_approx(ℳ, Ẑ)
    elseif est_method == "arithmetic_mean"
        α, 𝒩 = compute_α_arithmetic_mean(ℳ, Ẑ)
    elseif est_method == "constant"
        α, 𝒩 = tfrec.α, zero(tfrec.α)
    else
        error(
            "Unknown α estimation method: $(est_method). Please choose one of $(α_METHODS).",
        )
    end
    return α, 𝒩
end

"""
    compute_α_jacobian_upper_bound(ℳ)
	
Compute upper bound on `α` via parameters of model `ℳ`.
"""
function compute_α_jacobian_upper_bound(ℳ)
    𝒩 = norm_upper_bound(ℳ)
    return opt_α(𝒩), 𝒩
end

"""
    compute_α_batch_maximum(ℳ, Ẑ)

Compute `α` via the maximum Jacobian spectral norm of a given batch of sequences.
"""
function compute_α_batch_maximum(ℳ, Ẑ::AbstractArray{T, 3}) where {T}
    # flatten batch and time dimensions
    Ẑ_flat = reshape(Ẑ, size(Ẑ, 1), :)

    # compute jacobian spectral norms
    𝒩s = ThreadsX.map(z -> opnorm(jacobian(ℳ, z)), eachcol(Ẑ_flat))

    # max spectral norm across entire batch
    𝒩 = maximum(𝒩s)
    return opt_α(𝒩), 𝒩
end

"""
    compute_α_explog_approx(ℳ, Ẑ)

Compute `α` using and the "exp-log" approximation.
"""
function compute_α_explog_approx(ℳ, Ẑ::AbstractArray{T, 3}) where {T}
    # geomean of spectral norms along sequences
    𝒩s = ThreadsX.map(seq -> opnorm(geomean(jacobian(ℳ, seq))), eachslice(Ẑ, dims = 2))
    𝒩 = maximum(𝒩s)
    return opt_α(𝒩), 𝒩
end

"""
    compute_α_product_upper_bound(ℳ, Ẑ)

Compute `α` using the upper bound on the Jacobian product norm along
a training sequence.
"""
function compute_α_product_upper_bound(ℳ, Ẑ::AbstractArray{T, 3}) where {T}
    # geomean of spectral norms along sequences
    𝒩s = ThreadsX.map(seq -> geomean(opnorm.(jacobian(ℳ, seq))), eachslice(Ẑ, dims = 2))
    𝒩 = maximum(𝒩s)
    return opt_α(𝒩), 𝒩
end

"""
    compute_α_arithmetic_mean(ℳ, Ẑ)

Compute `α` using the arithmetic mean of Jacobians along a training sequence.
"""
function compute_α_arithmetic_mean(ℳ, Ẑ::AbstractArray{T, 3}) where {T}
    # geomean of spectral norms along sequences
    𝒩s = ThreadsX.map(seq -> opnorm(mean(jacobian(ℳ, seq))), eachslice(Ẑ, dims = 2))
    𝒩 = maximum(𝒩s)
    return opt_α(𝒩), 𝒩
end

"""
    opt_α(𝒩)

Compute optimal GTF `α` for given spectral norm `𝒩`.
"""
opt_α(𝒩) = max(0, 1 - 1 / 𝒩)
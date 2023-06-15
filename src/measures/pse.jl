using FFTW: rfft
using ImageFiltering: Kernel, imfilter!
using StatsBase: standardize

"""
    power_spectrum_error(X, X̃)

Compute the power spectrum error between ground truth `X` and
simulated trajectory `X̃`.
"""
function power_spectrum_error(
    X::AbstractMatrix{T},
    X̃::AbstractMatrix{T},
    σ::T = zero(T);
    return_unreduced::Bool = false,
) where {T}
    @assert all(size(X) .== size(X̃))

    X_ps = normalized_and_smoothed_power_spectrum(X, σ)
    X̃_ps = normalized_and_smoothed_power_spectrum(X̃, σ)
    hd = hellinger_distance(X_ps, X̃_ps)

    return return_unreduced ? (mean(hd), hd) : mean(hd)
end

"""
    power_spectrum_correlation(X, X̃)

[LEGACY] Compute the power spectrum correlation between ground truth `X` and
simulated trajectory `X̃`. 
"""
function power_spectrum_correlation(
    X::AbstractMatrix{T},
    X̃::AbstractMatrix{T},
    σ::T;
    return_unreduced::Bool = false,
) where {T}
    @assert all(size(X) .== size(X̃))
    X_ps = normalized_and_smoothed_power_spectrum(X, σ)
    X̃_ps = normalized_and_smoothed_power_spectrum(X̃, σ)

    r = cor.(eachcol(X_ps), eachcol(X̃_ps))
    return return_unreduced ? (mean(r), r) : mean(r)
end

function normalized_and_smoothed_power_spectrum(
    X::AbstractMatrix{T},
    σ::T;
    normalize::Bool = true,
) where {T}
    # standardize
    X_ = standardize(ZScoreTransform, X, dims = 1)

    # compute PS
    PS = abs2.(rfft(X_, 1))

    if σ > zero(T)
        # smooth PS
        smooth_dims!(PS, σ)

        # removes very small (negligable) negative values (artifacts) after smoothing
        PS[PS.<zero(T)] .= zero(T)
    end

    # normalize PS
    normalize ? PS ./= sum(PS, dims = 1) : nothing

    return PS
end

"""
    smooth_dims!(X::AbstractMatrix, σ::Real)

Smooth every observation (row) of matrix `X` using a gaussian kernel 
with standard deviation `σ`. Filtering parameters adjusted such that
method aligns with default behaviour of python's `scipy.gaussian_filter1d`,
specifically padding mode and kernel length.
"""
function smooth_dims!(X::AbstractMatrix{T}, σ::T) where {T}
    # set kernel radius to be equal to python implementation
    # which is 4 * standard deviation instead of default of 2
    # used by ImageFiltering.Kernel.gaussian()
    l = 2 * 4 * σ
    # kernel length has to be odd
    l = l % 2 == 0 ? l + 1 : l
    κ = Kernel.gaussian((σ,), (Int(l),))

    # reflect padding as used by scipy.gaussian_filter1d
    @views for i ∈ axes(X, 2)
        imfilter!(X[:, i], X[:, i], κ, "reflect")
    end
end

hellinger_distance(X::AbstractVecOrMat{T}, X̃::AbstractVecOrMat{T}) where {T} =
    one(T) / convert(T, √2) .* sqrt.(sum((sqrt.(X) .- sqrt.(X̃)) .^ 2, dims = 1))

using Tullio
using StatsBase: kldivergence

using ..Utilities

laplace_smoothing(hist::AbstractArray, α = 1.0f-5) =
    (hist .+ α) ./ (sum(hist) .+ α .* prod(size(hist)))

"""
    state_space_divergence(X, X̃)

Compute the ``D_{stsp}`` measure given two time series `X` and `X̃` using a binning approach. 

Measures the distance in probability space defined by visitation frequencies of trajectories
in state (observation) space using the Kullback-Leibler (KL) divergence. Assumes reference data 
`X` is a `T x N` Matrix, and `X̃` a `T' x N` Matrix. 

## Binning
This function first approximates a distribution over state space by constructing an 'N'-dimensional histogram
for each time series. The distance between the resulting probability mass functions is then computed by the KL
divergence. 

The `"default"` binning scheme uses the state space volume which completely includes the observed attractor.
If the `"legacy"` binning scheme is used, the histograms are constructed using the state space volume defined
by `[-2σ, 2σ]ᴺ` [Brenner et al (2022)].

## GMM
Instead of binning state space, Gaussian mixture models are fit to both ground truth and generated trajectories.
Reported is the KL divergence of the GMMs, which is approximated via a Monte-Carlo sampling approach introduced by
Hershey & Olsen (2007) [ https://ieeexplore.ieee.org/document/4218101 ]. The GMMs are constructed via assigning
a Gaussian to each data point with means `X`/`X̃` and diagonal covariance `Σ=I*σ²`.
"""
function state_space_divergence(
    X::Matrix,
    X̃::Matrix,
    n_bins::Int;
    binning::String = "default",
)
    N = size(X, 2)
    Ñ = size(X̃, 2)
    @assert N == Ñ

    # standard deviation
    σ = std(X, dims = 1)

    if binning == "default"
        lo = minimum(X, dims = 1) .- 0.1f0 .* σ
        hi = maximum(X, dims = 1) .+ 0.1f0 .* σ
    elseif binning == "legacy"
        hi = 2.0f0 .* σ
        lo = -hi
    end

    # bin edges
    bin_edges = Tuple(range.(lo, hi, n_bins + 1))

    # compute histograms (memory intense!)
    hist_true = fit(Histogram{UInt16}, Tuple(eachcol(X)), bin_edges)
    hist_gen = fit(Histogram{UInt16}, Tuple(eachcol(X̃)), bin_edges)

    # check if generated hist is empty
    if sum(hist_gen.weights) == 0
        return NaN
    end

    p = reshape(laplace_smoothing(hist_true.weights), :)
    q = reshape(laplace_smoothing(hist_gen.weights), :)

    return kldivergence(p, q)
end

@inbounds function state_space_divergence(
    X::AbstractMatrix,
    X̃::AbstractMatrix,
    σ²::Real;
    max_T::Int = 10000,
    mc_samples::Int = 1000,
)
    T, N = size(X)
    Ñ = size(X̃, 2)
    @assert N == Ñ
    @assert σ² >= 0.0f0
    σ = convert(Float32, √σ²)

    # time steps used for computation
    T̃ = min(T, max_T)
    X_ = X[1:T̃, :]
    X̃_ = X̃[1:T̃, :]

    # draw samples
    tᵢ = rand(1:T̃, mc_samples)
    samples = @views X_[tᵢ, :] .+ σ .* randn_like(X_[tᵢ, :])

    p = gmm_likelihood_diagonal_Σ(samples, X_, σ)
    q = gmm_likelihood_diagonal_Σ(samples, X̃_, σ)

    p, q, n = filter_outliers(p, q)
    #println("OUTLIERS: $n")

    return mean(log.(p) .- log.(q))
end

function gmm_likelihood_diagonal_Σ(samples::AbstractMatrix, μ::AbstractMatrix, σ::Float32)
    T, N = size(μ)
    # det and precision
    sqrt_det_Σ = σ^N
    invΣ = 1 / σ^2

    # compute likelihood
    @tullio E[i, t] := (samples[i, n] - μ[t, n])^2 * invΣ
    #E = dropdims(sum(D.^2 .* invΣ, dims=3), dims=3)
    L = @. exp(-0.5f0 * E) / sqrt_det_Σ
    return sum(L, dims = 2) ./ T
end

function filter_outliers(p::AbstractVecOrMat, q::AbstractVecOrMat)
    mask = p .> 0.0f0 .&& q .> 0.0f0
    return p[mask], q[mask], sum(.!mask)
end

decide_on_measure(scaling::Real, bins::Int, N::Int) =
    N < 6 ? (bins, "(BIN)") : (scaling, "(GMM)")

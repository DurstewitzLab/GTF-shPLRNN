"""
    normalized_positive_definite(M)

Build a positive definite matrix with maximum eigenvalue of 1.

RNN weight matrix initialized proposed by Talathi & Vartak (2016)
[ https://arxiv.org/abs/1511.03771 ].
"""
function normalized_positive_definite(M::Int)
    R = randn(Float32, M, M)
    K = R'R ./ M + I
    λ = maximum(abs.(eigvals(K)))
    return K ./ λ
end

function uniform_init(shape::Tuple; eltype::Type{T} = Float32) where {T <: AbstractFloat}
    @assert length(shape) < 3
    din = Float32(shape[end])
    r = 1 / √din
    return uniform(shape, -r, r)
end

"""
    uniform_threshold_init(shape, Dataset)

Return a Matrix of shape `shape` filled with values within the 
minimum and maximum extends of `D`.

Used to initialize basis thresholds `H` of the dendritic PLRNN.
"""
function uniform_threshold_init(shape::Tuple{Int, Int}, X::AbstractMatrix)
    # compute minima and maxima of dataset
    lo, hi = minimum(X), maximum(X)
    return uniform(shape, lo, hi)
end

function initialize_A_W_h(M::Int)
    AW = normalized_positive_definite(M)
    A, W = diag(AW), offdiagonal(AW)
    h = zeros(Float32, M)
    return A, W, h
end

initialize_L(M::Int, N::Int) =
    if M == N
        L = nothing
    else
        L = uniform_init((M - N, N))
    end
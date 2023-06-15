using CUDA
using Flux

num_params(m) = sum(length, Flux.params(m))

offdiagonal(X::AbstractMatrix) = X - Diagonal(X)
@inbounds offdiagonal!(X::AbstractMatrix) = X[diagind(X)] .= zero(eltype(X))

uniform(a, b) = rand(eltype(a)) * (b - a) + a
uniform(size, a, b) = rand(eltype(a), size) .* (b - a) .+ a

randn_like(X::AbstractArray{T, N}) where {T, N} = randn(T, size(X)...)
randn_like(X::CUDA.CuArray{T, N, B}) where {T, N, B} = CUDA.randn(T, size(X)...)
add_gaussian_noise!(X::AbstractArray{T, N}, noise_level::T) where {T, N} =
    X .+= noise_level .* randn_like(X)
add_gaussian_noise(X::AbstractArray{T, N}, noise_level::T) where {T, N} =
    X .+ noise_level .* randn_like(X)
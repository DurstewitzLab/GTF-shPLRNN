using ChainRulesCore: NoTangent, @thunk
import ChainRulesCore

@inbounds """
    force(z, x)

Replace the first `N = dim(x)` dimensions of `z` with `x` (id-TF).

Supplied with custom backward `ChainRulesCore.rrule`.
"""
function force(z::AbstractMatrix, x::AbstractMatrix)
    N = size(x, 1)
    return [x; z[N+1:end, :]]
end

@inbounds function ChainRulesCore.rrule(
    ::typeof(force),
    z::AbstractMatrix,
    x::AbstractMatrix,
)
    N = size(x, 1)
    function force_pullback(ΔΩ)
        ∂x = ΔΩ[1:N, :]
        # in-place here for speed
        @views ΔΩ[1:N, :] .= 0
        return (NoTangent(), ΔΩ, ∂x)
    end
    return force(z, x), force_pullback
end

"""
    force(z, z̄, α)

Generalized Teacher Forcing (GTF) with forcing parameter `α` as mentioned in
"Bifurcations in the learning of recurrent neural networks" [Doya (1992), https://ieeexplore.ieee.org/document/230622 ].
"""
force(z::AbstractMatrix{T}, z̄::AbstractMatrix{T}, α::T) where {T} = @. (1 - α) * z + α * z̄

function ChainRulesCore.rrule(
    ::typeof(force),
    z::AbstractMatrix{T},
    z̄::AbstractMatrix{T},
    α::T,
) where {T}
    force_pullback(ΔΩ) =
        (NoTangent(), @thunk((1 .- α) .* ΔΩ), @thunk(α .* ΔΩ), @thunk(sum((z̄ .- z) .* ΔΩ)))
    return force(z, z̄, α), force_pullback
end

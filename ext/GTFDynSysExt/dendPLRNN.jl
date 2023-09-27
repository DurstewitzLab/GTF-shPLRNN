using GTF, Flux, LinearAlgebra
using DynamicalSystems: DeterministicIteratedMap, TangentDynamicalSystem

function GTF.Utilities.wrap_as_dynamical_system(model::dendPLRNN, z₁ = nothing)
    !isnothing(model.C) && throw("Non-autonomous shallowPLRNNs are not supported.")

    # params
    A, W, h, α, H = cast_params_to_float64(Flux.params(model))

    # initial state
    z = isnothing(z₁) ? zeros(length(A)) : Float64.(z₁)

    ds = DeterministicIteratedMap(dendPLRNN_step!, z, (A, W, h, α, H))
    return TangentDynamicalSystem(ds, J = dendPLRNN_jacobian!)
end

# step & jacobian for DynamicalSystems.jl support
function dendPLRNN_step!(out, z, p, n)
    A, W, h, α, H = p
    out .= A .* z .+ W * GTF.PLRNNs.basis_expansion(z, α, H) .+ h
    return nothing
end

function dendPLRNN_jacobian!(out, z, p, n)
    A, W, h, α, H = p
    α_ = reshape(α, 1, :)
    z_ = reshape(z, :, 1)
    ∂Φ∂z = Diagonal(vec(sum(α_ .* (z_ .> H), dims = 2)))
    out .= Diagonal(A) + W * ∂Φ∂z
    return nothing
end

function GTF.Utilities.wrap_as_dynamical_system(model::clippedDendPLRNN, z₁ = nothing)
    !isnothing(model.C) && throw("Non-autonomous shallowPLRNNs are not supported.")

    # params
    A, W, h, α, H = cast_params_to_float64(Flux.params(model))

    # initial state
    z = isnothing(z₁) ? zeros(length(A)) : Float64.(z₁)

    ds = DeterministicIteratedMap(clippedDendPLRNN_step!, z, (A, W, h, α, H))
    return TangentDynamicalSystem(ds, J = clippedDendPLRNN_jacobian!)
end

# step & jacobian for DynamicalSystems.jl support
function clippedDendPLRNN_step!(out, z, p, n)
    A, W, h, α, H = p
    out .= A .* z .+ W * GTF.PLRNNs.clipping_basis_expansion(z, α, H) .+ h
    return nothing
end

function clippedDendPLRNN_jacobian!(out, z, p, n)
    A, W, h, α, H = p
    α_ = reshape(α, 1, :)
    z_ = reshape(z, :, 1)
    ∂Φ∂z = Diagonal(vec(sum(α_ .* ((z_ .> H) .- (z_ .> 0)), dims = 2)))
    out .= Diagonal(A) + W * ∂Φ∂z
    return nothing
end
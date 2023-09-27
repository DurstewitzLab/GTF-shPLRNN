using GTF, Flux, LinearAlgebra
using DynamicalSystems: DeterministicIteratedMap, TangentDynamicalSystem

function GTF.Utilities.wrap_as_dynamical_system(model::PLRNN, z₁ = nothing)
    !isnothing(model.C) && throw("Non-autonomous shallowPLRNNs are not supported.")

    # params
    A, W, h = cast_params_to_float64(Flux.params(model))

    # initial state
    z = isnothing(z₁) ? zeros(length(A)) : Float64.(z₁)

    ds = DeterministicIteratedMap(PLRNN_step!, z, (A, W, h))
    return TangentDynamicalSystem(ds, J = PLRNN_jacobian!)
end

# step & jacobian for DynamicalSystems.jl support
function PLRNN_step!(out, z, p, n)
    A, W, h = p
    out .= A .* z .+ W * relu.(z) .+ h
    return nothing
end

function PLRNN_jacobian!(out, z, p, n)
    A, W, h = p
    out .= Diagonal(A) + W * Diagonal(z .> 0)
    return nothing
end

function GTF.Utilities.wrap_as_dynamical_system(model::mcPLRNN, z₁ = nothing)
    !isnothing(model.C) && throw("Non-autonomous shallowPLRNNs are not supported.")

    # params
    A, W, h = cast_params_to_float64(Flux.params(model))

    # initial state
    z = isnothing(z₁) ? zeros(length(A)) : Float64.(z₁)

    ds = DeterministicIteratedMap(mcPLRNN_step!, z, (A, W, h))
    return TangentDynamicalSystem(ds, J = mcPLRNN_jacobian!)
end

# step & jacobian for DynamicalSystems.jl support
function mcPLRNN_step!(out, z, p, n)
    A, W, h = p
    out .= A .* z .+ W * relu.(GTF.PLRNNs.mean_center(z)) .+ h
    return nothing
end

function mcPLRNN_jacobian!(out, z, p, n)
    A, W, h = p
    M = length(z)
    ℳ = (M * I - ones(eltype(z), M, M)) / M
    out .= Diagonal(A) + W * Diagonal(ℳ * z .> 0) * ℳ
    return nothing
end
using GTF, Flux, LinearAlgebra
using DynamicalSystems: DeterministicIteratedMap, TangentDynamicalSystem

function GTF.Utilities.wrap_as_dynamical_system(model::shallowPLRNN, z₁ = nothing)
    non_autonomous_error(model)

    # params
    A, W₁, W₂, h₁, h₂ = cast_params_to_float64(Flux.params(model))

    # initial state
    z = isnothing(z₁) ? zeros(length(A)) : Float64.(z₁)

    ds = DeterministicIteratedMap(shallowPLRNN_step!, z, (A, W₁, W₂, h₁, h₂))
    return TangentDynamicalSystem(ds, J = shallowPLRNN_jacobian!)
end

# step & jacobian for DynamicalSystems.jl support
function clippedShallowPLRNN_step!(out, z, p, n)
    A, W₁, W₂, h₁, h₂ = p
    W₂z = W₂ * z
    out .= A .* z .+ W₁ * (relu.(W₂z .+ h₂) .- relu.(W₂z)) .+ h₁
    return nothing
end

function clippedShallowPLRNN_jacobian!(out, z, p, n)
    A, W₁, W₂, h₁, h₂ = p
    W₂z = W₂ * z
    out .= Diagonal(A) + W₁ * Diagonal((W₂z .> -h₂) .- (W₂z .> 0)) * W₂
    return nothing
end

function GTF.Utilities.wrap_as_dynamical_system(model::clippedShallowPLRNN, z₁ = nothing)
    non_autonomous_error(model)

    # params
    A, W₁, W₂, h₁, h₂ = cast_params_to_float64(Flux.params(model))

    # initial state
    z = isnothing(z₁) ? zeros(length(A)) : Float64.(z₁)

    ds = DeterministicIteratedMap(clippedShallowPLRNN_step!, z, (A, W₁, W₂, h₁, h₂))
    return TangentDynamicalSystem(ds, J = clippedShallowPLRNN_jacobian!)
end

# step & jacobian for DynamicalSystems.jl support
function shallowPLRNN_step!(out, z, p, n)
    A, W₁, W₂, h₁, h₂ = p
    out .= A .* z .+ W₁ * relu.(W₂ * z .+ h₂) .+ h₁
    return nothing
end

function shallowPLRNN_jacobian!(out, z, p, n)
    A, W₁, W₂, h₁, h₂ = p
    out .= Diagonal(A) + W₁ * Diagonal(W₂ * z .> -h₂) * W₂
    return nothing
end
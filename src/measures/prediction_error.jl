using Flux
using ..PLRNNs
using ..ObservationModels

function prediction_error(model, O::ObservationModel, X::AbstractMatrix, n::Int)
    T = size(X, 1)
    T̃ = T - n

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, X[1:T̃, :]')
    for _ = 1:n
        z = model(z)
    end

    # compute MSE
    mse = @views Flux.mse(O(z)', X[n+1:end, :])
    return mse
end

function prediction_error(
    model,
    O::ObservationModel,
    X::AbstractMatrix,
    n::Int,
    S::AbstractMatrix,
)
    T = size(X, 1)
    T̃ = T - n

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, X[1:T̃, :]')
    for i = 1:n
        z = @views model(z, S[i+1:T̃+i, :]')
    end

    # compute MSE
    mse = @views Flux.mse(O(z)', X[n+1:end, :])
    return mse
end
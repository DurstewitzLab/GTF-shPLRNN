using NPZ

abstract type AbstractDataset end
"""
    Dataset(args; kwargs)

Standard dataset storing a continuous time series of
size T × N, where N is the data dimension.
"""
struct Dataset{M <: AbstractMatrix} <: AbstractDataset
    X::M
    name::String
end

function Dataset(path::String, name::String; device = cpu, dtype = Float32)
    X = npzread(path) .|> dtype |> device
    @assert ndims(X) == 2 "Data must be 2-dimensional but is $(ndims(X))-dimensional."
    return Dataset(X, name)
end

Dataset(path::String; device = cpu, dtype = Float32) =
    Dataset(path, ""; device = device, dtype = dtype)

"""
    ExternalInputsDataset(args; kwargs)

Standard dataset storing a continuous time series of size
T × N, where N is the data dimension, and a corresponding time
series of exogeneous inputs of shape T × K.
"""
struct ExternalInputsDataset{M <: AbstractMatrix} <: AbstractDataset
    X::M
    S::M
    name::String
end

function ExternalInputsDataset(
    data_path::String,
    inputs_path::String,
    name::String;
    device = cpu,
    dtype = Float32,
)
    X = npzread(data_path) .|> dtype |> device
    S = npzread(inputs_path) .|> dtype |> device

    @assert ndims(X) == ndims(S) == 2 "Data and inputs must be 2D but are $(ndims(X))D and $(ndims(S))D."
    @assert size(X, 1) == size(S, 1) "Data and exogeneous inputs have to be of equal length."
    return ExternalInputsDataset(X, S, name)
end

ExternalInputsDataset(
    data_path::String,
    inputs_path::String;
    device = cpu,
    dtype = Float32,
) = ExternalInputsDataset(data_path, inputs_path, ""; device = device, dtype = dtype)

"""
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
"""
function sample_sequence(D::Dataset, T̃::Int)
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :]
end

function sample_sequence(D::ExternalInputsDataset, T̃::Int)
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :], D.S[i:i+T̃, :]
end

"""
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement!).
"""
function sample_batch(D::Dataset, T̃::Int, S::Int)
    N = size(D.X, 2)
    Xs = similar(D.X, N, S, T̃ + 1)
    for i = 1:S
        Xs[:, i, :] .= sample_sequence(D, T̃)'
    end
    return Xs
end

function sample_batch(D::ExternalInputsDataset, T̃::Int, S::Int)
    N, K = size(D.X, 2), size(D.S, 2)
    Xs = similar(D.X, N, S, T̃ + 1)
    Ss = similar(D.X, K, S, T̃ + 1)
    for i = 1:S
        X̃, S̃ = sample_sequence(D, T̃)
        Xs[:, i, :] .= X̃'
        Ss[:, i, :] .= S̃'
    end
    return Xs, Ss
end
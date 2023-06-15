module PLRNNs

using Flux, LinearAlgebra

using ..Utilities

export AbstractPLRNN,
    AbstractVanillaPLRNN,
    AbstractDendriticPLRNN,
    AbstractShallowPLRNN,
    PLRNN,
    mcPLRNN,
    dendPLRNN,
    clippedDendPLRNN,
    FCDendPLRNN,
    shallowPLRNN,
    clippedShallowPLRNN,
    generate,
    jacobian,
    uniform_init,
    norm_upper_bound,
    keep_connectivity_offdiagonal!

include("initialization.jl")
include("vanilla_plrnn.jl")
include("dendritic_plrnn.jl")
include("shallow_plrnn.jl")
include("model_utilities.jl")


end
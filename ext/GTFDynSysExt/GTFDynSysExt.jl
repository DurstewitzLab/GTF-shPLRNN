module GTFDynSysExt
using GTF, DynamicalSystems

cast_params_to_float64(params) = map(θ -> Float64.(θ), params)
non_autonomous_error(model::AbstractPLRNN) =
    !isnothing(model.C) && throw("Non-autonomous RNNs are not supported.")

include("shallowPLRNN.jl")
include("dendPLRNN.jl")
include("PLRNN.jl")

end
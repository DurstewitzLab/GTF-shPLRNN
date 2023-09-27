module GTFDynSysExt
using GTF, DynamicalSystems

cast_params_to_float64(params) = map(θ -> Float64.(θ), params)

include("shallowPLRNN.jl")
include("dendPLRNN.jl")
include("PLRNN.jl")

end
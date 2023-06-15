module ObservationModels

using ..Utilities

abstract type ObservationModel end
(O::ObservationModel)(z::AbstractArray) = forward(O, z)

export ObservationModel, Identity, Affine, apply_inverse, init_state

include("affine.jl")

end
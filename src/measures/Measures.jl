module Measures

using Statistics
using StatsBase

export state_space_divergence,
    decide_on_measure,
    normalized_and_smoothed_power_spectrum,
    power_spectrum_error,
    power_spectrum_correlation,
    prediction_error

include("stsp_measure.jl")
include("pse.jl")
include("prediction_error.jl")

end
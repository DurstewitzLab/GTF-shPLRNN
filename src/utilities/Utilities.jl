module Utilities

using LinearAlgebra
using Plots

const Maybe{T} = Union{T, Nothing}

export Maybe,
    offdiagonal,
    offdiagonal!,
    uniform,
    randn_like,
    num_params,
    add_gaussian_noise,
    add_gaussian_noise!,
    create_folder_structure,
    store_hypers,
    load_defaults,
    load_json_f32,
    save_model,
    load_model,
    format_run_ID,
    check_for_NaNs,
    plot_reconstruction,
    wrap_as_dynamical_system

include("helpers.jl")
include("utils.jl")
include("plotting.jl")

end
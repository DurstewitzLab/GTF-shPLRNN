using JSON
using BSON: @save
using Flux: cpu

const RES = "Results"

"""
    create_folder_structure(exp::String, run::Int)

Creates basic saving structure for a single run/experiment.
"""
function create_folder_structure(exp::String, name::String, run::Int)::String
    # create folder
    path_to_run = joinpath(RES, exp, name, format_run_ID(run))
    mkpath(joinpath(path_to_run, "checkpoints"))
    mkpath(joinpath(path_to_run, "plots"))
    return path_to_run
end

function format_run_ID(run::Int)::String
    # only allow three digit numbers
    @assert run < 1000
    return string(run, pad = 3)
end

store_hypers(dict::Dict, path::String) =
    open(joinpath(path, "args.json"), "w") do f
        JSON.print(f, dict, 4)
    end

function convert_to_Float32(dict::Dict)
    for (key, val) in dict
        dict[key] = val isa AbstractFloat ? Float32(val) : val
    end
    return dict
end

load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))

load_json_f32(path) = convert_to_Float32(JSON.parsefile(path))

save_model(model, path::String) = @save path model = cpu(model)

function check_for_NaNs(θ)
    nan = false
    for p in θ
        nan = nan || !isfinite(sum(p))
    end
    return nan
end
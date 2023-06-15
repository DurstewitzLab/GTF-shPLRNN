using DataFrames
using DataStructures
using CSV
using Statistics: mean, std
using StatsBase
using Flux
using BSON: load
using NPZ

using ..Measures
using ..Utilities

load_model(path::String) = load(path, @__MODULE__)[:model]

function add_runs_to_df_dict!(df_dict::AbstractDict, d::AbstractDict)
    for (key, metrics) in d
        name, run = extract_name_and_run(key)
        push!(df_dict["Name"], name)
        push!(df_dict["Run ID"], run)
        for (metric, val) in metrics
            push!(df_dict[get_key_name(df_dict, metric)], val)
        end
    end
end

get_key_name(d::AbstractDict, s::String) =
    for (key, _) in d
        if occursin(s, key)
            return key
        end
    end

function save_as_csv(df::DataFrame, path::String, csv_name::String)
    name = joinpath(path, "$csv_name.csv")
    CSV.write(name, df, delim = "\t")
end

function initialize_eval_dict(measure_settings::AbstractDict)
    bin_σ = measure_settings["Dstsp"]
    detail = typeof(bin_σ) <: Int ? "(#bins = $(bin_σ))" : "(σ = $(bin_σ))"
    return OrderedDict([
        "Name" => String[],
        "Run ID" => String[],
        "Dstsp " * detail => Union{Float32, Missing}[],
        "PSE (σ = $(measure_settings["PSE"]))" => Union{Float32, Missing}[],
        "PE (n = $(measure_settings["PE"]))" => Union{Float32, Missing}[],
    ])
end

function compute_summary_stats(df::DataFrame; filter_iqr::Bool = false)
    stat_dict = OrderedDict{String, Any}(
        "Statistic" => ["mean", "std", "sem", "median", "mad", "outlier_ratio"],
    )

    # select metric only columns
    metric_df = select(select(df, Not("Name")), Not("Run ID"))
    samples = nrow(metric_df)

    # filter NaN rows
    df_no_nans = filter(isfinite ∘ sum, metric_df)

    # all data NaN?
    if isempty(df_no_nans)
        @info "No data but NaNs found in `compute_summary_stats()`! \
            Returning empty DataFrame."
        return DataFrame()
    end

    # compute summary stats
    for metric in names(df_no_nans)
        v = df_no_nans[!, metric]

        # filter outliers via iqr method
        v_ = filter_iqr ? filter_outliers_iqr(v) : v

        # total ratio of discarded data
        r = 1.0 - length(v_) / samples
        stat_dict[metric] = [compute_stats(v_)..., round(r, digits = 3)]
    end

    return DataFrame(stat_dict)
end

compute_stats(x) = mean(x), std(x), sem(x), median(x), mad(x, normalize = false)

"""
    filter_outliers_iqr(x)

Use the inter quartile range (IQR) method to detect and discard
outliers for summary statistics.
"""
function filter_outliers_iqr(x)
    Q1, Q3 = nquantile(x, 4)[[2, 4]]
    IQR = Q3 - Q1
    r₋, r₊ = Q1 - 1.5IQR, Q3 + 1.5IQR
    filtered_x = x[(x.>r₋).&&(x.<r₊)]
    return filtered_x
end

function extract_name_and_run(s::String)
    pieces = split(s, "/")
    run = pieces[end]
    name = joinpath(pieces[1:end-1])
    return replace_win_path(name), run
end

function evaluate_experiment(
    path_to_exp::String,
    dataset::AbstractDataset;
    measure_settings::AbstractDict = Dict{String, Any}(
        "Dstsp" => nothing,
        "PSE" => nothing,
        "PE" => nothing,
    ),
)
    if isempty(measure_settings) || any(isnothing, values(measure_settings))
        @info "Not all measure settings specified. Inferring missing ones from `args.json`."
        infer_measure_settings!(path_to_exp, measure_settings)
    end
    @info "Measure settings: $measure_settings"

    subs = readdir(path_to_exp, join = true)
    df_dict = initialize_eval_dict(measure_settings)
    for sub in subs
        df_dict = initialize_eval_dict(measure_settings)
        d = Dict()
        runs = filter(isdir, readdir(sub, join = true))
        Threads.@threads for run in runs
            d[replace_win_path(run)] = evaluate_run(run, dataset, measure_settings)
        end
        add_runs_to_df_dict!(df_dict, d)

        # metric data
        df = DataFrame(df_dict)
        sort!(df, "Run ID")

        @info "Saving metric data to $(replace_win_path(sub))."
        save_as_csv(df, replace_win_path(sub), "metrics")

        # summary statistics
        df_sum = compute_summary_stats(df)
        @info "Saving summary statistics to $(replace_win_path(sub))."
        save_as_csv(df_sum, replace_win_path(sub), "statistics")
    end
end

function infer_measure_settings!(exp_path, settings)
    for (root, dirs, files) in walkdir(exp_path)
        if "args.json" in files
            args = load_json_f32(joinpath(root, "args.json"))
            if settings["Dstsp"] === nothing
                scal = args["D_stsp_scaling"]
                bins = args["D_stsp_bins"]
                # get observation dimension
                N = size(npzread(args["path_to_data"]), 2)
                settings["Dstsp"] = decide_on_measure(scal, bins, N)[1]
            end
            if settings["PSE"] === nothing
                settings["PSE"] = args["PSE_smoothing"]
            end
            if settings["PE"] === nothing
                settings["PE"] = args["PE_n"]
            end
            break
        end
    end
    return nothing
end

"""
    find_latest_model(run_path)

Search the folder given by `run_path` for the latest `model_[EPOCH].bson` and
return its path.
"""
function find_latest_model(run_path::String)::String
    bsons = filter(x -> endswith(x, ".bson"), readdir(joinpath(run_path, "checkpoints")))
    n = length(bsons)
    ep_vec = Vector{Int}(undef, n)
    for i = 1:n
        ep_bson = split(bsons[i], "_")[end]
        ep = parse(Int, split(ep_bson, ".")[1])
        ep_vec[i] = ep
    end
    return joinpath(run_path, "checkpoints", bsons[argmax(ep_vec)])
end

function evaluate_run(run_path::String, dataset::Dataset, measure_settings::AbstractDict)
    model_bson = find_latest_model(run_path)
    model, O = load_model(model_bson)

    # check for NaNs
    nan = check_for_NaNs(Flux.params(model, O))
    if nan
        return Dict("Dstsp" => NaN, "PSE" => NaN, "PE" => NaN)
    end

    # generate trajectory
    X = dataset.X
    T = size(X, 1)

    # generate trajectory and discard transients
    T̃ = floor(Int, 1.25 * T)
    X_gen = generate(model, O, @view(X[1, :]), T̃)[floor(Int, 0.25 * T)+1:end, :]

    # dstsp
    Dstsp = evaluate_Dstsp(X, X_gen, measure_settings["Dstsp"])
    # pse
    PSE = evaluate_PSE(X, X_gen, measure_settings["PSE"])
    # PE
    PE = evaluate_PE(model, O, X, measure_settings["PE"])

    return Dict("Dstsp" => Dstsp, "PSE" => PSE, "PE" => PE)
end

function evaluate_run(
    run_path::String,
    dataset::ExternalInputsDataset,
    measure_settings::AbstractDict,
)
    model_bson = find_latest_model(run_path)
    model, O = load_model(model_bson)

    # generate trajectory
    X, S = dataset.X, dataset.S
    T = size(X, 1)

    # generate trajectory 
    # no discarding of trainsients due to missing ext. inputs
    X_gen = generate(model, O, @view(X[1, :]), T, S)

    # dstsp
    Dstsp = evaluate_Dstsp(X, X_gen, measure_settings["Dstsp"])
    # pse
    PSE = evaluate_PSE(X, X_gen, measure_settings["PSE"])
    # PE
    PE = evaluate_PE(model, O, X, measure_settings["PE"], S)

    return Dict("Dstsp" => Dstsp, "PSE" => PSE, "PE" => PE)
end

function evaluate_Dstsp(X, X_gen, bins_or_scaling)
    compute = bins_or_scaling > zero(bins_or_scaling)
    return compute ? state_space_divergence(X, X_gen, bins_or_scaling) : missing
end

function evaluate_PSE(X, X_gen, smoothing)
    compute = smoothing > zero(smoothing)
    return compute ? power_spectrum_error(X, X_gen, smoothing) : missing
end

function evaluate_PE(m, O, X, n)
    compute = n > zero(n)
    return compute ? prediction_error(m, O, X, n) : missing
end

function evaluate_PE(m, O, X, S, n)
    compute = n > zero(n)
    return compute ? prediction_error(m, O, X, S, n) : missing
end

replace_win_path(s::String) = replace(s, "\\" => "/")

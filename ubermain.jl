using Distributed
using ArgParse

@everywhere using LinearAlgebra; BLAS.set_num_threads(1)

function parse_ubermain()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--procs", "-p"
        help = "Number of parallel processes/workers to spawn."
        arg_type = Int
        default = 1

        "--runs", "-r"
        help = "Number of runs per experiment setting."
        arg_type = Int
        default = 5
    end
    return parse_args(s)
end

# parse number of procs, number of runs
ub_args = parse_ubermain()

# start workers in GTF env
addprocs(
    ub_args["procs"];
    exeflags = `--threads=$(Threads.nthreads()) --project=$(Base.active_project())`,
)

# make pkgs available in all processes
@everywhere using GTF
@everywhere ENV["GKSwstype"] = "nul"

"""
    ubermain(n_runs)

Start multiple parallel trainings, with optional grid search and
multiple runs per experiment.
"""
function ubermain(n_runs::Int, args::GTF.ArgVec)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # prepare tasks
    tasks = prepare_tasks(defaults, args, n_runs)
    println(length(tasks))

    # run tasks
    pmap(main_routine, tasks)
end

# list arguments here
args = GTF.ArgVec([
    Argument("experiment", "ADAALPHA"),
    Argument("name", "Lorenz63"),
    Argument("path_to_data", "ICML2023_datasets/Lorenz63/lorenz63_on0.05_train.npy"),
    Argument("model", "clippedShallowPLRNN", "ℳ"), # clippedDendPLRNN: M=50, B=40
    Argument("gtf_alpha", 0.99999),
    Argument("gtf_alpha_decay", 0.999, "γ"),
    Argument(
        "gtf_alpha_method",
        ["arithmetic_mean"],
        "α_method",
    ),
    Argument("observation_model", "Identity", "𝒪"),
    #Argument("teacher_forcing_interval", 7, "τ"),
    Argument("latent_dim", 3, "M"),
    #Argument("num_bases", 100, "B"),
    Argument("hidden_dim", 50, "H"),
    Argument("sequence_length", 200, "T̃"),
    Argument("scalar_saving_interval", 10),
    Argument("image_saving_interval", 10),
    Argument("epochs", 2000),
    Argument("gradient_clipping_norm", 0.0),
    Argument("lat_model_regularization", 0.0),
    Argument("obs_model_regularization", 0.0),
    Argument("gaussian_noise_level", 0.0),
    # measures
    Argument("D_stsp_bins", 30),
    Argument("D_stsp_scaling", 1.0f0),
    Argument("PSE_smoothing", 20.0),
    Argument("PE_n", 20),
])

# run experiments
ubermain(ub_args["runs"], args)
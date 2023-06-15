using Distributed
using ArgParse
using GTF

@everywhere using LinearAlgebra;
BLAS.set_num_threads(1);

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

# name of the experiment
experiment_name = "Table1-ECG-shPLRNN+aGTF"

# list arguments here
args = GTF.ArgVec([
    # general
    Argument("experiment", experiment_name),
    Argument("name", "CRSettings"),
    Argument("device", "cpu"),
    Argument("path_to_data", "ICML2023_datasets/ECG/ECG_pecuzal_5d_train.npy"),

    # model
    Argument("model", "clippedShallowPLRNN", "ℳ"),
    Argument("observation_model", "Affine", "𝒪"),
    Argument("latent_dim", 5, "M"),
    Argument("hidden_dim", 250, "H"),

    # GTF
    Argument("gtf_alpha", 1.0),
    Argument("gtf_alpha_decay", 0.999),
    Argument("gtf_alpha_method", "product_upper_bound"),
    Argument("alpha_update_interval", 5),
    Argument("sequence_length", 200),
    Argument("batch_size", 16),
    Argument("lat_model_regularization", 1e-5),
    Argument("obs_model_regularization", 1e-6),
    Argument("partial_forcing", false),

    # training
    Argument("epochs", 5000),
    Argument("batches_per_epoch", 50),
    Argument("start_lr", 1e-3),
    Argument("end_lr", 1e-6),
    Argument("scalar_saving_interval", 500),
    Argument("image_saving_interval", 500),

    # measures
    Argument("D_stsp_bins", 30),
    Argument("D_stsp_scaling", 1.0f0),
    Argument("PSE_smoothing", 20.0),
    Argument("PE_n", 20),
])

# run experiments
ubermain(ub_args["runs"], args)

# run evaluation
evaluate_experiment(
    joinpath("Results", experiment_name),
    Dataset("ICML2023_datasets/ECG/ECG_pecuzal_5d_test.npy"),
)
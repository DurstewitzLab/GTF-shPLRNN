using ArgParse
using Flux

function initialize_model(args::AbstractDict, D::AbstractDataset; mod = @__MODULE__)
    # gather args
    M = args["latent_dim"]
    id_tf = args["observation_model"] == "Identity"

    B = args["num_bases"]
    model_name = args["model"]
    hidden_dim = args["hidden_dim"]

    # model type in correct module scope
    model_t = @eval mod $(Symbol(model_name))

    # specify model args based on model type
    if model_t <: AbstractVanillaPLRNN
        model_args = (M,)
    elseif model_t <: AbstractDendriticPLRNN
        model_args = id_tf ? (M, B, D.X) : (M, B)
    elseif model_t <: AbstractShallowPLRNN
        model_args = (M, hidden_dim)
    end

    # external inputs?
    K = typeof(D) <: ExternalInputsDataset ? (size(D.S, 2),) : ()

    # initialize model
    model = model_t(model_args..., K...)

    println("Model / # Parameters: $(typeof(model)) / $(num_params(model))")
    return model
end

function initialize_observation_model(args::AbstractDict, D::AbstractDataset)
    N = size(D.X, 2)
    M = args["latent_dim"]

    # initialize by default w/o bias
    if args["observation_model"] == "Affine"
        obs_model = Affine(N, M; use_bias = false)
    elseif args["observation_model"] == "Identity"
        obs_model = Identity(N, M)
    end

    println("Obs. Model / # Parameters: $(typeof(obs_model)) / $(num_params(obs_model))")
    return obs_model
end

function initialize_optimizer(args::Dict{String, Any})
    # optimizer chain
    opt_vec = []

    # vars
    κ = args["gradient_clipping_norm"]::Float32
    ηₛ = args["start_lr"]::Float32
    ηₑ = args["end_lr"]::Float32
    E = args["epochs"]::Int
    bpe = args["batches_per_epoch"]::Int

    # set gradient clipping
    if κ > zero(κ)
        push!(opt_vec, ClipNorm(κ))
    end

    # set SGD optimzier (ADAM, RADAM, etc)
    opt_sym = Symbol(args["optimizer"])
    opt = @eval $opt_sym($ηₛ)
    push!(opt_vec, opt)

    # set exponential decay learning rate scheduler
    γ = exp(log(ηₑ / ηₛ) / E)
    decay = ExpDecay(1, γ, bpe, ηₑ, 1)
    push!(opt_vec, decay)

    return Flux.Optimise.Optimiser(opt_vec...)
end

get_device(args::AbstractDict) =
    if args["device"] == "gpu"
        return gpu
    else
        return cpu
    end

"""
    argtable()

Prepare the argument table holding the information of all possible arguments
and correct datatypes.
"""
function argtable()
    settings = ArgParseSettings()
    defaults = load_defaults()

    @add_arg_table! settings begin
        # meta
        "--experiment"
        help = "The overall experiment name."
        arg_type = String
        default = defaults["experiment"] |> String

        "--name"
        help = "Name of a single experiment instance."
        arg_type = String
        default = defaults["name"] |> String

        "--run", "-r"
        help = "The run ID."
        arg_type = Int
        default = defaults["run"] |> Int

        "--scalar_saving_interval"
        help = "The interval at which scalar quantities are stored measured in epochs."
        arg_type = Int
        default = defaults["scalar_saving_interval"] |> Int

        "--image_saving_interval"
        help = "The interval at which images are stored measured in epochs."
        arg_type = Int
        default = defaults["image_saving_interval"] |> Int

        # data
        "--path_to_data", "-d"
        help = "Path to dataset used for training."
        arg_type = String
        default = defaults["path_to_data"] |> String

        "--path_to_inputs"
        help = "Path to external inputs used for training."
        arg_type = String
        default = defaults["path_to_inputs"] |> String

        # training
        "--teacher_forcing_interval"
        help = "The teacher forcing interval to use."
        arg_type = Int
        default = defaults["teacher_forcing_interval"] |> Int

        "--gtf_alpha"
        help = "α used for generalized TF. Starting value for annealing protocol and final value for constant protocol."
        arg_type = Float32
        default = defaults["gtf_alpha"] |> Float32

        "--gtf_alpha_decay"
        help = "α decay used for generalized TF."
        arg_type = Float32
        default = defaults["gtf_alpha_decay"] |> Float32

        "--gtf_alpha_method"
        help = "α estimation method used for generalized TF."
        arg_type = String
        default = defaults["gtf_alpha_method"] |> String

        "--alpha_update_interval"
        help = "The interval at which α is updated measured in parameter updates."
        arg_type = Int
        default = defaults["alpha_update_interval"] |> Int

        "--partial_forcing"
        help = "Whether to use partial forcing."
        arg_type = Bool
        default = defaults["partial_forcing"] |> Bool

        "--gaussian_noise_level"
        help = "Noise level of gaussian noise added to teacher signals."
        arg_type = Float32
        default = defaults["gaussian_noise_level"] |> Float32

        "--sequence_length", "-T"
        help = "Length of sequences sampled from the dataset during training."
        arg_type = Int
        default = defaults["sequence_length"] |> Int

        "--batch_size", "-S"
        help = "The number of sequences to pack into one batch."
        arg_type = Int
        default = defaults["batch_size"] |> Int

        "--epochs", "-e"
        help = "The number of epochs to train for."
        arg_type = Int
        default = defaults["epochs"] |> Int

        "--batches_per_epoch"
        help = "The number of batches processed in each epoch."
        arg_type = Int
        default = defaults["batches_per_epoch"] |> Int

        "--gradient_clipping_norm"
        help = "The norm at which to clip gradients during training."
        arg_type = Float32
        default = defaults["gradient_clipping_norm"] |> Float32

        "--optimizer"
        help = "The optimizer to use for SGD optimization. Must be one provided by Flux.jl."
        arg_type = String
        default = defaults["optimizer"] |> String

        "--start_lr"
        help = "Learning rate passed to the optimizer at the beginning of training."
        arg_type = Float32
        default = defaults["start_lr"] |> Float32

        "--end_lr"
        help = "Target learning rate at the end of training due to exponential decay."
        arg_type = Float32
        default = defaults["end_lr"] |> Float32

        "--device"
        help = "Training device to use."
        arg_type = String
        default = defaults["device"] |> String

        # model
        "--model", "-m"
        help = "RNN to use."
        arg_type = String
        default = defaults["model"] |> String

        "--latent_dim", "-M"
        help = "RNN latent dimension."
        arg_type = Int
        default = defaults["latent_dim"] |> Int

        "--hidden_dim"
        help = "hidden dimension for shallow PLRNN"
        arg_type = Int
        default = defaults["hidden_dim"] |> Int

        "--num_bases", "-B"
        help = "Number of bases to use in dendritic PLRNN"
        arg_type = Int
        default = defaults["num_bases"] |> Int

        "--observation_model", "-o"
        help = "Observation model to use."
        arg_type = String
        default = defaults["observation_model"] |> String

        # Manifold Attractor Regularization
        "--MAR_ratio"
        help = "Ratio of regularized states."
        arg_type = Float32
        default = defaults["MAR_ratio"] |> Float32

        "--MAR_lambda"
        help = "Regularization factor λ."
        arg_type = Float32
        default = defaults["MAR_lambda"] |> Float32

        "--lat_model_regularization"
        help = "Regularization λ for latent model parameters."
        arg_type = Float32
        default = defaults["lat_model_regularization"] |> Float32

        "--obs_model_regularization"
        help = "Regularization λ for observation model parameters."
        arg_type = Float32
        default = defaults["obs_model_regularization"] |> Float32

        # Metrics
        "--D_stsp_scaling"
        help = "GMM scaling parameter."
        arg_type = Float32
        default = defaults["D_stsp_scaling"] |> Float32

        "--D_stsp_bins"
        help = "Number of bins for D_stsp binning method."
        arg_type = Int
        default = defaults["D_stsp_bins"] |> Int

        "--PSE_smoothing"
        help = "Gaussian kernel smoothing σ for power spectrum smoothing."
        arg_type = Float32
        default = defaults["PSE_smoothing"] |> Float32

        "--PE_n"
        help = "n-step ahead prediction error."
        arg_type = Int
        default = defaults["PE_n"] |> Int
    end
    return settings
end

"""
    parse_commandline()

Parses all commandline arguments for execution of `main.jl`.
"""
parse_commandline() = parse_args(argtable())

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

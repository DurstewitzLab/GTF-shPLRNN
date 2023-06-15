module TFTraining

using ..Utilities
export AbstractTFRecur,
    TFRecur,
    GTFRecur,
    init_state!,
    force,
    train_!,
    mar_loss,
    AR_convergence_loss,
    sample_batch,
    sample_sequence,
    AbstractDataset,
    Dataset,
    ExternalInputsDataset,
    regularize

# α stuff
export α_METHODS,
    compute_α,
    compute_α_jacobian_upper_bound,
    compute_α_batch_maximum,
    compute_α_explog_approx,
    compute_α_product_upper_bound,
    compute_α_arithmetic_mean,
    opt_α

include("dataset.jl")
include("forcing.jl")
include("tfrecur.jl")
include("regularization.jl")
include("progress.jl")
include("training.jl")
include("alpha_estimation.jl")

end

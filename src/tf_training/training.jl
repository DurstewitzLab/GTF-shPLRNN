using Flux
using CUDA: @allowscalar
using BSON: @save
using DataStructures

using ..Measures
using ..PLRNNs
using ..ObservationModels
using ..Utilities

"""
    loss(tfrec, X̃, S̃)

Performs a forward pass using the teacher forced recursion wrapper `tfrec` and
computes and return the loss w.r.t. data `X̃`. Optionally external inputs `S̃`
can be provided.
"""
function loss(
    tfrec::AbstractTFRecur,
    X̃::AbstractArray{T, 3},
    Ẑ::AbstractArray{T, 3},
) where {T}
    Z = tfrec(X̃, Ẑ)
    X̂ = tfrec.O(Z)
    return @views Flux.mse(X̂, X̃[:, :, 2:end])
end

function loss(
    tfrec::AbstractTFRecur,
    X̃::AbstractArray{T, 3},
    Ẑ::AbstractArray{T, 3},
    S̃::AbstractArray{T, 3},
) where {T}
    Z = tfrec(X̃, Ẑ, S̃)
    X̂ = tfrec.O(Z)
    return @views Flux.mse(X̂, X̃[:, :, 2:end])
end

function train_!(
    m::AbstractPLRNN,
    O::ObservationModel,
    𝒟::AbstractDataset,
    opt::Flux.Optimise.Optimiser,
    args::AbstractDict,
    save_path::String,
)
    # hypers
    E = args["epochs"]::Int
    M = args["latent_dim"]::Int
    Sₑ = args["batches_per_epoch"]::Int
    S = args["batch_size"]::Int
    τ = args["teacher_forcing_interval"]::Int
    σ_noise = args["gaussian_noise_level"]::Float32
    T̃ = args["sequence_length"]::Int
    κ = args["MAR_ratio"]::Float32
    λₘₐᵣ = args["MAR_lambda"]::Float32
    σ²_scaling = args["D_stsp_scaling"]::Float32
    bins = args["D_stsp_bins"]::Int
    σ_smoothing = args["PSE_smoothing"]::Float32
    PE_n = args["PE_n"]::Int
    isi = args["image_saving_interval"]::Int
    ssi = args["scalar_saving_interval"]::Int
    exp = args["experiment"]::String
    name = args["name"]::String
    run = args["run"]::Int
    α = args["gtf_alpha"]::Float32
    α_method = args["gtf_alpha_method"]::String
    γ = args["gtf_alpha_decay"]::Float32
    λₒ = args["obs_model_regularization"]::Float32
    λₗ = args["lat_model_regularization"]::Float32
    partial_forcing = args["partial_forcing"]::Bool
    k = args["alpha_update_interval"]::Int
    use_gtf = args["use_gtf"]::Bool

    # data shape
    T, N = size(𝒟.X)

    # wheter external inputs are available
    ext_inputs = typeof(𝒟) <: ExternalInputsDataset

    # progress tracking
    prog = Progress(joinpath(exp, name), run, 20, E, 0.8)

    # decide on D_stsp scaling
    scal, stsp_name = decide_on_measure(σ²_scaling, bins, N)

    # initialize stateful model wrapper
    tfrec = nothing
    z₀ = similar(𝒟.X, M, S)
    if use_gtf
        tfrec = GTFRecur(m, O, z₀, α)
        println(
            "Using GTF with initial α = $α and annealing method: $α_method (γ = $γ, k = $k)",
        )
        println("Partial forcing set to: $partial_forcing (N = $N, M = $M)")
    else
        tfrec = TFRecur(m, O, z₀, τ)
        println("Using sparse TF with τ = $τ")
        println("Partial forcing set to: $partial_forcing (N = $N, M = $M)")
    end

    # model parameters
    θ = Flux.params(tfrec)

    # initial α
    α_est = α

    for e = 1:E
        # process a couple of batches
        t₁ = time_ns()
        for sₑ = 1:Sₑ
            # sample a batch
            X̃, S̃_exts = ext_inputs ? sample_batch(𝒟, T̃, S) : (sample_batch(𝒟, T̃, S), ())
            S̃_exts = ext_inputs ? (S̃_exts,) : S̃_exts

            # add noise noise if noise level > 0
            σ_noise > zero(σ_noise) ? add_gaussian_noise!(X̃, σ_noise) : nothing

            # precompute forcing signals
            Ẑ = estimate_forcing_signals(tfrec, X̃)

            # α estimation & annealing
            if sₑ % k == 0 && use_gtf
                α_est = compute_α(tfrec, @view(Ẑ[:, :, 2:end]), α_method)
                if α_est > tfrec.α
                    tfrec.α = α_est
                else
                    tfrec.α = γ * tfrec.α + (1 - γ) * α_est
                end
            end

            # partial forcing
            Ẑ_subset = @views partial_forcing ? Ẑ[1:N, :, 2:end] : Ẑ[:, :, 2:end]

            # forward and backward pass
            grads = Flux.gradient(θ) do
                Lₜᵣ = loss(tfrec, X̃, Ẑ_subset, S̃_exts...)
                Lᵣ = regularization_loss(tfrec, κ, λₘₐᵣ, λₗ, λₒ)
                return Lₜᵣ + Lᵣ
            end

            # keep W matrix offdiagonal for PLRNN, dendPLRNN
            keep_connectivity_offdiagonal!(tfrec.model, grads)

            # optimiser step
            Flux.Optimise.update!(opt, θ, grads)

            # check for NaNs in parameters (exploding gradients)
            if check_for_NaNs(θ)
                save_model(
                    [tfrec.model, tfrec.O],
                    joinpath(save_path, "checkpoints", "model_$e.bson"),
                )
                @warn "NaN(s) in parameters detected! \
                    This is likely due to exploding gradients. Aborting training..."
                return nothing
            end
        end
        t₂ = time_ns()
        Δt = (t₂ - t₁) / 1e9
        update!(prog, Δt, e)

        # plot trajectory
        if e % ssi == 0
            #@show tfrec.α, cond(tfrec.O.B)
            # loss
            # sample a batch
            X̃, S̃_exts = ext_inputs ? sample_batch(𝒟, T̃, S) : (sample_batch(𝒟, T̃, S), ())
            S̃_exts = ext_inputs ? (S̃_exts,) : S̃_exts

            # precompute forcing signals
            Ẑ = estimate_forcing_signals(tfrec, X̃)
            # partial forcing
            @views Ẑ_subset = partial_forcing ? Ẑ[1:N, :, 2:end] : Ẑ[:, :, 2:end]
            Lₜᵣ = loss(tfrec, X̃, Ẑ_subset, S̃_exts...)
            Lᵣ = regularization_loss(tfrec, κ, λₘₐᵣ, λₗ, λₒ)

            # generated trajectory
            S_exts = ext_inputs ? (𝒟.S,) : ()
            X_gen =
                @allowscalar @views generate(tfrec.model, tfrec.O, 𝒟.X[1, :], T, S_exts...)

            # move data to cpu for metrics and plotting
            X_cpu = 𝒟.X |> cpu
            X_gen_cpu = X_gen |> cpu

            # metrics
            D_stsp = state_space_divergence(X_cpu, X_gen_cpu, scal)
            pse = power_spectrum_error(X_cpu, X_gen_cpu, σ_smoothing)
            pe = prediction_error(tfrec.model, tfrec.O, 𝒟.X, PE_n, S_exts...)

            # progress printing
            scalars = gather_scalars(Lₜᵣ, Lᵣ, D_stsp, stsp_name, pse, pe, PE_n)
            typeof(tfrec) <: GTFRecur ? scalars["α"] = round(tfrec.α, digits = 3) : nothing
            typeof(O) <: Affine ? scalars["cond(B)"] = round(cond(O.B), digits = 3) :
            nothing

            print_progress(prog, Δt, scalars)

            save_model(
                [tfrec.model, tfrec.O],
                joinpath(save_path, "checkpoints", "model_$e.bson"),
            )
            if e % isi == 0
                # plot
                plot_reconstruction(
                    X_gen_cpu,
                    X_cpu,
                    joinpath(save_path, "plots", "generated_$e.png"),
                )
            end
        end
    end
    return nothing
end

function regularization_loss(
    tfrec::AbstractTFRecur,
    κ::Float32,
    λₘₐᵣ::Float32,
    λₗ::Float32,
    λₒ::Float32,
)
    # latent model regularization
    Lᵣ = 0.0f0
    if κ > zero(κ)
        Lᵣ += mar_loss(m, κ, λₘₐᵣ)
    elseif λₗ > 0
        Lᵣ += regularize(tfrec.model, λₗ)
    end

    # observation model regularization
    Lᵣ += (λₒ > 0) ? regularize(tfrec.O, λₒ) : 0
    return Lᵣ
end

function gather_scalars(Lₜᵣ, Lᵣ, D_stsp, stsp_name, pse, pe, PE_n)
    scalars = OrderedDict("∑L" => Lₜᵣ + Lᵣ)
    if Lᵣ > 0.0f0
        scalars["Lₜᵣ"] = Lₜᵣ
        scalars["Lᵣ"] = Lᵣ
    end
    scalars["Dₛₜₛₚ $stsp_name"] = round(D_stsp, digits = 3)
    scalars["Dₕ"] = round(pse, digits = 3)
    scalars["PE($PE_n)"] = pe
    return scalars
end

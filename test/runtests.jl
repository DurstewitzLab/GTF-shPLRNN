using GTF
using Test

@testset "tf_training/forcing.jl" begin
    using Flux: gradient
    # check GTF rrule
    let
        rnn = clippedShallowPLRNN(4, 100)
        z = randn(Float32, 4, 16)
        x = randn(Float32, 4, 16)
        α = 0.1f0

        # will be AD'd through
        force_check(z::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T) where {T} =
            @. (1 - α) * z + α * x

        # grads
        grad_AD_z = gradient(z -> sum(abs2, rnn(force_check(z, x, α))), z)[1]
        # uses custom rrule from forcing.jl
        grad_custom_z = gradient(z -> sum(abs2, rnn(force(z, x, α))), z)[1]
        @test grad_AD_z ≈ grad_custom_z

        grad_AD_x = gradient(x -> sum(abs2, rnn(force_check(z, x, α))), x)[1]
        grad_custom_x = gradient(x -> sum(abs2, rnn(force(z, x, α))), x)[1]
        @test grad_AD_x ≈ grad_custom_x

        grad_AD_α = gradient(α -> sum(abs2, rnn(force_check(z, x, α))), α)[1]
        grad_custom_α = gradient(α -> sum(abs2, rnn(force(z, x, α))), α)[1]
        @test grad_AD_α ≈ grad_custom_α
    end

    let
        rnn = clippedShallowPLRNN(4, 100)
        z = randn(Float32, 4, 16)
        x = randn(Float32, 2, 16)

        # will be AD'd through
        function force_check(z::AbstractMatrix, x::AbstractMatrix)
            N = size(x, 1)
            return [x; z[N+1:end, :]]
        end

        # grads
        grad_AD_z = gradient(z -> sum(abs2, rnn(force_check(z, x))), z)[1]
        grad_custom_z = gradient(z -> sum(abs2, rnn(force(z, x))), z)[1]
        @test grad_AD_z ≈ grad_custom_z

        grad_AD_x = gradient(x -> sum(abs2, rnn(force_check(z, x))), x)[1]
        grad_custom_x = gradient(x -> sum(abs2, rnn(force(z, x))), x)[1]
        @test grad_AD_x ≈ grad_custom_x
    end
end

@testset "Jacobians" begin
    using ForwardDiff
    let
        M = 10
        z = randn(Float32, M)

        # clippedShallowPLRNN
        H = 30
        F = clippedShallowPLRNN(M, H)

        # AD jacobian
        Jᴬᴰ = ForwardDiff.jacobian(F, z)
        # custom jacobian
        J = jacobian(F, z)
        @test Jᴬᴰ ≈ J
    end
end

@testset "ext/GTFDynSysExt" begin
    using DynamicalSystems

    function test_rnn(rnn, z₁)
        rnn_ds = wrap_as_dynamical_system(rnn, z₁)

        Z = generate(rnn, Float32.(z₁), 101)
        Z_ds = Matrix(trajectory(rnn_ds, 100)[1])

        @test Z ≈ Z_ds

        # check jacobian
        J_out = similar(z₁, 3, 3)
        p = current_parameters(rnn_ds)
        rnn_ds.J(J_out, z₁, p, 0)

        @test jacobian(rnn, z₁) ≈ J_out
    end

    # shallowPLRNN
    shplrnn = shallowPLRNN(3, 10)
    test_rnn(shplrnn, randn(3))

    # clippedShallowPLRNN
    clipped_shplrnn = clippedShallowPLRNN(3, 10)
    test_rnn(shplrnn, randn(3))

    # dendPLRNN
    dendplrnn = dendPLRNN(3, 10)
    test_rnn(dendplrnn, randn(3))

    # clippedDendPLRNN
    clippeddendplrnn = clippedDendPLRNN(3, 10)
    test_rnn(clippeddendplrnn, randn(3))

    # PLRNN
    plrnn = PLRNN(3)
    test_rnn(plrnn, randn(3))

    # mcPLRNN
    mcplrnn = mcPLRNN(3)
    test_rnn(mcplrnn, randn(3))
end
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

@testset "ext/GTFDynSysExt/shallowPLRNN.jl" begin
    using DynamicalSystems
    
    # shallowPLRNN
    begin 
        shplrnn = shallowPLRNN(3, 10)
        z₁ = randn(Float32, 3)

        shplrnn_ds = wrap_as_dynamical_system(shplrnn, z₁)

        Z = generate(shplrnn, z₁, 1001)
        Z_ds = Matrix{Float32}(trajectory(shplrnn_ds, 1000)[1])

        @test Z ≈ Z_ds
        
        # check jacobian
        J_out = similar(z₁, 3, 3)
        p = current_parameters(shplrnn_ds)
        shplrnn_ds.J(J_out, z₁, p, 0)

        @test jacobian(shplrnn, z₁) ≈ J_out
    end

    # clippedShallowPLRNN
    begin 
        clipped_shplrnn = clippedShallowPLRNN(3, 10)
        z₁ = randn(Float32, 3)

        clipped_shplrnn_ds = wrap_as_dynamical_system(clipped_shplrnn, z₁)

        Z = generate(clipped_shplrnn, z₁, 1001)
        Z_ds = Matrix{Float32}(trajectory(clipped_shplrnn_ds, 1000)[1])

        @test Z ≈ Z_ds
        
        # check jacobian
        J_out = similar(z₁, 3, 3)
        p = current_parameters(clipped_shplrnn_ds)
        clipped_shplrnn_ds.J(J_out, z₁, p, 0)

        @test jacobian(clipped_shplrnn, z₁) ≈ J_out
    end

end
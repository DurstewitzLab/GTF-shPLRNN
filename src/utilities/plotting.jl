function plot_reconstruction_2d(X::AbstractMatrix, X̃::AbstractMatrix)
    @assert size(X, 2) == size(X̃, 2) == 2
    fig = plot(X[:, 1], X[:, 2], label = "true", legend = true)
    plot!(fig, X̃[:, 1], X̃[:, 2], label = "generated", xlabel = "x", ylabel = "y")
    return fig
end

function plot_reconstruction_3d(X::AbstractMatrix, X̃::AbstractMatrix)
    @assert size(X, 2) == size(X̃, 2) == 3
    fig = plot(X[:, 1], X[:, 2], X[:, 3], label = "true", legend = true)
    plot!(
        fig,
        X̃[:, 1],
        X̃[:, 2],
        X̃[:, 3],
        label = "generated",
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
    )
    return fig
end

function plot_reconstruction_series(X::AbstractMatrix, X̃::AbstractMatrix)
    @assert size(X, 2) == size(X̃, 2) == 1
    t = 1:size(X, 1)
    fig = plot(t, X[:, 1], label = "true", legend = true)
    plot!(fig, t, X̃[:, 1], label = "generated", xlabel = "t", ylabel = "a.u.")
    return fig
end

function plot_reconstruction_multiple_series(
    X::AbstractMatrix,
    X̃::AbstractMatrix,
    n_plots::Int,
)
    @assert size(X, 2) == size(X̃, 2) >= n_plots
    t = 1:size(X, 1)
    ps = []
    for i = 1:n_plots
        ticks = i == n_plots ? true : false
        legend = i == 1 ? true : false
        p = plot(
            t,
            X[:, i],
            label = "true",
            legend = legend,
            xticks = ticks,
            yticks = false,
        )
        plot!(p, t, X̃[:, i], label = "generated")
        push!(ps, p)
    end
    plts = (ps[i] for i = 1:n_plots)
    fig = plot(plts..., layout = (n_plots, 1), link = :all)
    return fig
end

function plot_reconstruction(
    X_gen_cpu::AbstractMatrix,
    X_cpu::AbstractMatrix,
    save_path::String,
)
    if size(X_cpu, 2) == 3
        fig = plot_reconstruction_3d(X_cpu, X_gen_cpu)
    elseif size(X_cpu, 2) == 2
        fig = plot_reconstruction_2d(X_cpu, X_gen_cpu)
    elseif size(X_cpu, 2) == 1
        fig = plot_trajectory(X_cpu, X_gen_cpu)
    elseif size(X_cpu, 2) >= 3
        n_plots = size(X_cpu, 2) > 5 ? 5 : size(X_cpu, 2)
        fig = plot_reconstruction_multiple_series(X_cpu, X_gen_cpu, n_plots)
    end
    savefig(fig, save_path)
end
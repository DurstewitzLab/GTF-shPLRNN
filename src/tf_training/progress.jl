using Dates
using DataStructures

using Statistics: mean
using ..Utilities: format_run_ID

mutable struct Progress
    name::String
    q::Queue{Float64}
    qsize::Int
    E::Int
    e::Int
    α::Float64
    Δt::Float64

    function Progress(exp::String, run::Int, qsize::Int, E::Int, α::Float64)
        q = Queue{Float64}()
        name = "[$exp::$(format_run_ID(run))]"
        return new(name, q, qsize, E, 1, α, 1e10)
    end
end

function update!(prog::Progress, Δt::Real, e::Int)
    Δt_old = Δt
    if !isempty(prog.q)
        Δt_old = mean(prog.q)
    end

    # compute ema
    Δt_new = ema(Δt, Δt_old, prog.α)

    # update queue
    enqueue!(prog.q, Δt_new)
    if length(prog.q) > prog.qsize
        dequeue!(prog.q)
    end

    # update prog
    prog.e = e
    prog.Δt = Δt_new
end

function print_progress(prog::Progress, Δt::Float64, scalars::AbstractDict)
    eta = Time(0) + Second(round(Int, prog.Δt * (prog.E - prog.e)))

    scalar_str = ""
    for (key, val) in scalars
        scalar_str *= "| $key $val "
    end
    println(
        "$(prog.name) Epoch $(prog.e)/$(prog.E) took $(round(Δt, digits=2))s | ETA $eta " *
        scalar_str,
    )
end

ema(x_new, x_old, α) = α * x_old + (1.0 - α) * x_new

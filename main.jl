using LinearAlgebra; BLAS.set_num_threads(1)

using GTF
ENV["GKSwstype"] = "nul"
main_routine(parse_commandline())
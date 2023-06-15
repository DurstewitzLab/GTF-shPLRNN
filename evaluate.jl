using GTF, LinearAlgebra, ArgParse
BLAS.set_num_threads(1)
ENV["GKSwstype"] = "nul"

s = ArgParseSettings()
@add_arg_table! s begin
    "--data_path", "-d"
    help = "Path to (test) dataset."
    arg_type = String
    required = true

    "--results_path", "-r"
    help = "Path to experiment/results directory."
    arg_type = String
    required = true

    "--external_inputs_path", "-e"
    help = "Path to external inputs."
    arg_type = String
    default = ""

    "--Dstsp_setting"
    help = "Dstsp setting. Will be inferred from results `args.json` if not specified."
    arg_type = Number

    "--PSE_setting"
    help = "DH setting. Will be inferred from results `args.json` if not specified."
    arg_type = Number

    "--PE_setting"
    help = "PE(n) setting. Will be inferred from results `args.json` if not specified."
    arg_type = Int
end
args = parse_args(s)

# data paths
D_path = args["data_path"]
Ext_path = args["external_inputs_path"]

# model/experiment directory path
exp_path = args["results_path"]

# data
D = isempty(Ext_path) ? Dataset(D_path) : ExternalInputsDataset(D_path, Ext_path)

# metric settings
measure_settings = Dict{String, Any}(
    "Dstsp" => args["Dstsp_setting"],
    "PSE" => args["PSE_setting"],
    "PE" => args["PE_setting"],
)

# run evaluation
evaluate_experiment(exp_path, D, measure_settings=measure_settings)

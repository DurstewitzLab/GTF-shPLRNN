using Distributed
using LinearAlgebra

mutable struct Argument
    name::String
    value::Any
    id_str::String

    function Argument(name::String, value)
        if value isa Vector
            error("Please add an identifier name \
                  for arguments that are subject to grid search!")
        end
        return new(name, value, "")
    end
    Argument(name::String, value, id_str::String) = new(name, value, id_str)
end

# type aliases
ArgVec = Vector{Argument}
ArgDict = Dict{String, Any}
TaskVec = Vector{ArgDict}

add_to_name(name::String, arg::Argument) = name * "-" * arg.id_str * "_" * string(arg.value)

function prepare_base_name(default_name::String, args::ArgVec)::String
    name = default_name
    # filter name argument if specified by user
    arg = filter(arg -> arg.name == "name", args)
    if !isempty(arg)
        name = arg[1].value
    end
    return name
end

function check_arguments(defaults::ArgDict, args::ArgVec)
    for arg in args
        # check if arg exists
        @assert haskey(defaults, arg.name) "Argument/Setting <$(arg.name)> does not exist."

        # cast to correct type
        arg.value = arg.value .|> typeof(defaults[arg.name])
    end
end

function prepare_tasks(defaults::ArgDict, args::ArgVec, n_runs::Int)
    # check if arguments passed actually exist in default settings
    check_arguments(defaults, args)

    # extract multitasking name
    name = prepare_base_name(defaults["name"], args)

    # split arguments into the ones that are subject to 
    # undergo grid search and the ones constant
    const_args = filter(arg -> !(arg.value isa Vector), args)
    gs_args = filter(arg -> arg.value isa Vector, args)

    # overwrite default args with const args
    baseline_args = copy(defaults)
    for arg in const_args
        baseline_args[arg.name] = arg.value
        name = isempty(arg.id_str) ? name : add_to_name(name, arg)
    end
    baseline_args["name"] = name

    # done here, if no gs is performed
    tasks = [baseline_args]
    if !isempty(gs_args)
        tasks = generate_grid_search_tasks(baseline_args, gs_args)
    end

    # add multiple runs per task
    tasks = add_runs_to_tasks(tasks, n_runs)

    return tasks
end

function add_runs_to_tasks(tasks::TaskVec, n_runs::Int)
    tasks_w_runs = TaskVec()
    for task in tasks
        for r = 1:n_runs
            task_cp = copy(task)
            task_cp["run"] = r
            push!(tasks_w_runs, task_cp)
        end
    end
    @assert length(tasks_w_runs) == length(tasks) * n_runs
    return tasks_w_runs
end

function generate_grid_search_tasks(args::ArgDict, gs_args::ArgVec)
    # initialize with first gs variable
    tasks = TaskVec()
    init_arg = gs_args[1]
    add_values_to_task!(tasks, args, init_arg)

    # loop over other variables
    for arg in gs_args[2:end]
        new_tasks = copy(tasks)
        for task in tasks
            add_values_to_task!(new_tasks, task, arg)
        end
        # keep "mix terms"
        tasks = new_tasks[length(tasks)+1:end]
    end
    return tasks
end

function replace_arg(args::ArgDict, arg::Argument)
    args_cp = copy(args)
    args_cp[arg.name] = arg.value
    args_cp["name"] = add_to_name(args_cp["name"], arg)
    return args_cp
end

function add_values_to_task!(tasks::TaskVec, task::ArgDict, arg::Argument)
    for v in arg.value
        push!(tasks, replace_arg(task, Argument(arg.name, v, arg.id_str)))
    end
end

"""
    main_routine(args)

Function executed by every worker process.
"""
function main_routine(args::AbstractDict)
    # num threads
    n_threads = Threads.nthreads()
    println("Running on $n_threads Julia thread(s) [BLAS threads: $(BLAS.get_num_threads())]")

    # get computing device
    device = get_device(args)

    # check if external inputs are provided
    if !isempty(args["path_to_inputs"])
        println("Path to external inputs provided, initializing ExternalInputsDataset.")
        D = ExternalInputsDataset(
            args["path_to_data"],
            args["path_to_inputs"];
            device = device,
        )
    else
        println("No path to external inputs provided, initializing vanilla Dataset.")
        D = Dataset(args["path_to_data"]; device = device)
    end

    # model
    plrnn = initialize_model(args, D) |> device

    # observation_model
    O = initialize_observation_model(args, D) |> device

    # optimizer
    opt = initialize_optimizer(args)

    # create directories
    save_path = create_folder_structure(args["experiment"], args["name"], args["run"])

    # store hypers
    store_hypers(args, save_path)

    train_!(plrnn, O, D, opt, args, save_path)
end

"""
    ubermain(n_runs)

Start multiple parallel trainings, with optional grid search and
multiple runs per experiment.
"""
function ubermain(n_runs::Int, args::GTF.ArgVec)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # prepare tasks
    tasks = prepare_tasks(defaults, args, n_runs)
    println(length(tasks))

    # run tasks
    pmap(main_routine, tasks)
end
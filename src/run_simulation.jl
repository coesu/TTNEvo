using ITensors: inner
using ITensorNetworks: AbstractTreeTensorNetwork, inner, contraction_sequence
using OMEinsumContractionOrders: OMEinsumContractionOrders
using Printf
using JLD2
using Dates

export run_simulation, resume_from_file
export TimeEvolutionConfig

# Accessor functions for configuration
get_L(config::TreeConfig) = config.graph.L
get_h(config::TreeConfig) = config.model.h
get_total_time(config::TreeConfig) = config.time_evolution.t_range[2]
get_time_step(config::TreeConfig) = config.time_evolution.t_step
get_maxdim(config::TreeConfig) = config.time_evolution.maxdim
get_time_evo_method(config::TreeConfig) = config.time_evolution.method
get_initial_state_type(config::TreeConfig) = config.initial_state.state_type
get_gridnum(config::TreeConfig) = config.graph.gridnum

function _get_initial_state_params(config::InitialStateConfig)
  if !haskey(config.parameters, :initial_maxdim)
    error("InitialStateConfig parameters must contain :initial_maxdim")
  end
  initial_maxdim = config.parameters[:initial_maxdim]

  if !haskey(config.parameters, :qns)
    error("InitialStateConfig parameters must contain :qns")
  end
  qns_val = config.parameters[:qns]
  return initial_maxdim, qns_val
end

# Functions to generate state strings using Val-dispatch
function _get_state_string(graph, state_type::Symbol)
  return _get_state_string(graph, Val(state_type))
end

function _get_state_string(graph, ::Val{:neel})
  return Dict(v => iseven(sum(v)) ? "↑" : "↓" for v in vertices(graph))
end

function _get_state_string(graph, ::Val{:polarized})
  return Dict(v => "↑" for v in vertices(graph))
end

function _get_state_string(graph, ::Val{:columnar_neel})
  return Dict(v => iseven(v[1]) ? "↑" : "↓" for v in vertices(graph))
end

function _get_state_string(graph, ::Val{:random})
  return Dict(v => rand(rng, ["↑", "↓"]) for v in vertices(graph))
end

function _get_state_string(graph, state_val::Val)
  error("Unknown initial state type: $state_val")
end

function _get_free_graph_state_string(graph, state_type::Symbol)
  return _get_free_graph_state_string(graph, Val(state_type))
end

function _get_free_graph_state_string(graph, ::Val{:neel})
  return Dict(v => iseven(sum(v[2:3])) ? "↑" : "↓" for v in vertices(graph) if v[1] == 1)
end

function _get_free_graph_state_string(graph, ::Val{:columnar_neel})
  return Dict(v => iseven(v[2]) ? "↑" : "↓" for v in vertices(graph))
end

function _get_free_graph_state_string(graph, state_val::Val)
  error("Unknown initial state type for FreeGraph: $state_val")
end


function build_initial_state(config::InitialStateConfig, graph, graph_type, sites)
  initial_maxdim, qns_val = _get_initial_state_params(config)

  println("Determine sequence")
  contract_sq = get_contraction_sequence(siteinds("S=1/2", graph; conserve_qns=qns_val), initial_maxdim)
  println("finish sequence")

  state_string = _get_state_string(graph, config.state_type)

  L = Int(sqrt(nv(graph)))
  dims = (L, L)

  graph_grid = NamedGraph(grid(dims), Tuple.(CartesianIndices(dims)))

  return expanded_random_product_state(
    L,
    graph,
    graph_grid,
    sites,
    contract_sq,
    state_string;
    maxdim=initial_maxdim,
    qns=qns_val,
  )
end

function build_initial_state(config::InitialStateConfig, graph, graph_type::Union{FreeGraph,HierarchicalTree}, sites)
  initial_maxdim, qns_val = _get_initial_state_params(config)

  contract_sq = get_contraction_sequence(siteinds(v -> v[1] == 1 ? "S=1/2" : "a", graph; conserve_qns=false), initial_maxdim)

  state_string = _get_free_graph_state_string(graph, config.state_type)

  L = graph_type.L
  dims = (L, L)
  graph_grid = NamedGraph(grid(dims), [(1, v...) for v in Tuple.(CartesianIndices(dims))])

  return expanded_random_product_state(
    L,
    graph,
    graph_grid,
    graph_type,
    sites,
    contract_sq,
    state_string;
    maxdim=initial_maxdim,
    qns=qns_val,
  )
end

function init_simulation(config::AbstractSimulationConfig)
  graph, field = build_graph(config.graph)
  @visualize graph

  qns = config.time_evolution.qns
  sites = if isa(config.graph, FreeGraph) || isa(config.graph, HierarchicalTree)
    siteinds(v -> v[1] == 1 ? "S=1/2" : "a", graph; conserve_qns=qns)
  else
    siteinds("S=1/2", graph; conserve_qns=qns)
  end

  ham_graph = graph
  if config.graph.full_interaction
    dims = (config.graph.L, config.graph.L)
    ham_graph = if isa(config.graph, FreeGraph) || isa(config.graph, HierarchicalTree)
      NamedGraph(grid(dims), [(1, v...) for v in Tuple.(CartesianIndices(dims))])
    else
      g = NamedGraph(grid(dims), Tuple.(CartesianIndices(dims)))
      @show g
      g
    end
  end
  hamilt = build_hamiltonian(config.model, ham_graph, field)
  H = ttn(hamilt, sites)

  ψ = build_initial_state(config.initial_state, graph, config.graph, sites)

  return ψ, H, hamilt, field
end

save_path_training = "data/train"

function get_end_time_from_file(filepath::String)
  if !isfile(filepath)
    return -1.0
  end
  try
    return jldopen(filepath, "r") do file
      if haskey(file, "config")
        config = file["config"]
        if hasproperty(config, :observer) &&
           hasproperty(config.observer, :times) &&
           !isempty(config.observer.times)
          return config.observer.times[end]
        end
      end
      return -1.0
    end
  catch e
    println("Warning: Could not read time from $filepath. It might be corrupt. Error: $e")
    return -1.0
  end
end

function run_simulation(config::AbstractSimulationConfig; check_for_previous_run::Bool=false)
  if check_for_previous_run
    finished_filepath = joinpath(
      "data", config.save_dir, generate_filename(config; finished=true)
    )
    preliminary_filepath = joinpath(
      "data", config.save_dir, generate_filename(config; finished=false)
    )

    finished_time = get_end_time_from_file(finished_filepath)
    preliminary_time = get_end_time_from_file(preliminary_filepath)

    resume_filepath = ""
    if preliminary_time > finished_time
      println(
        "Found preliminary file at $preliminary_filepath with later time (t=$preliminary_time), resuming simulation.",
      )
      resume_filepath = preliminary_filepath
    elseif finished_time >= 0
      println(
        "Found finished file at $finished_filepath with later or equal time (t=$finished_time), resuming simulation.",
      )
      resume_filepath = finished_filepath
    end

    if !isempty(resume_filepath)
      return resume_from_file(resume_filepath, config)
    else
      println("Found no previous run file to resume from, starting new simulation.")
    end
  end

  # No previous run found or check is disabled, start a new simulation
  if isa(config, PepsConfig)
    return _run_simulation(config, nothing, nothing, nothing)
  end
  ψ, H, hamilt, field = init_simulation(config)

  @show save_path_training
  if !isnothing(save_path_training)
    return _run_simulation(config, ψ, H, hamilt; save_path=save_path_training, disorder=field)
  end
  return _run_simulation(config, ψ, H, hamilt)
end

function resume_from_file(filepath::String, new_config::PepsConfig)
  old_config, ψ, opsum = load_peps_simulation_state(filepath)
  start_time = old_config.observer.times[end]
  new_config.observer = old_config.observer
  return _run_simulation(new_config, ψ, opsum, nothing; start_time=start_time)
end

function resume_from_file(filepath::String, new_config::TreeConfig)
  old_config, ψ, H, hamilt = load_simulation_state(filepath)
  start_time = old_config.observer.times[end]
  # Carry over the observer from the old config
  new_config.observer = old_config.observer
  return _run_simulation(new_config, ψ, H, hamilt; start_time=start_time)
end

function load_peps_simulation_state(filepath::String)
  if !isfile(filepath)
    error("Simulation state file not found at $filepath")
  end
  return jldopen(filepath, "r") do file
    if !haskey(file, "config") || !haskey(file, "psi") || !haskey(file, "hamilt")
      error("Incomplete simulation state in $filepath")
    end
    config = file["config"]
    ψ = file["psi"]
    opsum = file["hamilt"]
    return config, ψ, opsum
  end
end

function load_simulation_state(filepath::String)
  if !isfile(filepath)
    error("Simulation state file not found at $filepath")
  end
  return jldopen(filepath, "r") do file
    if !haskey(file, "config") || !haskey(file, "psi") || !haskey(file, "H") || !haskey(file, "hamilt")
      error("Incomplete simulation state in $filepath")
    end
    config = file["config"]
    ψ = file["psi"]
    H = file["H"]
    hamilt = file["hamilt"]
    return config, ψ, H, hamilt
  end
end

function time_evolve(method::Symbol, H, dt, ψ; cutoff, maxdim)
  if method == :cbe
    return my_tdvp(H, dt, ψ; cutoff=cutoff, use_expansion=false, nsites=1, maxdim=maxdim)
  elseif method == :twosite
    return tdvp(H, dt, ψ; cutoff=cutoff, nsites=2, maxdim=maxdim)
  elseif method == :onesite
    return tdvp(H, dt, ψ; cutoff=cutoff, nsites=1, maxdim=maxdim)
  else
    error("Unknown time evolution method: $method")
  end
end

# Dispatched function for structure and bond dimension optimization
function optimize_structure_and_bond!(ψ, H, hamilt, graph::AbstractQuantumGraph, config; kwargs...)
  return ψ, H, 0.0, 0.0 # No change, no error for generic graphs
end

function optimize_structure_and_bond!(ψ, H, hamilt, graph::FreeGraph, config; save_path=nothing, disorder=nothing)
  error_tree_opt = 0.0
  error_adapt_bond = 0.0

  time_opti = @elapsed if graph.optimize_structure
    ψ, H, changed, err = tree_optimization_sweep_free(ψ, hamilt; maxdim=config.time_evolution.maxdim, save_path, disorder)
    @show changed
    error_tree_opt += err
  end
  @show time_opti
  flush(stdout)

  time_trunc = @elapsed if graph.optimize_bonddim
    ψ, err = adapt_bond_dimension(ψ; maxdim=config.time_evolution.maxdim, maxtensorsize=config.time_evolution.maxdim^2 * 8)
    error_adapt_bond += err
  end
  @show time_trunc
  flush(stdout)

  return ψ, H, error_tree_opt, error_adapt_bond
end

function _run_simulation(config::PepsConfig, ψ, opsum, hamilt; start_time=0.0)
  # return peps_simulation(config; psi=ψ, opsum=opsum, start_time=start_time)
  return peps_simulation(config)
end

function _run_simulation(config::TreeConfig, ψ, H, hamilt; start_time=0.0, kwargs...)
  t_start, t_end = config.time_evolution.t_range
  t_step = config.time_evolution.t_step
  opt_interval = config.time_evolution.optimization_interval
  checkpoint_interval = config.time_evolution.checkpoint_interval

  if start_time == 0.0
    measure!(config.observer, t_start, ψ, H, nothing, 0.0)
  end

  time_elapsed_since_checkpoint = 0
  error_tree_opt = 0.0
  error_adapt_bond = 0.0

  for t in range(start_time + t_step; stop=t_end, step=t_step)
    total_time = @elapsed begin
      flush(stdout)

      ex_time = @elapsed ψ = time_evolve(
        config.time_evolution.method,
        H,
        -im * t_step,
        ψ;
        cutoff=config.time_evolution.cutoff,
        maxdim=config.time_evolution.maxdim,
      )

      perform_optimization = !isnothing(opt_interval) && ((t >= 0.2 && t <= 0.5) || t % opt_interval == 0)
      if perform_optimization
        @show ex_time
        flush(stdout)
        ψ, H, err_opt, err_bond = optimize_structure_and_bond!(ψ, H, hamilt, config.graph, config; kwargs...)
        error_tree_opt += err_opt
        error_adapt_bond += err_bond
      end

      errors = Dict("adapt_bond_dim" => error_adapt_bond, "tree_opt" => error_tree_opt)
      time_measure = @elapsed measure!(
        config.observer, t, ψ, H, nothing, ex_time; errors=errors, save_linkdims=perform_optimization
      )

      time_elapsed_since_checkpoint += ex_time
      if !isnothing(checkpoint_interval) && time_elapsed_since_checkpoint > checkpoint_interval
        save_simulation_data(joinpath("data", config.save_dir), config, ψ, H, hamilt; finished=false)
        time_elapsed_since_checkpoint = 0
      end
      println("t=$t, ex_time=$ex_time, measure_time=$time_measure, maxbond=$(max_linkdim(ψ))")
    end
    @show total_time
  end

  if isa(config.graph, FreeGraph) && config.graph.save_tree
    tree_path = generate_tree_filename(config)
    save_tree(ψ, tree_path)
  end

  save_simulation_data(joinpath("data", config.save_dir), config, ψ, H, hamilt; finished=true)
  return config, ψ, H
end

function generate_tree_filename(L_val, h_val, gridnum, maxdim)
  formatted_h = @sprintf("%.2f", h_val)

  graph_l = "free"

  base_string = "$(graph_l)_L=$(L_val)_gridnum=$(gridnum)_h=$(formatted_h)_maxdim=$(maxdim)"
  base_string = replace(base_string, "/" => "_", "\\" => "_", ":" => "_")

  return joinpath("data", "tree_structures", "$(base_string)_tree.jld2")
end

export generate_tree_filename
function generate_tree_filename(config)
  L_val = get_L(config)
  h_val = get_h(config)
  gridnum = get_gridnum(config)
  maxdim = get_maxdim(config)

  formatted_h = @sprintf("%.2f", h_val)

  graph_l = graph_label(config.graph)

  base_string = "$(graph_l)_L=$(L_val)_gridnum=$(gridnum)_h=$(formatted_h)_maxdim=$(maxdim)"
  base_string = replace(base_string, "/" => "_", "\\" => "_", ":" => "_")

  return joinpath("data", "tree_structures", "$(base_string)_tree.jld2")
end

graph_label(c::TreeGraph) = "tree"
graph_label(c::SnakeGraph) = "snake"
graph_label(c::HilbertCurve) = "hilbert"
graph_label(c::FreeGraph) = "free"
graph_label(c::HierarchicalTree) = "hierch"

function generate_filename(config; finished=true)
  L_val = get_L(config)
  h_val = get_h(config)
  time_step_val = get_time_step(config)
  maxdim_val = get_maxdim(config)
  time_evo_method_val = get_time_evo_method(config)
  init_state_val = get_initial_state_type(config)
  gridnum = get_gridnum(config)

  formatted_h = @sprintf("%.2f", h_val)
  formatted_time_step = @sprintf("%.4f", time_step_val)

  graph_l = graph_label(config.graph)

  base_string = "$(graph_l)_L=$(L_val)_gridnum=$(gridnum)_h=$(formatted_h)_dt=$(formatted_time_step)_D=$(maxdim_val)_method=$(time_evo_method_val)_init=$(init_state_val)"
  base_string = replace(base_string, "/" => "_", "\\" => "_", ":" => "_")

  if finished
    return "$(base_string)_finished_results.jld2"
  else
    return "$(base_string)_preliminary_results.jld2"
  end
end

function config_to_dict(config::Union{TreeConfig,PepsConfig})
  params = Dict{String,Any}()

  # Time evolution parameters
  for field in fieldnames(typeof(config.time_evolution))
    params[string(field)] = getfield(config.time_evolution, field)
  end

  # Initial state parameters
  params["initial_state_type"] = config.initial_state.state_type
  for (k, v) in config.initial_state.parameters
    params["initial_state_$(k)"] = v
  end

  # Model parameters
  for field in fieldnames(typeof(config.model))
    params["model_$(field)"] = getfield(config.model, field)
  end

  # Graph parameters
  params["graph_type"] = string(typeof(config.graph))
  for field in fieldnames(typeof(config.graph))
    params["graph_$(field)"] = getfield(config.graph, field)
  end

  # Observer data
  observables = Dict{String,Any}()
  for field in fieldnames(typeof(config.observer))
    value = getfield(config.observer, field)
    if !isa(value, Function)
      observables[string(field)] = value
    end
  end

  return Dict("parameters" => params, "observables" => observables)
end

function save_simulation_data(dir::String, config::Union{TreeConfig,PepsConfig}, ψ, H, hamilt; finished=true)
  mkpath(dir)

  if finished
    filepath = joinpath(dir, generate_filename(config; finished=true))
    jldopen(filepath, "w") do file
      file["config"] = config
      file["psi"] = ψ
      file["H"] = H
      file["hamilt"] = hamilt
      file["results"] = config_to_dict(config)
    end
    # Clean up preliminary file
    prelim_path = joinpath(dir, generate_filename(config; finished=false))
    if isfile(prelim_path)
      rm(prelim_path)
    end
  else # Checkpoint
    filepath = joinpath(dir, generate_filename(config; finished=false))
    jldopen(filepath, "w") do file
      file["config"] = config
      file["psi"] = ψ
      file["H"] = H
      file["hamilt"] = hamilt
      file["results"] = config_to_dict(config)
    end
  end
  println("Simulation data saved to $filepath")
end

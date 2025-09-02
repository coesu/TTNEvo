using Base
using ITensors: inner
using ITensorNetworks: AbstractTreeTensorNetwork, inner, contraction_sequence
using OMEinsumContractionOrders: OMEinsumContractionOrders
using Printf # Added for formatting floating-point numbers in filenames
using JLD2 # Added for saving data

export run_simulation
export TimeEvolutionConfig
export TreeConfig
export TreeGraph
export TimeEvolution
export InitialStateConfig
export Observer

abstract type AbstractSimulationConfig end
abstract type AbstractSimulationResult end
abstract type AbstractTimeEvolutionConfig end
abstract type AbstractTimeEvolution end
abstract type AbstractInitialState end


Base.@kwdef struct MpsConfig <: AbstractSimulationConfig
  time_evolution::AbstractTimeEvolutionConfig
  save_dir::String
  model::AbstractModel
  graph::MPSGraph
end

Base.@kwdef struct InitialStateConfig <: AbstractInitialState
  state_type::Symbol # e.g., :neel, :random, :polarized
  parameters::Dict{Symbol,Any} = Dict{Symbol,Any}() # Additional parameters if needed
end

Base.@kwdef struct TreeConfig <: AbstractSimulationConfig
  time_evolution::AbstractTimeEvolutionConfig
  initial_state::InitialStateConfig
  save_dir::String
  model::AbstractModel
  graph::TreeGraph
  observer::Observer
end

Base.@kwdef struct TimeEvolutionConfig <: AbstractTimeEvolutionConfig
  t_range::Tuple{Float64,Float64} = (0, 0)
  t_step::Float64 = (0.1)
  maxdim::Int = 100
  nsite::Int = 1
  cutoff::Float64 = 1e-14
  qns::Bool = false
  method::Symbol = :cbe
end

struct TimeEvolution <: AbstractTimeEvolution
  config::TimeEvolutionConfig
  initial_state::AbstractTreeTensorNetwork
  hamiltonian::AbstractTreeTensorNetwork
  observer::Observer
end

# Assuming build_graph, build_hamiltonian, expanded_random_product_state,
# AbstractModel, MPSGraph, TreeGraph, my_tdvp, and tdvp are defined elsewhere.
# For this update, we only modify the Observer and measure! function.

function build_initial_state(config::InitialStateConfig, graph, sites)
  L = nv(graph)
  # Ensure initial_maxdim is present in parameters
  if !haskey(config.parameters, :initial_maxdim)
    error("InitialStateConfig parameters must contain :initial_maxdim")
  end
  initial_maxdim = config.parameters[:initial_maxdim]

  # Ensure qns is present in parameters
  if !haskey(config.parameters, :qns)
    error("InitialStateConfig parameters must contain :qns")
  end
  qns_val = config.parameters[:qns]


  contract_sq = get_contraction_sequence(siteinds("S=1/2", graph; conserve_qns=qns_val), initial_maxdim * 10)
  state_string = if config.state_type == :neel
    # Assuming vertices returns something iterable like tuples or vectors that can be summed
    Dict(v => iseven(sum(v)) ? "↑" : "↓" for v in vertices(graph)) # Corrected to use vertices(graph) and sum(v)
  elseif config.state_type == :polarized
    Dict(v => "↑" for v in vertices(graph)) # Corrected to use vertices(graph)
  elseif config.state_type == :random
    # Assuming rng is defined and available in this scope
    Dict(v => rand(rng, ["↑", "↓"]) for v in vertices(graph)) # Corrected to use vertices(graph)
  else
    error("Unknown initial state type: $(config.state_type)")
  end
  return expanded_random_product_state(
    L,
    graph, # Pass the graph object
    graph, # Pass the graph object again if needed by expanded_random_product_state
    sites,
    contract_sq,
    state_string;
    maxdim=initial_maxdim, # Use the extracted initial_maxdim
    qns=qns_val, # Use the extracted qns_val
  )
end

function run_simulation(config::AbstractSimulationConfig)
  return _run_simulation(config)
end

function _run_simulation(config::TreeConfig)
  graph, field = build_graph(config.graph)
  hamilt = build_hamiltonian(config.model, graph, field)

  sites = siteinds("S=1/2", graph; conserve_qns=config.time_evolution.qns)
  H = ttn(hamilt, sites)

  initial_state = build_initial_state(
    config.initial_state, graph, sites
  )

  evo = TimeEvolution(config.time_evolution, initial_state, H, config.observer)
  # The time_evolution function will now return the observer object directly
  final_observer = time_evolution(evo)

  # Save the configuration which now contains the updated observer with results
  save_simulation_data(config.save_dir, config)

  # Return the observer which contains both times and results
  return final_observer
end

function save_simulation_data(dir::String, current_config::AbstractSimulationConfig)
  # Ensure the save directory exists
  mkpath(dir)

  # Construct filename based on parameters
  # Extract parameters from current_config
  # Added checks for nested fields before accessing
  L_val = getfield(current_config.graph, :L) # Access L field safely
  # Assuming h is a scalar or can be represented simply.
  # If h is a complex structure, this might need adjustment.
  # For now, assuming h from HeisenbergModel is a number or simple function.
  # If it's a function, you might need a different representation in the filename.
  h_val = getfield(current_config.model, :h) # Access h field safely
  total_time_val = current_config.time_evolution.t_range[2]
  time_step_val = current_config.time_evolution.t_step
  maxdim_val = current_config.time_evolution.maxdim
  time_evo_method_val = current_config.time_evolution.method
  init_state_val = current_config.initial_state.state_type # Assuming state_type is a Symbol

  # Format floating-point numbers for filename
  formatted_h = @sprintf("%.2f", h_val)
  formatted_total_time = @sprintf("%.2f", total_time_val)
  formatted_time_step = @sprintf("%.4f", time_step_val)


  # Create a parameter string, replacing potentially invalid characters
  param_string = "L=$(L_val)_h=$(formatted_h)_t=$(formatted_total_time)_dt=$(formatted_time_step)_D=$(maxdim_val)_method=$(time_evo_method_val)_init=$(init_state_val)"
  param_string = replace(param_string, "/" => "_", "\\" => "_", ":" => "_") # Replace common invalid characters


  filename = "$(param_string)_results.jld2"
  filepath = joinpath(dir, filename)


  # Data to be saved: the entire configuration object.
  # This object now contains the simulation results within its 'observer.results' field.
  # JLD2 will attempt to save all fields, preserving their types.
  # Anonymous functions in config_object_with_results.observer.observables
  # will likely not be saved or will be saved as `nothing`.
  # data_to_save = Dict("simulation_config_and_results" => current_config) # No longer needed to wrap in Dict

  try
    # Using jldopen for robust saving
    jldopen(filepath, "w") do file
      # Save the entire config object, which includes the observer with results
      file["config_with_results"] = current_config
    end
    println("Simulation data and configuration saved to $filepath")
  catch e
    println("Error saving simulation data using JLD2: $e")
    @warn "Failed to save data to $filepath. Anonymous functions in Observer might be the cause. Error: $e"
    # Depending on requirements, you might rethrow the error or attempt a fallback.
  end
  return nothing
end


function time_evolution(evo::TimeEvolution)
  ψ = evo.initial_state
  H = evo.hamiltonian
  observer = evo.observer

  t_start = evo.config.t_range[1]
  t_step = evo.config.t_step
  t_end = evo.config.t_range[2]

  # Measure at the initial time t_start
  measure!(observer, ψ, H, t_start)

  step = 0
  # Iterate from t_start + t_step up to t_end, inclusive of t_end if it's a multiple of t_step
  for t in range(t_start + t_step, stop=t_end, step=t_step)
    step += 1
    @show t

    # Evolution step
    if evo.config.method == :cbe
      @time ψ = my_tdvp(
        H,
        ψ,
        -im * t_step;
        cutoff=evo.config.cutoff,
        use_expansion=true,
        nsites=1,
        maxdim=evo.config.maxdim,
      )
    elseif evo.config.method == :twosite
      ex_time = @elapsed ψ = tdvp(
        H, -im * t_step, ψ; cutoff=1e-14, nsites=evo.config.nsite, maxdim=evo.config.maxdim
      )

      @show evo.config.maxdim
      @show ex_time
    end

    # Check if measurement is needed at this step
    if should_measure(observer, step)
      @time measure!(observer, ψ, H, t)
    end
  end

  # Return the observer object containing both times and results
  return observer
end

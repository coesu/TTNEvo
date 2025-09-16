export PepsConfig, PepsObserver, PepsGraph
export TreeConfig
export TreeGraph
export FreeGraph
export SnakeGraph
export HilbertCurve
export HierarchicalTree
export InitialStateConfig
export Observer


abstract type AbstractSimulationConfig end
abstract type AbstractSimulationResult end
abstract type AbstractTimeEvolutionConfig end
abstract type AbstractTimeEvolution end
abstract type AbstractInitialState end


Base.@kwdef struct InitialStateConfig <: AbstractInitialState
  state_type::Symbol
  parameters::Dict{Symbol,Any} = Dict{Symbol,Any}()
end


Base.@kwdef struct TimeEvolutionConfig <: AbstractTimeEvolutionConfig
  t_range::Tuple{Float64,Float64} = (0, 0)
  t_step::Float64 = (0.1)
  maxdim::Int = 100
  nsite::Int = 1
  cutoff::Float64 = 1e-14
  qns::Bool = false
  method::Symbol = :cbe
  optimization_interval::Union{Float64,Nothing} = 5.0
  checkpoint_interval::Union{Int,Nothing} = 1200 # in seconds
end


Base.@kwdef mutable struct TreeConfig <: AbstractSimulationConfig
  time_evolution::AbstractTimeEvolutionConfig
  initial_state::InitialStateConfig
  save_dir::String
  model::AbstractModel
  graph::AbstractQuantumGraph
  observer::Observer
end
Base.@kwdef struct PepsGraph <: AbstractQuantumGraph
  L::Int
  gridnum::Int
end

Base.@kwdef mutable struct PepsObserver <: AbstractObserver
  times::Vector{Float64} = Vector{Float64}()
  sz::Vector{Any} = Vector{Any}()
  ee::Vector{Float64} = Vector{Float64}()
  ex_times::Vector{Float64} = Vector{Float64}()
  maxdim::Vector{Pair{Float64,Any}} = Vector{Pair{Float64,Any}}()
  memory::Vector{Int} = Vector{Int}()
  num_size::Vector{Int} = Vector{Int}()
  errors::Vector{Pair{Float64,Any}} = Vector{Pair{Float64,Any}}()
end

Base.@kwdef mutable struct PepsConfig <: AbstractSimulationConfig
  time_evolution::AbstractTimeEvolutionConfig
  initial_state::InitialStateConfig
  save_dir::String
  model::AbstractModel
  graph::AbstractQuantumGraph
  observer::PepsObserver
end

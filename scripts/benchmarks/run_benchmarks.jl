#!/usr/bin/env julia

# Standalone benchmark driver for TTNEvo variants.
# - Baseline: one-site TDVP, high initial bond dim (initial_maxdim = maxdim)
# - Shrewd: my_tdvp with use_expansion=true, low initial bond dim (initial_maxdim = initial_dim)
# - Global Krylov: expand(...; alg="global_krylov") between one-site TDVP steps, low initial bond dim
# - Two-site: two-site TDVP from ITensorNetworks, low initial bond dim (initial_maxdim = initial_dim)
#
# You can limit which variants run via CLI flags:
#   --variants=baseline,shrewd,krylov,twosite
# or individual flags: --baseline --shrewd --krylov --twosite
# If no flags are provided, all variants run.
#
# Note: This script does not modify any src code. It reuses TTNEvo APIs and mirrors
# parts of the run loop from src/run_simulation.jl for the custom variants.

using Dates
using Printf
using ITensors
using ITensors: Algorithm
using ITensorNetworks
using ITensorNetworks: AbstractTTN, ttn, maxlinkdim
using NamedGraphs
using Graphs

using TTNEvo


# Include expand methods (defines `expand` and global_krylov logic)
# include(joinpath(@__DIR__, "../../src/expand.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "results")
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)
const CSV_PATH = joinpath(RESULTS_DIR, "benchmark_results.csv")

function write_csv_header(path)
  if !isfile(path)
    open(path, "w") do io
      println(io, join([
          "timestamp",
          "variant",
          "L",
          "gridnum",
          "h",
          "dt",
          "T",
          "maxdim",
          "initial_maxdim",
          "method",
          "total_wall_time_s",
          "steps",
          "mean_step_time_s",
          "max_step_time_s",
          "final_maxbond",
          "max_memory_MB",
          "save_dir",
        ], ","))
    end
  end
end

function append_csv(path; kwargs...)
  open(path, "a") do io
    row = [
      string(get(kwargs, :timestamp, now())),
      string(get(kwargs, :variant, "")),
      string(get(kwargs, :L, "")),
      string(get(kwargs, :gridnum, "")),
      string(get(kwargs, :h, "")),
      string(get(kwargs, :dt, "")),
      string(get(kwargs, :T, "")),
      string(get(kwargs, :maxdim, "")),
      string(get(kwargs, :initial_maxdim, "")),
      string(get(kwargs, :method, "")),
      @sprintf("%.6f", get(kwargs, :total_wall_time_s, NaN)),
      string(get(kwargs, :steps, "")),
      @sprintf("%.6f", get(kwargs, :mean_step_time_s, NaN)),
      @sprintf("%.6f", get(kwargs, :max_step_time_s, NaN)),
      string(get(kwargs, :final_maxbond, "")),
      string(get(kwargs, :max_memory_MB, "")),
      string(get(kwargs, :save_dir, "")),
    ]
    println(io, join(row, ","))
  end
end

"""
Build a TreeConfig with provided knobs.
"""
function build_tree_config(; L=4, gridnum=1, h=10.0, dt=0.1, T=10.0,
  maxdim=32, initial_maxdim=32, method=:onesite,
  save_dir="bench_default")
  evo = TTNEvo.TimeEvolutionConfig(
    t_range=(0.0, T),
    t_step=dt,
    maxdim=maxdim,
    nsite=1,
    cutoff=1e-12,
    qns=false,
    method=method,
    optimization_interval=0.1,
  )
  init = TTNEvo.InitialStateConfig(
    :columnar_neel, Dict(:initial_maxdim => initial_maxdim, :qns => false)
  )
  model = TTNEvo.HeisenbergModel(J1=1.0, JZ=1.0, h=h)
  graph = TTNEvo.FreeGraph(L=L, gridnum=gridnum, with_ancilla=false, full_interaction=true, optimize_structure=false, optimize_bonddim=false)
  # graph = TTNEvo.SnakeGraph(L=L, gridnum=gridnum, with_ancilla=false, full_interaction=true)
  obs = TTNEvo.Observer()
  return TTNEvo.TreeConfig(evo, init, save_dir, model, graph, obs)
end

function maxdim_for_maxdim_graph(maxd)
  return maximum(maxd[e] for e in edges(maxd))
end

function run_baseline(; L=4, gridnum=1, h=10.0, dt=0.1, T=10.0, maxdim=32)
  cfg = build_tree_config(
    ; L, gridnum, h, dt, T, maxdim, initial_maxdim=maxdim,
    method=:onesite, save_dir="bench_baseline_L$(L)_D$(maxdim)"
  )
  t_wall = @elapsed begin
    TTNEvo.run_simulation(cfg; check_for_previous_run=false)
  end
  steps = length(cfg.observer.ex_times)
  mean_step = steps > 0 ? sum(cfg.observer.ex_times) / steps : NaN
  max_step = steps > 0 ? maximum(cfg.observer.ex_times) : NaN
  max_mem = isempty(cfg.observer.memory) ? 0 : maximum(cfg.observer.memory) ÷ 10^6
  # We don't have psi here; estimate from saved observer maxdim if available
  final_maxbond = NaN
  if !isempty(cfg.observer.maxdim)
    final_maxbond = maxdim_for_maxdim_graph(last(cfg.observer.maxdim)[2])
  end
  return (; cfg, t_wall, steps, mean_step, max_step, max_mem, final_maxbond)
end

function run_twosite(; L=4, gridnum=1, h=10.0, dt=0.1, T=10.0, maxdim=32, initial_dim=4)
  cfg = build_tree_config(
    ; L, gridnum, h, dt, T, maxdim, initial_maxdim=initial_dim,
    method=:twosite, save_dir="bench_twosite_L$(L)_D$(maxdim)"
  )
  t_wall = @elapsed begin
    TTNEvo.run_simulation(cfg; check_for_previous_run=false)
  end
  steps = length(cfg.observer.ex_times)
  mean_step = steps > 0 ? sum(cfg.observer.ex_times) / steps : NaN
  max_step = steps > 0 ? maximum(cfg.observer.ex_times) : NaN
  max_mem = isempty(cfg.observer.memory) ? 0 : maximum(cfg.observer.memory) ÷ 10^6
  final_maxbond = NaN
  if !isempty(cfg.observer.maxdim)
    final_maxbond = maxdim_for_maxdim_graph(last(cfg.observer.maxdim)[2])
  end
  return (; cfg, t_wall, steps, mean_step, max_step, max_mem, final_maxbond)
end

function init_for_custom(cfg)
  ψ, H, hamilt, field = TTNEvo.init_simulation(cfg)
  return ψ, H, hamilt, field
end

function run_shrewd(; L=4, gridnum=1, h=10.0, dt=0.1, T=10.0, maxdim=32, initial_dim=4)
  cfg = build_tree_config(
    ; L, gridnum, h, dt, T, maxdim, initial_maxdim=initial_dim,
    method=:onesite, save_dir="bench_shrewd_L$(L)_D$(maxdim)"
  )
  ψ, H, hamilt, field = init_for_custom(cfg)
  t_start, t_end = cfg.time_evolution.t_range
  # opt_interval = cfg.time_evolution.optimization_interval
  error_tree_opt = 0.0
  error_adapt_bond = 0.0
  step_times = Float64[]

  for t in range(t_start + cfg.time_evolution.t_step; stop=t_end, step=cfg.time_evolution.t_step)
    ex_time = @elapsed begin
      ψ = TTNEvo.my_tdvp(
        H, -im * cfg.time_evolution.t_step, ψ;
        cutoff=cfg.time_evolution.cutoff, maxdim=cfg.time_evolution.maxdim,
        nsites=1, use_expansion=true,
      )
    end

    push!(step_times, ex_time)

    # perform_optimization = !isnothing(opt_interval) && ((t >= 0.2 && t <= 0.5) || t % opt_interval == 0)
    # if perform_optimization
    #   ψ, H, err_opt, err_bond = TTNEvo.optimize_structure_and_bond!(ψ, H, hamilt, cfg.graph, cfg)
    #   error_tree_opt += err_opt
    #   error_adapt_bond += err_bond
    # end
    # errors = Dict("adapt_bond_dim" => error_adapt_bond, "tree_opt" => error_tree_opt)
    TTNEvo.measure!(cfg.observer, t, ψ, H, nothing, ex_time; save_linkdims=true)
  end
  t_wall = sum(step_times)
  steps = length(step_times)
  mean_step = steps > 0 ? sum(step_times) / steps : NaN
  max_step = steps > 0 ? maximum(step_times) : NaN
  max_mem = isempty(cfg.observer.memory) ? 0 : maximum(cfg.observer.memory) ÷ 10^6
  final_maxbond = TTNEvo.max_linkdim(ψ)
  # Save results using the same method as src/run_simulation.jl
  TTNEvo.save_simulation_data(joinpath("data", cfg.save_dir), cfg, ψ, H, hamilt; finished=true)
  return (; cfg, t_wall, steps, mean_step, max_step, max_mem, final_maxbond)
end

function run_krylov(; L=4, gridnum=1, h=10.0, dt=0.1, T=10.0, maxdim=32, krylovdim=2, initial_dim=4)
  cfg = build_tree_config(
    ; L, gridnum, h, dt, T, maxdim, initial_maxdim=initial_dim,
    method=:onesite, save_dir="bench_krylov_L$(L)_D$(maxdim)_k$(krylovdim)"
  )
  ψ, H, hamilt, field = init_for_custom(cfg)
  g, _ = TTNEvo.build_graph(cfg.graph)
  t_start, t_end = cfg.time_evolution.t_range
  # opt_interval = cfg.time_evolution.optimization_interval
  error_tree_opt = 0.0
  error_adapt_bond = 0.0
  step_times = Float64[]

  @show norm(ψ)
  for t in range(t_start + cfg.time_evolution.t_step; stop=t_end, step=cfg.time_evolution.t_step)
    ex_time = @elapsed begin
      ψ = ITensorNetworks.tdvp(
        H, -im * cfg.time_evolution.t_step, ψ;
        cutoff=cfg.time_evolution.cutoff, nsites=1, maxdim=cfg.time_evolution.maxdim,
      )
      # Expansion between updates using global_krylov
      ψ = TTNEvo.expand(ψ, H, g; alg="global_krylov", krylovdim=krylovdim, cutoff=1e-14, apply_kwargs=(; maxdim=cfg.time_evolution.maxdim, cutoff=cfg.time_evolution.cutoff))
      # ψ = expand(ψ, H, g; alg="global_krylov", cutoff=cfg.time_evolution.cutoff)
      normalize!(ψ)
      ψ = truncate(ψ; maxdim=cfg.time_evolution.maxdim, cutoff=1e-14)
      normalize!(ψ)
    end
    push!(step_times, ex_time)

    # perform_optimization = !isnothing(opt_interval) && ((t >= 0.2 && t <= 0.5) || t % opt_interval == 0)
    # if perform_optimization
    #   ψ, H, err_opt, err_bond = TTNEvo.optimize_structure_and_bond!(ψ, H, hamilt, cfg.graph, cfg)
    #   error_tree_opt += err_opt
    #   error_adapt_bond += err_bond
    # end
    TTNEvo.measure!(cfg.observer, t, ψ, H, nothing, ex_time; save_linkdims=true)
  end
  t_wall = sum(step_times)
  steps = length(step_times)
  mean_step = steps > 0 ? sum(step_times) / steps : NaN
  max_step = steps > 0 ? maximum(step_times) : NaN
  max_mem = isempty(cfg.observer.memory) ? 0 : maximum(cfg.observer.memory) ÷ 10^6
  final_maxbond = TTNEvo.max_linkdim(ψ)
  # Save results using the same method as src/run_simulation.jl
  TTNEvo.save_simulation_data(joinpath("data", cfg.save_dir), cfg, ψ, H, hamilt; finished=true)
  return (; cfg, t_wall, steps, mean_step, max_step, max_mem, final_maxbond)
end

"""
Parse which variants to run from command line arguments.
Supports:
- --variants=baseline,shrewd,krylov,twosite
- individual flags: --baseline --shrewd --krylov --twosite
If no selection is provided, all variants are run.
"""
function parse_variants(args::Vector{String})
  valid = Set(["baseline", "shrewd", "krylov", "twosite"])
  selected = String[]
  for arg in args
    if startswith(arg, "--variants=")
      vals = split(replace(arg, "--variants=" => ""), ",")
      append!(selected, [strip(v) for v in vals if !isempty(strip(v))])
    elseif startswith(arg, "--")
      flag = replace(arg, "--" => "")
      push!(selected, flag)
    end
  end
  if isempty(selected)
    return ["baseline", "shrewd", "krylov", "twosite"]
  end
  out = String[]
  for v in selected
    if v in valid && !(v in out)
      push!(out, v)
    elseif !(v in valid)
      @warn "Unknown variant flag '$v' ignored" v
    end
  end
  return isempty(out) ? ["baseline", "shrewd", "krylov", "twosite"] : out
end

function main_bench(args)
  # Defaults per your instructions
  L = 4
  gridnum = 1
  T = 10.0
  dt = 0.1
  h = 10.0
  maxdim = 128
  krylovdim = 2

  initial_dim = 8

  # Disable training-data HDF5 writes from tree optimization to avoid HDF5
  # interactions during benchmarking (prevents segfaults if a directory path
  # is used instead of a file). Final JLD2 results are still saved as usual.
  try
    TTNEvo.save_path_training = nothing
  catch
    # If not available, ignore.
  end

  write_csv_header(CSV_PATH)

  # Determine which variants to run from CLI
  variants = parse_variants(args)

  @show variants
  for v in variants
    if v == "baseline"
      @info "Running baseline (one-site, high-D start)"
      res = run_baseline(; L, gridnum, h, dt, T, maxdim)
      append_csv(CSV_PATH;
        timestamp=now(), variant="baseline_onesite",
        L, gridnum, h, dt, T, maxdim,
        initial_maxdim=maxdim, method=:onesite,
        total_wall_time_s=res.t_wall, steps=res.steps,
        mean_step_time_s=res.mean_step, max_step_time_s=res.max_step,
        final_maxbond=res.final_maxbond, max_memory_MB=res.max_mem,
        save_dir=res.cfg.save_dir,
      )
    elseif v == "shrewd"
      @info "Running shrewd (my_tdvp + expansion, low-D start)"
      res = run_shrewd(; L, gridnum, h, dt, T, maxdim, initial_dim)
      append_csv(CSV_PATH;
        timestamp=now(), variant="shrewd_onesite",
        L, gridnum, h, dt, T, maxdim,
        initial_maxdim=initial_dim, method=:onesite,
        total_wall_time_s=res.t_wall, steps=res.steps,
        mean_step_time_s=res.mean_step, max_step_time_s=res.max_step,
        final_maxbond=res.final_maxbond, max_memory_MB=res.max_mem,
        save_dir=res.cfg.save_dir,
      )
    elseif v == "krylov"
      @info "Running global_krylov (1-site TDVP + expand, low-D start)"
      res = run_krylov(; L, gridnum, h, dt, T, maxdim, krylovdim, initial_dim)
      append_csv(CSV_PATH;
        timestamp=now(), variant="global_krylov_onesite",
        L, gridnum, h, dt, T, maxdim,
        initial_maxdim=initial_dim, method=:onesite,
        total_wall_time_s=res.t_wall, steps=res.steps,
        mean_step_time_s=res.mean_step, max_step_time_s=res.max_step,
        final_maxbond=res.final_maxbond, max_memory_MB=res.max_mem,
        save_dir=res.cfg.save_dir,
      )
    elseif v == "twosite"
      @info "Running two-site TDVP (low-D start)"
      res = run_twosite(; L, gridnum, h, dt, T, maxdim, initial_dim)
      append_csv(CSV_PATH;
        timestamp=now(), variant="twosite",
        L, gridnum, h, dt, T, maxdim,
        initial_maxdim=initial_dim, method=:twosite,
        total_wall_time_s=res.t_wall, steps=res.steps,
        mean_step_time_s=res.mean_step, max_step_time_s=res.max_step,
        final_maxbond=res.final_maxbond, max_memory_MB=res.max_mem,
        save_dir=res.cfg.save_dir,
      )
    end
  end

  @info "Benchmarking complete. Results written to: $CSV_PATH"
end

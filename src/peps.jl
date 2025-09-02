using ITensorNetworks: ITensorNetwork, tebd, vidal_tebd, insert_linkinds, apply, max_linkdim, VidalITensorNetwork, gauge_error, BeliefPropagationCache, update, bond_tensors, default_norm_cache, Trotter, BilinearFormNetwork, default_cache_update_kwargs, gauge
using ITensorNetworks.ITensorsExtensions: group_terms
using ITensors: op
using TTNEvo
using JLD2
using Printf
using ITensorMPS: ITensorMPS
using Dates


# Accessor functions for configuration
get_L(config::PepsConfig) = config.graph.L
get_h(config::PepsConfig) = config.model.h
get_total_time(config::PepsConfig) = config.time_evolution.t_range[2]
get_time_step(config::PepsConfig) = config.time_evolution.t_step
get_maxdim(config::PepsConfig) = config.time_evolution.maxdim
get_time_evo_method(config::PepsConfig) = config.time_evolution.method
get_initial_state_type(config::PepsConfig) = config.initial_state.state_type
get_gridnum(config::PepsConfig) = config.graph.gridnum

function measure!(obs::PepsObserver, t, ψ, H, ex_time; save_linkdims=false, errors=nothing)
  if isa(ψ, VidalITensorNetwork)
    ϕ = ITensorNetwork(ψ)
  else
    ϕ = ψ
  end
  sz = expect(ϕ, "Sz"; alg="bp")

  push!(obs.sz, sz)
  push!(obs.times, t)
  push!(obs.ex_times, ex_time)
  if save_linkdims
    push!(obs.maxdim, t => linkdims(ψ))
  end
  push!(obs.memory, Base.summarysize(ψ))
  push!(obs.num_size, number_size(ψ))
  if !isnothing(errors)
    push!(obs.errors, t => errors)
  end
end

function expect_Sz(ψ_orig; verts=vertices(ψ_orig))
  L = Int(sqrt(nv(ψ_orig)))
  sz = zeros(ComplexF64, L, L)
  norm = boundary_mps(ψ_orig, ψ_orig)
  for v in verts
    @show v
    ψ = copy(ψ_orig)
    ψ[v] = noprime(op("Sz", siteinds(ψ, v)) * ψ[v])
    @time sz[v...] = boundary_mps(ψ_orig, ψ) / norm
  end
  return sz
end

function my_apply(mpo, init; maxdim=20, cutoff=1e-10)
  L = length(mpo)
  res = Vector{ITensor}(undef, L)

  for i in 1:L
    # println("_"^100)
    # @show inds(init[i])
    # @show inds(mpo[i][1])
    # @show inds(mpo[i][2])
    res[i] = init[i] * mpo[i][1] * mpo[i][2]
    # @show dim(res[i])
    # @show inds(res[i])
  end

  psi = ITensorMPS.MPS(res)

  ITensorMPS.truncate!(psi; maxdim=maxdim, cutoff=cutoff)
  return psi
end

function boundary_mps(tn1, tn2)
  tn2dag = sim(dag(tn2); sites=[])
  L = Int(sqrt(nv(tn1)))
  init = [tn1[1, i] * tn2dag[1, i] for i in 1:L]
  init = ITensorMPS.MPS(init)
  ITensorMPS.truncate!(init; maxdim=20, cutoff=1e-10)
  for j in 2:L
    mpo = [(tn1[j, i], tn2dag[j, i]) for i in 1:L]
    init = my_apply(mpo, init)
  end
  res = foldl(*, init)
  return res[1]
end


function peps_simulation(config::PepsConfig; kwargs...)
  L = config.graph.L
  gridnum = config.graph.gridnum
  model = config.model

  data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))
  h_grid = data["grid"]
  display(h_grid)

  dims = (L, L)
  g = NamedGraph(grid(dims), Tuple.(CartesianIndices(dims)))
  opsum = build_hamiltonian(model, g, h_grid)

  state_string = _get_state_string(g, config.initial_state.state_type)
  sites = siteinds("S=1/2", g)
  psi = ITensorNetwork(state_string, sites)

  psi = VidalITensorNetwork(psi)

  t_start, t_end = config.time_evolution.t_range
  t_step = config.time_evolution.t_step
  checkpoint_interval = config.time_evolution.checkpoint_interval

  measure!(config.observer, t_start, psi, nothing, 0.0)

  measure_step = 0.1

  time_elapsed_since_checkpoint = 0

  errors = []
  truncerr = 0.0
  singular_values = ITensor()
  function callback(; singular_values, truncation_error)
    truncerr += truncation_error
    singular_values = singular_values
  end
  # envs = environment(psi, collect(vertices(psi)))
  # @show envs

  sz = expect_Sz(noprime(psi))
  sz_bp = expect(psi, "Sz"; alg="bp")
  @show maximum(abs.(vec(sz) .- sz_bp))

  @show sz_bp
  @show vec(sz)

  @show collect(range(t_start + measure_step; stop=t_end, step=measure_step))
  for t in range(t_start + measure_step; stop=t_end, step=measure_step)
    total_time = @elapsed begin
      flush(stdout)

      ex_time = @elapsed begin
        psi = vidal_tebd(
          group_terms(opsum, g),
          psi;
          β=-measure_step * im,
          Δβ=-t_step * im,
          cutoff=1e-14,
          maxdim=config.time_evolution.maxdim,
          print_frequency=typemax(Int),
          callback,
          kwargs...
        )
      end
      # println("_")
      # VidalITensorNetwork(ITensorNetwork(psi))
      # @show gauge_error(psi)
      # time_measure = 0
      time_measure = @elapsed measure!(
        config.observer, t, noprime(psi), nothing, ex_time; errors=truncerr, save_linkdims=true
      )

      push!(errors, truncerr)

      time_elapsed_since_checkpoint += ex_time
      if !isnothing(checkpoint_interval) &&
         time_elapsed_since_checkpoint > checkpoint_interval
        save_simulation_data(
          joinpath("data", config.save_dir), config, psi, nothing, opsum; finished=false
        )
        time_elapsed_since_checkpoint = 0
      end
      println("t=$t, ex_time=$ex_time, measure_time=$time_measure, maxbond=$(max_linkdim(psi))")
    end
  end

  # psi = ITensorNetwork(psi)
  # # println("-"^100)
  # @show truncerr
  # # @show errors
  # # @show singular_values
  # sz = expect_Sz(noprime(psi))
  # sz_exact = expect(psi, "Sz"; alg="exact")
  # sz_bp = expect(psi, "Sz"; alg="bp")
  # # @show maximum(abs.(sz_exact .- vec(sz)))
  # # @show maximum(abs.(sz_exact .- sz_bp))
  # @show maximum(abs.(vec(sz) .- sz_bp))
  #
  # @show sz_bp
  # @show vec(sz)
  # @show columnar_imbalance(sz_bp)
  # @show columnar_imbalance(vec(sz))
  # @show columnar_imbalance(sz_exact)

  save_simulation_data(joinpath("data", config.save_dir), config, psi, nothing, opsum; finished=true)
  return config, psi, opsum, errors
end

export peps_main
function peps_main()
  maxdim = 4
  all_errors = []
  hs = [0, 10, 100, 1000]
  hs = [10.0]
  JZ = 0
  for h in hs
    config = PepsConfig(
      TimeEvolutionConfig((0.0, 1.0), 0.01, maxdim, 1, 1e-14, false, :peps, nothing, nothing),
      InitialStateConfig(:columnar_neel, Dict(:initial_maxdim => maxdim, :qns => false)),
      "test_peps",
      HeisenbergModel(1, JZ, h / 2),
      PepsGraph(4, 1),
      PepsObserver(),
    )
    _, _, _, errors = peps_simulation(config)
    push!(all_errors, errors)
  end


  filepath = "data/peps_trunc/all_errors_JZ=$(JZ)_D=$maxdim.jld2"
  mkpath(basename(filepath))
  jldopen(filepath, "w") do file
    file["errors"] = all_errors
    file["h_values"] = hs
  end
  # c, psi, opsum = peps_simulation(config)
  # @show number_size(psi)
end

using CairoMakie
using JLD2

function plot_peps_trunc()
  JZ = 0
  D = 4
  filepath = "data/peps_trunc/all_errors_JZ=$(JZ)_D=$D.jld2"

  all_errors, hs = jldopen(filepath, "r") do file
    file["errors"], file["h_values"]
  end

  fig = Figure()
  ax = Axis(fig[1, 1];
    xlabel="h",
    ylabel="Error",
    title="PEPS Truncation Errors"
  )

  for (i, errs) in enumerate(all_errors)
    lines!(ax, errs, label="h=$(hs[i])")
  end

  axislegend(ax)

  display(fig)
  save("plots/peps_trunc/peps_truncation_errors.png", fig)
  return fig
end

function plot_peps_trunc_derivative()
  JZ = 0
  D = 4
  filepath = "data/peps_trunc/all_errors_JZ=$(JZ)_D=$D.jld2"

  all_errors, hs = jldopen(filepath, "r") do file
    file["errors"], file["h_values"]
  end

  dt = 0.1

  fig = Figure()
  ax = Axis(fig[1, 1];
    xlabel="d",
    ylabel="dε/dt",
    title="Derivative of PEPS Truncation Errors"
  )

  for (i, errs) in enumerate(all_errors)
    deriv = diff(errs) ./ dt
    t_idx = (1:length(deriv)) .* dt
    lines!(ax, t_idx, deriv, label="h=$(hs[i])")
  end

  axislegend(ax)
  display(fig)
  save("plots/peps_trunc/peps_truncation_errors_derivative.png", fig)
  return fig
end

function graph_label(c::PepsGraph)
  return "peps"
end

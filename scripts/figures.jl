using TTNEvo: build_graph, FreeGraph, SnakeGraph
using ITensorNetworks: siteinds, ttn
using NamedGraphs
using JLD2
using Graphs

function save_snake_graph(L)
  T = 1.0
  L = 8
  dt = 0.1
  maxdim = 10
  nsite = 2
  cutoff = 1e-12
  qns = false
  config = TreeConfig(
    TimeEvolutionConfig(t_range=(0, T), t_step=0.1, maxdim=maxdim, nsite=1, cutoff=1e-12, qns=false, method=:onesite),
    InitialStateConfig(:columnar_neel, Dict(:initial_maxdim => maxdim, :qns => false)),
    "test2",
    HeisenbergModel(J1=1, JZ=1, h=10.0),
    SnakeGraph(L, 1, false, true),
    Observer()
  )

  graph, field = build_graph(config.graph)

  qns = config.time_evolution.qns
  sites = if isa(config.graph, FreeGraph)
    siteinds(v -> v[1] == 1 ? "S=1/2" : "a", graph; conserve_qns=qns)
  else
    siteinds("S=1/2", graph; conserve_qns=qns)
  end

  ham_graph = graph
  if config.graph.full_interaction
    dims = (config.graph.L, config.graph.L)
    ham_graph = if isa(config.graph, FreeGraph)
      NamedGraph(grid(dims), [(1, v...) for v in Tuple.(CartesianIndices(dims))])
    else
      g = NamedGraph(grid(dims), Tuple.(CartesianIndices(dims)))
      @show g
      g
    end
  end
  hamilt = TTNEvo.build_hamiltonian(config.model, ham_graph, field)

  H = ttn(hamilt, sites)

  ψ = TTNEvo.build_initial_state(config.initial_state, graph, config.graph, sites)

  jldopen("../figure-creation/tn/snake_L$L.jld2", "w") do file
    file["tn"] = ψ
  end
end

function save_tree_graph(L)
  T = 1.0
  L = 8
  dt = 0.1
  maxdim = 10
  nsite = 2
  cutoff = 1e-12
  qns = false
  config = TreeConfig(
    TimeEvolutionConfig(t_range=(0, T), t_step=0.1, maxdim=maxdim, nsite=1, cutoff=1e-12, qns=false, method=:onesite),
    InitialStateConfig(:columnar_neel, Dict(:initial_maxdim => maxdim, :qns => false)),
    "test2",
    HeisenbergModel(J1=1, JZ=1, h=10.0),
    FreeGraph(L=L, gridnum=1, with_ancilla=false, full_interaction=true, optimize_structure=true, optimize_bonddim=true),
    Observer()
  )

  graph, field = build_graph(config.graph)

  qns = config.time_evolution.qns
  sites = if isa(config.graph, FreeGraph)
    siteinds(v -> v[1] == 1 ? "S=1/2" : "a", graph; conserve_qns=qns)
  else
    siteinds("S=1/2", graph; conserve_qns=qns)
  end

  ham_graph = graph
  if config.graph.full_interaction
    dims = (config.graph.L, config.graph.L)
    ham_graph = if isa(config.graph, FreeGraph)
      NamedGraph(grid(dims), [(1, v...) for v in Tuple.(CartesianIndices(dims))])
    else
      g = NamedGraph(grid(dims), Tuple.(CartesianIndices(dims)))
      @show g
      g
    end
  end
  hamilt = TTNEvo.build_hamiltonian(config.model, ham_graph, field)

  H = ttn(hamilt, sites)

  ψ = TTNEvo.build_initial_state(config.initial_state, graph, config.graph, sites)

  jldopen("../figure-creation/tn/tree_L$L.jld2", "w") do file
    file["tn"] = ψ
  end
end

function save_peps_graph(L=8)
end

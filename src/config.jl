using ITensorNetworks: inner, contraction_sequence, ⊗, expect, dag, siteinds, random_ttn, vertices
using ITensorNetworks: AbstractTreeTensorNetwork
using ITensors

L = [6]
h = [0.0]

time_step = [0.1]
total_time = [100.0]
time_evo_method = [:onesite]

maxdim = [32, 64, 128, 196]
# maxdim = [64]
cutoff = [1e-12]
nsite = [1]
qns = [false]
grid_number = collect(1:1)

init_state = [:columnar_neel]
initial_maxdim = [2]

observe_every = [1]

save_path_training = "data/train"

tree_pre_det = false
tree_opt = false
snake = true

parameter_sets = []
for Li in L,
  total_timei in total_time,
  time_stepi in time_step,
  hi in h,
  grid_numberi in grid_number,
  cutoffi in cutoff,
  nsitei in nsite,
  maxdimi in maxdim,
  qnsi in qns,
  init_statei in init_state,
  initial_maxdimi in initial_maxdim,
  methodi in time_evo_method

  if tree_pre_det
    push!(parameter_sets, TreeConfig(
      TimeEvolutionConfig(t_range=(0, total_timei), t_step=time_stepi, maxdim=maxdimi, nsite=nsitei, cutoff=cutoffi, qns=qnsi, method=methodi),
      InitialStateConfig(init_statei, Dict(:initial_maxdim => maxdimi, :qns => qnsi)),
      "L$Li-column-tree-new-pre-det",
      HeisenbergModel(J1=1.0, JZ=1.0, h=hi),
      FreeGraph(L=Li, gridnum=grid_numberi, with_ancilla=false, full_interaction=true, optimize_structure=false, optimize_bonddim=true, save_tree=false, load_tree_path=generate_tree_filename(Li, hi, grid_numberi, 32)),
      Observer()
    ))
  end
  if tree_opt
    push!(parameter_sets, TreeConfig(
      TimeEvolutionConfig(t_range=(0, total_timei), t_step=time_stepi, maxdim=maxdimi, nsite=nsitei, cutoff=cutoffi, qns=qnsi, method=methodi),
      InitialStateConfig(init_statei, Dict(:initial_maxdim => maxdimi, :qns => qnsi)),
      "L$Li-column-tree-new",
      HeisenbergModel(J1=1.0, JZ=1.0, h=hi),
      FreeGraph(L=Li, gridnum=grid_numberi, with_ancilla=false, full_interaction=true, optimize_structure=true, optimize_bonddim=true, save_tree=true),
      Observer()
    ))
  end
  if snake
    push!(parameter_sets, TreeConfig(
      TimeEvolutionConfig(t_range=(0, total_timei), t_step=time_stepi, maxdim=maxdimi, nsite=nsitei, cutoff=cutoffi, qns=qnsi, method=methodi),
      InitialStateConfig(init_statei, Dict(:initial_maxdim => maxdimi, :qns => qnsi)),
      "L$Li-column-snake-new",
      HeisenbergModel(J1=1.0, JZ=1.0, h=hi),
      SnakeGraph(Li, grid_numberi, false, true),
      Observer()
    ))
  end
  # push!(parameter_sets, PepsConfig(
  #   TimeEvolutionConfig(t_range=(0, total_timei), t_step=time_stepi, maxdim=maxdimi, nsite=nsitei, cutoff=cutoffi, qns=qnsi, method=methodi),
  #   InitialStateConfig(init_statei, Dict(:initial_maxdim => maxdimi, :qns => qnsi)),
  #   "L$Li-column-peps",
  #   HeisenbergModel(J1=1.0, JZ=1.0, h=hi),
  #   PepsGraph(Li, grid_numberi),
  #   PepsObserver()
  # ))
end

total_combinations = length(parameter_sets)
@show total_combinations

export test_run
function test_run(config)

  check_for_previous_run = false
  t1 = @elapsed (c_tree, ψ, H) = run_simulation(config; check_for_previous_run)
  # t2 = @elapsed (c_tree2, ψ, H) = run_simulation(parameter_set2[2]; check_for_previous_run)
  plot_free_graph_with_maxdim(linkdims(ψ))
  # t2 = @elapsed (c_snake, ψ, H) = run_simulation(parameter_set2[3]; check_for_previous_run)

  return 0
  L = 4

  h = 10.0
  gridnum = 1
  ed_data_path = joinpath(@__DIR__, "../../heisenberg_ed/data/results_columnar_Lx=$(L)_Ly=$(L)_hmax=$(h)_gridnum=$(gridnum).jld2")
  data_ed = load(ed_data_path)
  sz_ed = data_ed["sz_expectations"]
  imbalance_ed = columnar_imbalance_ed(sz_ed, L)

  imbalance_tree = [real(columnar_imbalance(x)) for x in c_tree.observer.sz]
  diff_tree = imbalance_ed[1:length(imbalance_tree)-1] .- imbalance_tree[2:end]
  imbalance_tree2 = [real(columnar_imbalance(x)) for x in c_tree2.observer.sz]
  diff_tree2 = imbalance_ed[1:length(imbalance_tree2)-1] .- imbalance_tree2[2:end]
  display(diff_tree)

  imbalance_snake = [real(columnar_imbalance(x)) for x in c_snake.observer.sz]
  diff_snake = imbalance_ed[1:length(imbalance_tree)-1] .- imbalance_snake[2:end]
  display(diff_snake)

  return diff_snake, diff_tree, diff_tree2
end

function flatten_state(psi)
  s = siteinds(psi)
  cb = combiner([dag(s[v]) for v in vertices(s)]...)
  @show cb
  return prod(psi) * cb
end

function flatten_ttn(H, psi)
  s = siteinds(psi)
  cb = combiner([dag(s[v]) for v in vertices(s)]...)
  cbp = combiner([s[v]' for v in vertices(s)]...)
  H = prod(H) * cb * cbp
  return H
end


function test_dmrg_preselection_config()
  L = 6
  T = 1

  config = TreeConfig(
    TimeEvolutionConfig(t_range=(0, T), t_step=0.1, maxdim=64, nsite=1, cutoff=1e-12, qns=false, method=:onesite),
    InitialStateConfig(:columnar_neel, Dict(:initial_maxdim => 64, :qns => false)),
    "test3",
    HeisenbergModel(J1=1, JZ=1, h=10.0),
    FreeGraph(L=L, gridnum=1, with_ancilla=false, full_interaction=true, optimize_structure=false, optimize_bonddim=true),
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
  hamilt = build_hamiltonian(config.model, ham_graph, field)

  H = ttn(hamilt, sites)

  initial_maxdim, qns_val = _get_initial_state_params(config.initial_state)

  contract_sq = get_contraction_sequence(siteinds(v -> v[1] == 1 ? "S=1/2" : "a", graph; conserve_qns=false), initial_maxdim)

  state_string = _get_free_graph_state_string(graph, config.initial_state.state_type)

  dims = (L, L)
  graph_grid = NamedGraph(grid(dims), [(1, v...) for v in Tuple.(CartesianIndices(dims))])

  return L, graph, graph_grid, config.graph, sites, contract_sq, state_string
end

function test_dmrg_preselection(c)
  initial_maxdim = 50
  qns_val = false
  ψ1 = expanded_random_product_state(
    c...;
    maxdim=initial_maxdim,
    qns=qns_val,
    nsweeps=3
  )
  ψ2 = expanded_random_product_state(
    c...;
    maxdim=initial_maxdim,
    qns=qns_val,
    nsweeps=10
  )
  v = (1, 2, 3)
  @show maximum(ψ[v] - ψ[v])
  @show inner(ψ1, ψ2)
end

using TTNEvo

function test_cbe_twosite()
  L = 6
  total_time = 1.0
  time_step = 0.1
  maxdim = 100
  nsite = 1
  cutoff = 0.0
  qns = false
  h = 50
  grid_number = 1
  initial_maxdim = 100
  init_state = :neel
  tree_opti = false
  save_dir = "testing"

  cbe_config = TreeConfig(
    TimeEvolutionConfig((0, total_time), time_step, maxdim, nsite, cutoff, qns, :cbe),
    InitialStateConfig(init_state, Dict(:initial_maxdim => initial_maxdim, :qns => qns)),
    save_dir,
    HeisenbergModel(J1=1.0, JZ=1.0, h=h),
    FreeGraph(L, grid_number, false, true, tree_opti),
    Observer()
  )
  twosite_config = TreeConfig(
    TimeEvolutionConfig((0, total_time), time_step, maxdim, nsite, cutoff, qns, :twosite),
    InitialStateConfig(init_state, Dict(:initial_maxdim => initial_maxdim, :qns => qns)),
    save_dir,
    HeisenbergModel(J1=1.0, JZ=1.0, h=h),
    FreeGraph(L, grid_number, false, true, tree_opti),
    Observer()
  )
  onesite_config = TreeConfig(
    TimeEvolutionConfig((0, total_time), time_step, maxdim, nsite, cutoff, qns, :onesite),
    InitialStateConfig(init_state, Dict(:initial_maxdim => initial_maxdim, :qns => qns)),
    save_dir,
    HeisenbergModel(J1=1.0, JZ=1.0, h=h),
    FreeGraph(L, grid_number, false, true, tree_opti),
    Observer()
  )

  t1 = @elapsed res_cbe, ψ_cbe = run_simulation(cbe_config)
  t2 = @elapsed res_twosite, ψ_twosite = run_simulation(twosite_config)
  t3 = @elapsed res_onesite, ψ_onesite = run_simulation(onesite_config)
  @show t1
  @show t2
  @show t3

  # match_indices(ψ_cbe, ψ_onesite)
  # match_indices(ψ_cbe, ψ_twosite)
  # @show inner(ψ_cbe, ψ_twosite)
  # @show test_inner(ψ_cbe, ψ_onesite)
  # @show inner(ψ_onesite, ψ_twosite)
  return res_cbe, res_twosite, res_onesite, ψ_cbe, ψ_twosite, ψ_onesite


end

function match_indices(a, b)
  sa = siteinds(a)
  sb = siteinds(b)
  for v in vertices(a)
    replaceinds!(b[v], sb[v] => sa[v])
  end
end

function test_inner(a, b)
  for e in edges(a)
    if physical_index(e)
      if src(e)[1] == 1
        a = contract(a::ITensorNetworks.AbstractITensorNetwork, src(e) => dst(e))
      else
        a = contract(a::ITensorNetworks.AbstractITensorNetwork, dst(e) => src(e))
      end
    end
  end
  for e in edges(b)
    if physical_index(e)
      if src(e)[1] == 1
        b = contract(b::ITensorNetworks.AbstractITensorNetwork, src(e) => dst(e))
      else
        b = contract(b::ITensorNetworks.AbstractITensorNetwork, dst(e) => src(e))
      end
    end
  end
  @show inner(a, b)
end

function contract_aux_edges(a)
  for e in edges(a)
    if physical_index(e)
      if src(e)[1] == 1
        a = contract(a::ITensorNetworks.AbstractITensorNetwork, src(e) => dst(e))
      else
        a = contract(a::ITensorNetworks.AbstractITensorNetwork, dst(e) => src(e))
      end
    end
  end
  return a
end

function test_tdvp_free()
  L = 4
  total_time = 1.0
  time_step = 0.1
  maxdim = 100
  nsite = 1
  cutoff = 0.0
  qns = false
  h = 50
  grid_number = 1
  initial_maxdim = 100
  init_state = :neel
  tree_opti = false
  save_dir = "testing"

  config = TreeConfig(
    TimeEvolutionConfig((0, total_time), time_step, maxdim, nsite, cutoff, qns, :cbe),
    InitialStateConfig(init_state, Dict(:initial_maxdim => initial_maxdim, :qns => qns)),
    save_dir,
    HeisenbergModel(J1=1.0, JZ=1.0, h=h),
    FreeGraph(L, grid_number, false, true, tree_opti),
    Observer()
  )

  graph, field = build_graph(config.graph)
  @visualize graph

  if config.graph.full_interaction
    dims = (config.graph.L, config.graph.L)
    if isa(config.graph, FreeGraph)
      graph_grid = NamedGraph(grid(dims), [(1, v...) for v in Tuple.(CartesianIndices(dims))])
      hamilt = build_hamiltonian(config.model, graph_grid, field)
      sites = siteinds(v -> v[1] == 1 ? "S=1/2" : "a", graph; conserve_qns=config.time_evolution.qns)
    else
      graph_grid = NamedGraph(grid(dims), Tuple.(CartesianIndices(dims)))
      hamilt = build_hamiltonian(config.model, graph_grid, field)
      sites = siteinds("S=1/2", graph; conserve_qns=config.time_evolution.qns)
    end
  else
    hamilt = build_hamiltonian(config.model, graph, field)
    sites = siteinds("S=1/2", graph; conserve_qns=config.time_evolution.qns)
  end

  H = ttn(hamilt, sites)

  ψ = build_initial_state(
    config.initial_state, graph, config.graph, sites
  )

  ψ1 = tdvp(H, -1.0 * im, ψ; time_step=-0.1 * im, nsites=2, cutoff=1e-12, maxdim=60)
  ψ2 = copy(ψ)
  for _ in 1:10
    ψ2 = my_tdvp(H, -0.1 * im, ψ2; nsites=2, cutoff=1e-12, maxdim=60)
  end
  @show Base.summarysize(ψ1)
  @show number_size(ψ1)
  @show Base.summarysize(ψ2)
  @show number_size(ψ2)

  @show inner(ψ1, ψ2)
end


function test_bad_snake_good_tree()
  L = 4
  total_time = 10.0
  time_step = 0.1
  maxdim = 100
  nsite = 1
  cutoff = 1e-12
  qns = false
  h = 5
  grid_number = 1
  initial_maxdim = 16
  init_state = :neel
  tree_opti = false
  save_dir = "bad_snake_good_tree"

  config = TreeConfig(
    TimeEvolutionConfig((0, total_time), time_step, maxdim, nsite, cutoff, qns, :twosite),
    InitialStateConfig(init_state, Dict(:initial_maxdim => initial_maxdim, :qns => qns)),
    save_dir,
    HeisenbergModel(J1=1.0, JZ=1.0, h=h),
    TreeGraph(L, grid_number, false, false),
    Observer()
  )
  config_snake = TreeConfig(
    TimeEvolutionConfig((0, total_time), time_step, maxdim, nsite, cutoff, qns, :twosite),
    InitialStateConfig(init_state, Dict(:initial_maxdim => initial_maxdim, :qns => qns)),
    save_dir,
    HeisenbergModel(J1=1.0, JZ=1.0, h=h),
    SnakeGraph(L, grid_number, false, false),
    Observer()
  )
  graph, field = build_graph(config.graph)

  hamilt = build_hamiltonian(config.model, graph, field)
  sites = siteinds("S=1/2", graph; conserve_qns=config.time_evolution.qns)

  H_tree = ttn(hamilt, sites)

  ψ_tree = build_initial_state(
    config.initial_state, graph, config.graph, sites
  )

  graph, field = build_graph(config_snake.graph)

  sites = siteinds("S=1/2", graph; conserve_qns=config.time_evolution.qns)

  H_snake = ttn(hamilt, sites)

  ψ_snake = build_initial_state(
    config.initial_state, graph, config_snake.graph, sites
  )

  @show inner(ψ_tree', H_tree, ψ_tree)
  @show inner(ψ_snake', H_snake, ψ_snake)

  t_start = config.time_evolution.t_range[1]
  t_step = config.time_evolution.t_step
  t_end = config.time_evolution.t_range[2]

  measure!(config.observer, t_start, ψ_tree, H_tree, nothing, 0.0)
  measure!(config_snake.observer, t_start, ψ_snake, H_snake, nothing, 0.0)

  for t in range(t_start + t_step, stop=t_end, step=t_step)
    flush(stdout)

    ex_time = @elapsed ψ_tree = tdvp(
      H_tree, -im * t_step, ψ_tree; cutoff=config.time_evolution.cutoff, nsites=2, maxdim=config.time_evolution.maxdim
    )
    ex_time_snake = @elapsed ψ_snake = tdvp(
      H_snake, -im * t_step, ψ_snake; cutoff=config.time_evolution.cutoff, nsites=2, maxdim=config.time_evolution.maxdim
    )
    @time measure!(config.observer, t, ψ_tree, H_tree, nothing, ex_time)
    @time measure!(config_snake.observer, t, ψ_snake, H_snake, nothing, ex_time_snake)
    println("t=$t, maxbond=$(max_linkdim(ψ_tree)),$(max_linkdim(ψ_snake))")
  end

  @show inner(ψ_tree', H_tree, ψ_tree)
  @show inner(ψ_snake', H_snake, ψ_snake)

  save_simulation_data(joinpath("data", config.save_dir), config)
  sleep(1)
  save_simulation_data(joinpath("data", config_snake.save_dir), config_snake)
end

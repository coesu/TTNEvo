using ITensorNetworks
using ITensorNetworks: underlying_graph, AbstractTreeTensorNetwork, linkinds
using NamedGraphs.GraphsExtensions: rem_vertex
using Graphs: neighbors
using NamedGraphs: incident_edges
using StatsBase
using JLD2
using HDF5
using JSON

function physical_index(edge)
  s = src(edge)
  d = dst(edge)
  s_is_physical_layer = isa(s, Tuple) && !isempty(s) && s[1] == 1
  d_is_physical_layer = isa(d, Tuple) && !isempty(d) && d[1] == 1
  return s_is_physical_layer || d_is_physical_layer
end

function tree_optimization_sweep_free(ψ, os, N=nv(ψ) * 2; maxdim=Inf, save_path=nothing, disorder=nothing, origin=rand(edges(ψ)))
  structure_has_changed = false
  recently_checked = Vector{edgetype(ψ)}()

  current = origin

  maxtensorsize = 0
  total_error = 0.0

  for _ in 1:N
    n_edges = neighbouring_edges(ψ, current)
    preferred_edges = setdiff(n_edges, recently_checked)

    if !isempty(preferred_edges)
      current = rand(preferred_edges)
    else
      if length(recently_checked) > N / 2
        empty!(recently_checked)
      end
      current = rand(n_edges)
    end

    if physical_index(current)
      push!(recently_checked, current)
      continue
    end

    ψ = orthogonalize(ψ, region(current))
    local_tensor = prod(ψ[v] for v in region(current))
    si = [only(siteinds(ψ)[v]) for v in region(current)]
    li = noncommoninds(inds(local_tensor), si)

    if dim(local_tensor) > maxtensorsize
      maxtensorsize = dim(local_tensor)
    end
    @show length(li)
    bipartitions = unique_bipartitions_4(length(li))
    # bipartitions = free_bipartitions(li)

    if !isempty(bipartitions)
      inds_src = inds(ψ[src(current)])
      current_partition_a = intersect(li, inds_src)

      min_ee = Inf
      min_bi = nothing
      min_idx = nothing
      min_u = nothing
      min_s = nothing
      min_v = nothing
      min_spec = nothing
      bipartition_svs = []

      for (i, (a, b)) in enumerate(bipartitions)
        ltags = tags(ψ, current)
        u, s, v, spec = svd(local_tensor, vcat(li[a], si[1]); lefttags=ltags, maxdim, cutoff=1e-12, use_absolute_cutoff=true)
        push!(bipartition_svs, diag(array(s)))
        ee = entanglement_entropy(array(diag(s)))
        if ee < min_ee
          min_ee = ee
          min_bi = (a, b)
          min_idx = i
          min_u = u
          min_s = s
          min_v = v
          min_spec = spec
        end
      end

      current_partition_a_indices = findall(in(current_partition_a), li)
      structure_unchanged = Set(min_bi[1]) == Set(current_partition_a_indices)
      @show structure_unchanged
      # if !isnothing(save_path)
      #   @show save_path
      #   println("saving data")
      #   save_training_data_simple(
      #     ψ, current, li, bipartitions, min_idx, disorder, save_path, bipartition_svs, structure_unchanged
      #   )
      # end
      if !structure_unchanged
        remove_edges_local!(ψ, region(current))
        ψ[src(current)] = min_u
        ψ[dst(current)] = min_s * min_v
        restore_edges!(ψ)
        ψ = set_ortho_region(ψ, [dst(current)])
        structure_has_changed = true
        total_error += truncerror(min_spec)
      end


      push!(recently_checked, current)
    end
  end

  H = ttn(os, siteinds(ψ))
  println("tree opt")
  @show maxtensorsize
  println("tree opt error")
  @show total_error
  return ψ, H, structure_has_changed, total_error
end

function get_singular_values(ψ::AbstractTTN, edge)
  ψ_ortho = orthogonalize(ψ, src(edge))

  C = ψ_ortho[src(edge)]

  bond_ind = commonind(C, ψ_ortho[dst(edge)])
  if isnothing(bond_ind)
    error("No common index found for edge $edge. Are the vertices connected?")
  end

  left_inds = noncommoninds(C, ψ_ortho[dst(edge)])

  _, S, _ = svd(C, left_inds)

  return diag(array(S))
end

function save_training_data_simple(
  ψ, current, li, bipartitions, min_idx, disorder, save_path, bipartition_svs, structure_unchanged
)
  g = underlying_graph(ψ)
  v_list = vertices(g)
  vertex_map = Dict(v => i - 1 for (i, v) in enumerate(v_list))

  edge_list = edges(g)
  edge_index = hcat([[vertex_map[src(e)], vertex_map[dst(e)]] for e in edge_list]...)

  involved_edges = neighbouring_edges(ψ, current)
  involved_edges_index = hcat([[vertex_map[src(e)], vertex_map[dst(e)]] for e in involved_edges]...)

  # Get singular values for each edge in the graph
  edge_svs = [get_singular_values(ψ, e) for e in involved_edges]
  max_sv_len = maximum(length, edge_svs)
  padded_edge_svs = [vcat(sv, zeros(max_sv_len - length(sv))) for sv in edge_svs]
  edge_features = collect(hcat(padded_edge_svs...)')
  @show edge_features
  @show size(edge_features)

  println("saving data to", save_path)
  h5open(save_path, "cw") do file
    group_id = length(keys(file)) + 1
    group = create_group(file, "sample_$group_id")

    group["sv"] = edge_features
    group["label"] = min_idx - 1
  end

end

function save_training_data(
  ψ, current, li, bipartitions, min_idx, disorder, save_path, bipartition_svs, structure_unchanged
)
  g = underlying_graph(ψ)
  v_list = vertices(g)
  vertex_map = Dict(v => i - 1 for (i, v) in enumerate(v_list))

  # Get coordinates for each vertex
  vertex_coordinates = hcat([[v...] for v in v_list]...)'
  @show vertex_coordinates

  # Create edge_index for the full graph
  edge_list = edges(g)
  edge_index = hcat([[vertex_map[src(e)], vertex_map[dst(e)]] for e in edge_list]...)

  # Get singular values for each edge in the graph
  edge_svs = [get_singular_values(ψ, e) for e in edge_list]

  # Pad singular values to the same length
  max_sv_len = maximum(length, edge_svs)
  padded_edge_svs = [vcat(sv, zeros(max_sv_len - length(sv))) for sv in edge_svs]
  edge_features = collect(hcat(padded_edge_svs...)')

  @show edge_index
  @show [vertex_map[src(current)], vertex_map[dst(current)]]
  @show edge_features

  # Map bipartitions of ITensor indices to bipartitions of graph vertices
  li_to_vertex = []
  for idx in li
    found = false
    for v in setdiff(vertices(ψ), region(current))
      if hascommoninds(ψ[v], idx)
        push!(li_to_vertex, vertex_map[v])
        found = true
        break
      end
    end
    if !found
      error("Could not find a vertex for index $idx")
    end
  end

  vertex_bipartitions = []
  for (a, b) in bipartitions
    v_a = [li_to_vertex[i] for i in a]
    v_b = [li_to_vertex[i] for i in b]
    @show (v_a, v_b)
    push!(vertex_bipartitions, (v_a, v_b))
  end

  return 0
  # Save to HDF5
  h5open(save_path, "cw") do file
    # Create a new unique group for this training sample
    group_id = length(keys(file)) + 1
    group = create_group(file, "sample_$group_id")

    group["edge_index"] = edge_index
    group["edge_features"] = edge_features
    group["vertex_coordinates"] = vertex_coordinates

    if !isempty(bipartition_svs)
      max_len = maximum(length, bipartition_svs)
      padded_svs = [vcat(sv, zeros(max_len - length(sv))) for sv in bipartition_svs]
      sv_features = collect(hcat(padded_svs...)')
      group["bipartition_svs"] = sv_features
    else
      group["bipartition_svs"] = Matrix{Float64}(undef, 0, 0)
    end
    group["label"] = min_idx - 1 # Make 0-indexed
    group["structure_unchanged"] = structure_unchanged

    # Store vertex mapping
    group["vertex_map_keys"] = JSON.json([string(v) for v in v_list])
    group["vertex_map_values"] = collect(0:(length(v_list)-1))

    # Store local problem info
    group["current_edge"] = [vertex_map[src(current)], vertex_map[dst(current)]]
    group["bipartition_choices"] = JSON.json(vertex_bipartitions)

    if !isnothing(disorder)
      # Ensure disorder is a flat array for saving
      if typeof(disorder) <: AbstractArray
        group["disorder"] = reshape(disorder, length(disorder))
      else
        group["disorder"] = [disorder]
      end
    end
  end
end

function unique_bipartitions_4(x)
  if x == 0
    return []
  end
  return [
    ([1, 2], [3, 4]),
    ([1, 3], [2, 4]),
    ([1, 4], [2, 3]),
  ]
end

function free_bipartitions(set)
  n = length(set)
  result = []
  for i in 1:(2^n-1)
    A = eltype(set)[]
    B = eltype(set)[]
    for j in 1:n
      if (i >> (j - 1)) & 1 == 1
        push!(A, set[j])
      else
        push!(B, set[j])
      end
    end
    if !isempty(A) && !isempty(B) && abs(length(A) - length(B)) <= 1
      push!(result, (A, B))
    end
  end
  return result
end

function neighbouring_edges(ψ, edge)
  return setdiff([incident_edges(ψ, src(edge)); incident_edges(ψ, dst(edge))], [edge])
end

function restore_edges!(ψ)
  et = edgetype(ψ)
  for v in vertices(ψ)
    for w in vertices(ψ)
      if v != w && hascommoninds(ψ[v], ψ[w])
        add_edge!(ψ, et((v, w)))
      end
    end
  end
end

function select_new_bond(ψ, current, visited)
  in = incident_edges(ψ, current)
  @show [i for i in in if !visited[i]]
end

function NamedGraphs.incident_edges(g::AbstractTTN, e::NamedEdge)
  res = incident_edges(g, src(e))
  for i in incident_edges(g, dst(e))
    if src(i) == src(e) || dst(i) == src(e)
    else
      push!(res, i)
    end
  end
  return res
end

function Graphs.neighbors(g::AbstractTTN, e::NamedEdge)
  return [neighbors(g, src(e)); neighbors(g, dst(e))]
end

function region(e::NamedEdge)
  return [src(e), dst(e)]
end

function test_mps()
  config = TreeConfig(
    TimeEvolutionConfig((0, 0.5), 0.1, 120, 1, 1e-10, false, :onesite),
    InitialStateConfig(:neel, Dict(:initial_maxdim => 120, :qns => false)),
    "testing",
    HeisenbergModel(J1=1, JZ=1, h=50.0),
    SnakeGraph(6, 1, false, true),
    Observer()
  )
  rng = Xoshiro(5)

  println("building graph")
  graph, field = build_graph(config.graph)

  dims = (config.graph.L, config.graph.L)
  graph_grid = NamedGraph(grid(dims), Tuple.(CartesianIndices(dims)))
  println("building hamiltonian")
  hamilt = build_hamiltonian(config.model, graph_grid, field)

  sites = siteinds("S=1/2", graph; conserve_qns=config.time_evolution.qns)

  H = ttn(hamilt, sites)
  @show max_linkdim(H)

  println("building initial state")
  ψ = build_initial_state(
    config.initial_state, graph, config.graph, sites
  )
  println("finished initial state")
  flush(stdout)

  return ψ, H, hamilt, config
end

function test_tree()
  T = 1.0
  L = 2
  config = TreeConfig(
    TimeEvolutionConfig(t_range=(0, T), t_step=0.1, maxdim=16, nsite=1, cutoff=1e-12, qns=false, method=:onesite),
    InitialStateConfig(:columnar_neel, Dict(:initial_maxdim => 16, :qns => false)),
    "test2",
    HeisenbergModel(J1=1, JZ=1, h=10.0),
    FreeGraph(L=L, gridnum=1, with_ancilla=false, full_interaction=true, optimize_structure=true, optimize_bonddim=true),
    Observer()
  )
  rng = Xoshiro(5)

  println("building graph")
  graph, field = build_graph(config.graph)

  dims = (config.graph.L, config.graph.L)
  graph_grid = NamedGraph(grid(dims), [(1, v...) for v in Tuple.(CartesianIndices(dims))])
  println("building hamiltonian")
  hamilt = build_hamiltonian(config.model, graph_grid, field)

  sites = siteinds(v -> v[1] == 1 ? "S=1/2" : "a", graph; conserve_qns=config.time_evolution.qns)

  H = ttn(hamilt, sites)
  @show max_linkdim(H)

  println("building initial state")
  ψ = build_initial_state(
    config.initial_state, graph, config.graph, sites
  )
  println("finished initial state")
  flush(stdout)

  return ψ, H, hamilt, config
end

function test_opt(ψ, H, hamilt, config)

  t_start = config.time_evolution.t_range[1]
  t_step = config.time_evolution.t_step
  t_end = config.time_evolution.t_range[2]

  measure!(config.observer, t_start, ψ, H, nothing, 0.0)

  for t in range(t_start + t_step, stop=t_end, step=t_step)
    flush(stdout)

    if config.time_evolution.method == :cbe
      ex_time = @elapsed ψ = my_tdvp(
        H,
        -im * t_step,
        ψ;
        cutoff=config.time_evolution.cutoff,
        use_expansion=true,
        nsites=1,
        maxdim=config.time_evolution.maxdim,
      )
    elseif config.time_evolution.method == :twosite
      ex_time = @elapsed ψ = tdvp(
        H, -im * t_step, ψ; cutoff=config.time_evolution.cutoff, nsites=2, maxdim=config.time_evolution.maxdim, outputlevel=0
      )
      @show ex_time
    elseif config.time_evolution.method == :onesite
      ex_time = @elapsed ψ = tdvp(
        H, -im * t_step, ψ; cutoff=config.time_evolution.cutoff, nsites=1, maxdim=config.time_evolution.maxdim, outputlevel=0
      )
      @show ex_time
    end

    println("starting opti")
    flush(stdout)
    # display(plot_free_graph_with_maxdim(ψ))
    t_opti = @elapsed ψ, H = tree_optimization_sweep_free(ψ, hamilt; maxdim=config.time_evolution.maxdim)
    # @show t_opti


    println("starting measure")
    flush(stdout)

    @time measure!(config.observer, t, ψ, H, nothing, ex_time)
    # display(plot_free_graph_with_maxdim(linkdims(ψ)))
    println("t=$t, ex_time=$ex_time, maxbond=$(max_linkdim(ψ))")
    flush(stdout)
  end
  display(plot_free_graph_with_maxdim(linkdims(ψ)))
  return ψ
end

function test_ttnos()
  config = TreeConfig(
    TimeEvolutionConfig((0, 0.05), 0.05, 60, 1, 1e-14, false, :cbe),
    InitialStateConfig(:neel, Dict(:initial_maxdim => 2, :qns => false)),
    "testing",
    HeisenbergModel(J1=1, J2=0, h=5.0),
    TreeGraph(3, 2, false, true),
    Observer()
  )
  rng = Xoshiro(5)

  graph_int = NamedGraph(Graph(4), Tuple.(CartesianIndices((2, 2))))
  rem_vertex!(graph_int, (2, 1))
  add_edge!(graph_int, (1, 1) => (2, 2))
  add_edge!(graph_int, (2, 2) => (1, 2))
  add_edge!(graph_int, (1, 1) => (1, 2))
  @visualize graph_int

  graph = NamedGraph(Graph(4), Tuple.(CartesianIndices((2, 2))))
  add_edge!(graph, (1, 1) => (2, 1))
  add_edge!(graph, (2, 1) => (1, 2))
  add_edge!(graph, (2, 1) => (2, 2))
  @visualize graph

  graph_unc = NamedGraph(Graph(3), [(1, 1), (1, 2), (2, 2)])
  add_edge!(graph_unc, (1, 1) => (2, 2))
  add_edge!(graph_unc, (2, 2) => (1, 2))

  sites = siteinds(x -> x != (2, 1) ? "S=1/2" : "a", graph)
  os = build_hamiltonian(HeisenbergModel(J1=1, J2=0, h=0), graph_int, 0)
  H = ttn(os, sites)

  @show [iseven(v[1] + v[2]) ? "↑" : "↓" for v in vertices(graph)]
  psi = ttn(v -> iseven(v[1] + v[2]) ? "↑" : "↓", sites)
  @show inner(psi', H, psi)
  @show siteinds(psi)
end

function test_new_sweep()
  config = TreeConfig(
    TimeEvolutionConfig((0, 0.05), 0.05, 60, 1, 1e-14, false, :cbe),
    InitialStateConfig(:neel, Dict(:initial_maxdim => 2, :qns => false)),
    "testing",
    HeisenbergModel(J1=1, J2=0, h=5.0),
    TreeGraph(3, 2, false, true),
    Observer()
  )

  rng = Xoshiro(5)
  graph, field = build_random_graph(config.graph; rng)
  display(plot_coordinate_graph(graph))

  if config.graph.full_interaction
    dims = (config.graph.L, config.graph.L)
    graph_grid = NamedGraph(grid(dims), Tuple.(CartesianIndices(dims)))
    hamilt = build_hamiltonian(config.model, graph_grid, field)
  else
    hamilt = build_hamiltonian(config.model, graph, field)
  end

  sites = siteinds("S=1/2", graph; conserve_qns=config.time_evolution.qns)
  H = ttn(hamilt, sites)

  ψ = build_initial_state(
    config.initial_state, graph, config.graph, sites
  )
  @show inner(ψ', H, ψ)

  @disable_warn_order ee_sequence = entanglement_entropy_sequence(random_ttn(sites; link_space_physical=config.time_evolution.maxdim, link_space_ancilla=2)
  )

  a, order = tree_optimization_sweep(ψ, hamilt)
  @show a
  @show order
end

function tree_optimization_sweep_sa(ψ, os; β=1, root_vertex=rand(vertices(ψ)))
  sweep = forward_sweep(
    direction(1),
    ψ;
    root_vertex,
    nsites=2,
    reverse_step=false,
    region_kwargs=(),
  )

  #TODO just use a random edge. Its the easiest although not too efficient
  for s in sweep
    region = s[1]
    ψ = orthogonalize(ψ, region)
    local_tensor = prod(ψ[v] for v in region)

    si = [only(siteinds(ψ)[v]) for v in region]
    li = noncommoninds(inds(local_tensor), si)

    pre1 = setdiff(neighbors(ψ, region[1]), region)
    pre2 = setdiff(neighbors(ψ, region[2]), region)

    temp = vcat([neighbors(ψ, region[i]) for i in 1:2]...)
    neigh = setdiff(temp, region)

    bipartitions = equal_bipartitions(neigh)

    if !isempty(bipartitions)
      min = Inf
      us = []
      ss = []
      vs = []
      ees = []
      for (a, b) in bipartitions
        u, s, v = svd(local_tensor, vcat(vertices_to_inds(ψ, local_tensor, a), si[1]))
        ee = entanglement_entropy(array(diag(s)))
        push!(ees, ee)
        push!(us, u)
        push!(ss, s)
        push!(vs, v)
        @show a, b
      end

      min = argmin(ees)
      @show bipartitions[min]
      if bipartitions[min][1] != pre1
        remove_edges_local!(ψ, region, pre1, pre2)
        add_edges_local!(ψ, region, bipartitions[min][1], bipartitions[min][2])
        ψ[region[1]] = us[min]
        ψ[region[2]] = ss[min] * vs[min]
        ψ = set_ortho_region(ψ, [region[2]])
        plot_coordinate_graph(ψ)
      else
        # i = sample(collect(1:length(bipartitions)), AnalyticWeights(exp.(-β .* ees)))
        # @show i
        # remove_edges_local!(ψ, region, pre1, pre2)
        # add_edges_local!(ψ, region, bipartitions[i][1], bipartitions[i][2])
        # ψ[region[1]] = us[i]
        # ψ[region[2]] = ss[i] * vs[i]
        # ψ = set_ortho_region(ψ, [region[2]])
      end
    end
  end
  H = ttn(os, siteinds(ψ))
  return ψ, H
end


function tree_optimization_sweep_old(ψ, os; root_vertex=rand(vertices(ψ)))
  sweep = forward_sweep(
    direction(1),
    ψ;
    root_vertex,
    nsites=2,
    reverse_step=false,
    region_kwargs=(),
  )

  for s in sweep
    region = s[1]
    ψ = orthogonalize(ψ, region)
    local_tensor = prod(ψ[v] for v in region)

    si = [only(siteinds(ψ)[v]) for v in region]
    li = noncommoninds(inds(local_tensor), si)

    pre1 = setdiff(neighbors(ψ, region[1]), region)
    pre2 = setdiff(neighbors(ψ, region[2]), region)

    temp = vcat([neighbors(ψ, region[i]) for i in 1:2]...)
    neigh = setdiff(temp, region)

    bipartitions = equal_bipartitions(neigh)

    if !isempty(bipartitions)
      min = Inf
      min_bi = nothing
      min_u = nothing
      min_s = nothing
      min_v = nothing
      for (a, b) in bipartitions
        u, s, v = svd(local_tensor, vcat(vertices_to_inds(ψ, local_tensor, a), si[1]))
        ee = entanglement_entropy(array(diag(s)))
        if ee < min
          min = ee
          min_bi = (a, b)
          min_u = u
          min_s = s
          min_v = v
        end
      end


      if min_bi[1] != pre1
        println("Changing tree structure")
        remove_edges_local!(ψ, region, pre1, pre2)
        add_edges_local!(ψ, region, min_bi[1], min_bi[2])
        ψ[region[1]] = min_u
        ψ[region[2]] = min_s * min_v
        ψ = set_ortho_region(ψ, [region[2]])
      end
    end
  end
  H = ttn(os, siteinds(ψ))
  @visualize ψ
  return ψ, H
end

function add_edges_local!(ψ, region, a, b)
  et = edgetype(ψ)
  for ai in a
    add_edge!(ψ, et((ai, region[1])))
  end
  for bi in b
    add_edge!(ψ, et((bi, region[2])))
  end
end

function remove_edges_local!(ψ, region)
  et = edgetype(ψ)
  rem_edge!(ψ, et((region[1], region[2])))
  for e in incident_edges(ψ, et((region[1], region[2])))
    rem_edge!(ψ, e)
  end
end

function remove_edges_local!(ψ, region, pre1, pre2)
  et = edgetype(ψ)
  for p in [pre1..., pre2...]
    for r in region
      rem_edge!(ψ, et((p, r)))
    end
  end
end

function vertices_to_inds(ψ, local_tensor, a)
  res = []
  for i in a
    push!(res, commonind(local_tensor, ψ[i]))
  end
  return res
end

function equal_bipartitions(set)
  n = length(set)
  result = []
  for i in 1:2^n-1
    A = [set[j] for j in 1:n if (i >> (j - 1)) & 1 == 1]
    B = [set[j] for j in 1:n if (i >> (j - 1)) & 1 == 0]
    if !isempty(A) && !isempty(B) && abs(length(A) - length(B)) < 1
      push!(result, [A, B])
    end
  end
  return result
end

function save_tree(ψ::AbstractTreeTensorNetwork, path::String)
  mkpath(dirname(path))
  println("Saving tree structure to $path")
  graph = underlying_graph(ψ)
  jldopen(path, "w") do file
    file["graph"] = graph
  end
end

export save_tree

using Graphs
using NamedGraphs: NamedGraph
using NamedGraphs.GraphsExtensions: add_edge
using ITensorNetworks
using LinearAlgebra
using JLD2

export AbstractQuantumGraph,
  MPSGraph, TreeGraph, TreeWithAncillaGraph, HierarchicalTreeGraph
export build_graph, add_fields

# Abstract base type for all quantum graph types
abstract type AbstractQuantumGraph end

# Concrete graph types
Base.@kwdef struct MPSGraph <: AbstractQuantumGraph
  L::Int  # Number of sites
  gridnum::Int  # Grid number for field generation
  with_ancilla::Bool = false  # Whether to include ancilla sites
end

Base.@kwdef struct TreeGraph <: AbstractQuantumGraph
  L::Int  # Linear dimension of the 2D square grid
  gridnum::Int  # Grid number for field generation
  with_ancilla::Bool = false  # Whether to include ancilla sites
  full_interaction::Bool = true
end

Base.@kwdef struct SnakeGraph <: AbstractQuantumGraph
  L::Int  # Linear dimension of the 2D square grid
  gridnum::Int  # Grid number for field generation
  with_ancilla::Bool = false  # Whether to include ancilla sites
  full_interaction::Bool = true
end

Base.@kwdef struct HilbertCurve <: AbstractQuantumGraph
  L::Int  # Linear dimension of the 2D square grid
  gridnum::Int  # Grid number for field generation
  with_ancilla::Bool = false  # Whether to include ancilla sites
  full_interaction::Bool = true
end

Base.@kwdef struct FreeGraph <: AbstractQuantumGraph
  L::Int  # Linear dimension of the 2D square grid
  gridnum::Int  # Grid number for field generation
  with_ancilla::Bool = false  # Whether to include ancilla sites
  full_interaction::Bool = true
  optimize_structure::Bool = true
  optimize_bonddim::Bool = true
  load_tree_path::Union{String,Nothing} = nothing
  save_tree::Bool = false
end

Base.@kwdef struct HierarchicalTreeGraph <: AbstractQuantumGraph
  L::Int
  gridnum::Int
end

function build_graph(graph::AbstractQuantumGraph)
  # Dispatch to specific implementation based on graph type
  return _build_graph(graph)
end

function _build_graph(graph_type::Union{FreeGraph,HierarchicalTreeGraph})
  L = graph_type.L
  gridnum = graph_type.gridnum

  data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))
  h_grid = data["grid"]

  if !isnothing(graph_type.load_tree_path)
    println("Loading tree from $(graph_type.load_tree_path)")
    g = load(graph_type.load_tree_path, "graph")
    @visualize g
    return g, h_grid
  end

  k = 1
  current_layer = Tuple.(CartesianIndices((L, L)))
  g = NamedGraph()

  for v in current_layer
    add_vertex!(g, (k, v...))
  end

  function left(g, cur_dim, k)
    new_dim = (cld(cur_dim[1], 2), cur_dim[2])
    new_layer = Tuple.(CartesianIndices(new_dim))
    for v in new_layer
      add_vertex!(g, (k, v...))
      add_edge!(g, (k - 1, v[1] * 2 - 1, v[2]) => (k, v...))
      if v[1] * 2 <= cur_dim[1]
        add_edge!(g, (k - 1, v[1] * 2, v[2]) => (k, v...))
      end
    end
    return new_dim
  end
  function right(g, cur_dim, k)
    new_dim = (cur_dim[1], cld(cur_dim[2], 2))
    new_layer = Tuple.(CartesianIndices(new_dim))
    for v in new_layer
      add_vertex!(g, (k, v...))
      add_edge!(g, (k - 1, v[1], v[2] * 2 - 1) => (k, v...))
      if v[2] * 2 <= cur_dim[2]
        add_edge!(g, (k - 1, v[1], v[2] * 2) => (k, v...))
      end
    end
    return new_dim
  end

  dim = (L, L)
  k = 2
  while any(d > 1 for d in dim)
    if iseven(k)
      dim = left(g, dim, k)
    else
      dim = right(g, dim, k)
    end
    k += 1
  end

  # Prune the tree to remove intermediate nodes with degree 2
  while true
    node_to_remove = nothing
    for v in vertices(g)
      # We are looking for intermediate nodes of the tree that are not branching points.
      # Physical sites are at layer 1 and are leaves of this construction (degree 1),
      # so they are not affected by the degree==2 check.
      # We can remove any non-physical node (layer > 1) if its degree is 2.
      if v[1] > 1 && degree(g, v) == 2
        node_to_remove = v
        break
      end
    end

    if isnothing(node_to_remove)
      break # No more nodes to remove
    else
      v = node_to_remove
      n1, n2 = neighbors(g, v)
      rem_vertex!(g, v)
      add_edge!(g, n1 => n2)
    end
  end

  return g, h_grid
end

function test_hier()
  tree = FreeGraph(12, 1, false, true, false, false, nothing, false)
  graph, _ = TTNEvo.build_graph(tree)

  jldopen("tree.jld2", "w") do file
    file["graph"] = graph
  end
end

function _build_graph2(graph_type::FreeGraph)
  L = graph_type.L
  gridnum = graph_type.gridnum

  data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))
  h_grid = data["grid"]

  if !isnothing(graph_type.load_tree_path)
    println("Loading tree from $(graph_type.load_tree_path)")
    g = load(graph_type.load_tree_path, "graph")
    @visualize g
    return g, h_grid
  end

  verts = Tuple.(CartesianIndices((L, L)))
  g = NamedGraph()

  for v in verts
    add_vertex!(g, (2, v...))
    add_vertex!(g, (1, v...))
  end

  for i in 1:(L-1)
    for j in 1:L
      add_edge!(g, (2, j, i) => (2, j, i + 1))
    end
  end
  for j in 1:(L-1)
    if isodd(j)
      add_edge!(g, (2, j, L) => (2, j + 1, L))
    else
      add_edge!(g, (2, j, 1) => (2, j + 1, 1))
    end
  end
  for i in 1:L
    for j in 1:L
      add_edge!(g, (1, i, j) => (2, i, j))
    end
  end


  rem_vertex!(g, (2, 1, 1))
  add_edge!(g, (1, 1, 1) => (2, 1, 2))
  if isodd(L)
    rem_vertex!(g, (2, L, L))
    add_edge!(g, (1, L, L) => (2, L, L - 1))
  else
    rem_vertex!(g, (2, L, 1))
    add_edge!(g, (1, L, 1) => (2, L, 2))
  end
  return g, h_grid
end


function _build_graph(graph_type::SnakeGraph)
  L = graph_type.L
  gridnum = graph_type.gridnum

  data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))
  h_grid = data["grid"]

  g_snake = NamedGraph(Graph(L^2), Tuple.(CartesianIndices((L, L))))

  for i in 1:(L-1)
    for j in 1:L
      add_edge!(g_snake, (j, i) => (j, i + 1))
    end
  end
  for j in 1:(L-1)
    if isodd(j)
      add_edge!(g_snake, (j, L) => (j + 1, L))
    else
      add_edge!(g_snake, (j, 1) => (j + 1, 1))
    end
  end

  # Add ancilla if requested
  if graph_type.with_ancilla
    @error "Not implementated snake with ancilla"
    g_snake = add_ancilla_system(L, g_snake)
  end

  return g_snake, h_grid
end

function _build_graph(graph_type::HilbertCurve)
  L = graph_type.L
  gridnum = graph_type.gridnum

  local h_grid
  try
    data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))
    h_grid = data["grid"]
  catch
    h_grid = rand(L, L)
  end

  s = hilbert_curve("A", log2(L))
  g = parse_hilbert_curve(s, L)
  return g, h_grid
end

function hilbert_expand(seed::AbstractString, order::Int)
  rules = Dict(
    'A' => "+BF-AFA-FB+",
    'B' => "-AF+BFB+FA-",
  )
  s = seed
  for _ in 1:order
    s = join(get(rules, c, string(c)) for c in s)  # passthrough F,+,-
  end
  return s
end
function parse_hilbert_curve(s, L)
  x, y = 1, 1
  # directions: 0=right, 1=up, 2=left, 3=down
  dir_ = 1
  dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
  points = [(x, y)]
  for ch in s
    if ch == 'F'
      dx, dy = dirs[dir_]
      x, y = x + dx, y + dy
      push!(points, (x, y))
    elseif ch == '+'  # left turn 90°
      dir_ = mod1(dir_ + 1, 4)
    elseif ch == '-'  # right turn 90°
      dir_ = mod1(dir_ - 1, 4)
    end
  end

  g = NamedGraph()
  for p in points
    add_vertex!(g, p)
  end
  for (i, p) in enumerate(points[1:end-1])
    add_edge!(g, points[i] => points[i+1])
  end
  return g
end

function hilbert_curve(current, order)
  if order <= 0
    return current
  end
  rules = Dict(
    'A' => "+BF-AFA-FB+",
    'B' => "-AF+BFB+FA-",
  )
  new = join([get(rules, c, string(c)) for c in current])
  return hilbert_curve(new, order - 1)
end

function build_random_graph(graph_type::TreeGraph; rng=nothing)
  L = graph_type.L
  gridnum = graph_type.gridnum

  # Load grid data
  data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))
  h_grid = data["grid"]  # Not used here, but you might want this later.

  dims = (L, L)
  g = NamedGraph(grid(dims), Tuple.(CartesianIndices((L, L))))
  for i in 1:L-1
    for j in 1:L-1
      add_edge!(g, (i, j) => (i + 1, j + 1))
    end
  end
  for i in 2:L
    for j in 1:L-1
      add_edge!(g, (i, j) => (i - 1, j + 1))
    end
  end

  edges = randomized_dfs_tree(g; rng)

  g = NamedGraph(Graph(L^2), Tuple.(CartesianIndices((L, L))))
  for e in edges
    add_edge!(g, e)
  end
  return g, h_grid
end

function randomized_dfs_tree(g::NamedGraph; rng=nothing)
  visited = Set{Tuple{Int,Int}}()
  tree_edges = Set{Tuple{Tuple{Int,Int},Tuple{Int,Int}}}()
  if isnothing(rng)
    start = rand(vertices(g))
  else
    start = rand(rng, vertices(g))
  end
  stack = [start]

  while !isempty(stack)
    current = stack[end]
    push!(visited, current)
    unvisited_neighbors = [v for v in neighbors(g, current) if v ∉ visited]
    if !isempty(unvisited_neighbors)
      if isnothing(rng)
        next_node = rand(unvisited_neighbors)
      else
        next_node = rand(rng, unvisited_neighbors)
      end
      push!(stack, next_node)
      push!(tree_edges, (current, next_node))
    else
      pop!(stack)
    end
  end

  return tree_edges
end

function _build_graph(graph_type::TreeGraph)
  L = graph_type.L
  gridnum = graph_type.gridnum

  # Create tree structure
  graph, h_grid = construct_mst_tree(L, gridnum)

  # Add ancilla if requested
  if graph_type.with_ancilla
    graph = add_ancilla_system(L, graph)
  end

  return graph, h_grid
end

function add_fields(graph::AbstractGraph, field_data)
  field_dict = Dict{eltype(vertices(graph)),Float64}()

  # Handle different field data types
  if field_data isa Matrix
    # For 2D field data on a 2D graph
    for v in vertices(graph)
      if length(v) == 2 && all(i -> 1 <= i <= size(field_data, 1), v)
        field_dict[v] = field_data[v...]
      end
    end
  elseif field_data isa Vector
    # For 1D field data on a 1D graph
    for (i, v) in enumerate(vertices(graph))
      if i <= length(field_data)
        field_dict[v] = field_data[i]
      end
    end
  end

  return field_dict
end

export load_h_chain, construct_tree, construct_mst_tree, add_ancilla_system

function load_h_chain(L, gridnum)
  data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))

  chain = data["chain"]
  return chain
end

function construct_tree(L, gridnum)
  data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))
  h_grid = data["grid"]

  g = NamedGraph(Graph(L^2), Tuple.(CartesianIndices((L, L))))
  ref_g = NamedGraph(grid((L, L)), Tuple.(CartesianIndices((L, L))))

  return add_edges(g, ref_g, h_grid, L), h_grid
end

function add_edges(g, ref_g, h_grid, L)
  initial_site = (L ÷ 2, L ÷ 2)
  visited = Set([initial_site])
  frontier = [(initial_site, n) for n in neighbors(ref_g, initial_site)]

  while !isempty(frontier)
    min_diff = Inf
    min_edge = nothing

    for (current, next) in frontier
      height_diff = h_grid[next[1], next[2]] - h_grid[current[1], current[2]]
      if height_diff < min_diff
        min_diff = height_diff
        min_edge = (current, next)
      end
    end

    current, next = min_edge
    add_edge!(g, current, next)
    push!(visited, next)

    filter!(e -> e[2] != next, frontier)

    for neighbor in neighbors(ref_g, next)
      if neighbor ∉ visited
        push!(frontier, (next, neighbor))
      end
    end
  end

  return g
end

function add_ancilla_system(L, g)
  g_anc = deepcopy(g)
  for v in vertices(g)
    add_vertex!(g_anc, v .+ (L, L))
    add_edge!(g_anc, v, v .+ (L, L))
  end
  return g_anc
end

function construct_mst_tree(L, gridnum)
  data = load(joinpath("data", "grid", "L=$(L)_gridnum=$gridnum.jld2"))
  h_grid = data["grid"]

  g_mst = NamedGraph(Graph(L^2), Tuple.(CartesianIndices((L, L))))
  for e in degree_limited_mst(h_grid, L, 3)[2]
    add_edge!(g_mst, e[1] => e[2])
  end
  return g_mst, h_grid
end

function add_ancilla_system!(g)
  vs = collect(vertices(g))
  for v in copy(vs)
    add_vertex!(g, v .+ (-1, 0, 0))
    add_edge!(g, v => v .+ (-1, 0, 0))
  end
end

struct UnionFind
  parent::Vector{Int}
  rank::Vector{Int}
end

UnionFind(size::Int) = UnionFind(collect(1:size), zeros(Int, size))

function find!(uf::UnionFind, x::Int)
  if uf.parent[x] != x
    uf.parent[x] = find!(uf, uf.parent[x])
  end
  return uf.parent[x]
end

function union!(uf::UnionFind, x::Int, y::Int)
  x_root = find!(uf, x)
  y_root = find!(uf, y)
  x_root == y_root && return false
  if uf.rank[x_root] < uf.rank[y_root]
    uf.parent[x_root] = y_root
  else
    uf.parent[y_root] = x_root
    uf.rank[x_root] += (uf.rank[x_root] == uf.rank[y_root])
  end
  return true
end

function degree_limited_mst(grid, N::Int, max_degree::Int)
  edges = []

  # Generate all edges with [i,j] vertices and weights
  for i in 1:N
    for j in 1:N
      # Right neighbor
      if j < N
        push!(edges, ((i, j), (i, j + 1), abs(grid[i, j] - grid[i, j+1])))
      end
      # Down neighbor
      if i < N
        push!(edges, ((i, j), (i + 1, j), abs(grid[i, j] - grid[i+1, j])))
      end
    end
  end

  # for i in 1:N-1
  #   for j in 1:N-1
  #     push!(edges, ((i, j), (i + 1, j + 1), abs(grid[i, j] - grid[i+1, j+1])))
  #   end
  # end
  # for i in 2:N
  #   for j in 1:N-1
  #     push!(edges, ((i, j), (i - 1, j + 1), abs(grid[i, j] - grid[i-1, j+1])))
  #   end
  # end


  # Sort edges by weight
  sort!(edges; by=x -> x[3])

  # Initialize data structures
  uf = UnionFind(N * N)
  degrees = Dict{Tuple{Int,Int},Int}([(i, j) => 0 for i in 1:N for j in 1:N])
  mst_edges = []

  # Convert [i,j] to linear index
  ij_to_linear(i, j) = (i - 1) * N + j

  for edge in edges
    u, v, weight = edge
    u_deg = degrees[u]
    v_deg = degrees[v]

    # Check degree constraints
    if u_deg >= max_degree || v_deg >= max_degree
      continue
    end

    # Check connectivity
    u_lin = ij_to_linear(u...)
    v_lin = ij_to_linear(v...)
    if find!(uf, u_lin) != find!(uf, v_lin)
      union!(uf, u_lin, v_lin)
      degrees[u] += 1
      degrees[v] += 1
      push!(mst_edges, edge)
    end
  end

  # Verify connectivity
  root = find!(uf, 1)
  is_connected = all(find!(uf, i) == root for i in 1:(N*N))

  return grid, mst_edges, is_connected
end

export build_graph


function generate_grids(L, number_of_grids)
  for gridnum in 1:number_of_grids
    grid = rand(rng, L, L)
    file = "data/grid/L=$(L)_gridnum=$gridnum.jld2"
    jldopen(file, "w") do file
      file["grid"] = grid
    end
  end
end

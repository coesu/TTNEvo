using ITensorUnicodePlots: @visualize
using ITensorNetworks: linkdims, IndsNetwork
using ITensorNetworks
using CairoMakie
using GraphMakie
using Graphs
using Graphs.SimpleGraphs: adj
using NetworkLayout
using NamedGraphs


function visualize_tree(L, grid_number)
  g, _ = construct_mst_tree(L, grid_number)
  sites = ITensorNetworks.siteinds("S=1/2", g)
  psi = ITensorNetworks.ttn(sites)
  @visualize psi
end

function vert_to_ind(graph, v)
  is = collect(vertices(graph))
  return findfirst(==(v), is)
end

function ind_to_vert(graph, ind::Integer)
  is = collect(vertices(graph))
  return is[ind]
end

function namedgraph_to_graph(graph)
  g = SimpleGraph(nv(graph))
  for edge in edges(graph)
    add_edge!(g, vert_to_ind(graph, src(edge)) => vert_to_ind(graph, dst(edge)))
  end
  return g
end

function plot_free_graph(graph)
  labels = [ind_to_vert(graph, i)[1] == 2 ? "  " : "$(ind_to_vert(graph, i)[2:3])" for i in 1:nv(graph)]
  cols = Dict(i => ind_to_vert(graph, i)[1] == 1 ? :red : :green for i in 1:nv(graph))
  # f, ax, p = graphplot(namedgraph_to_graph(graph, L); ilabels=labels, node_color=cols)
  f, ax, p = graphplot(namedgraph_to_graph(graph); node_color=cols, ilabels=labels)
  # f, ax, p = graphplot(namedgraph_to_graph(graph); node_color=[ind_to_vert(L, i)[1] == 1 ? :red : :black for i in 1:nv(graph)])
  return f
end

function plot_free_graph_with_maxdim(linkdim_graph; pin_spins=true)
  graph = linkdim_graph
  maxdims = [graph[e] for e in collect(edges(graph))]
  maxdims_label = ["$m" for m in maxdims]
  edge_width = [log2(m) for m in maxdims]
  edge_color = [1 - ew / maximum(edge_width) for ew in edge_width]
  labels = [v[1] == 2 ? "  " : "$(v[2:3])" for v in collect(vertices(graph))]
  # labels = ["$(v[2:3])" for v in collect(vertices(graph))]

  L = 4
  pin = Dict(v[1] == 1 ? vert_to_ind(graph, v) => 2 .* v[2:3] : vert_to_ind(graph, v) => false for v in collect(vertices(graph)))

  cols = Dict(i => ind_to_vert(graph, i)[1] == 1 ? :red : :green for i in 1:nv(graph))
  fig = Figure(; resolution=(1000, 1000))
  ax = Axis(fig[1, 1]; aspect=1)
  if pin_spins
    graphplot!(ax, namedgraph_to_graph(graph); layout=NetworkLayout.Spring(; pin), elabels=maxdims_label, edge_width, edge_color, node_color=cols, ilabels=labels)
  else
    graphplot!(ax, namedgraph_to_graph(graph); layout=NetworkLayout.Stress(), elabels=maxdims_label, edge_width, edge_color, node_color=cols, ilabels=labels)
  end
  return fig
end

function plot_coordinate_graph(graph)
  verts = vertices(graph)
  es = edges(graph)
  n = nv(graph)

  fig = Figure(; resolution=(800, 600))
  ax = Axis(fig[1, 1]; aspect=1, xlabel="X", ylabel="Y", title="Graph Visualization")

  if length(first(verts)) == 2
    x_coords = [v[1] for v in verts]
    y_coords = [v[2] for v in verts]
    for e in es
      lines!([src(e)[1], dst(e)[1]], [src(e)[2], dst(e)[2]]; color=:steelblue, linewidth=2)
    end
    for v in verts
      text!(v[1], v[2]; text="$(v)", align=(:center, :center), color=:black, fontsize=12)
    end
  elseif length(first(verts)) == 3
    x_coords = [v[2] for v in verts if v[1] == 2]
    y_coords = [v[3] for v in verts if v[1] == 2]
    for e in es
      lines!([src(e)[2], dst(e)[2]], [src(e)[3], dst(e)[3]]; color=:steelblue, linewidth=2)
    end
    for v in verts
      text!(v[2], v[3]; text="$(v)", align=(:center, :center), color=:black, fontsize=12)
    end
  end

  # Plot the vertices as scatter points
  scatter!(
    x_coords, y_coords; color=:skyblue, markersize=20, strokewidth=1, strokecolor=:black
  )

  # Add vertex labels

  # Set axis limits with some padding
  padding = 0.5
  xlims!(minimum(x_coords) - padding, maximum(x_coords) + padding)
  ylims!(minimum(y_coords) - padding, maximum(y_coords) + padding)

  return fig
end

export max_linkdim
function max_linkdim(psi)
  l = linkdims(psi)
  return maximum([l[e] for e in edges(l)])
end

function tdvp_tikz(graph)
  sweep = forward_sweep(
    direction(1),
    graph;
    root_vertex=(4, 3),
    nsites=1,
    reverse_step=false,
    region_kwargs=(),
    reverse_kwargs=(),
  )
  @show sweep

  res = ""
  processed = Set{Tuple{Int,Int}}()

  for (current, _) in sweep
    current = current[1]
    colors = Dict{Tuple{Int,Int},String}()

    for v in vertices(graph)
      if v == current
        colors[v] = "yellow!50"
      elseif v in processed
        colors[v] = "red!50"
      else
        colors[v] = "blue!50"
      end
    end

    res *= graph_tikz(graph; colors)
    push!(processed, current)
  end

  return res
end

export graph_tikz
function graph_tikz(graph; yoff=0, xoff=0, phys=false, colors=nothing)
  L = Int(sqrt(nv(graph)))

  res = "\\begin{tikzpicture}[tensornetwork, yscale=1.33]\n"
  for v in vertices(graph)
    if isnothing(colors)
      res *= "\\node[atensor]     (A$(v[1])-$(v[2])) at ($(v[1]-(L+1)/2), $(v[2]-(L+1)/2)) {};\n"
    else
      res *= "\\node[atensor,fill=$(colors[v])]     (A$(v[1])-$(v[2])) at ($(v[1]-(L+1)/2), $(v[2]-(L+1)/2)) {};\n"
    end
    if phys
      res *= "\\draw (A$(v[1])-$(v[2])) -- +(0.3,0.3);\n"
    end
  end
  for e in edges(graph)
    v = src(e)
    w = dst(e)

    res *= "\\draw (A$(v[1])-$(v[2])) -- (A$(w[1])-$(w[2]));\n"
  end
  res *= "\\end{tikzpicture}"
  println(res)
  return res
end

function graph_opt()
  L = 3
  g = NamedGraph(Graph(6), Tuple.(CartesianIndices((L - 1, L))))

  add_edge!(g, (1, 1) => (1, 2))
  add_edge!(g, (1, 2) => (1, 3))
  add_edge!(g, (1, 2) => (2, 2))
  add_edge!(g, (2, 1) => (2, 2))
  add_edge!(g, (2, 3) => (2, 2))
  display(plot_coordinate_graph(g))

  g = NamedGraph(Graph(6), Tuple.(CartesianIndices((L - 1, L))))
  add_edge!(g, (1, 1) => (1, 2))
  add_edge!(g, (2, 1) => (1, 2))
  add_edge!(g, (1, 2) => (2, 2))
  add_edge!(g, (2, 2) => (1, 3))
  add_edge!(g, (2, 2) => (2, 3))
  display(plot_coordinate_graph(g))
end

function number_size(ψ)
  sum = 0
  for v in vertices(ψ)
    sum += dim(ψ[v])
  end
  return sum
end

function max_possible_bonddimension(ψ)
  psi = deepcopy(ψ)
  curr = linkdims(ψ)
  for i in edges(curr)
    curr[i] = 0
  end

  for v in leaf_vertices(psi)
    ex_ind = uniqueinds(psi, v)
    e = only(incident_edges(psi, v))
    curr[e] = dim(ex_ind)
    rem_vertex!(psi, v)
  end
  while nv(psi) != 1
    for v in leaf_vertices(psi)
      ex_ind = uniqueinds(psi, v)
      if nv(psi) == 1
        continue
      end
      d = 1
      for e in incident_edges(curr, v)
        if curr[e] != 0
          d *= curr[e]
        end
      end
      @show d
      e = first(incident_edges(psi, v))
      curr[e] = d
      rem_vertex!(psi, v)
    end
  end
  return curr
end

function soft_max_dim(maxdim, maxdim_th, min=8)
  if maxdim_th <= min
    return maxdim_th
  end
end

function test_max_possible_bonddimension()
  config = TreeConfig(
    TimeEvolutionConfig((0, 0.2), 0.05, 120, 1, 1e-10, false, :twosite),
    InitialStateConfig(:neel, Dict(:initial_maxdim => 8, :qns => false)),
    "testing",
    HeisenbergModel(J1=1, JZ=1, h=5.0),
    FreeGraph(6, 1, false, true, true),
    Observer()
  )
  graph, field = build_graph(config.graph)
  dims = (config.graph.L, config.graph.L)
  graph_grid = NamedGraph(grid(dims), [(1, v...) for v in Tuple.(CartesianIndices(dims))])
  hamilt = build_hamiltonian(config.model, graph_grid, field)
  sites = siteinds(v -> v[1] == 1 ? "S=1/2" : "a", graph; conserve_qns=config.time_evolution.qns)
  H = ttn(hamilt, sites)
  ψ = build_initial_state(
    config.initial_state, graph, config.graph, sites
  )
  for v in vertices(ψ)
    @show dim(ψ[v])
  end
  return ψ
  max_possible_bonddimension(ψ)
end

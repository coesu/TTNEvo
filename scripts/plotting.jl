using CairoMakie
using Printf
using TTNEvo
using NamedGraphs
using ITensorNetworks
using Statistics
using HDF5
using Graphs

export plot_data_middle_sz

include("load.jl")

function plot_data_middle_sz(data::Vector{Any})
  fig = Figure()

  ax = Axis(
    fig[1, 1],
    title="Time Evolution",
    xlabel="Time",
    ylabel="⟨Sz(t)⟩",
    titlealign=:center,
    xticklabelsize=10,
    yticklabelsize=10
  )
  for r in data
    if r.graph.L == 10
      d = r.observer
      middle = r.graph.L ÷ 2
      sz_middle = real([x[(middle, middle)] for x in d.sz])

      lines!(ax, d.times, sz_middle, label="L=$(r.graph.L), h=$(r.model.h)")

    end

  end
  Legend(fig[1, 2], ax, "Legend")
  display(fig)
end

function plot_diff_sz(data)
  fig = Figure()

  ax = Axis(
    fig[1, 1],
    title="Difference between bond dimension 40 and 60",
    xlabel="Time",
    ylabel="χ₄₀ - χ₆₀",
    titlealign=:center,
    xticklabelsize=10,
    yticklabelsize=10
  )

  Ls = unique([d.graph.L for d in data])
  hs = unique([d.model.h for d in data])

  for L in Ls
    for h in hs
      # Find entries for this (L, h) pair
      i40 = findfirst(d -> d.graph.L == L && d.model.h == h && d.time_evolution.maxdim == 40, data)
      i60 = findfirst(d -> d.graph.L == L && d.model.h == h && d.time_evolution.maxdim == 60, data)

      if isnothing(i40) || isnothing(i60)
        @warn "Missing data for L=$L, h=$h"
        continue
      end

      d40 = data[i40]
      d60 = data[i60]

      if isnothing(d40) || isnothing(d60)
        @warn "Missing data for L=$L, h=$h"
        continue
      end

      # Extract time and observable
      middle = d40.graph.L ÷ 2
      t40, χ40 = d40.observer.times, real([x[(middle, middle)] for x in d40.observer.sz])
      t60, χ60 = d60.observer.times, real([x[(middle, middle)] for x in d60.observer.sz])

      # Match time grids if necessary (here assuming they are identical)
      if t40 != t60
        @warn "Mismatched time grids for L=$L, h=$h"
        continue
      end

      χ_diff = χ40 .- χ60

      lines!(ax, t40, χ_diff, label="L=$L, h=$h")
    end
  end

  axislegend(ax)
  fig
end


function plot_diff_sz(data)
  fig = Figure()

  ax = Axis(
    fig[1, 1],
    title="Difference between bond dimension 40 and 60",
    xlabel="Time",
    ylabel="χ₄₀ - χ₆₀",
    titlealign=:center,
    xticklabelsize=10,
    yticklabelsize=10
  )

  Ls = unique([d.graph.L for d in data])
  hs = unique([d.model.h for d in data])

  for L in Ls
    for h in hs
      # Find entries for this (L, h) pair
      i40 = findfirst(d -> d.graph.L == L && d.model.h == h && d.time_evolution.maxdim == 40, data)
      i60 = findfirst(d -> d.graph.L == L && d.model.h == h && d.time_evolution.maxdim == 60, data)

      if isnothing(i40) || isnothing(i60)
        @warn "Missing data for L=$L, h=$h"
        continue
      end

      d40 = data[i40]
      d60 = data[i60]

      if isnothing(d40) || isnothing(d60)
        @warn "Missing data for L=$L, h=$h"
        continue
      end

      # Extract time and observable
      middle = d40.graph.L ÷ 2
      t40, χ40 = d40.observer.times, real([x[(middle, middle)] for x in d40.observer.sz])
      t60, χ60 = d60.observer.times, real([x[(middle, middle)] for x in d60.observer.sz])

      # Match time grids if necessary (here assuming they are identical)
      if t40 != t60
        @warn "Mismatched time grids for L=$L, h=$h"
        continue
      end

      χ_diff = χ40 .- χ60

      lines!(ax, t40, χ_diff, label="L=$L, h=$h")
    end
  end

  axislegend(ax)
  fig
end


function plot_data_ee(data::Vector{Any})
  fig = Figure()

  ax = Axis(
    fig[1, 1],
    title="Entanglement Entropy",
    xlabel="Time",
    ylabel="EE",
    titlealign=:center,
    xticklabelsize=10,
    yticklabelsize=10
  )
  xlims!(ax, 0, 1)
  for r in data
    d = r.observer

    lines!(ax, d.times, d.ee)


  end
  display(fig)

end

function plot_all(data)
  fig = Figure(size=(1200, 800))

  ax_sz = Axis(fig[1, 1], title="Sz", xlabel="t", ylabel="S_z")
  ax_ee = Axis(fig[1, 2], title="EE", xlabel="t", ylabel="EE")
  ylims!(ax_ee, (0, 2))
  ax_maxdim = Axis(fig[2, 1], title="maxdim", xlabel="t", ylabel="maxdim")
  ax_extime = Axis(fig[2, 2], title="Execution time", xlabel="t", ylabel="Execution time")

  for d in data
    o = d.observer
    middle = d.graph.L ÷ 2
    sz = [real(x[(middle, middle)]) for x in o.sz]
    lines!(ax_sz, o.times, sz)
    lines!(ax_ee, o.times, o.ee)
    # lines!(ax_maxdim, o.times, o.maxdim)
    # lines!(ax_extime, o.times, o.extimes)
  end

  display(fig)
end

function plot_L(data, L)
  fig = Figure(size=(1200, 800))

  # ax_sz = Axis(fig[1, 1], title="L=$L, Sz expectation value of a middle site", xlabel="t", ylabel="S_z", xscale=log10)
  ax_sz = Axis(fig[1, 1], title="L=$L, Sz expectation value of a middle site", xlabel="t", ylabel="S_z")

  for d in data
    if d.graph.L == L
      o = d.observer
      middle = d.graph.L ÷ 2
      sz = [real(x[(middle, middle)]) for x in o.sz]
      lines!(ax_sz, o.times[2:end], sz[2:end], label="h=$(d.model.h), maxdim=$(d.time_evolution.maxdim), type=$(typeof(d.graph))")
    end
  end
  Legend(fig[1, 2], ax_sz)
  display(fig)
end

function average_by_bond(bond_tree, time_tree)
  if length(bond_tree) != length(time_tree)
    error("Input arrays bond_tree and time_tree must have the same length.")
  end

  aggregates = Dict{eltype(bond_tree),Tuple{Float64,Int}}()

  for i in eachindex(bond_tree)
    bond = bond_tree[i]
    time = time_tree[i]

    current_sum, current_count = get(aggregates, bond, (0.0, 0))

    aggregates[bond] = (current_sum + Float64(time), current_count + 1)
  end

  unique_bonds_sorted = sort(collect(keys(aggregates)))

  averaged_times = Float64[] # Initialize an empty array for averaged times
  sizehint!(averaged_times, length(unique_bonds_sorted)) # Optional: preallocate memory for efficiency

  for bond in unique_bonds_sorted
    sum_t, count_t = aggregates[bond]
    push!(averaged_times, sum_t / count_t)
  end

  return unique_bonds_sorted, averaged_times
end

function plot_tree_snake_time_bond(data)
  fig = Figure(size=(1200, 800))
  ax = Axis(fig[1, 1], title="Execution time vs bond dimensin", xlabel="D", ylabel="t", yscale=log10, xscale=log10)
  bond_snake = []
  time_snake = []

  bond_tree = []
  time_tree = []
  for d in data
    if d.graph.L == 6 && d.graph.gridnum == 4
      if typeof(d.graph) == TTNEvo.SnakeGraph
        push!(bond_snake, d.observer.maxdim)
        push!(time_snake, d.observer.ex_times)
      else
        push!(bond_tree, d.observer.maxdim)
        push!(time_tree, d.observer.ex_times)
      end
    end
  end
  bond_snake = reduce(vcat, bond_snake)
  time_snake = reduce(vcat, time_snake)
  bond_tree = reduce(vcat, bond_tree)
  time_tree = reduce(vcat, time_tree)

  bond_tree, time_tree = average_by_bond(bond_tree, time_tree)
  bond_snake, time_snake = average_by_bond(bond_snake, time_snake)

  scatter!(ax, bond_snake, time_snake, label="Snake")
  scatter!(ax, bond_tree, time_tree, label="Tree")
  Legend(fig[1, 2], ax)
  display(fig)

end

function plot_tree_snake_sz_only(data)
  fig = Figure(size=(1200, 800))
  ax_sz = Axis(fig.layout[1, 1], title=L"$\Delta S_z$ compared to time evolution with $D=160$ (Snake) and $D=80$ (Tree)", xlabel=L"t", ylabel=L"\Delta S_z")

  gridnum = 2
  h = 10.0
  snake = []
  tree = []
  for d in data
    if d.graph.L == 4 && d.model.h == h && d.graph.gridnum == gridnum
      label = nothing
      if typeof(d.graph) == SnakeGraph
        label = "Snake"
        push!(snake, d)
      else
        push!(tree, d)
      end
    end
  end
  sort!(snake, by=x -> x.time_evolution.maxdim)
  sort!(tree, by=x -> x.time_evolution.maxdim)

  times = snake[1].observer.times
  snake_sz = [[real(x[(2, 2)]) for x in y.observer.sz] for y in snake]

  @show tree[3]
  tree_sz = [[real(x[(2, 2)]) for x in y.observer.sz] for y in tree]

  @show length(tree_sz)
  @show length(snake_sz)
  @show [length(x) for x in snake_sz]
  @show [length(x) for x in tree_sz]

  for i in 1:length(tree_sz)-1
    lines!(ax_sz, times, snake_sz[i] .- snake_sz[end], label="Snake D=$(snake[i].time_evolution.maxdim)")
  end
  for i in 1:length(tree_sz)-1
    lines!(ax_sz, times, tree_sz[i] .- tree_sz[end], label="Tree D=$(tree[i].time_evolution.maxdim)")
  end

  Legend(fig[1, 2], ax_sz)
  save("plots/tree_snake_diff_grid$(gridnum)_h$h.pdf", fig)
  save("plots/tree_snake_diff_grid$(gridnum)_h$h.png", fig)
  display(fig)
end

function plot_tree_snake_sz(data)
  fig = Figure(size=(1200, 800))
  ax_sz = Axis(fig.layout[1, 1:2], title=L"S_z", xlabel=L"t", ylabel=L"S_z")
  ax_ee = Axis(fig.layout[2, 1:2], title="EE", xlabel=L"t", ylabel="EE", xscale=log10)
  ax_time = Axis(fig.layout[3, 1], title="Execution time per bond dimension", xlabel=L"D", ylabel=L"t_{ex}")
  ax_bond = Axis(fig.layout[3, 2], title="Execution time over time", xlabel=L"t", ylabel=L"t_{ex}")
  ax_bond2 = Axis(fig.layout[4, 1:2], title="Bond dimension", xlabel=L"t", ylabel=L"D")

  gridnum = 2
  h = 10.0
  ylims!(ax_time, (0, 4))
  ylims!(ax_bond, (0, 4))
  xlims!(ax_ee, (0.1, 50))
  for d in data
    if d.graph.L == 4 && d.model.h == h && d.graph.gridnum == gridnum
      label = nothing
      if typeof(d.graph) == SnakeGraph
        label = "Snake, D=$(d.time_evolution.maxdim)"
      else
        label = "Tree, D=$(d.time_evolution.maxdim)"
      end
      times = d.observer.times

      sz = [real(x[(2, 2)]) for x in d.observer.sz]
      scatter!(ax_time, d.observer.maxdim, d.observer.ex_times, label=label)
      lines!(ax_bond, times, d.observer.ex_times, label=label)
      lines!(ax_bond2, times, d.observer.maxdim, label=label)
      lines!(ax_ee, times, d.observer.ee, label=label)
      lines!(ax_sz, times, sz, label=label)
    end
  end

  Legend(fig[2, 3], ax_sz)
  save("plots/tree_snake_comparison_grid$(gridnum)_h$h.pdf", fig)
  save("plots/tree_snake_comparison_grid$(gridnum)_h$h.png", fig)
  display(fig)
end

function plot_tree_diff_h(data, L, maxdim)
  fig = Figure(size=(1600, 600))
  ax_sz = Axis(fig.layout[1, 1], title=L"L=%$L, averaged $S_z$ and entanglement entropy for different disorder strength $h$", xlabel=L"t", ylabel=L"S_z")
  ax_ee = Axis(fig.layout[1, 2], xlabel=L"t", ylabel="EE", xscale=log10)

  tree = []
  xlims!(ax_ee, (0.1, 50))
  for d in data
    if d.graph.L == L && d.time_evolution.maxdim == maxdim
      if typeof(d.graph) == TreeGraph
        push!(tree, d)
      end
    end
  end
  h = [1.0, 5.0, 10.0]
  tr = [filter(x -> x.model.h == hi, tree) for hi in h]
  @show length(tr[1])

  middle = L ÷ 2
  # tree_sz = [[real(x[(2, 2)]) for x in y.observer.sz] for y in tree]
  ts = [sum([[real(y[(middle, middle)]) ./ 4 for y in z.observer.sz] for z in x]) for x in tr]
  ee = [sum([z.observer.ee for z in x]) ./ 4 for x in tr]
  for i in 1:3
    lines!(ax_sz, tr[1][1].observer.times, ts[i], label=L"$h$=%$(h[i])")

    lines!(ax_ee, tr[1][1].observer.times, ee[i], label=L"$h$=%$(h[i])")
  end


  axislegend(ax_sz)
  save("plots/tree_snake_avereged_L=$L.pdf", fig)
  save("plots/tree_snake_avereged_L=$L.png", fig)
  display(fig)
end

using CairoMakie

function plot_cbe_twosite()
  data = load_dir("data/cbe_2site_full/")

  set_theme!(palette=(color=["#4263eb",
    "#f03e3e",
    "#0a9a84",
    "#191e44",
    "#f76707",
    "#7143e0",
    "#f2cc35",
    "#791457",
    "#6adad3",
    "#cf6db0"
  ],))
  fig = Figure(resolution=(1200, 800))

  fig.layout[1, 1] = GridLayout()
  fig.layout[1, 2] = GridLayout()
  fig.layout[2, 1] = GridLayout()
  fig.layout[2, 2] = GridLayout()
  fig.layout[3, 1:2] = GridLayout()

  # Create axes
  ax_sz = Axis(fig.layout[1, 1], title=L"S_z", xlabel=L"t", ylabel=L"S_z")
  ax_ee = Axis(fig.layout[1, 2], title="EE", xlabel=L"t", ylabel="EE")
  ax_time = Axis(fig.layout[2, 1], title="Execution time per bond dimension", xlabel=L"D", ylabel=L"t_{ex}")
  ax_bond = Axis(fig.layout[2, 2], title="Execution time over time", xlabel=L"t", ylabel=L"t_{ex}")
  ax_bond2 = Axis(fig.layout[3, 1:2], title="Bond dimension", xlabel=L"t", ylabel=L"D")

  for d in data
    if d.time_evolution.t_step == 0.02 &&
       d.time_evolution.t_range[2] == 2 &&
       d.graph.L == 4 &&
       d.graph.gridnum == 2

      label = String(d.time_evolution.method)
      times = d.observer.times

      sz = [real(x[(2, 2)]) for x in d.observer.sz]

      scatter!(ax_time, d.observer.maxdim, d.observer.ex_times, label=label)
      lines!(ax_bond, times, d.observer.ex_times, label=label)
      lines!(ax_bond2, times, d.observer.maxdim, label=label)
      lines!(ax_ee, times, d.observer.ee, label=label)
      lines!(ax_sz, times, sz, label=label)  # Use sz for S_z plot
    end
  end

  Legend(fig[1, 3], ax_sz)
  save("plots/cbe_comparison.pdf", fig)
  save("plots/cbe_comparison.png", fig)
  display(fig)
end

function filter_configs(
  data::Vector{Any};
  t_step::Union{Nothing,Float64}=nothing,
  t_range_end::Union{Nothing,Real}=nothing,
  L::Union{Nothing,Int}=nothing,
  gridnum::Union{Nothing,Int}=nothing,
  method::Union{Nothing,Symbol}=nothing,
  h::Union{Nothing,Float64}=nothing,
  free::Union{Nothing,Bool}=nothing,
  tree_opt::Union{Nothing,Bool}=nothing,
  maxdim_opt::Union{Nothing,Bool}=nothing,
  maxdim::Union{Nothing,Int}=nothing,
)
  # Use Julia's built-in filter function with a do-block for the predicate
  return filter(data) do d
    match_t_step = isnothing(t_step) || (d.time_evolution.t_step == t_step)
    match_t_range_end = isnothing(t_range_end) || (d.time_evolution.t_range[2] == t_range_end)
    match_L = isnothing(L) || (d.graph.L == L)
    match_gridnum = isnothing(gridnum) || (d.graph.gridnum == gridnum)
    match_method = isnothing(method) || (d.time_evolution.method == method)
    match_h = isnothing(h) || (d.model.h == h)
    match_maxdim = isnothing(maxdim) || (d.time_evolution.maxdim == maxdim)
    match_free = isnothing(free) || (isa(d.graph, FreeGraph) == free)
    match_tree_opt = isnothing(tree_opt) || (isa(d.graph, FreeGraph) && (d.graph.optimize_structure == tree_opt)) || (isa(d.graph, TreeGraph) && !tree_opt)
    match_maxdim_opt = isnothing(maxdim_opt) || (isa(d.graph, FreeGraph) && (d.graph.optimize_bonddim == maxdim_opt))

    return match_t_step && match_t_range_end && match_L && match_gridnum && match_method && match_h && match_tree_opt && match_free && match_maxdim && match_maxdim_opt
  end
end

function plot_cbe_full_comp(data)
  fig = Figure(resolution=(800, 1200))

  ax_sz = Axis(fig[1, 1], title="h=50", xlabel=L"t", ylabel=L"\chi")
  for d in data
    times = d.observer.times

    sz = [real(x[(1, 3, 3)][2]) for x in d.observer.sz]

    lines!(ax_sz, times, sz, label="$(d.time_evolution.method)")
  end
  ax_diff = Axis(fig[2, 1], title="h=50", xlabel=L"t", ylabel=L"\chi")
  times = data[1].observer.times
  sz1 = [real(x[(1, 3, 3)][2]) for x in data[1].observer.sz]
  sz2 = [real(x[(1, 3, 3)][2]) for x in data[2].observer.sz]
  sz3 = [real(x[(1, 3, 3)][2]) for x in data[2].observer.sz]
  # lines!(ax_diff, times, sz1 .- sz2)
  @show sz2 .- sz3
  lines!(ax_diff, times, sz2 .- sz3)
  ylims!(ax_diff, (-1e-14, 1e-14))
  # lines!(ax_diff, times, sz3 .- sz1)
  Legend(fig[1, 2], ax_sz)
  display(fig)
end

function plot_tree_opt(data, L=4, gridnum=1)
  fig = Figure(resolution=(800, 1200))

  hs = [0.0, 1.0, 2.0, 5.0, 10.0]
  for (i, h) in enumerate(hs)
    ax_maxdim = Axis(fig[i, 1], title="h=$h", xlabel=L"t", ylabel=L"\chi")
    d1 = only(filter_configs(data; L, gridnum, h, tree_opt=false, free=false))
    d2 = only(filter_configs(data; L, gridnum, h, tree_opt=true, free=true))
    d3 = only(filter_configs(data; L, gridnum, h, tree_opt=false, free=true))

    labels = ["Heuristic Tree", "Tree opt", "Snake"]

    for (d, l) in zip([d1, d2, d3], labels)
      times = d.observer.times
      maxdim = d.observer.maxdim
      lines!(ax_maxdim, times, maxdim, label=l)
    end

    Legend(fig[1, 2], ax_maxdim)
  end
  display(fig)
  save("plots/free/L=$(L)_gridnum=$(gridnum)_maxdim.png", fig)
  save("plots/free/L=$(L)_gridnum=$(gridnum)_maxdim.pdf", fig)

  fig = Figure(resolution=(800, 1200))
  for (i, h) in enumerate(hs)
    ax_maxdim = Axis(fig[i, 1], title="h=$h", xlabel=L"t", ylabel="Memory")
    d1 = only(filter_configs(data; L, gridnum, h, tree_opt=false, free=false))
    d2 = only(filter_configs(data; L, gridnum, h, tree_opt=true, free=true))
    d3 = only(filter_configs(data; L, gridnum, h, tree_opt=false, free=true))

    labels = ["Heuristic Tree", "Tree opt", "Snake"]

    for (d, l) in zip([d1, d2, d3], labels)
      times = d.observer.times
      maxdim = d.observer.memory
      lines!(ax_maxdim, times, maxdim, label=l)
    end

    Legend(fig[1, 2], ax_maxdim)
  end
  display(fig)
  save("plots/free/L=$(L)_gridnum=$(gridnum)_memory.png", fig)
  save("plots/free/L=$(L)_gridnum=$(gridnum)_memory.pdf", fig)

  fig = Figure(resolution=(800, 1200))
  for (i, h) in enumerate(hs)
    ax_maxdim = Axis(fig[i, 1], title="h=$h", xlabel=L"t", ylabel="Execution time", yscale=log10)
    d1 = only(filter_configs(data; L, gridnum, h, tree_opt=false, free=false))
    d2 = only(filter_configs(data; L, gridnum, h, tree_opt=true, free=true))
    d3 = only(filter_configs(data; L, gridnum, h, tree_opt=false, free=true))

    labels = ["Heuristic Tree", "Tree opt", "Snake"]

    for (d, l) in zip([d1, d2, d3], labels)
      times = d.observer.times
      maxdim = d.observer.ex_times
      lines!(ax_maxdim, times, maxdim, label=l)
    end

    Legend(fig[1, 2], ax_maxdim)
  end
  display(fig)
  save("plots/free/L=$(L)_gridnum=$(gridnum)_ex_time.png", fig)
  save("plots/free/L=$(L)_gridnum=$(gridnum)_ex_time.pdf", fig)
end

function plot_all_the_shit(data)
  fig = Figure(resolution=(800, 1200))
  ax_imbalance = Axis(fig[1, 1], title="Imbalance", xlabel=L"t", ylabel="Imbalance", yscale=log10)

  t = [d.observer.times for d in data]
  imb = [[imbalance(x) for x in d.observer.sz] for d in data]

  println(abs.(imb[1] - imb[2]))
  lines!(ax_imbalance, t[1], abs.(imb[1] - imb[2]) .+ 1e-10, label="cbe")
  lines!(ax_imbalance, t[2], abs.(imb[1] - imb[3]), label="twosite")
  lines!(ax_imbalance, t[2], abs.(imb[2] - imb[3]), label="twosite")

  ax_memory = Axis(fig[2, 1], title="Number of params", xlabel=L"t", ylabel="N")

  t = [d.observer.times for d in data]
  imb = [d.observer.num_size for d in data]

  lines!(ax_memory, t[1], imb[1], label="cbe")
  lines!(ax_memory, t[2], imb[2], label="2site")
  lines!(ax_memory, t[2], imb[3], label="1site")

  ax_maxdim = Axis(fig[3, 1], title="Maxdim", xlabel=L"t", ylabel="N")

  t = [d.observer.times for d in data]
  x = data[1].observer.maxdim[1]

  @show edges(x)

  imb = [[max_linkdim_graph(x) for x in d.observer.maxdim] for d in data]

  lines!(ax_maxdim, t[1], imb[1], label="cbe")
  lines!(ax_maxdim, t[2], imb[2], label="2site")
  lines!(ax_maxdim, t[2], imb[3], label="1site")
  Legend(fig, ax_memory)

  display(fig)
end

function max_linkdim_graph(link)
  return maximum([link[e] for e in edges(link)])
end

function plot_tree_snake_comp_sz(data_tree, data_snake)
  fig = Figure(resolution=(800, 1200))

  # Define the maxdim values and corresponding colors
  maxdim_values = [60, 120]
  # 1. Define a list of colors to cycle through.
  #    You can choose any colors you like, e.g., from Makie.ColorSchemes
  colors = [:blue, :red, :green]

  for (i, h) in enumerate([10.0, 20.0, 30.0])
    ax_imbalance = Axis(fig[i, 1],
      title="h=$h",
      xlabel=L"t",
      ylabel="Imbalance",
    )
    # Get the time vector once per subplot
    t = data[1].observer.times

    # Use enumerate to get an index `j` for our color list
    for (j, maxdim) in enumerate(maxdim_values)
      num_grids = 10
      # all_imbalances = zeros(length(t), num_grids)
      all_imbalances_snake = Vector{Vector{Float64}}()
      all_imbalances_tree = Vector{Vector{Float64}}()

      for gridnum in 1:num_grids
        # This assumes `only` will always find exactly one match
        d_snake = try
          only(filter_configs(data_snake; h, maxdim, gridnum))
        catch
          continue
        end

        d_tree = try
          only(filter_configs(data_tree; h, maxdim, gridnum))
        catch
          continue
        end

        imbalance_trajectory_snake = [real(x[3, 3]) for x in d_snake.observer.sz]
        imbalance_trajectory_tree = [real(x[1, 3, 3]) for x in d_tree.observer.sz]
        # all_imbalances[:, gridnum] = imbalance_trajectory
        push!(all_imbalances_snake, imbalance_trajectory_snake)
        push!(all_imbalances_tree, imbalance_trajectory_tree)
      end

      all_imbalances_snake = hcat(all_imbalances_snake...)
      # Calculate statistics
      avrg_imb_snake = vec(mean(all_imbalances_snake, dims=2))
      # This is the Standard Error of the Mean (SEM), which is great for plots!
      std_err_imb_snake = vec(std(all_imbalances_snake, dims=2)) / sqrt(size(all_imbalances_snake)[2])

      line_color = colors[j]
      band_color = (line_color, 0.2)

      band!(ax_imbalance, t, avrg_imb_snake .- std_err_imb_snake, avrg_imb_snake .+ std_err_imb_snake, color=band_color)
      lines!(ax_imbalance, t, avrg_imb_snake, label="maxdim=$maxdim_snake, snake", color=line_color)


      all_imbalances_tree = hcat(all_imbalances_tree...)
      # Calculate statistics
      avrg_imb_tree = vec(mean(all_imbalances_tree, dims=2))
      # This is the Standard Error of the Mean (SEM), which is great for plots!
      std_err_imb_tree = vec(std(all_imbalances_tree, dims=2)) / sqrt(size(all_imbalances_tree)[2])

      line_color = colors[j]
      band_color = (line_color, 0.2)

      band!(ax_imbalance, t, avrg_imb_tree .- std_err_imb_tree, avrg_imb_tree .+ std_err_imb_tree, color=band_color)
      lines!(ax_imbalance, t, avrg_imb_tree, label="maxdim=$maxdim_tree, tree", color=line_color)
    end

    axislegend(ax_imbalance)
  end
  return fig # It's good practice to return the figure object
end

function print_config(data_tree, data_snake)
  maxdim_values = [60, 120]
  for (i, h) in enumerate([10.0, 20.0, 30.0])
    for (j, maxdim) in enumerate(maxdim_values)
      c_snake = only(filter_configs(data_snake; h, maxdim, gridnum=1))
      c_tree = only(filter_configs(data_tree; h, maxdim, gridnum=1))
      @show c_snake.time_evolution == c_tree.time_evolution
      @show c_snake.model == c_tree.model
      @show c_snake.initial_state == c_tree.initial_state
    end
  end

end

function plot_tree_snake_comp(data_tree, data_snake)
  fig = Figure(resolution=(800, 1200))

  # Define the maxdim values and corresponding colors
  maxdim_values_snake = [64, 128]
  maxdim_values_tree = [60, 120]
  # 1. Define a list of colors to cycle through.
  #    You can choose any colors you like, e.g., from Makie.ColorSchemes
  colors = [:blue, :red, :green]

  for (i, h) in enumerate([10.0, 20.0, 30.0, 50.0])
    ax_imbalance = Axis(fig[i, 1],
      title="h=$h",
      xlabel=L"t",
      ylabel="Imbalance",
    )
    # Get the time vector once per subplot
    t = data_tree[1].observer.times

    # Use enumerate to get an index `j` for our color list
    for (j, (maxdim_snake, maxdim_tree)) in enumerate(zip(maxdim_values_snake, maxdim_values_tree))
      num_grids = 10
      # all_imbalances = zeros(length(t), num_grids)
      all_imbalances_snake = Vector{Vector{Float64}}()
      all_imbalances_tree = Vector{Vector{Float64}}()

      for gridnum in 1:num_grids
        # This assumes `only` will always find exactly one match
        d_snake = try
          only(filter_configs(data_snake; h, maxdim=maxdim_snake, gridnum))

        catch
          continue
        end

        d_tree = try
          only(filter_configs(data_tree; h, maxdim=maxdim_tree, gridnum))
        catch
          continue
        end

        imbalance_trajectory_snake = [real(columnar_imbalance_snake(x)) for x in d_snake.observer.sz]
        imbalance_trajectory_tree = [real(columnar_imbalance(x)) for x in d_tree.observer.sz]
        # all_imbalances[:, gridnum] = imbalance_trajectory
        push!(all_imbalances_snake, imbalance_trajectory_snake)
        push!(all_imbalances_tree, imbalance_trajectory_tree)
      end
      @show length(all_imbalances_snake)
      @show length(all_imbalances_tree)

      all_imbalances_snake = hcat(all_imbalances_snake...)
      # Calculate statistics
      avrg_imb_snake = vec(mean(all_imbalances_snake, dims=2))
      # This is the Standard Error of the Mean (SEM), which is great for plots!
      std_err_imb_snake = vec(std(all_imbalances_snake, dims=2)) / sqrt(size(all_imbalances_snake)[2])

      line_color = colors[j]
      band_color = (line_color, 0.2)

      band!(ax_imbalance, t, avrg_imb_snake .- std_err_imb_snake, avrg_imb_snake .+ std_err_imb_snake, color=band_color)
      lines!(ax_imbalance, t, avrg_imb_snake, label="maxdim=$maxdim_snake, snake", color=line_color)


      all_imbalances_tree = hcat(all_imbalances_tree...)
      # Calculate statistics
      avrg_imb_tree = vec(mean(all_imbalances_tree, dims=2))
      # This is the Standard Error of the Mean (SEM), which is great for plots!
      std_err_imb_tree = vec(std(all_imbalances_tree, dims=2)) / sqrt(size(all_imbalances_tree)[2])

      line_color = colors[j]
      band_color = (line_color, 0.2)

      band!(ax_imbalance, t, avrg_imb_tree .- std_err_imb_tree, avrg_imb_tree .+ std_err_imb_tree, color=band_color)
      lines!(ax_imbalance, t, avrg_imb_tree, label="maxdim=$maxdim_tree, tree", color=line_color)
    end

    axislegend(ax_imbalance)
  end
  return fig # It's good practice to return the figure object
end

function plot_large_tree(data)
  fig = Figure(resolution=(800, 1200))

  # Define the maxdim values and corresponding colors
  maxdim_values = [60, 120]
  # 1. Define a list of colors to cycle through.
  #    You can choose any colors you like, e.g., from Makie.ColorSchemes
  colors = [:blue, :red, :green]

  for (i, h) in enumerate([10.0, 20.0, 30.0])
    ax_imbalance = Axis(fig[i, 1],
      title="h=$h",
      xlabel=L"t",
      ylabel="Imbalance",
    )
    # Get the time vector once per subplot
    t = data[1].observer.times

    # Use enumerate to get an index `j` for our color list
    for (j, maxdim) in enumerate(maxdim_values)
      num_grids = 10
      # all_imbalances = zeros(length(t), num_grids)
      all_imbalances = Vector{Vector{Float64}}()

      for gridnum in 1:num_grids
        # This assumes `only` will always find exactly one match
        d = try
          only(filter_configs(data; h, maxdim, gridnum))
        catch
          continue
        end

        imbalance_trajectory = [real(columnar_imbalance(x)) for x in d.observer.sz]
        # all_imbalances[:, gridnum] = imbalance_trajectory
        push!(all_imbalances, imbalance_trajectory)
      end

      all_imbalances = hcat(all_imbalances...)
      # Calculate statistics
      avrg_imb = vec(mean(all_imbalances, dims=2))
      # This is the Standard Error of the Mean (SEM), which is great for plots!
      std_err_imb = vec(std(all_imbalances, dims=2)) / sqrt(size(all_imbalances)[2])

      line_color = colors[j]
      band_color = (line_color, 0.2)

      band!(ax_imbalance, t, avrg_imb .- std_err_imb, avrg_imb .+ std_err_imb, color=band_color)
      lines!(ax_imbalance, t, avrg_imb, label="maxdim=$maxdim", color=line_color)
    end

    axislegend(ax_imbalance)
  end
  return fig # It's good practice to return the figure object
end

function plot_large_snake(data)
  fig = Figure(resolution=(800, 1200))

  # Define the maxdim values and corresponding colors
  maxdim_values = [60, 100, 120]
  # 1. Define a list of colors to cycle through.
  #    You can choose any colors you like, e.g., from Makie.ColorSchemes
  colors = [:blue, :red, :green]

  for (i, h) in enumerate([10.0, 20.0, 30.0])
    ax_imbalance = Axis(fig[i, 1],
      title="h=$h",
      xlabel=L"t",
      ylabel="Imbalance",
    )
    # Get the time vector once per subplot
    t = data[1].observer.times

    # Use enumerate to get an index `j` for our color list
    for (j, maxdim) in enumerate(maxdim_values)
      num_grids = 10
      all_imbalances = zeros(length(t), num_grids)

      for gridnum in 1:num_grids
        # This assumes `only` will always find exactly one match
        d = only(filter_configs(data; h, maxdim, gridnum))
        imbalance_trajectory = [real(columnar_imbalance_snake(x)) for x in d.observer.sz]
        all_imbalances[:, gridnum] = imbalance_trajectory
      end

      # Calculate statistics
      avrg_imb = vec(mean(all_imbalances, dims=2))
      # This is the Standard Error of the Mean (SEM), which is great for plots!
      std_err_imb = vec(std(all_imbalances, dims=2)) / sqrt(num_grids)

      line_color = colors[j]
      band_color = (line_color, 0.2)

      band!(ax_imbalance, t, avrg_imb .- std_err_imb, avrg_imb .+ std_err_imb, color=band_color)
      lines!(ax_imbalance, t, avrg_imb, label="maxdim=$maxdim", color=line_color)
    end

    axislegend(ax_imbalance)
  end
  display(fig)
  return fig # It's good practice to return the figure object
end

function plot_comp_snake(data_snake)
  fig = Figure(resolution=(800, 1200))

  ax_imbalance = Axis(fig[1, 1], title="Imbalance", xlabel=L"t", ylabel="Imbalance")

  tree = only(load_dir("data/free_test/"; time_after=DateTime(2025, 6, 26, 10, 0, 0)))

  snake = only(filter_configs(data_snake; h=20.0, maxdim=120, gridnum=1))

  t = snake.observer.times
  imb = [columnar_imbalance_snake(x) for x in snake.observer.sz]

  lines!(ax_imbalance, t, imb, label="snake")

  t = tree.observer.times
  imb = [columnar_imbalance(x) for x in tree.observer.sz]

  lines!(ax_imbalance, t, imb, label="tree")
  # xlims!(ax_imbalance, (0, 1.1))
  Legend(fig, ax_imbalance)
  return fig
end

function plot_comp_snake2(data)
  fig = Figure(resolution=(800, 1200))

  ax_imbalance = Axis(fig[1, 1], title="Imbalance", xlabel=L"t", ylabel="Imbalance", yscale=log10)

  tree = data[1]
  tree2 = data[2]

  snake = data[3]
  snake2 = data[4]

  t = snake.observer.times
  imb_snake = [columnar_imbalance_snake(x) for x in snake.observer.sz]
  imb_snake2 = [columnar_imbalance_snake(x) for x in snake2.observer.sz]

  # lines!(ax_imbalance, t, imb, label="snake")

  t = tree.observer.times
  imb_tree = [columnar_imbalance(x) for x in tree.observer.sz]
  imb_tree2 = [columnar_imbalance(x) for x in tree2.observer.sz]

  lines!(ax_imbalance, t, abs.(imb_snake - imb_tree), label="large tree")
  lines!(ax_imbalance, t, abs.(imb_tree2 - imb_tree), label="tree")
  lines!(ax_imbalance, t, abs.(imb_snake2 - imb_tree), label="small tree")
  lines!(ax_imbalance, t, abs.(imb_snake2 - imb_snake), label="snakes")
  xlims!(ax_imbalance, (0, 1.1))
  ylims!(ax_imbalance, (1e-13, 1e-5))
  Legend(fig, ax_imbalance)
  return fig
end


function plot_test_bad_snake_good_tree(data)
  fig = Figure(resolution=(800, 1200))
  ax_imbalance = Axis(fig[1, 1], title="Imbalance", xlabel=L"t", ylabel="Imbalance", yscale=log10)

  t = [d.observer.times for d in data]
  imb = [[imbalance_snake(x) for x in d.observer.sz] for d in data]

  println(abs.(imb[1] - imb[2]))
  lines!(ax_imbalance, t[1], abs.(imb[1] - imb[2]) .+ 1e-10, label="diff")
  # lines!(ax_imbalance, t[2], abs.(imb[1] - imb[3]), label="twosite")

  ax_memory = Axis(fig[2, 1], title="Number of params", xlabel=L"t", ylabel="N")

  t = [d.observer.times for d in data]
  imb = [d.observer.num_size for d in data]

  labels = ["$(typeof(d.graph))" for d in data]

  for i in 1:2
    lines!(ax_memory, t[i], imb[i], label=labels[i])
  end

  ax_maxdim = Axis(fig[3, 1], title="Maxdim", xlabel=L"t", ylabel="N")

  t = [d.observer.times for d in data]
  imb = [[max_linkdim_graph(x) for x in d.observer.maxdim] for d in data]

  for i in 1:2
    lines!(ax_maxdim, t[i], imb[i], label=labels[i])
  end

  ax_ex_time = Axis(fig[4, 1], title="Execution time", xlabel=L"t", ylabel="T")


  t = [d.observer.times for d in data]
  imb = [d.observer.ex_times for d in data]

  for i in 1:2
    lines!(ax_ex_time, t[i], imb[i], label=labels[i])
  end

  Legend(fig, ax_memory)

  display(fig)
end


function plot_ttn_comparison(data)
  sz = h5read("../TTN.jl/results/obs_J_-1_g_-0.5_h_20_chi_64.h5", "sz")
  time = h5read("../TTN.jl/results/obs_J_-1_g_-0.5_h_20_chi_64.h5", "time")
  @show sz
  @show time
  sorted_keys = sort(collect(keys(time)), by=k -> parse(Int, k))

  sz_array = [sz[k] for k in sorted_keys]
  time_array = [time[k] for k in sorted_keys]

  @show sz_array
  @show typeof(sz_array)

  fig = Figure(resolution=(800, 1200))
  ax_imbalance = Axis(fig[1, 1], title="Imbalance", xlabel=L"t", ylabel="Imbalance")

  lines!(ax_imbalance, time_array, abs.([columnar_imbalance(x) for x in sz_array]), label="hirarchical")

  for d in data
    times = d.observer.times
    if typeof(d.graph) == SnakeGraph
      sz = [columnar_imbalance_snake(x) for x in d.observer.sz]
    else
      sz = [columnar_imbalance(x) for x in d.observer.sz]
    end
    lines!(ax_imbalance, times, abs.(sz), label="$(typeof(d.graph))")
  end
  Legend(fig, ax_imbalance)

  return fig
end


function plot_diff_tree_snake(data)
  fig = Figure(resolution=(800, 800))

  ax_imbalance = Axis(fig[1, 1], title="Imbalance", xlabel=L"t", ylabel="Imbalance")
  ax_sz = Axis(fig[2, 1], title="Sz", xlabel=L"t", ylabel="Sz")

  t = data[1].observer.times
  imb_tree = [columnar_imbalance(x) for x in data[1].observer.sz]
  imb_snake = [columnar_imbalance_snake(x) for x in data[2].observer.sz]

  sz_tree = [real(x[(1, 2, 2)][2]) for x in data[1].observer.sz]
  sz_snake = [real(x[(2, 2)]) for x in data[2].observer.sz]

  lines!(ax_imbalance, t, imb_tree - imb_snake, label="tree")
  # lines!(ax_imbalance, t, imb_snake, label="snake")

  lines!(ax_sz, t, sz_tree - sz_snake, label="tree")
  # lines!(ax_sz, t, sz_snake, label="snake")
  Legend(fig, ax_sz)
  return fig
end

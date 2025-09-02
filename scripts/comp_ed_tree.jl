using CairoMakie
using Printf
using TTNEvo
using NamedGraphs
using ITensorNetworks
using Statistics
using HDF5
using Graphs
using JLD2

function columnar_imbalance_ed(sz_matrix, L)
  imbalances = Float64[]
  for t in 1:size(sz_matrix, 1)
    sz_t = sz_matrix[t, :]

    N = L * L
    @assert length(sz_t) == N

    sublattice_A = 0.0
    sublattice_B = 0.0

    for i in 1:N
      x = (i - 1) % L + 1
      y = div(i - 1, L) + 1

      if x % 2 == 0
        sublattice_A += sz_t[i]
      else
        sublattice_B += sz_t[i]
      end
    end

    imbalance = (sublattice_A - sublattice_B) / (0.5 * N)
    push!(imbalances, imbalance)
  end
  return imbalances
end

function plot_ed_tree(data; L, gridnum, h)
  # Load ED data
  ed_data_path = joinpath(@__DIR__, "../../heisenberg_ed/data/results_columnar_Lx=$(L)_Ly=$(L)_hmax=$(h)_gridnum=$(gridnum).jld2")
  if !isfile(ed_data_path)
    @warn "ED data not found for L=$L, h=$h, gridnum=$gridnum. Path: $(ed_data_path)"
    return
  end
  data_ed = load(ed_data_path)

  t_ed = 0.1:0.1:100
  sz_ed = data_ed["sz_expectations"]
  imbalance_ed = columnar_imbalance_ed(sz_ed, L)
  site_index = (2 - 1) * L + 2
  sz_22_ed = sz_ed[:, site_index]

  # Find corresponding tree and snake data
  tree_sims = filter_configs(data; h, L, gridnum, free=true)
  if isempty(tree_sims)
    @warn "Could not find any tree data for h=$h, L=$L, gridnum=$gridnum."
  end

  snake_sims = filter_configs(data; h, L, gridnum, free=false)
  if isempty(snake_sims)
    @warn "Could not find any snake data for h=$h, L=$L, gridnum=$gridnum."
  end

  fig = Figure(resolution=(800, 1200))
  ax_imbalance = Axis(fig[1, 1],
    title="Imbalance, h=$h, gridnum=$gridnum, L=$L",
    xlabel=L"t",
    ylabel="Imbalance",
  )
  lines!(ax_imbalance, t_ed, imbalance_ed, label="ED")

  ax_sz = Axis(fig[2, 1],
    title="Sz at (2,2)",
    xlabel=L"t",
    ylabel=L"⟨S^z_{2,2}⟩",
  )
  lines!(ax_sz, t_ed, sz_22_ed, label="ED")

  for d_tree in tree_sims
    t_tree = d_tree.observer.times
    imbalance_tree = [real(columnar_imbalance(x)) for x in d_tree.observer.sz]
    sz_22_tree = [real(x[(1, 2, 2)][2]) for x in d_tree.observer.sz]
    lines!(ax_imbalance, t_tree, imbalance_tree, label="Tree (maxdim=$(d_tree.time_evolution.maxdim))")
    lines!(ax_sz, t_tree, sz_22_tree, label="Tree (maxdim=$(d_tree.time_evolution.maxdim))")
  end

  for d_snake in snake_sims
    t_snake = d_snake.observer.times
    imbalance_snake = [real(columnar_imbalance_snake(x)) for x in d_snake.observer.sz]
    sz_22_snake = [real(x[(2, 2)]) for x in d_snake.observer.sz]
    lines!(ax_imbalance, t_snake, imbalance_snake, label="Snake (maxdim=$(d_snake.time_evolution.maxdim))", linestyle=:dash)
    lines!(ax_sz, t_snake, sz_22_snake, label="Snake (maxdim=$(d_snake.time_evolution.maxdim))", linestyle=:dash)
  end

  axislegend(ax_imbalance)
  axislegend(ax_sz)

  plots_dir = joinpath(@__DIR__, "../plots/comp_ed")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end

  filename_png = "comparison_L=$(L)_h=$(h)_grid=$(gridnum).png"
  filename_pdf = "comparison_L=$(L)_h=$(h)_grid=$(gridnum).pdf"
  save(joinpath(plots_dir, filename_png), fig)
  save(joinpath(plots_dir, filename_pdf), fig)
  println("Saved plot to $(joinpath(plots_dir, filename_png)) and $(joinpath(plots_dir, filename_pdf))")
  display(fig)
end

function plot_single_comparison(data)
    L = 4

    # Get unique h and gridnum for L=4
    params = unique([(Float64(c.model.h), c.graph.gridnum) for c in data if c.graph.L == L])
    if isempty(params)
        @warn "No configurations found for L=$L"
        return
    end

    println("Found $(length(params)) parameter combinations for L=$L to plot.")

    for (h, gridnum) in params
        println("Processing h=$h, gridnum=$gridnum")
        plot_ed_tree(data; L=L, gridnum=gridnum, h=h)
    end
end

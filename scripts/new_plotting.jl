using CairoMakie
using Printf
using TTNEvo
using NamedGraphs
using ITensorNetworks
using Statistics
using HDF5
using Graphs
using NetworkLayout

function plot_tree_snake_comp(data_tree, data_snake)
  fig = Figure(resolution=(800, 1200))

  # Define the maxdim values and corresponding colors
  maxdim_values_snake = [64, 128, 196]
  maxdim_values_tree = [60, 120, 200]
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

        imbalance_trajectory_snake = [real(columnar_imbalance(x)) for x in d_snake.observer.sz]
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

function plot_and_save_single_run(data_tree, data_snake, h, gridnum)
  fig = Figure(resolution=(800, 600))

  maxdim_values_snake = [32, 64, 128, 196]
  maxdim_values_tree = [32, 64, 128, 196]
  colors = [:blue, :red, :green, :grey]

  ax_imbalance = Axis(fig[1, 1],
    title="h=$h, gridnum=$gridnum",
    xlabel=L"t",
    ylabel="Imbalance",
    xscale=log10,
  )
  xlims!(ax_imbalance, (0.1, 100))

  for (j, (maxdim_snake, maxdim_tree)) in enumerate(zip(maxdim_values_snake, maxdim_values_tree))
    # Plot Snake
    d_snake = try
      only(filter_configs(data_snake; h, maxdim=maxdim_snake, gridnum))
    catch
      @warn "Could not find snake data for h=$h, maxdim=$maxdim_snake, gridnum=$gridnum"
      continue
    end
    imbalance_snake = [real(columnar_imbalance(x)) for x in d_snake.observer.sz]
    lines!(ax_imbalance, d_snake.observer.times, imbalance_snake, label="maxdim=$maxdim_snake, snake", color=colors[j])

    # Plot Tree
    d_tree = try
      only(filter_configs(data_tree; h, maxdim=maxdim_tree, gridnum))
    catch
      @warn "Could not find tree data for h=$h, maxdim=$maxdim_tree, gridnum=$gridnum"
      continue
    end
    imbalance_tree = [real(columnar_imbalance(x)) for x in d_tree.observer.sz]
    lines!(ax_imbalance, d_tree.observer.times, imbalance_tree, label="maxdim=$maxdim_tree, tree", color=colors[j], linestyle=:dash)
  end

  axislegend(ax_imbalance)

  # Ensure the plots directory exists
  plots_dir = joinpath(@__DIR__, "../plots/L8")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end

  filename_png = "imbalance_h=$(h)_grid=$(gridnum).png"
  filename_pdf = "imbalance_h=$(h)_grid=$(gridnum).pdf"
  save(joinpath(plots_dir, filename_png), fig)
  save(joinpath(plots_dir, filename_pdf), fig)

  return fig
end

function plot_all_single_runs(data_tree, data_snake)
  h_values = unique([Float64(d.model.h) for d in data_tree])
  @show h_values
  gridnum_values = unique([d.graph.gridnum for d in data_tree])

  for h in h_values
    for gridnum in gridnum_values
      println("Plotting for h=$h, gridnum=$gridnum")
      plot_and_save_single_run(data_tree, data_snake, h, gridnum)
    end
  end
end

function plot_accuracy_convergence(data_tree, data_snake)
  h_values = [10.0, 20.0, 30.0, 50.0]
  maxdim_values_snake = [64, 128, 196]
  maxdim_values_tree = [60, 120, 200]

  fig = Figure(resolution=(800, 1200))

  for (i, h) in enumerate(h_values)
    ax = Axis(fig[i, 1],
      title="h=$h",
      xlabel="Maxdim",
      ylabel="Error (Max Abs Diff)",
      yscale=log10
    )

    # --- Snake Method ---
    benchmark_snake_all_data = filter_configs(data_snake; h, maxdim=maximum(maxdim_values_snake))
    if isempty(benchmark_snake_all_data)
      @warn "No benchmark data for snake with h=$h"
    else
      benchmark_snake_gridnums = Set([d.graph.gridnum for d in benchmark_snake_all_data])
      errors_snake = []
      dims_snake = []

      for maxdim in maxdim_values_snake[1:end-1]
        current_snake_all_data = filter_configs(data_snake; h, maxdim)
        if isempty(current_snake_all_data)
          continue
        end

        current_snake_gridnums = Set([d.graph.gridnum for d in current_snake_all_data])
        common_gridnums = intersect(benchmark_snake_gridnums, current_snake_gridnums)

        if isempty(common_gridnums)
          @warn "No common gridnums for snake h=$h between maxdim=$maxdim and benchmark"
          continue
        end

        benchmark_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, benchmark_snake_all_data)
        current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_snake_all_data)

        benchmark_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in benchmark_data_filtered]...)
        avg_benchmark_traj = vec(mean(benchmark_trajectories, dims=2))

        current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
        avg_current_traj = vec(mean(current_trajectories, dims=2))

        error = maximum(abs.(avg_current_traj .- avg_benchmark_traj))
        push!(errors_snake, error + 1e-16)
        push!(dims_snake, maxdim)
      end
      scatter!(ax, dims_snake, errors_snake, label="Snake", color=:red)
      lines!(ax, dims_snake, errors_snake, color=:red)
    end

    # --- Tree Method ---
    benchmark_tree_all_data = filter_configs(data_tree; h, maxdim=maximum(maxdim_values_tree))
    if isempty(benchmark_tree_all_data)
      @warn "No benchmark data for tree with h=$h"
    else
      benchmark_tree_gridnums = Set([d.graph.gridnum for d in benchmark_tree_all_data])
      errors_tree = []
      dims_tree = []

      for maxdim in maxdim_values_tree[1:end-1]
        current_tree_all_data = filter_configs(data_tree; h, maxdim)
        if isempty(current_tree_all_data)
          continue
        end

        current_tree_gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
        common_gridnums = intersect(benchmark_tree_gridnums, current_tree_gridnums)

        if isempty(common_gridnums)
          @warn "No common gridnums for tree h=$h between maxdim=$maxdim and benchmark"
          continue
        end

        benchmark_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, benchmark_tree_all_data)
        current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)

        benchmark_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in benchmark_data_filtered]...)
        avg_benchmark_traj = vec(mean(benchmark_trajectories, dims=2))

        current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
        avg_current_traj = vec(mean(current_trajectories, dims=2))

        error = maximum(abs.(avg_current_traj .- avg_benchmark_traj))
        push!(errors_tree, error + 1e-16)
        push!(dims_tree, maxdim)
      end
      scatter!(ax, dims_tree, errors_tree, label="Tree", color=:blue)
      lines!(ax, dims_tree, errors_tree, color=:blue)
    end

    axislegend(ax)
  end

  plots_dir = joinpath(@__DIR__, "../plots/large")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  display(fig)
  save(joinpath(plots_dir, "accuracy_convergence.png"), fig)
  save(joinpath(plots_dir, "accuracy_convergence.pdf"), fig)

  return fig
end


function plot_accuracy_vs_parameters(data_tree, data_snake)
  h_values = [10.0, 20.0, 30.0, 50.0]
  maxdim_values_snake = [64, 128, 196]
  maxdim_values_tree = [60, 120, 200]

  fig = Figure(resolution=(800, 1200))

  for (i, h) in enumerate(h_values)
    ax = Axis(fig[i, 1],
      title="h = $h",
      xlabel="Number of Parameters",
      ylabel="Error (Max Abs Diff)",
      xscale=log10,
      yscale=log10,
    )

    # --- Snake Method ---
    benchmark_snake_all_data = filter_configs(data_snake; h, maxdim=maximum(maxdim_values_snake))
    if isempty(benchmark_snake_all_data)
      @warn "No benchmark data for snake with h=$h"
    else
      benchmark_snake_gridnums = Set([d.graph.gridnum for d in benchmark_snake_all_data])
      errors_snake = []
      params_snake = []

      for maxdim in maxdim_values_snake[1:end-1]
        current_snake_all_data = filter_configs(data_snake; h, maxdim)
        if isempty(current_snake_all_data)
          continue
        end

        current_snake_gridnums = Set([d.graph.gridnum for d in current_snake_all_data])
        common_gridnums = intersect(benchmark_snake_gridnums, current_snake_gridnums)

        if isempty(common_gridnums)
          continue
        end

        benchmark_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, benchmark_snake_all_data)
        current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_snake_all_data)

        benchmark_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in benchmark_data_filtered]...)
        avg_benchmark_traj = vec(mean(benchmark_trajectories, dims=2))

        current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
        avg_current_traj = vec(mean(current_trajectories, dims=2))

        error = maximum(abs.(avg_current_traj .- avg_benchmark_traj))
        avg_params = mean([minimum(d.observer.num_size) for d in current_data_filtered])

        push!(errors_snake, error + 1e-16)
        push!(params_snake, avg_params)
      end
      scatter!(ax, params_snake, errors_snake, label="Snake", color=:red)
      lines!(ax, params_snake, errors_snake, color=:red)
    end

    # --- Tree Method ---
    benchmark_tree_all_data = filter_configs(data_tree; h, maxdim=maximum(maxdim_values_tree))
    benchmark_tree_all_data = filter_configs(data_snake; h, maxdim=maximum(maxdim_values_snake))
    if isempty(benchmark_tree_all_data)
      @warn "No benchmark data for tree with h=$h"
    else
      benchmark_tree_gridnums = Set([d.graph.gridnum for d in benchmark_tree_all_data])
      errors_tree = []
      params_tree = []

      for maxdim in maxdim_values_tree[1:end-1]
        current_tree_all_data = filter_configs(data_tree; h, maxdim)
        if isempty(current_tree_all_data)
          continue
        end

        current_tree_gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
        common_gridnums = intersect(benchmark_tree_gridnums, current_tree_gridnums)

        if isempty(common_gridnums)
          continue
        end

        benchmark_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, benchmark_tree_all_data)
        current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)

        benchmark_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in benchmark_data_filtered]...)
        avg_benchmark_traj = vec(mean(benchmark_trajectories, dims=2))

        current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
        avg_current_traj = vec(mean(current_trajectories, dims=2))

        error = maximum(abs.(avg_current_traj .- avg_benchmark_traj))
        avg_params = mean([minimum(d.observer.num_size) for d in current_data_filtered])

        push!(errors_tree, error + 1e-16)
        push!(params_tree, avg_params)
      end
      scatter!(ax, params_tree, errors_tree, label="Tree", color=:blue)
      lines!(ax, params_tree, errors_tree, color=:blue)
    end

    axislegend(ax)
  end

  plots_dir = joinpath(@__DIR__, "../plots/large")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  save(joinpath(plots_dir, "accuracy_vs_parameters.pdf"), fig)
  save(joinpath(plots_dir, "accuracy_vs_parameters.png"), fig)

  return fig
end

function plot_runtime_vs_parameters(data_tree, data_snake)
  h_values = [10.0, 20.0, 30.0, 50.0]
  maxdim_values_snake = [64, 128, 196]
  maxdim_values_tree = [60, 120, 200]

  fig = Figure(resolution=(800, 1200))

  for (i, h) in enumerate(h_values)
    ax = Axis(fig[i, 1],
      title="h = $h",
      xlabel="Number of Parameters",
      ylabel="Execution Time (s)",
      xscale=log10,
      yscale=log10,
    )

    # --- Snake Method ---
    runtimes_snake = []
    params_snake = []
    for maxdim in maxdim_values_snake
      current_data = filter_configs(data_snake; h, maxdim)
      if isempty(current_data)
        continue
      end
      avg_runtime = mean([mean(d.observer.ex_times) for d in current_data])
      avg_params = mean([mean(d.observer.num_size) for d in current_data])
      push!(runtimes_snake, avg_runtime)
      push!(params_snake, avg_params)
    end
    scatter!(ax, params_snake, runtimes_snake, label="Snake", color=:red)
    lines!(ax, params_snake, runtimes_snake, color=:red)

    # --- Tree Method ---
    runtimes_tree = []
    params_tree = []
    for maxdim in maxdim_values_tree
      current_data = filter_configs(data_tree; h, maxdim)
      if isempty(current_data)
        continue
      end
      avg_runtime = mean([mean(d.observer.ex_times) for d in current_data])
      avg_params = mean([mean(d.observer.num_size) for d in current_data])
      push!(runtimes_tree, avg_runtime)
      push!(params_tree, avg_params)
    end
    scatter!(ax, params_tree, runtimes_tree, label="Tree", color=:blue)
    lines!(ax, params_tree, runtimes_tree, color=:blue)

    axislegend(ax)
  end

  plots_dir = joinpath(@__DIR__, "../plots/large")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  save(joinpath(plots_dir, "runtime_vs_parameters.pdf"), fig)
  save(joinpath(plots_dir, "runtime_vs_parameters.png"), fig)

  return fig
end

function plot_accuracy_vs_runtime(data_tree, data_snake)
  h_values = [10.0, 20.0, 30.0, 50.0]
  maxdim_values_snake = [64, 128, 196]
  maxdim_values_tree = [60, 120, 200]

  fig = Figure(resolution=(800, 1200), title="accuracy_vs_runtime")

  for (i, h) in enumerate(h_values)
    ax = Axis(fig[i, 1],
      title="h = $h",
      xlabel="Execution Time (s)",
      ylabel="Error (Max Abs Diff)",
      xscale=log10,
      yscale=log10,
    )

    # --- Snake Method ---
    benchmark_snake_all_data = filter_configs(data_snake; h, maxdim=maximum(maxdim_values_snake))
    if isempty(benchmark_snake_all_data)
      @warn "No benchmark data for snake with h=$h"
    else
      benchmark_snake_gridnums = Set([d.graph.gridnum for d in benchmark_snake_all_data])
      errors_snake = []
      runtimes_snake = []

      for maxdim in maxdim_values_snake[1:end-1]
        current_snake_all_data = filter_configs(data_snake; h, maxdim)
        if isempty(current_snake_all_data)
          continue
        end

        current_snake_gridnums = Set([d.graph.gridnum for d in current_snake_all_data])
        common_gridnums = intersect(benchmark_snake_gridnums, current_snake_gridnums)

        if isempty(common_gridnums)
          continue
        end

        benchmark_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, benchmark_snake_all_data)
        current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_snake_all_data)

        benchmark_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in benchmark_data_filtered]...)
        avg_benchmark_traj = vec(mean(benchmark_trajectories, dims=2))

        current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
        avg_current_traj = vec(mean(current_trajectories, dims=2))

        error = maximum(abs.(avg_current_traj .- avg_benchmark_traj))
        avg_runtime = mean([sort(d.observer.ex_times)[2] for d in current_data_filtered])

        push!(errors_snake, error + 1e-16)
        push!(runtimes_snake, avg_runtime)
      end
      scatter!(ax, runtimes_snake, errors_snake, label="Snake", color=:red)
      lines!(ax, runtimes_snake, errors_snake, color=:red)
    end

    # --- Tree Method ---
    benchmark_tree_all_data = filter_configs(data_tree; h, maxdim=maximum(maxdim_values_tree))
    benchmark_tree_all_data = filter_configs(data_snake; h, maxdim=maximum(maxdim_values_snake))
    if isempty(benchmark_tree_all_data)
      @warn "No benchmark data for tree with h=$h"
    else
      benchmark_tree_gridnums = Set([d.graph.gridnum for d in benchmark_tree_all_data])
      errors_tree = []
      runtimes_tree = []

      for maxdim in maxdim_values_tree[1:end-1]
        current_tree_all_data = filter_configs(data_tree; h, maxdim)
        if isempty(current_tree_all_data)
          continue
        end


        current_tree_gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
        common_gridnums = intersect(benchmark_tree_gridnums, current_tree_gridnums)

        if isempty(common_gridnums)
          continue
        end

        benchmark_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, benchmark_tree_all_data)
        current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)

        benchmark_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in benchmark_data_filtered]...)
        avg_benchmark_traj = vec(mean(benchmark_trajectories, dims=2))

        current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
        avg_current_traj = vec(mean(current_trajectories, dims=2))

        error = maximum(abs.(avg_current_traj .- avg_benchmark_traj))
        avg_runtime = mean([sort(d.observer.ex_times)[2] for d in current_data_filtered])
        @show avg_runtime
        # avg_runtime = mean([mean(d.observer.ex_times) for d in current_data_filtered])
        # @show avg_runtime

        push!(errors_tree, error + 1e-16)
        push!(runtimes_tree, avg_runtime)
      end
      scatter!(ax, runtimes_tree, errors_tree, label="Tree", color=:blue)
      lines!(ax, runtimes_tree, errors_tree, color=:blue)
    end

    axislegend(ax)
  end

  plots_dir = joinpath(@__DIR__, "../plots/large")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  save(joinpath(plots_dir, "accuracy_vs_runtime.pdf"), fig)
  save(joinpath(plots_dir, "accuracy_vs_runtime.png"), fig)

  return fig
end

function plot_tree_structure(data_tree; pin_spins=true)
  plots_dir = joinpath(@__DIR__, "../plots/L4")
  for d in data_tree
    fig = TTNEvo.plot_free_graph_with_maxdim(d.observer.maxdim[end]; pin_spins)
    if pin_spins
      save(joinpath(plots_dir, "tree_plots", "pinned_L=$(d.graph.L)_h=$(d.model.h)_maxdim=$(d.time_evolution.maxdim)_gridnum=$(d.graph.gridnum).png"), fig)
      save(joinpath(plots_dir, "tree_plots", "pinned_L=$(d.graph.L)_h=$(d.model.h)_maxdim=$(d.time_evolution.maxdim)_gridnum=$(d.graph.gridnum).pdf"), fig)
    else
      save(joinpath(plots_dir, "tree_plots", "not_pinned_L=$(d.graph.L)_h=$(d.model.h)_maxdim=$(d.time_evolution.maxdim)_gridnum=$(d.graph.gridnum).png"), fig)
      save(joinpath(plots_dir, "tree_plots", "not_pinned_L=$(d.graph.L)_h=$(d.model.h)_maxdim=$(d.time_evolution.maxdim)_gridnum=$(d.graph.gridnum).pdf"), fig)
    end
  end
end

function plot_tree_structure_over_time(config; pin_spins=true, max_plots=9)
  tree_data = config.observer.maxdim

  # Select a subset of time steps to plot
  num_data_points = length(tree_data)
  if num_data_points == 0
    @warn "No tree structure data to plot."
    return
  end

  indices_to_plot = round.(
    Int, range(1; stop=num_data_points, length=min(num_data_points, max_plots))
  )
  data_to_plot = tree_data[indices_to_plot]

  num_plots = length(data_to_plot)
  cols = Int(ceil(sqrt(num_plots)))
  rows = Int(ceil(num_plots / cols))

  fig = Figure(; resolution=(400 * cols, 400 * rows))

  for (i, (t, linkdim_graph)) in enumerate(data_to_plot)
    row, col = divrem(i - 1, cols) .+ (1, 1)

    ax = Axis(fig[row, col]; title=@sprintf("t = %.2f", t), aspect=1)
    hidedecorations!(ax)
    hidespines!(ax)

    # Replicating logic from plot_free_graph_with_maxdim
    graph = linkdim_graph
    maxdims = [graph[e] for e in collect(edges(graph))]
    maxdims_label = ["$m" for m in maxdims]
    edge_width = [log2(m) for m in maxdims]
    edge_color = [1 - ew / maximum(edge_width) for ew in edge_width]
    labels = [v[1] == 2 ? "  " : "$(v[2:3])" for v in collect(vertices(graph))]

    # These functions are not exported, so we need to qualify them
    g_simple = TTNEvo.namedgraph_to_graph(graph)
    pin = Dict(
      v[1] == 1 ? TTNEvo.vert_to_ind(graph, v) => 2 .* v[2:3] :
      TTNEvo.vert_to_ind(graph, v) => false for v in collect(vertices(graph))
    )
    cols_nodes = Dict(
      i => (TTNEvo.ind_to_vert(graph, i)[1] == 1 ? :red : :green) for i in 1:nv(graph)
    )

    layout_alg = pin_spins ? NetworkLayout.Spring(; pin) : NetworkLayout.Spring()

    graphplot!(
      ax,
      g_simple;
      layout=layout_alg,
      elabels=maxdims_label,
      edge_width,
      edge_color,
      node_color=cols_nodes,
      ilabels=labels,
    )
  end

  # Save the plot
  plots_dir = joinpath(@__DIR__, "../plots/L4/tree_plots")
  mkpath(plots_dir)
  filename = "tree_structure_over_time_L=$(config.graph.L)_h=$(config.model.h)_maxdim=$(config.time_evolution.maxdim)_gridnum=$(config.graph.gridnum).png"
  save(joinpath(plots_dir, filename), fig)
  save(replace(joinpath(plots_dir, filename), ".png" => ".pdf"), fig)

  return fig
end

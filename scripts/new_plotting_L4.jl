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

function get_ed_benchmark(L, h, gridnums)
  ed_trajectories = []
  common_gridnums_with_ed = []
  for gridnum in gridnums
    ed_data_path = joinpath(@__DIR__, "../../heisenberg_ed/data/fun_results_columnar_Lx=$(L)_Ly=$(L)_hmax=$(h)_gridnum=$(gridnum).jld2")
    if isfile(ed_data_path)
      data_ed = load(ed_data_path)
      sz_ed = data_ed["sz_expectations"]
      imbalance_ed = columnar_imbalance_ed(sz_ed, L)
      push!(ed_trajectories, imbalance_ed)
      push!(common_gridnums_with_ed, gridnum)
    else
      @warn "ED data not found for L=$L, h=$h, gridnum=$gridnum. Path: $(ed_data_path)"
    end
  end

  if isempty(ed_trajectories)
    return nothing, nothing
  end

  avg_benchmark_traj = vec(mean(hcat(ed_trajectories...), dims=2))
  return avg_benchmark_traj, common_gridnums_with_ed
end

function calculate_accuracy_convergence_data(data)
  L = 4
  h_values = [0.0, 10.0, 20.0, 30.0, 50.0]
  maxdim_values_snake = [16, 32, 48, 64, 128, 196]
  maxdim_values_tree = [round(Int, 1.5 * x) for x in maxdim_values_snake]

  results = Dict()
  time_snake = nothing

  for h in h_values
    # --- Snake Method ---
    errors_snake = []
    dims_snake = []
    for maxdim in maxdim_values_snake
      current_snake_all_data = filter_configs(data; h, maxdim, free=false)
      if isempty(current_snake_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_snake_all_data])
      gridnums = [1]
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for snake h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_snake_all_data)
      time_snake = first(current_data_filtered).observer.times
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      min_len = min(length(avg_current_traj), length(avg_benchmark_traj))
      error = maximum(abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:min_len]))
      push!(errors_snake, error + 1e-16)
      push!(dims_snake, maxdim)
    end

    # --- Tree Method ---
    errors_tree = []
    dims_tree = []
    for maxdim in maxdim_values_tree
      current_tree_all_data = filter_configs(data; h, maxdim, free=true, maxdim_opt=true)
      if isempty(current_tree_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for tree h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      min_len = min(length(avg_current_traj), length(avg_benchmark_traj))
      error = maximum(abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:min_len]))
      push!(errors_tree, error + 1e-16)
      push!(dims_tree, maxdim)
    end

    # --- Tree Method no trunc ---
    errors_tree_no_trunc = []
    dims_tree_no_trunc = []
    for maxdim in maxdim_values_tree
      current_tree_all_data = filter_configs(data; h, maxdim, free=true, maxdim_opt=false)
      if isempty(current_tree_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for tree h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      min_len = min(length(avg_current_traj), length(avg_benchmark_traj))
      error = maximum(abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:min_len]))
      push!(errors_tree_no_trunc, error + 1e-16)
      push!(dims_tree_no_trunc, maxdim)
    end

    results[h] = Dict(
      "snake" => Dict("errors" => errors_snake, "dims" => dims_snake),
      "tree" => Dict("errors" => errors_tree, "dims" => dims_tree),
      "tree_no_trunc" => Dict("errors" => errors_tree_no_trunc, "dims" => dims_tree_no_trunc),
    )
  end
  return results
end

function calculate_accuracy_vs_parameters_data(data_snake, data_tree)
  L = 4
  h_values = [0.0, 10.0, 20.0, 30.0, 50.0]
  maxdim_values_snake = [16, 32, 48, 64, 128, 196]
  maxdim_values_tree = [32, 48, 64, 128, 192, 294]

  results = Dict()

  for h in h_values
    # --- Snake Method ---
    errors_snake = []
    params_snake = []
    for maxdim in maxdim_values_snake
      current_snake_all_data = filter_configs(data_snake; h, maxdim, free=false)
      if isempty(current_snake_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_snake_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for snake h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_snake_all_data)

      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      min_len = min(length(avg_current_traj), length(avg_benchmark_traj))
      error = maximum(abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:min_len]))
      avg_params = mean([minimum(d.observer.num_size) for d in current_data_filtered])

      push!(errors_snake, error + 1e-16)
      push!(params_snake, avg_params)
    end

    # --- Tree Method ---
    errors_tree = []
    params_tree = []
    for maxdim in maxdim_values_tree
      current_tree_all_data = filter_configs(data_tree; h, maxdim, free=true, maxdim_opt=true)
      if isempty(current_tree_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for tree h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      min_len = min(length(avg_current_traj), length(avg_benchmark_traj))
      error = maximum(abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:min_len]))
      avg_params = mean([minimum(d.observer.num_size) for d in current_data_filtered])

      push!(errors_tree, error + 1e-16)
      push!(params_tree, avg_params)
    end

    results[h] = Dict(
      "snake" => Dict("errors" => errors_snake, "params" => params_snake),
      "tree" => Dict("errors" => errors_tree, "params" => params_tree),
    )
  end
  return results
end

function calculate_accuracy_vs_runtime_data(data_snake, data_tree)
  L = 4
  h_values = [0.0, 10.0, 20.0, 30.0, 50.0]
  maxdim_values_snake = [16, 32, 48, 64, 128, 196]
  maxdim_values_tree = [32, 48, 64, 128, 192, 294]

  results = Dict()

  for h in h_values
    # --- Snake Method ---
    errors_snake = []
    runtimes_snake = []
    for maxdim in maxdim_values_snake
      current_snake_all_data = filter_configs(data_snake; h, maxdim, free=false)
      if isempty(current_snake_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_snake_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for snake h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_snake_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      min_len = min(length(avg_current_traj), length(avg_benchmark_traj))
      error = maximum(abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:min_len]))
      avg_runtime = mean([mean(d.observer.ex_times) for d in current_data_filtered])

      push!(errors_snake, error + 1e-16)
      push!(runtimes_snake, avg_runtime)
    end

    # --- Tree Method ---
    errors_tree = []
    runtimes_tree = []
    for maxdim in maxdim_values_tree
      current_tree_all_data = filter_configs(data_tree; h, maxdim, free=true, maxdim_opt=true)
      if isempty(current_tree_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for tree h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      min_len = min(length(avg_current_traj), length(avg_benchmark_traj))
      error = maximum(abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:min_len]))
      avg_runtime = mean([mean(d.observer.ex_times) for d in current_data_filtered])

      push!(errors_tree, error + 1e-16)
      push!(runtimes_tree, avg_runtime)
    end

    results[h] = Dict(
      "snake" => Dict("errors" => errors_snake, "runtimes" => runtimes_snake),
      "tree" => Dict("errors" => errors_tree, "runtimes" => runtimes_tree),
    )
  end
  return results
end

function calculate_accuracy_over_time_data(data)
  L = 4
  h_values = [10.0]
  maxdim_values_snake = [196]
  maxdim_values_tree = maxdim_values_snake
  T = 100
  time = 0.1:0.1:T

  results = Dict()

  for h in h_values
    h_results = Dict()

    # --- Snake Method ---
    snake_results = Dict()
    for maxdim in maxdim_values_snake
      current_snake_all_data = filter_configs(data; h, maxdim, free=false)
      if isempty(current_snake_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_snake_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for snake h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_snake_all_data)
      current_trajectories = hcat([[real(columnar_imbalance_snake(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      error = abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:end])
      snake_results[maxdim] = Dict("time" => time, "error" => error)
    end
    h_results["snake"] = snake_results

    # --- Tree Method (struct and trunc) ---
    tree_results = Dict()
    for maxdim in maxdim_values_tree
      current_tree_all_data = filter_configs(data; h, maxdim, free=true, maxdim_opt=true)
      if isempty(current_tree_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for tree h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      error = abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:end])
      tree_results[maxdim] = Dict("time" => time, "error" => error)
    end
    h_results["tree_struct_trunc"] = tree_results

    # --- Tree Method (no trunc, no struct) ---
    tree_no_opt_results = Dict()
    for maxdim in maxdim_values_tree
      current_tree_all_data = filter_configs(data; h, maxdim, free=true, tree_opt=false, maxdim_opt=false)
      if isempty(current_tree_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for tree h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      error = abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:end])
      tree_no_opt_results[maxdim] = Dict("time" => time, "error" => error)
    end
    h_results["tree_no_trunc_no_struct"] = tree_no_opt_results

    # --- Tree Method (no struct, trunc) ---
    tree_no_struct_results = Dict()
    for maxdim in maxdim_values_tree
      current_tree_all_data = filter_configs(data; h, maxdim, free=true, tree_opt=false, maxdim_opt=true)
      if isempty(current_tree_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for tree h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      error = abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:end])
      tree_no_struct_results[maxdim] = Dict("time" => time, "error" => error)
    end
    h_results["tree_no_struct_trunc"] = tree_no_struct_results

    # --- Tree Method (struct, no trunc) ---
    tree_no_trunc_results = Dict()
    for maxdim in maxdim_values_tree
      current_tree_all_data = filter_configs(data; h, maxdim, free=true, maxdim_opt=false)
      if isempty(current_tree_all_data)
        continue
      end

      gridnums = Set([d.graph.gridnum for d in current_tree_all_data])
      avg_benchmark_traj, common_gridnums = get_ed_benchmark(L, h, gridnums)

      if isnothing(avg_benchmark_traj)
        @warn "No ED data for tree h=$h, maxdim=$maxdim"
        continue
      end

      current_data_filtered = filter(d -> d.graph.gridnum in common_gridnums, current_tree_all_data)
      current_trajectories = hcat([[real(columnar_imbalance(x)) for x in d.observer.sz] for d in current_data_filtered]...)
      avg_current_traj = vec(mean(current_trajectories, dims=2))

      error = abs.(avg_current_traj[2:end] .- avg_benchmark_traj[1:end])
      tree_no_trunc_results[maxdim] = Dict("time" => time, "error" => error)
    end
    h_results["tree_struct_no_trunc"] = tree_no_trunc_results

    results[h] = h_results
  end
  return results
end

function plot_accuracy_convergence(calculated_data)
  h_values = sort(collect(keys(calculated_data)))
  fig = Figure(resolution=(800, 1200))

  for (i, h) in enumerate(h_values)
    ax = Axis(fig[i, 1], title="h=$h", xlabel="Maxdim", ylabel="Error vs ED (Max Abs Diff)", yscale=log10)

    h_data = calculated_data[h]

    # --- Snake Method ---
    if !isempty(h_data["snake"]["dims"])
      scatter!(ax, h_data["snake"]["dims"], h_data["snake"]["errors"], label="Snake", color=:red)
      lines!(ax, h_data["snake"]["dims"], h_data["snake"]["errors"], color=:red)
    end

    # --- Tree Method ---
    if !isempty(h_data["tree"]["dims"])
      scatter!(ax, h_data["tree"]["dims"], h_data["tree"]["errors"], label="Tree", color=:blue)
      lines!(ax, h_data["tree"]["dims"], h_data["tree"]["errors"], color=:blue)
    end

    # --- Tree Method no trunc ---
    if !isempty(h_data["tree_no_trunc"]["dims"])
      scatter!(ax, h_data["tree_no_trunc"]["dims"], h_data["tree_no_trunc"]["errors"], label="Tree, no trunc", color=:green)
      lines!(ax, h_data["tree_no_trunc"]["dims"], h_data["tree_no_trunc"]["errors"], color=:green)
    end

    axislegend(ax)
  end

  plots_dir = joinpath(@__DIR__, "../plots/L4")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  display(fig)
  save(joinpath(plots_dir, "accuracy_convergence_vs_ed.png"), fig)
  save(joinpath(plots_dir, "accuracy_convergence_vs_ed.pdf"), fig)

  return fig
end

function plot_accuracy_vs_parameters(calculated_data)
  h_values = sort(collect(keys(calculated_data)))
  fig = Figure(resolution=(800, 1200))

  for (i, h) in enumerate(h_values)
    ax = Axis(fig[i, 1], title="h = $h", xlabel="Number of Parameters", ylabel="Error vs ED (Max Abs Diff)", xscale=log10, yscale=log10)

    h_data = calculated_data[h]

    # --- Snake Method ---
    if !isempty(h_data["snake"]["params"])
      scatter!(ax, h_data["snake"]["params"], h_data["snake"]["errors"], label="Snake", color=:red)
      lines!(ax, h_data["snake"]["params"], h_data["snake"]["errors"], color=:red)
    end

    # --- Tree Method ---
    if !isempty(h_data["tree"]["params"])
      scatter!(ax, h_data["tree"]["params"], h_data["tree"]["errors"], label="Tree", color=:blue)
      lines!(ax, h_data["tree"]["params"], h_data["tree"]["errors"], color=:blue)
    end

    axislegend(ax)
  end

  plots_dir = joinpath(@__DIR__, "../plots/L4")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  display(fig)
  save(joinpath(plots_dir, "accuracy_vs_parameters_vs_ed.png"), fig)
  save(joinpath(plots_dir, "accuracy_vs_parameters_vs_ed.pdf"), fig)

  return fig
end

function plot_accuracy_vs_runtime(calculated_data)
  h_values = sort(collect(keys(calculated_data)))
  fig = Figure(resolution=(800, 1200), title="accuracy_vs_runtime")

  for (i, h) in enumerate(h_values)
    ax = Axis(fig[i, 1], title="h = $h", xlabel="Execution Time per timestep (s)", ylabel="Error vs ED (Max Abs Diff)", yscale=log10)

    h_data = calculated_data[h]

    # --- Snake Method ---
    if !isempty(h_data["snake"]["runtimes"])
      scatter!(ax, h_data["snake"]["runtimes"], h_data["snake"]["errors"], label="Snake", color=:red)
      lines!(ax, h_data["snake"]["runtimes"], h_data["snake"]["errors"], color=:red)
    end

    # --- Tree Method ---
    if !isempty(h_data["tree"]["runtimes"])
      scatter!(ax, h_data["tree"]["runtimes"], h_data["tree"]["errors"], label="Tree", color=:blue)
      lines!(ax, h_data["tree"]["runtimes"], h_data["tree"]["errors"], color=:blue)
    end

    axislegend(ax)
  end

  plots_dir = joinpath(@__DIR__, "../plots/L4")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  display(fig)
  save(joinpath(plots_dir, "accuracy_vs_runtime_vs_ed.png"), fig)
  save(joinpath(plots_dir, "accuracy_vs_runtime_vs_ed.pdf"), fig)

  return fig
end

function plot_accuracy_over_time(calculated_data)
  h_values = sort(collect(keys(calculated_data)))
  fig = Figure(resolution=(1200, 800), title="accuracy_vs_runtime")

  for (i, h) in enumerate(h_values)
    ax = Axis(fig[i, 1], title="h = $h", xlabel="Time", ylabel="Error vs ED", xscale=log10, yscale=log10)

    h_data = calculated_data[h]

    # --- Snake Method ---
    for (maxdim, d) in h_data["snake"]
      scatter!(ax, d["time"], d["error"], label="Snake, $maxdim", color=:red)
      lines!(ax, d["time"], d["error"], color=:red)
    end

    # --- Tree Method (struct and trunc) ---
    for (maxdim, d) in h_data["tree_struct_trunc"]
      scatter!(ax, d["time"], d["error"], label="Tree, struct and trunc, $maxdim", color=:blue)
      lines!(ax, d["time"], d["error"], color=:blue)
    end

    # --- Tree Method (no trunc, no struct) ---
    for (maxdim, d) in h_data["tree_no_trunc_no_struct"]
      scatter!(ax, d["time"], d["error"], label="Tree, no truncation, no struct, $maxdim", color=:grey)
      lines!(ax, d["time"], d["error"], color=:grey)
    end

    # --- Tree Method (no struct, trunc) ---
    for (maxdim, d) in h_data["tree_no_struct_trunc"]
      scatter!(ax, d["time"], d["error"], label="Tree, no struct, truncation $maxdim", color=:black)
      lines!(ax, d["time"], d["error"], color=:black)
    end

    # --- Tree Method (struct, no trunc) ---
    for (maxdim, d) in h_data["tree_struct_no_trunc"]
      scatter!(ax, d["time"], d["error"], label="Tree, struct, no truncation, $maxdim", color=:green)
      lines!(ax, d["time"], d["error"], color=:green)
    end

    axislegend(ax)
  end

  plots_dir = joinpath(@__DIR__, "../plots/L4")
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  display(fig)
  save(joinpath(plots_dir, "accuracy_over_time_vs_ed.png"), fig)
  save(joinpath(plots_dir, "accuracy_over_time_vs_ed.pdf"), fig)
end


function plot_accuracy_over_time_seperated(data_snake, data_tree)
  L = 4
  h_values = [0.0, 10.0, 20.0, 30.0, 50.0]
  gridnums = 1:10
  maxdim_values_snake = [48, 64, 96, 128, 192]
  maxdim_values_tree = maxdim_values_snake

  datasets = [data_snake, data_tree]
  titles = ["Snake", "Tree"]

  for h in h_values
    for gridnum in gridnums
      fig = Figure(resolution=(1200, 200 + 400 * 2))
      Label(fig[1, 1], "Accuracy over time for h=$h, grid=$gridnum", fontsize=24, tellwidth=false)

      for (i, data) in enumerate(datasets)
        ax = Axis(fig[i+1, 1], title=titles[i], xlabel="Time", ylabel="Error vs ED", xscale=log10, yscale=log10)

        c = filter_configs(data; h, gridnum)

        colors = cgrad(:viridis, max(2, length(c)), categorical=true)

        for (j, ci) in enumerate(sort(c, by=x -> x.time_evolution.maxdim, rev=true))
          ci_imb = [columnar_imbalance(x) for x in ci.observer.sz]

          ed_data_path = joinpath(@__DIR__, "../../heisenberg_ed/data/results_columnar_Lx=$(L)_Ly=$(L)_hmax=$(h)_gridnum=$(gridnum).jld2")
          if isfile(ed_data_path)
            data_ed = load(ed_data_path)
            sz_ed = data_ed["sz_expectations"]
            imbalance_ed = columnar_imbalance_ed(sz_ed, L)
          end

          lines!(ax, ci.observer.times[2:end], abs.(ci_imb[2:end] .- imbalance_ed), label="$(ci.graph), maxdim=$(ci.time_evolution.maxdim)", color=colors[j])
        end
        if length(keys(c)) > 1
          axislegend(ax, position=:lt)
        end
      end

      plots_dir = joinpath(@__DIR__, "../plots/L4/error_over_time")
      if !isdir(plots_dir)
        mkpath(plots_dir)
      end
      save(joinpath(plots_dir, "accuracy_over_time_vs_ed_h=$(h)_gridnum=$(gridnum)_seperated.png"), fig)
      save(joinpath(plots_dir, "accuracy_over_time_vs_ed_h=$(h)_gridnum=$(gridnum)_seperated.pdf"), fig)
    end
  end
end


function plot_parameters_over_time_seperated(data_snake, data_tree)
  L = 4
  h_values = [0.0, 10.0, 20.0, 30.0, 50.0]
  gridnums = 1:10
  maxdim_values_snake = [48, 64, 96, 128, 192]
  maxdim_values_tree = maxdim_values_snake

  datasets = [data_snake, data_tree]
  titles = ["Snake", "Tree"]

  for h in h_values
    for gridnum in gridnums
      fig = Figure(resolution=(1200, 200 + 400 * 2))
      Label(fig[1, 1], "Parameters over time for h=$h, grid=$gridnum", fontsize=24, tellwidth=false)

      for (i, data) in enumerate(datasets)
        ax = Axis(fig[i+1, 1], title=titles[i], xlabel="Time", ylabel="Parameters", xscale=log10, yscale=log10)

        c = filter_configs(data; h, gridnum)

        colors = cgrad(:viridis, max(2, length(c)), categorical=true)

        for (j, ci) in enumerate(sort(c, by=x -> x.time_evolution.maxdim, rev=true))
          lines!(ax, ci.observer.times[2:end], ci.observer.num_size[2:end], label="$(ci.graph), maxdim=$(ci.time_evolution.maxdim)", color=colors[j])
        end
        if length(keys(c)) > 1
          axislegend(ax, position=:lt)
        end
      end

      plots_dir = joinpath(@__DIR__, "../plots/L4/parameters_over_time")
      if !isdir(plots_dir)
        mkpath(plots_dir)
      end
      save(joinpath(plots_dir, "parameters_over_time_vs_ed_h=$(h)_gridnum=$(gridnum)_seperated.png"), fig)
      save(joinpath(plots_dir, "parameters_over_time_vs_ed_h=$(h)_gridnum=$(gridnum)_seperated.pdf"), fig)
    end
  end
end


function plot_ex_time_over_time_seperated(data_snake, data_tree)
  L = 4
  h_values = [0.0, 10.0, 20.0, 30.0, 50.0]
  gridnums = 1:10
  maxdim_values_snake = [48, 64, 96, 128, 192]
  maxdim_values_tree = maxdim_values_snake

  datasets = [data_snake, data_tree]
  titles = ["Snake", "Tree"]

  for h in h_values
    for gridnum in gridnums
      fig = Figure(resolution=(1200, 200 + 400 * 2))
      Label(fig[1, 1], "Execution time over time for h=$h, grid=$gridnum", fontsize=24, tellwidth=false)

      for (i, data) in enumerate(datasets)
        ax = Axis(fig[i+1, 1], title=titles[i], xlabel="Time", ylabel="Ex time")

        c = filter_configs(data; h, gridnum)

        colors = cgrad(:viridis, max(2, length(c)), categorical=true)

        for (j, ci) in enumerate(sort(c, by=x -> x.time_evolution.maxdim, rev=true))
          lines!(ax, ci.observer.times[2:end], ci.observer.ex_times[2:end], label="$(ci.graph), maxdim=$(ci.time_evolution.maxdim)", color=colors[j])
        end
        if length(keys(c)) > 1
          axislegend(ax, position=:lt)
        end
      end

      plots_dir = joinpath(@__DIR__, "../plots/L4/ex_time_over_time")
      if !isdir(plots_dir)
        mkpath(plots_dir)
      end
      save(joinpath(plots_dir, "parameters_over_time_vs_ed_h=$(h)_gridnum=$(gridnum)_seperated.png"), fig)
      save(joinpath(plots_dir, "parameters_over_time_vs_ed_h=$(h)_gridnum=$(gridnum)_seperated.pdf"), fig)
    end
  end
end


function plot_test_diff(diff_snake, diff_tree, diff_tree_no_trunc)
  fig = Figure()
  ax = Axis(fig[1, 1], yscale=log10)
  lines!(ax, abs.(diff_snake), label="snake")
  lines!(ax, abs.(diff_tree), label="tree")
  lines!(ax, abs.(diff_tree_no_trunc), label="tree, no trunc")
  axislegend(ax)
  return fig
end

function L4_single_conv_over_time(; data_testing=load_dir("data/L4-testing6"; time_after=DateTime(2025, 1, 1, 0, 0, 0)))
  accuracy_over_time_data = calculate_accuracy_over_time_data(data_testing)
  plot_accuracy_over_time(accuracy_over_time_data)
  plot_accuracy_over_time_seperated(accuracy_over_time_data)
end

function L4_main(;
  data_snake=load_dir("data/L4-column-comp/"; time_after=DateTime(2025, 1, 1, 0, 0, 0)),
  data_tree=load_dir("data/L4-column-comp2/"; time_after=DateTime(2025, 1, 1, 0, 0, 0)),
)

  # Calculate all data upfront
  # accuracy_convergence_data = calculate_accuracy_convergence_data(data_snake)
  accuracy_vs_params_data = calculate_accuracy_vs_parameters_data(data_snake, data_tree)
  accuracy_vs_runtime_data = calculate_accuracy_vs_runtime_data(data_snake, data_tree)

  # Pass calculated data to plotting functions
  # plot_accuracy_convergence(accuracy_convergence_data)
  plot_accuracy_vs_parameters(accuracy_vs_params_data)
  plot_accuracy_vs_runtime(accuracy_vs_runtime_data)
end

using CairoMakie
using Printf
using TTNEvo
using NamedGraphs
using ITensorNetworks
using Statistics
using HDF5
using Graphs
using JLD2
using Base.Threads
using Dates
using DataFrames
using NetworkLayout
using ColorSchemes

"""
    logticks1(kmin::Int, kmax::Int)

Return tick positions and labels for {1,2}×10^k, k in kmin:kmax.
"""
function logticks1(kmin::Int, kmax::Int)
  positions = [10.0^k for k in kmin:kmax]
  labels = [L"10^{%$k}" for k in kmin:kmax]
  return positions, labels
end

"""
    logticks1_with_minors(kmin::Int, kmax::Int)

Major ticks at {1,2}×10^k with labels, and unlabeled minor ticks at 3–9×10^k.
"""
function logticks1_with_minors(kmin::Int, kmax::Int)
  major_pos = [10.0^k for k in kmin:kmax]
  major_labels = [m == 1 ?
                  L"10^{ %$k }" : L"$m \times 10^{ %$k }"
                  for k in kmin:kmax for m in (1)]

  minor_pos = [m * 10.0^k for k in kmin:kmax for m in 2:9]

  # Combine: majors with labels, minors without
  positions = vcat(major_pos, minor_pos)
  labels = vcat(major_labels, fill("", length(minor_pos)))

  return positions, labels
end

function publication_theme()
  return Theme(
    Axis=(
      # xlabelsize=14,
      # ylabelsize=14,
      # titlesize=16,
      # xticklabelsize=12,
      # yticklabelsize=12,
      xtickalign=1,
      ytickalign=1,
      spinewidth=1,
    ),
    Legend=(
      framevisible=false,
    )
  )
end

inch = 200
pt = 4 / 3
cm = inch / 2.54

width_cm, height_cm = 4cm, 12cm   # single-column figure size
pixel = (width_cm, height_cm)
individual_size = (8cm, 4cm)

set_theme!(merge(publication_theme(), theme_latexfonts()))

function moving_average(data::AbstractVector, window::Int)
  if window <= 1
    return convert(Vector{Float64}, data)
  end
  n = length(data)
  if n < window
    return Float64[]
  end

  m = n - window + 1
  out = Vector{Float64}(undef, m)

  s = sum(view(data, 1:window))
  out[1] = s / window

  for i in 2:m
    s = s - data[i-1] + data[i+window-1]
    out[i] = s / window
  end
  return out
end

"""
  aggregate_series_mean_std(times_vec, series_vec)

Align multiple time/value series by the minimum common length and return
  (times, mean(values), stderr(values)). Returns `nothing` if invalid.
"""
function aggregate_series_mean_std(times_vec, series_vec)
  if any(isempty, series_vec) || any(isempty, times_vec)
    return nothing
  end
  min_len_times = minimum(length, times_vec)
  min_len_vals = minimum(length, series_vec)
  min_len = min(min_len_times, min_len_vals)
  if min_len == 0
    return nothing
  end
  times = times_vec[1][1:min_len]
  mat = hcat([v[1:min_len] for v in series_vec]...)
  mean_vec = vec(mean(mat, dims=2))
  stderr_vec = vec(std(mat, dims=2)) ./ sqrt(size(mat, 2))
  return times, mean_vec, stderr_vec
end

"""
  compute_mean_imbalance_stats(df)

For rows at fixed (L, h), compute mean±stderr trajectories grouped by
  (graph_type, initial_state_initial_maxdim).
Returns Dict[(type::String, maxdim::Int)] => (times, mean, stderr).
"""
function compute_mean_imbalance_stats(df::DataFrame)
  results = Dict{Tuple{String,Int},NamedTuple}()
  graph_types = sort(unique(df.graph_type))
  for type in graph_types
    df_type = filter(row -> row.graph_type == type, df)
    if isempty(df_type)
      continue
    end
    maxdims = sort(unique(df_type.initial_state_initial_maxdim))
    for maxdim in maxdims
      df_maxdim = filter(row -> row.initial_state_initial_maxdim == maxdim, df_type)
      if isempty(df_maxdim)
        continue
      end
      times_vec = df_maxdim[:, :times]
      values_vec = df_maxdim[:, :imbalance]
      stats = aggregate_series_mean_std(times_vec, values_vec)
      if stats === nothing
        continue
      end
      times, mean_vals, stderr_vals = stats
      results[(type, maxdim)] = (times=times, mean=mean_vals, stderr=stderr_vals)
    end
  end
  return results
end

"""
  compute_ed_error_and_rows(df_current, L, h)

Compute mean absolute error of `df_current` trajectories against ED.
Returns (error::Union{Nothing,Float64}, df_current_common::DataFrame).
"""
function compute_ed_error_and_rows(df_current::DataFrame, L::Int, h)
  if isempty(df_current)
    return nothing, df_current
  end
  common_gridnums = Set(df_current.graph_gridnum)
  if isempty(common_gridnums)
    return nothing, df_current
  end
  df_current_common = filter(r -> r.graph_gridnum in common_gridnums, df_current)
  sort!(df_current_common, :graph_gridnum)
  current_imbalances = [r.imbalance for r in eachrow(df_current_common)]
  if isempty(current_imbalances)
    return nothing, df_current_common
  end
  min_len = minimum(length, current_imbalances)
  if min_len < 2
    return nothing, df_current_common
  end
  errors = []
  for gridnum in common_gridnums
    ed_imb, _ = get_ed_benchmark(L, h, [gridnum])
    sim = filter(r -> r.graph_gridnum == gridnum, df_current_common)
    imb = only(sim).imbalance
    error = ed_imb[1:min_len-1] .- imb[2:min_len]
    push!(errors, error)
  end
  errors = hcat(errors...)
  errors = mean(abs.(errors), dims=1)
  mean_error = exp.(mean(log.(errors)))
  std_error = min(std(errors), mean_error * 0.9) / sqrt(length(errors))
  return mean_error, std_error, df_current_common
end

"""
  compute_benchmark_error_and_rows(df_current, df_benchmark)

Compute mean absolute error of `df_current` against `df_benchmark` (e.g. larger D).
Returns (error::Union{Nothing,Float64}, df_current_common::DataFrame).
"""
function compute_benchmark_error_and_rows(df_current::DataFrame, df_benchmark::DataFrame)
  if isempty(df_current) || isempty(df_benchmark)
    return nothing, df_current
  end
  benchmark_gridnums = Set(df_benchmark.graph_gridnum)
  current_gridnums = Set(df_current.graph_gridnum)
  common_gridnums = intersect(benchmark_gridnums, current_gridnums)
  if isempty(common_gridnums)
    return nothing, df_current
  end
  df_benchmark_common = filter(r -> r.graph_gridnum in common_gridnums, df_benchmark)
  df_current_common = filter(r -> r.graph_gridnum in common_gridnums, df_current)
  sort!(df_benchmark_common, :graph_gridnum)
  sort!(df_current_common, :graph_gridnum)
  benchmark_imbalances = [r.imbalance for r in eachrow(df_benchmark_common)]
  current_imbalances = [r.imbalance for r in eachrow(df_current_common)]
  if isempty(benchmark_imbalances) || isempty(current_imbalances)
    return nothing, df_current_common
  end
  min_len_benchmark = minimum(length, benchmark_imbalances)
  min_len_current = minimum(length, current_imbalances)
  min_len = min(min_len_benchmark, min_len_current)
  if min_len == 0
    return nothing, df_current_common
  end
  benchmark_matrix = hcat([imb[1:min_len] for imb in benchmark_imbalances]...)
  current_matrix = hcat([imb[1:min_len] for imb in current_imbalances]...)
  error = abs.(benchmark_matrix .- current_matrix)
  error = mean(error, dims=1)
  mean_error = exp.(mean(log.(error)))
  std_error = min(std(error), mean_error * 0.9) / sqrt(length(error))

  return mean_error, std_error, df_current_common
end

"""
  average_runtime(df_rows)
Return mean runtime across rows.
"""
function average_runtime(df_rows::DataFrame)
  return mean([minimum(t[2:end]) for t in df_rows.ex_times])
end

"""
  average_num_params(df_rows)
Return mean number of parameters across rows.
"""
function average_num_params(df_rows::DataFrame)
  return exp.(mean(log.([mean(p) for p in df_rows.num_size])))
end

"""
  compute_ed_error_series(d_times, d_vals, ed_imbalance)

Given trajectory (d_times, d_vals) and an ED imbalance series sampled at
times 0.1:0.1:100, return (common_times, error_series) or `nothing` if
not enough overlap.
"""
function compute_ed_error_series(d_times, d_vals, ed_imbalance)
  ed_times = collect(0.1:0.1:100)
  common_times = intersect(d_times, ed_times)
  if isempty(common_times)
    return nothing
  end
  ed_idx = findall(t -> t in common_times, ed_times)
  d_idx = findall(t -> t in common_times, d_times)
  if isempty(ed_idx) || isempty(d_idx)
    return nothing
  end
  # ensure lengths match
  len = min(length(ed_idx), length(d_idx))
  if len == 0
    return nothing
  end
  ct = common_times[1:len]
  err = abs.(ed_imbalance[ed_idx[1:len]] .- d_vals[d_idx[1:len]])
  return ct, err
end

"""
  compute_benchmark_error_series(d_times, d_vals, b_times, b_vals)

Align two trajectories by minimum length and compute absolute error series.
Returns (times_to_plot, error_series) or `nothing` if empty.
"""
function compute_benchmark_error_series(d_times, d_vals, b_times, b_vals)
  len = min(length(d_times), length(b_vals))
  if len == 0
    return nothing
  end
  return d_times[1:len], abs.(d_vals[1:len] .- b_vals[1:len])
end

"""
  earliest_divergence(df_graph; threshold=0.01)

Given a DataFrame `df_graph` (fixed L, h, grid, and graph_type) containing
trajectories for multiple bond dimensions (column `initial_state_initial_maxdim`),
return the earliest time and value on the largest-χ curve where it diverges from
the next-lower χ by more than `threshold`. Returns `(tdiv, y_at_maxχ)` or `nothing`.
"""
function earliest_divergence(df_graph::DataFrame; threshold::Float64=0.01)
  if nrow(df_graph) < 2
    return nothing
  end
  Ds_local = sort(unique(df_graph.initial_state_initial_maxdim))
  if length(Ds_local) < 2
    return nothing
  end
  Dmax = Ds_local[end]
  Dnext = Ds_local[end-1]

  rows_max = filter(row -> row.initial_state_initial_maxdim == Dmax, df_graph)
  rows_next = filter(row -> row.initial_state_initial_maxdim == Dnext, df_graph)
  isempty(rows_max) && return nothing
  isempty(rows_next) && return nothing

  rmax = rows_max[argmax(length.(rows_max.times)), :]
  rnext = rows_next[argmax(length.(rows_next.times)), :]

  # Align by common times if grids differ; otherwise use min shared length
  t1 = rmax.times
  t2 = rnext.times
  if t1 === t2 || (length(t1) == length(t2) && all(t1 .== t2))
    len_common = min(length(t1), length(t2))
    len_common <= 0 && return nothing
    t = t1[1:len_common]
    y1 = rmax.imbalance[1:len_common]
    y2 = rnext.imbalance[1:len_common]
  else
    common_times = intersect(t1, t2)
    isempty(common_times) && return nothing
    # Preserve time order by sorting common times as they appear in t1
    idx1_map = Dict(v => i for (i, v) in pairs(t1))
    idx2_map = Dict(v => i for (i, v) in pairs(t2))
    # Filter to times present in both with valid indices
    ct = [τ for τ in t1 if haskey(idx2_map, τ)]
    isempty(ct) && return nothing
    y1 = [rmax.imbalance[idx1_map[τ]] for τ in ct]
    y2 = [rnext.imbalance[idx2_map[τ]] for τ in ct]
    t = ct
  end

  Δ = abs.(y1 .- y2)
  idx = findfirst(>(threshold), Δ)
  isnothing(idx) && return nothing
  return (t[idx], y1[idx])
end

"""
  earliest_divergence_stats(stat_max, stat_next; threshold=0.01)

Given two mean trajectories `stat_max` and `stat_next` (with fields `times` and
`mean` as returned by `compute_mean_imbalance_stats`), return the earliest time
and corresponding value on the max-χ mean curve where the absolute difference of
means exceeds `threshold`. Returns `(tdiv, y_at_maxχ)` or `nothing`.
"""
function earliest_divergence_stats(stat_max::NamedTuple, stat_next::NamedTuple; threshold::Float64=0.01)
  t1 = stat_max.times
  t2 = stat_next.times
  y1 = stat_max.mean
  y2 = stat_next.mean

  if isempty(t1) || isempty(t2) || isempty(y1) || isempty(y2)
    return nothing
  end

  # Align time grids if needed
  if (length(t1) == length(t2)) && all(t1 .== t2)
    len_common = min(length(t1), length(t2), length(y1), length(y2))
    len_common <= 0 && return nothing
    t = t1[1:len_common]
    a = y1[1:len_common]
    b = y2[1:len_common]
  else
    # Use common times present in both series
    idx1_map = Dict(v => i for (i, v) in pairs(t1))
    idx2_map = Dict(v => i for (i, v) in pairs(t2))
    ct = [τ for τ in t1 if haskey(idx2_map, τ)]
    isempty(ct) && return nothing
    a = [y1[idx1_map[τ]] for τ in ct]
    b = [y2[idx2_map[τ]] for τ in ct]
    t = ct
  end

  Δ = abs.(a .- b)
  idx = findfirst(>(threshold), Δ)
  isnothing(idx) && return nothing
  return (t[idx], a[idx])
end

"""
  earliest_divergence_mean_time_by_grid(df_graph; threshold=0.01)

Compute the divergence time per grid (between largest and next-largest χ) and
return the mean divergence time across grids. Uses the longest trajectory per χ
for each grid and aligns by common times if needed. Returns a NamedTuple
`(tmean=t̄, count=n, ymean=ȳ)` or `nothing` if no divergences found.
"""
function earliest_divergence_mean_time_by_grid(df_graph::DataFrame; threshold::Float64=0.01)
  if nrow(df_graph) < 2
    return nothing
  end
  Ds_local = sort(unique(df_graph.initial_state_initial_maxdim))
  if length(Ds_local) < 2
    return nothing
  end
  Dmax = Ds_local[end]
  Dnext = Ds_local[end-1]

  gridnums = sort(unique(df_graph.graph_gridnum))
  isempty(gridnums) && return nothing

  tdivs = Float64[]
  ydivs = Float64[]

  for g in gridnums
    rows_max = filter(row -> row.initial_state_initial_maxdim == Dmax && row.graph_gridnum == g, df_graph)
    rows_next = filter(row -> row.initial_state_initial_maxdim == Dnext && row.graph_gridnum == g, df_graph)
    isempty(rows_max) && continue
    isempty(rows_next) && continue

    rmax = rows_max[argmax(length.(rows_max.times)), :]
    rnext = rows_next[argmax(length.(rows_next.times)), :]

    t1 = rmax.times
    t2 = rnext.times
    if (length(t1) == length(t2)) && all(t1 .== t2)
      len_common = min(length(t1), length(t2))
      len_common <= 0 && continue
      t = t1[1:len_common]
      y1 = rmax.imbalance[1:len_common]
      y2 = rnext.imbalance[1:len_common]
    else
      idx1_map = Dict(v => i for (i, v) in pairs(t1))
      idx2_map = Dict(v => i for (i, v) in pairs(t2))
      ct = [τ for τ in t1 if haskey(idx2_map, τ)]
      isempty(ct) && continue
      y1 = [rmax.imbalance[idx1_map[τ]] for τ in ct]
      y2 = [rnext.imbalance[idx2_map[τ]] for τ in ct]
      t = ct
    end

    Δ = abs.(y1 .- y2)
    idx = findfirst(>(threshold), Δ)
    isnothing(idx) && continue
    push!(tdivs, t[idx])
    push!(ydivs, y1[idx])
  end

  isempty(tdivs) && return nothing
  return (tmean=mean(tdivs), count=length(tdivs), ymean=mean(ydivs))
end

function load_dicts(dir; start=nothing)
  local entries
  try
    entries = readdir(dir)
  catch e
    @warn "Could not load directory $dir: $e"
    return []
  end

  jld2_files = String[]
  for entry in entries
    full_path = joinpath(dir, entry)
    if isfile(full_path) && endswith(entry, ".jld2")
      if isnothing(start) || startswith(entry, start)
        push!(jld2_files, full_path)
      end
    end
  end
  println("Found $(length(jld2_files)) files to load.")

  results = Vector{Any}(nothing, length(jld2_files))

  @threads for i in 1:length(jld2_files)
    filepath = jld2_files[i]
    try
      results[i] = load(filepath)["results"]
    catch e
      @show load(filepath)
      return 0
      @warn "Could not load file '$filepath' on thread $(threadid()): $e"
    end
  end
  loaded_data = filter(!isnothing, results)
  println("Finished loading. Loaded $(length(loaded_data)) files successfully.")
  return loaded_data
end

function filter_configs(
  data::Vector{Any};
  t_step::Union{Nothing,Float64}=nothing,
  t_range_end::Union{Nothing,Real}=nothing,
  L::Union{Nothing,Int}=nothing,
  gridnum::Union{Nothing,Int}=nothing,
  method::Union{Nothing,String,Symbol}=nothing,
  h::Union{Nothing,Float64}=nothing,
  free::Union{Nothing,Bool}=nothing,
  tree_opt::Union{Nothing,Bool}=nothing,
  maxdim_opt::Union{Nothing,Bool}=nothing,
  maxdim::Union{Nothing,Int}=nothing,
)
  return filter(data) do d
    p = d["parameters"]
    g = p["graph"]
    t = p["time_evolution"]
    m = p["model"]

    match_t_step = isnothing(t_step) || (t["t_step"] == t_step)
    match_t_range_end = isnothing(t_range_end) || (t["t_range"][2] == t_range_end)
    match_L = isnothing(L) || (g["L"] == L)
    match_gridnum = isnothing(gridnum) || (g["gridnum"] == gridnum)
    match_method = isnothing(method) || (String(t["method"]) == String(method))
    match_h = isnothing(h) || (m["h"] == h)
    match_maxdim = isnothing(maxdim) || (t["maxdim"] == maxdim)

    # Graph type checks
    is_free_graph = g["type"] == "FreeGraph"
    is_tree_graph = g["type"] == "TreeGraph"

    match_free_cond = isnothing(free) || (is_free_graph == free)

    match_tree_opt_cond = true
    if !isnothing(tree_opt)
      if is_free_graph
        match_tree_opt_cond = g["optimize_structure"] == tree_opt
      elseif is_tree_graph
        match_tree_opt_cond = !tree_opt # Based on original logic
      else
        match_tree_opt_cond = false
      end
    end

    match_maxdim_opt_cond = true
    if !isnothing(maxdim_opt)
      if is_free_graph
        match_maxdim_opt_cond = g["optimize_bonddim"] == maxdim_opt
      else
        match_maxdim_opt_cond = false
      end
    end

    return match_t_step &&
           match_t_range_end &&
           match_L &&
           match_gridnum &&
           match_method &&
           match_h &&
           match_maxdim &&
           match_free_cond &&
           match_tree_opt_cond &&
           match_maxdim_opt_cond
  end
end

function reconstructed_conf(reconstructed)
  time_evo_reco = getfield(getfield(reconstructed, :fields)[1], :fields)
  time_evo = TimeEvolutionConfig(time_evo_reco..., 5.0, 1200)
  initial_state = getfield(reconstructed, :fields)[2]
  save_dir = getfield(reconstructed, :fields)[3]
  model = getfield(reconstructed, :fields)[4]
  if isa(getfield(reconstructed, :fields)[5], SnakeGraph)
    graph = getfield(getfield(reconstructed, :fields)[5], :fields)
  else
    graph = FreeGraph(getfield(getfield(reconstructed, :fields)[5], :fields)..., nothing, false)
  end
  observer_rec = getfield(getfield(reconstructed, :fields)[6], :fields)
  maxdims = [i * 5.0 => x for (i, x) in enumerate(observer_rec[5])]
  observer = Observer(times=observer_rec[1], sz=observer_rec[2], ee=observer_rec[3], ex_times=observer_rec[4], maxdim=maxdims, memory=observer_rec[6], num_size=observer_rec[7], errors=[0 => Dict()])
  TreeConfig(time_evo, initial_state, save_dir, model, graph, observer)
end

function turn_sim_into_dict(dir, out_dir)
  mkpath(out_dir)
  entries = readdir(dir)
  for entry in entries
    full_path = joinpath(dir, entry)
    out_path = joinpath(out_dir, entry)
    @show full_path
    if isfile(full_path) && endswith(entry, ".jld2")
      try
        config = reconstructed_conf(load(full_path)["results"])
        keep_every_nth!(config.observer.maxdim, 50)
        for (i, sz) in enumerate(config.observer.sz)
          config.observer.sz[i] = sz_dict_to_array(sz)
        end
        jldopen(out_path, "w") do file
          file["config"] = config
          file["results"] = TTNEvo.config_to_dict(config)
        end
      catch e
        rethrow(e)
        return
      end
    end
  end
end

function save_tree_structure_for_plotting(df_row, dir)
  for r in df_row.maxdim
    file = joinpath(dir, "tn_t=$(r[1]).jld2")
    mkpath(dir)
    jldopen(file, "w") do file
      file["tn"] = r[2]
    end
  end
end

function load_L_new(L; old_snake=true)
  tree_pre = "data/L$L-column-tree-new-pre-det"
  tree = "data/L$L-column-tree-new"
  local snake
  if old_snake
    snake = "data/proc/L$L-column-snake/"
  else
    snake = "data/L$L-column-snake-new"
  end
  if L == 12
    snake = "data/L$L-column-snake"
    tree_pre = "data/L$L-column-tree-pre-det"
  end
  if L == 8
    snake = "data/L$L-column-snake"
    tree_pre = "data/L$L-column-tree-new-pre-det"
  end
  return load_general_dirs([tree_pre, snake])
end

function load_L4_peps()
  peps = "data/L4-column-peps"
  return load_general_dirs([peps]; remove_dup=false, remove_small_time=false)
end

function load_L4_tree_demo()
  tree = "data/L4-column-tree"
  return load_general_dirs([tree])
end

function load_L4_special_comp()
  snake = "data/L4-column-snake-new"
  tree = "data/L4-column-tree-new-pre-det"
  return load_general_dirs([snake, tree])
end

function load_L10()
  snake_L10 = "data/L10.0-column-snake/"
  tree_L10 = "data/L10-column-tree-pre-det/"
  return load_general_dirs([snake_L10, tree_L10]; remove_small_time=false)
end

function load_L(L)
  snake = "data/L$L-column-snake/"
  tree = "data/L$L-column-tree-pre-det/"
  # return load_general_dirs([tree]; remove_small_time=false)
  return load_general_dirs([snake, tree]; remove_small_time=false)
end

function load_general_dirs(dirs; remove_dup=true, remove_small_time=true)
  all_data = []
  for dir in dirs
    println("Loading $dir")
    push!(all_data, load_dicts(dir))
  end

  df = DataFrame()
  for data in all_data
    for d in data
      flat_dict = merge(d["parameters"], d["observables"])
      push!(df, flat_dict, cols=:union)
    end
  end
  if remove_small_time
    df = remove_small_time_simulations(df; t_end=50.0)
  end
  if remove_dup
    df = remove_duplicates(df)
  end
  df.imbalance = TTNEvo.columnar_imbalance_total.(df.sz)
  # jldopen("data/dataframe.jld2", "w") do file
  #   file["dataframe"] = df
  # end
  return df
end

function heatmaps(dir)
  entries = readdir(dir)
  for entry in entries
    h = load(joinpath(dir, entry))["grid"]
    fig = Figure()
    ax = Axis(fig[1, 1])
    hm = heatmap!(ax, h)
    Colorbar(fig[1, 2], hm)

    # Safely strip extension and append .svg
    name = splitext(entry)[1]
    outbase = joinpath("plots", "grid", name)
    save(outbase * ".svg", fig)
    save(outbase * ".png", fig)
    save(outbase * ".pdf", fig)
  end
end

function load_all_data()
  snake_L4 = "data/proc/L4-column-comp/"
  tree_L4 = "data/proc/L4-column-tree/"
  snake_L4_2 = "data/L4-column-snake/"
  tree_L4_2 = "data/L4-column-tree/"
  snake_L6 = "data/proc/L6-column-snake/"
  snake_L8 = "data/L8-column-snake/"
  tree_L6 = "data/L6-column-tree/"
  tree_L8 = "data/L8-column-tree/"
  snake_L10 = "data/L10.0-column-snake/"
  tree_L10 = "data/L10-column-tree-pre-det/"

  all_data = []
  for dir in [tree_L4, snake_L6, snake_L8, tree_L6, tree_L8, snake_L4_2, tree_L4_2, snake_L10, tree_L10]
    println("Loading $dir")
    push!(all_data, load_dicts(dir))
  end
  push!(all_data, load_dicts(snake_L4; start="snake"))

  df = DataFrame()
  for data in all_data
    for d in data
      flat_dict = merge(d["parameters"], d["observables"])
      push!(df, flat_dict, cols=:union)
    end
  end
  # df = remove_small_time_simulations(df)
  df = remove_duplicates(df)
  df.imbalance = TTNEvo.columnar_imbalance_total.(df.sz)
  jldopen("data/dataframe.jld2", "w") do file
    file["dataframe"] = df
  end
  return df
end

function dicts_to_dataframe(data)
  df = DataFrame()
  for d in data
    flat_dict = merge(d["parameters"], d["observables"])
    push!(df, flat_dict, cols=:union)
  end
  return df
end

using DataFrames

function remove_small_time_simulations(df; t_end::Real=100.0, atol::Real=1e-8)
  return filter(row -> !isempty(row.times) && maximum(row.times) >= t_end - atol, df)
end

function remove_duplicates(df)
  grouped = groupby(df, [:graph_L, :model_h, :graph_type, :initial_state_initial_maxdim, :graph_gridnum])
  # Replace :accuracy with the actual column you're maximizing
  result = combine(grouped) do subdf
    subdf[argmax(length.(subdf.times)), :]  # select row with max accuracy in each group
  end
  return result
end

function wong_colors()
  return [
    colorant"#E69F00",  # orange
    colorant"#56B4E9",  # sky blue
    colorant"#009E73",  # bluish green
    colorant"#F0E442",  # yellow
    colorant"#0072B2",  # blue
    colorant"#D55E00",  # vermillion
    colorant"#CC79A7",  # reddish purple
  ]
end

function plot_accuracy_convergence(df::DataFrame; L::Int, dir)
  h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
  graph_types = sort(unique(filter(row -> row.graph_L == L, df).graph_type))

  if isempty(h_values)
    @warn "No data for L=$L"
    return
  end

  use_ed_benchmark = L == 4 ? true : false

  fig = Figure(size=pixel, fontsize=11pt)
  colors = wong_colors()

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(fig[h_idx, 1],
      title="h=$h, L=$L",
      xlabel="Maxdim",
      ylabel="Error (Max Abs Diff)",
      yscale=log10
    )

    df_h_filtered = filter(row -> row.graph_L == L && row.model_h == h, df)

    for (type_idx, type) in enumerate(graph_types)
      df_type = filter(row -> row.graph_type == type, df_h_filtered)
      if isempty(df_type)
        continue
      end

      maxdims = sort(unique(df_type.initial_state_initial_maxdim))
      if length(maxdims) < 2
        continue
      end

      benchmark_maxdim = maximum(maxdims)
      df_benchmark = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim, df_type)

      errors = Float64[]
      dims = Int[]

      end_idx = use_ed_benchmark ? 0 : 1
      for maxdim in maxdims[1:end-end_idx]
        df_current = filter(row -> row.initial_state_initial_maxdim == maxdim, df_type)
        if isempty(df_current)
          continue
        end

        if use_ed_benchmark
          error, _ = compute_ed_error_and_rows(df_current, L, h)
        else
          error, _ = compute_benchmark_error_and_rows(df_current, df_benchmark)
        end
        isnothing(error) && continue
        push!(errors, error + 1e-16)
        push!(dims, maxdim)
      end

      if !isempty(dims)
        line_color = colors[type_idx%length(colors)+1]
        scatter!(ax, dims, errors, label=type, color=line_color)
        lines!(ax, dims, errors, color=line_color)
      end
    end
    try
      axislegend(ax)
    catch
    end
  end

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)
  filename_pdf = "accuracy_convergence_L$(L).pdf"
  filename_png = "accuracy_convergence_L$(L).png"
  save(joinpath(plots_dir, filename_pdf), fig)
  save(joinpath(plots_dir, filename_png), fig)
  println("Saved plot to $(joinpath(plots_dir, filename_pdf)) and $(joinpath(plots_dir, filename_png))")

  return fig
end


function plot_accuracy_vs_parameters(df::DataFrame; L::Int, dir)
  h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
  graph_types = sort(unique(filter(row -> row.graph_L == L, df).graph_type))

  if isempty(h_values)
    @warn "No data for L=$L"
    return
  end

  use_ed_benchmark = L == 4

  fig = Figure(size=multiplot_size(), fontsize=11)

  # Two-column outer layout: shared ylabel + axes grid
  scaling = 1e5
  grid = fig[1, 2] = GridLayout()
  fig[1, 1] = Label(fig, "Error"; rotation=π / 2, tellheight=false)
  fig[2, 2] = Label(fig, L"\text{Number of Parameters} / 10^5"; tellwidth=false)

  colors = wong_colors()
  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]
  axes = Axis[]

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(grid[1, h_idx];
      yscale=log10,
      # xscale=log10,
    )
    push!(axes, ax)

    # improve tick formatting
    ax.xticks = WilkinsonTicks(5; simplicity_weight=1)
    ax.yticks = LogTicks(WilkinsonTicks(5))

    # hide y ticks/labels except for first subplot
    if h_idx != 1
      hideydecorations!(ax, grid=false)
    end

    # in-plot title (only h)
    text!(ax, 0.05, 0.10,
      text=L"h=%$h",
      align=(:left, :bottom),
      fontsize=11,
      space=:relative,
    )

    df_h_filtered = filter(row -> row.graph_L == L && row.model_h == h, df)

    for (type_idx, type) in enumerate(graph_types)
      df_type = filter(row -> row.graph_type == type, df_h_filtered)
      if isempty(df_type)
        continue
      end

      maxdims = sort(unique(df_type.initial_state_initial_maxdim))
      if L in (6, 8)
        maxdims = [64, 128, 196]
      end

      if length(maxdims) < 2 && !use_ed_benchmark
        continue
      end

      benchmark_maxdim = maximum(maxdims)
      df_benchmark = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim, df_type)

      errors = Float64[]
      stds = Float64[]
      params = Float64[]

      end_idx = use_ed_benchmark ? 0 : 1
      for maxdim in maxdims[1:end-end_idx]
        df_current = filter(row -> row.initial_state_initial_maxdim == maxdim, df_type)
        if isempty(df_current)
          continue
        end

        local error, serr, df_current_common
        if use_ed_benchmark
          error, serr, df_current_common = compute_ed_error_and_rows(df_current, L, h)
        else
          error, serr, df_current_common = compute_benchmark_error_and_rows(df_current, df_benchmark)
        end
        isnothing(error) && continue
        avg_params = average_num_params(df_current_common)

        push!(errors, error + 1e-16)
        push!(stds, serr)
        push!(params, avg_params / scaling)
      end

      if !isempty(params)
        line_color = colors[(type_idx-1)%length(colors)+1]
        marker = marker_shapes[(type_idx-1)%length(marker_shapes)+1]

        # plot line + points + errorbars
        lines!(ax, params, errors; color=line_color, linewidth=1.5)
        scatter!(ax, params, errors; label=type,
          color=line_color, marker=marker, markersize=6)
        errorbars!(ax, params, errors, stds; direction=:y, color=line_color)
      end
    end

    # put legend only once
    if h_idx == 1
      try
        axislegend(ax; position=:rt, orientation=:vertical,
          nbanks=1, framevisible=false, fontsize=8.0)
      catch
      end
    end
  end

  colgap!(fig.layout, 4.0)
  rowgap!(fig.layout, 4.0)
  colgap!(grid, 2.0)
  rowgap!(grid, 0.0)
  linkyaxes!(axes...)

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)
  filename_pdf = "accuracy_vs_params_L$(L).pdf"
  filename_png = "accuracy_vs_params_L$(L).png"
  save(joinpath(plots_dir, filename_pdf), fig)
  save(joinpath(plots_dir, filename_png), fig)
  println("Saved plot to $(joinpath(plots_dir, filename_pdf)) and $(joinpath(plots_dir, filename_png))")

  return fig
end



function plot_params_error_color_runtime_allpoints(df::DataFrame; L::Int, dir)
  h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
  graph_types = sort(unique(filter(row -> row.graph_L == L, df).graph_type))

  if isempty(h_values)
    @warn "No data for L=$L"
    return
  end

  use_ed_benchmark = L == 4

  scaling = 1e5
  data_by_h = Dict{Float64,Vector{NamedTuple}}()
  global_rt_min = Inf
  global_rt_max = -Inf

  for h in h_values
    series_list = NamedTuple[]
    df_h_filtered = filter(row -> row.graph_L == L && row.model_h == h, df)

    for (type_idx, type) in enumerate(graph_types)
      df_type = filter(row -> row.graph_type == type, df_h_filtered)
      if isempty(df_type)
        continue
      end

      maxdims = sort(unique(df_type.initial_state_initial_maxdim))
      if length(maxdims) < 2 && !use_ed_benchmark
        continue
      end

      benchmark_maxdim = maximum(maxdims)
      df_benchmark = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim, df_type)

      params = Float64[]
      errors = Float64[]
      runtimes = Float64[]

      end_idx = use_ed_benchmark ? 0 : 1
      for maxdim in maxdims[1:end-end_idx]
        df_current = filter(row -> row.initial_state_initial_maxdim == maxdim, df_type)
        if isempty(df_current)
          continue
        end

        for d in eachrow(df_current)
          # Compute per-row error
          local err_val
          if use_ed_benchmark
            ed_imbalance, _ = get_ed_benchmark(L, h, [d.graph_gridnum])
            if isempty(ed_imbalance)
              continue
            end
            series = compute_ed_error_series(d.times, d.imbalance, ed_imbalance)
            series === nothing && continue
            common_times, err_series = series
            if isempty(err_series)
              continue
            end
            err_val = exp(mean(log.(err_series .+ 1e-16)))
          else
            bench_rows = filter(row -> row.graph_gridnum == d.graph_gridnum && row.initial_state_initial_maxdim == benchmark_maxdim, df_type)
            if isempty(bench_rows)
              continue
            end
            bench = first(bench_rows)
            series = compute_benchmark_error_series(d.times, d.imbalance, bench.times, bench.imbalance)
            series === nothing && continue
            times_to_plot, err_series = series
            if isempty(err_series)
              continue
            end
            err_val = exp(mean(log.(err_series .+ 1e-16)))
          end

          # Per-row params and runtime
          p_row = d.num_size
          local p_avg
          if isempty(p_row)
            continue
          else
            p_avg = mean(p_row)
          end

          t_row = d.ex_times
          if isempty(t_row)
            continue
          end
          rt_val = length(t_row) > 1 ? minimum(t_row[2:end]) : minimum(t_row)

          push!(params, p_avg / scaling)
          push!(errors, err_val + 1e-16)
          push!(runtimes, rt_val)

          if rt_val < global_rt_min
            global_rt_min = rt_val
          end
          if rt_val > global_rt_max
            global_rt_max = rt_val
          end
        end
      end

      if !isempty(params)
        push!(series_list, (type=type, type_idx=type_idx, params=params, errors=errors, runtimes=runtimes))
      end
    end
    data_by_h[h] = series_list
  end

  if !isfinite(global_rt_min) || !isfinite(global_rt_max)
    @warn "No valid runtime range computed for L=$L"
    return
  end

  # Plot: x=params, y=error, color=runtime
  fig = Figure(size=multiplot_size(), fontsize=11)
  grid = fig[1, 2] = GridLayout()
  fig[1, 1] = Label(fig, "Error"; rotation=π / 2, tellheight=false)
  fig[2, 2] = Label(fig, L"\text{Number of Parameters} / 10^5"; tellwidth=false)

  colors = wong_colors()
  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]
  axes = Axis[]

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(grid[1, h_idx]; yscale=log10)
    push!(axes, ax)

    ax.xticks = WilkinsonTicks(5)
    ax.yticks = LogTicks(WilkinsonTicks(3))
    if h_idx != 1
      hideydecorations!(ax, grid=false)
    end

    text!(ax, 0.05, 0.95,
      text=L"h=%$h",
      align=(:left, :top),
      fontsize=11,
      space=:relative,
    )

    for s in get(data_by_h, h, NamedTuple[])
      marker = marker_shapes[(s.type_idx-1)%length(marker_shapes)+1]
      scatter!(ax, s.params, s.errors;
        color=s.runtimes,
        colormap=:magma,
        colorrange=(global_rt_min + 1e-16, global_rt_max + 1e-16),
        marker=marker,
        markersize=4.5,
        label=s.type,
      )
    end

    if h_idx == 1
      axislegend(ax; position=:rt, orientation=:vertical, nbanks=1, framevisible=false)
    end
  end

  Colorbar(fig[1, 3], colormap=:viridis, limits=(global_rt_min + 1e-16, global_rt_max + 1e-16), label="Execution Time (s)")

  colgap!(fig.layout, 4.0)
  rowgap!(fig.layout, 4.0)
  colgap!(grid, 2.0)
  rowgap!(grid, 0.0)
  linkyaxes!(axes...)

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)
  filename_pdf = "params_error_color_runtime_allpoints_L$(L).pdf"
  filename_png = "params_error_color_runtime_allpoints_L$(L).png"
  save(joinpath(plots_dir, filename_pdf), fig)
  save(joinpath(plots_dir, filename_png), fig)
  println("Saved plot to $(joinpath(plots_dir, filename_pdf)) and $(joinpath(plots_dir, filename_png))")

  return fig
end

function plot_params_runtime_colored_by_runtime(df::DataFrame; L::Int, dir)
  h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
  graph_types = sort(unique(filter(row -> row.graph_L == L, df).graph_type))

  if isempty(h_values)
    @warn "No data for L=$L"
    return
  end

  use_ed_benchmark = L == 4

  # First pass: compute data and global runtime range
  scaling = 1e5
  data_by_h = Dict{Float64,Vector{NamedTuple}}()
  global_rt_min = Inf
  global_rt_max = -Inf

  for h in h_values
    series_list = NamedTuple[]
    df_h_filtered = filter(row -> row.graph_L == L && row.model_h == h, df)

    for (type_idx, type) in enumerate(graph_types)
      df_type = filter(row -> row.graph_type == type, df_h_filtered)
      if isempty(df_type)
        continue
      end

      maxdims = sort(unique(df_type.initial_state_initial_maxdim))
      if length(maxdims) < 2 && !use_ed_benchmark
        continue
      end

      benchmark_maxdim = maximum(maxdims)
      df_benchmark = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim, df_type)

      params = Float64[]
      runtimes = Float64[]
      errors = Float64[]

      end_idx = use_ed_benchmark ? 0 : 1
      for maxdim in maxdims[1:end-end_idx]
        df_current = filter(row -> row.initial_state_initial_maxdim == maxdim, df_type)
        if isempty(df_current)
          continue
        end

        local error, serr, df_current_common
        if use_ed_benchmark
          error, serr, df_current_common = compute_ed_error_and_rows(df_current, L, h)
        else
          error, serr, df_current_common = compute_benchmark_error_and_rows(df_current, df_benchmark)
        end
        isnothing(error) && continue

        push!(errors, error + 1e-16)
        rt = average_runtime(df_current_common)
        push!(runtimes, rt)
        push!(params, average_num_params(df_current_common) / scaling)

        if rt < global_rt_min
          global_rt_min = rt
        end
        if rt > global_rt_max
          global_rt_max = rt
        end
      end

      if !isempty(params)
        push!(series_list, (type=type, type_idx=type_idx, params=params, runtimes=runtimes, errors=errors))
      end
    end
    data_by_h[h] = series_list
  end

  if !isfinite(global_rt_min) || !isfinite(global_rt_max)
    @warn "No valid runtime range computed for L=$L"
    return
  end

  # Plot with standard size to align with other figures/LaTeX
  fig = Figure(size=multiplot_size(), fontsize=11)
  grid = fig[1, 2] = GridLayout()
  fig[1, 1] = Label(fig, "Error"; rotation=π / 2, tellheight=false)
  fig[2, 2] = Label(fig, L"\text{Number of Parameters} / 10^5"; tellwidth=false)

  colors = wong_colors()
  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]
  axes = Axis[]

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(grid[1, h_idx]; yscale=log10)
    push!(axes, ax)

    ax.xticks = WilkinsonTicks(5)
    ax.yticks = LogTicks(WilkinsonTicks(3))
    if h_idx != 1
      hideydecorations!(ax, grid=false)
    end

    text!(ax, 0.95, 0.95,
      text=L"h=%$h",
      align=(:right, :top),
      fontsize=11,
      space=:relative,
    )

    for s in get(data_by_h, h, NamedTuple[])
      marker = marker_shapes[(s.type_idx-1)%length(marker_shapes)+1]
      # connect points by increasing params for readability
      if !isempty(s.params)
        order = sortperm(s.params)
        lines!(ax, s.params[order], s.errors[order]; color=:black, linewidth=1.0, transparency=true, alpha=0.5)
      end
      scatter!(ax, s.params, s.errors;
        color=s.runtimes,
        colormap=:magma,
        colorrange=(global_rt_min + 1e-16, global_rt_max + 1e-16),
        marker=marker,
        markersize=8,
        strokecolor=:black,
        strokewidth=0.8,
        label=s.type,
      )
    end
    if h_idx == 1
      axislegend(ax; position=(0.6, 0.12), framevisible=false, patchlabelgap=-4)
    end
  end

  Colorbar(fig[1, 3], colormap=:magma, limits=(global_rt_min + 1e-16, global_rt_max + 1e-16), label="Execution Time (s)")

  colgap!(fig.layout, 4.0)
  rowgap!(fig.layout, 4.0)
  colgap!(grid, 2.0)
  rowgap!(grid, 0.0)
  linkyaxes!(axes...)

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)
  filename_pdf = "params_runtime_color_runtime_L$(L).pdf"
  filename_png = "params_runtime_color_runtime_L$(L).png"
  save(joinpath(plots_dir, filename_pdf), fig)
  save(joinpath(plots_dir, filename_png), fig)
  println("Saved plot to $(joinpath(plots_dir, filename_pdf)) and $(joinpath(plots_dir, filename_png))")

  return fig
end

function multiplot_size()
  return (500, 200)
end


function plot_runtime_vs_parameters(df::DataFrame; L::Int, dir)
  h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
  graph_types = sort(unique(filter(row -> row.graph_L == L, df).graph_type))

  if isempty(h_values)
    @warn "No data for L=$L"
    return
  end

  fig = Figure(size=multiplot_size(), fontsize=11)

  # Outer layout: col 1 = shared ylabel, col 2 = plots grid
  scaling = 1e5
  grid = fig[1, 2] = GridLayout()
  fig[1, 1] = Label(fig, "Execution Time (s)"; rotation=π / 2, tellheight=false)
  fig[2, 2] = Label(fig, L"\text{Number of Parameters} / 10^5"; tellwidth=false)

  colors = wong_colors()
  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]
  axes = Axis[]

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(grid[1, h_idx];
      # xscale=log10,
      yscale=log10,
    )
    push!(axes, ax)

    # clean ticks
    ax.xticks = LogTicks(WilkinsonTicks(5))
    ax.yticks = LogTicks(WilkinsonTicks(3))

    # hide y-decorations except for the first plot
    if h_idx != 1
      hideydecorations!(ax, grid=false)
    end

    # add subplot title (only h)
    text!(ax, 0.05, 0.95,
      text=L"h=%$h",
      align=(:left, :top),
      fontsize=11,
      space=:relative,
    )

    df_h_filtered = filter(row -> row.graph_L == L && row.model_h == h, df)

    for (type_idx, type) in enumerate(graph_types)
      df_type = filter(row -> row.graph_type == type, df_h_filtered)
      if isempty(df_type)
        continue
      end

      maxdims = sort(unique(df_type.initial_state_initial_maxdim))

      runtimes = Float64[]
      params = Float64[]

      for maxdim in maxdims
        df_current = filter(row -> row.initial_state_initial_maxdim == maxdim, df_type)
        if isempty(df_current)
          continue
        end

        avg_runtime = mean([mean(t) for t in df_current.ex_times])
        avg_params = mean([p[end] for p in df_current.num_size])

        push!(runtimes, avg_runtime + 1e-16)
        push!(params, avg_params / scaling)
      end

      if !isempty(params)
        line_color = colors[(type_idx-1)%length(colors)+1]
        marker = marker_shapes[(type_idx-1)%length(marker_shapes)+1]

        lines!(ax, params, runtimes; color=line_color, linewidth=1.5)
        scatter!(ax, params, runtimes; label=type,
          color=line_color, marker=marker, markersize=6)
      end
    end

    if h_idx == 1
      axislegend(ax; position=:rt, orientation=:vertical,
        nbanks=1, framevisible=false)
    end
  end

  colgap!(fig.layout, 4.0)
  rowgap!(fig.layout, 4.0)
  colgap!(grid, 2.0)
  rowgap!(grid, 0.0)
  linkyaxes!(axes...)

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)
  filename_pdf = "runtime_vs_params_L$(L).pdf"
  filename_png = "runtime_vs_params_L$(L).png"
  save(joinpath(plots_dir, filename_pdf), fig)
  save(joinpath(plots_dir, filename_png), fig)
  println("Saved plot to $(joinpath(plots_dir, filename_pdf)) and $(joinpath(plots_dir, filename_png))")

  return fig
end



function plot_accuracy_vs_runtime(df::DataFrame; L::Int, dir)
  h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
  graph_types = sort(unique(filter(row -> row.graph_L == L, df).graph_type))

  if isempty(h_values)
    @warn "No data for L=$L"
    return
  end

  use_ed_benchmark = L == 4

  fig = Figure(size=multiplot_size(), fontsize=11)

  # Outer layout: col 1 = shared ylabel, col 2 = plots grid
  grid = fig[1, 2] = GridLayout()
  fig[1, 1] = Label(fig, "Error"; rotation=π / 2, tellheight=false, padding=(0, 0, 0, 0))

  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]
  colors = wong_colors()
  axes = Axis[]

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(grid[1, h_idx];
      xlabel="",    # shared xlabel later
      ylabel="",
      yscale=log10,
    )
    push!(axes, ax)

    ax.xticks = WilkinsonTicks(3)
    ax.yticks = LogTicks(WilkinsonTicks(3))
    # hide y-decorations except for the first plot
    if h_idx != 1
      hideydecorations!(ax, grid=false)
    end

    # in-plot title
    text!(ax, 0.05, 0.1,
      text=L"h = %$h",
      align=(:left, :bottom),
      fontsize=11,
      space=:relative,
    )

    df_h_filtered = filter(row -> row.graph_L == L && row.model_h == h, df)

    for (type_idx, type) in enumerate(graph_types)
      df_type = filter(row -> row.graph_type == type, df_h_filtered)
      if isempty(df_type)
        continue
      end

      maxdims = sort(unique(df_type.initial_state_initial_maxdim))
      if length(maxdims) < 2 && !use_ed_benchmark
        continue
      end

      benchmark_maxdim = maximum(maxdims)
      df_benchmark = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim, df_type)

      errors = Float64[]
      stds = Float64[]
      runtimes = Float64[]

      end_idx = use_ed_benchmark ? 0 : 1
      for maxdim in maxdims[1:end-end_idx]
        df_current = filter(row -> row.initial_state_initial_maxdim == maxdim, df_type)
        if isempty(df_current)
          continue
        end

        local error
        local df_current_common
        if use_ed_benchmark
          error, serr, df_current_common = compute_ed_error_and_rows(df_current, L, h)
        else
          error, serr, df_current_common = compute_benchmark_error_and_rows(df_current, df_benchmark)
        end
        isnothing(error) && continue

        push!(errors, error + 1e-16)
        push!(stds, serr)
        push!(runtimes, average_runtime(df_current_common))
      end

      if !isempty(runtimes)
        line_color = colors[(type_idx-1)%length(colors)+1]
        marker = marker_shapes[(type_idx-1)%length(marker_shapes)+1]
        lines!(ax, runtimes, errors; color=line_color, linewidth=1.5)
        scatter!(ax, runtimes, errors; label=type,
          color=line_color, marker=marker, markersize=6)
        errorbars!(ax, runtimes, errors, stds; direction=:y, color=line_color)
      end
    end

    if h_idx == 1
      try
        axislegend(ax, position=:rt)
      catch
      end
    end
  end

  # shared xlabel
  fig[2, 2] = Label(fig, "Execution Time (s)"; tellwidth=false)

  colgap!(fig.layout, 4.0)
  rowgap!(fig.layout, 4.0)
  colgap!(grid, 2.0)
  rowgap!(grid, 0.0)
  linkyaxes!(axes...)   # shared y-axis

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)
  filename_pdf = "accuracy_vs_runtime_L$(L).pdf"
  filename_png = "accuracy_vs_runtime_L$(L).png"
  save(joinpath(plots_dir, filename_pdf), fig)
  save(joinpath(plots_dir, filename_png), fig)
  println("Saved plot to $(joinpath(plots_dir, filename_pdf)) and $(joinpath(plots_dir, filename_png))")
  return fig
end


function plot_individual_imbalance(df, L_values=[4, 6, 8]; dir)
  for L in L_values
    h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
    for h in h_values
      gridnum_values = sort(unique(filter(row -> row.model_h == h && row.graph_L == L, df).graph_gridnum))
      for grid in gridnum_values
        println("Plotting L=$(L) h=$(h) grid=$(grid)")
        dfi = filter(row -> row.graph_L == L && row.model_h == h && row.graph_gridnum == grid, df)
        dfi = sort(dfi, [:graph_type, :initial_state_initial_maxdim])

        graph_types = sort(unique(dfi.graph_type))
        if isempty(graph_types)
          continue
        end

        fig = Figure(fontsize=8pt)
        axes = [Axis(fig[i, 1],
          title=L"%$(gtype) $h=%$h$, $L=%$L$",
          xlabel=L"t",
          ylabel=L"I",
          xscale=log10,
          xticks=[0.1, 1, 10, 100],
          limits=(0.1, 100, nothing, nothing)
        ) for (i, gtype) in enumerate(graph_types)
        ]

        Ds = unique(dfi.initial_state_initial_maxdim)
        Ds_sorted = sort(Ds, rev=true)
        D_to_val = if length(Ds_sorted) > 1
          Dict(D => (i - 1) / (length(Ds_sorted) - 1) for (i, D) in enumerate(Ds_sorted))
        else
          Dict(D => 0.5 for (i, D) in enumerate(Ds_sorted))
        end

        for (i, gtype) in enumerate(graph_types)
          ax = axes[i]
          df_graph = filter(row -> row.graph_type == gtype, dfi)

          for d in eachrow(df_graph)
            Dval = D_to_val[d.initial_state_initial_maxdim]
            color = get(ColorSchemes.viridis, Dval)
            lines!(ax, d.times, d.imbalance,
              label=L"\chi=%$(d.initial_state_initial_maxdim)",
              color=color)
          end

          if L == 4
            ed, _ = get_ed_benchmark(L, h, [grid])
            max_T = maximum(maximum.(df_graph.times))
            time = 0.1:0.1:max_T
            max_ind = length(time)
            if !isempty(ed)
              lines!(ax, time, ed[1:max_ind], label="ED", color=:black)
            end
          end

          # Mark earliest divergence time between largest and next-largest χ
          begin
            threshold = 0.01 / 2.0
            divpt = earliest_divergence(df_graph; threshold)
            if divpt !== nothing
              tdiv, ydiv = divpt
              # draw vertical marker and a star at the max-D curve
              vlines!(ax, [tdiv]; color=:black, linestyle=:dash, linewidth=1.5)
              scatter!(ax, [tdiv], [ydiv]; color=:black, marker=:star5, markersize=9)
              # Optional small annotation
              text!(ax, tdiv, ydiv; text=L"\Delta>%$(threshold)", align=(:left, :top), color=:black, fontsize=7)
            end
          end
          try
            Legend(fig[1:2, 2], axes[1])
          catch
          end
        end

        plot_dir = joinpath("plots", dir, "individual")
        mkpath(plot_dir)
        save(joinpath(plot_dir, "imbalance_L=$(L)_h=$(h)_grid=$(grid).pdf"), fig)
        save(joinpath(plot_dir, "imbalance_L=$(L)_h=$(h)_grid=$(grid).png"), fig)
      end
    end
  end
end

function plot_individual_imbalance_error(df, L_values=[4, 6, 8]; dir)
  for L in L_values
    h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
    for h in h_values
      gridnum_values = sort(unique(filter(row -> row.model_h == h && row.graph_L == L, df).graph_gridnum))
      for grid in gridnum_values
        println("Plotting Imbalance Error for L=$(L) h=$(h) grid=$(grid)")
        dfi = filter(row -> row.graph_L == L && row.model_h == h && row.graph_gridnum == grid, df)
        dfi = sort(dfi, [:graph_type, :initial_state_initial_maxdim])

        if nrow(dfi) == 0
          continue
        end

        graph_types = sort(unique(dfi.graph_type))
        if isempty(graph_types)
          continue
        end

        fig = Figure(fontsize=8pt)
        axes = [Axis(fig[i, 1],
          title=L"%$(gtype) $h=%$h$, $L=%$L$",
          xlabel=L"t",
          ylabel=L"\mathrm{Error}(\,I\,)",
          xscale=log10,
          xticks=logticks1_with_minors(-1, 2),
          yscale=log10) for (i, gtype) in enumerate(graph_types)
        ]

        ax_insets = [Axis(fig[i, 1];
          width=Relative(0.4),
          height=Relative(0.4),
          halign=0.95,
          valign=0.20,
          title=L"\text{Moving Avg.}",
          yscale=log10,
          # xticks=LogTicks([1, 5, 10, 50, 100]),
          xticks=logticks1(-1, 2),
          xscale=log10,
        ) for i in 1:length(graph_types)]

        if length(axes) > 1
          linkyaxes!(axes...)
          linkyaxes!(ax_insets...)
        end

        Ds = unique(dfi.initial_state_initial_maxdim)
        Ds_sorted = sort(Ds, rev=true)
        D_to_val = if length(Ds_sorted) > 1
          Dict(D => (i - 1) / (length(Ds_sorted) - 1) for (i, D) in enumerate(Ds_sorted))
        else
          Dict(D => 0.5 for (i, D) in enumerate(Ds_sorted))
        end

        if L == 4
          ed_imbalance, _ = get_ed_benchmark(L, h, [grid])
          if isempty(ed_imbalance)
            @warn "No ED benchmark for L=4, h=$h, grid=$grid. Skipping."
            continue
          end
          for (i, gtype) in enumerate(graph_types)
            ax = axes[i]
            ax_inset = ax_insets[i]
            df_graph = filter(row -> row.graph_type == gtype, dfi)
            for d in eachrow(df_graph)
              series = compute_ed_error_series(d.times, d.imbalance, ed_imbalance)
              series === nothing && continue
              common_times, error = series
              Dval = D_to_val[d.initial_state_initial_maxdim]
              color = get(ColorSchemes.viridis, Dval)
              lines!(ax, common_times, error .+ 1e-16,
                label=L"\chi=%$(d.initial_state_initial_maxdim)",
                color=color)
              window = 50
              if length(error) > window
                ma_error = moving_average(error, window)
                ma_times = common_times[window÷2:length(ma_error)+window÷2-1]
                lines!(ax_inset, ma_times, ma_error .+ 1e-16; color=color)
              end
            end
          end
        else
          for (i, gtype) in enumerate(graph_types)
            ax = axes[i]
            ax_inset = ax_insets[i]
            df_graph = filter(row -> row.graph_type == gtype, dfi)
            if nrow(df_graph) < 2
              continue
            end
            maxdims = sort(unique(df_graph.initial_state_initial_maxdim))
            benchmark_maxdim = maximum(maxdims)
            benchmark_row_list = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim, df_graph)
            isempty(benchmark_row_list) && continue
            benchmark_row = first(benchmark_row_list)
            b_times = benchmark_row.times
            b_vals = benchmark_row.imbalance
            for d in eachrow(df_graph)
              if d.initial_state_initial_maxdim == benchmark_maxdim
                continue
              end
              series = compute_benchmark_error_series(d.times, d.imbalance, b_times, b_vals)
              series === nothing && continue
              times_to_plot, error = series
              Dval = D_to_val[d.initial_state_initial_maxdim]
              color = get(ColorSchemes.viridis, Dval)
              lines!(ax, times_to_plot, error .+ 1e-16,
                label=L"\chi=%$(d.initial_state_initial_maxdim)",
                color=color)
              window = 50
              if length(error) > window
                ma_error = moving_average(error, window)
                ma_times = times_to_plot[window÷2:length(ma_error)+window÷2-1]
                lines!(ax_inset, ma_times, ma_error .+ 1e-16; color=color)
              end
            end
          end
        end

        try
          Legend(fig[1:2, 2], axes[1])
        catch
        end

        plots_dir = joinpath("plots", dir, "individual_error")
        mkpath(plots_dir)
        filename = "imbalance_error_L=$(L)_h=$(h)_grid=$(grid).pdf"
        save(joinpath(plots_dir, filename), fig)
        filename = "imbalance_error_L=$(L)_h=$(h)_grid=$(grid).png"
        save(joinpath(plots_dir, filename), fig)
        println("Saved plot to $(joinpath(plots_dir, filename))")
      end
    end
  end
end



function plot_ed(; L=4, grid)
  fig = Figure(size=pixel, fontsize=11pt)
  for (i, h) in enumerate([10.0])
    ax = Axis(fig[i, 1], xlabel="Time", ylabel="Imbalance", title="Imbalance vs. Time for L=$L, h=$h, grid=$grid")
    imb, _ = get_ed_benchmark(L, h, [grid])
    imb_new = zeros(length(imb) + 1)
    imb_new[1] = 1
    imb_new[2:end] = imb
    lines!(ax, 0:0.1:100, imb_new)
  end
end


function plot_tree_structure_over_time(df; dir, L, h, gridnum, maxdim)
  dfi = only(filter(row -> row.initial_state_initial_maxdim == maxdim && row.model_h == h && row.graph_L == L && row.graph_gridnum == gridnum && row.graph_type == "FreeGraph", df))

  for d in dfi.maxdim
    t = d[1]
    links = d[2]
    out_dir = joinpath("plots", dir, "L=$(L)_h=$(h)_grid=$(gridnum)_maxdim=$(maxdim)")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "t=$t.pdf")
    save(out_path, TTNEvo.plot_free_graph_with_maxdim(links))
  end
end

function plot_error_vs_disorder(df::DataFrame; dir)
  L_values = sort(unique(df.graph_L))

  for L in L_values
    use_ed_benchmark = (L == 4)

    fig = Figure(fontsize=11pt)
    ax = Axis(fig[1, 1],
      title="Error vs Disorder for L=$L",
      xlabel="Disorder strength (h)",
      ylabel="Error (Mean Abs Diff)",
      yscale=log10
    )

    df_L = filter(row -> row.graph_L == L, df)
    graph_types = sort(unique(df_L.graph_type))
    color_schemes = [:acton, :bamako]
    linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond, :star5]

    type_idx = 1
    for type in graph_types
      df_type = filter(row -> row.graph_type == type, df_L)
      if isempty(df_type)
        continue
      end

      maxdims = sort(unique(df_type.initial_state_initial_maxdim))

      if !use_ed_benchmark && length(maxdims) < 2
        continue
      end

      benchmark_maxdim = maximum(maxdims)

      maxdims_to_plot = if use_ed_benchmark
        maxdims
      else
        filter(d -> d != benchmark_maxdim, maxdims)
      end

      if isempty(maxdims_to_plot)
        continue
      end

      D_to_val = if length(maxdims_to_plot) > 1
        Dict(D => (i - 1) / (length(maxdims_to_plot) - 1) * 0.6 + 0.2 for (i, D) in enumerate(sort(maxdims_to_plot)))
      else
        Dict(D => 0.5 for D in maxdims_to_plot)
      end

      for maxdim in maxdims_to_plot
        h_values = sort(unique(filter(row -> row.initial_state_initial_maxdim == maxdim, df_type).model_h))
        errors = Float64[]
        hs_for_plot = Float64[]

        for h in h_values
          df_h = filter(row -> row.model_h == h, df_type)

          df_current = filter(row -> row.initial_state_initial_maxdim == maxdim, df_h)
          if isempty(df_current)
            continue
          end

          local error
          if use_ed_benchmark
            error, _ = compute_ed_error_and_rows(df_current, L, h)
          else
            df_benchmark = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim && row.model_h == h, df_type)
            error, _ = compute_benchmark_error_and_rows(df_current, df_benchmark)
          end
          isnothing(error) && continue
          push!(errors, error + 1e-16)
          push!(hs_for_plot, h)
        end

        if !isempty(hs_for_plot)
          cscheme = color_schemes[mod1(type_idx, length(color_schemes))]
          if cscheme == :acton
            line_color = get(ColorSchemes.acton, D_to_val[maxdim])
          else
            line_color = get(ColorSchemes.bamako, D_to_val[maxdim])
          end
          linestyle = :solid
          marker = markers[mod1(type_idx, length(markers))]
          label = "$type, D=$maxdim"
          lines!(ax, hs_for_plot, errors, label=label, color=line_color, linestyle=linestyle)
          scatter!(ax, hs_for_plot, errors, color=line_color, label=label, marker=marker, markersize=15)
        end
      end
      type_idx += 1
    end

    Legend(fig[1, 2], ax, merge=true; groupgap=10)
    plots_dir = joinpath("plots", dir)
    mkpath(plots_dir)
    filename = "error_vs_disorder_L$(L).pdf"
    save(joinpath(plots_dir, filename), fig)
    println("Saved plot to $(joinpath(plots_dir, filename))")
  end
end

function plot_error_vs_system_size(df::DataFrame; dir)
  h_values = sort(unique(df.model_h))

  for h in h_values
    fig = Figure(size=pixel, fontsize=11pt)
    ax = Axis(fig[1, 1],
      title="Error vs System Size for h=$h",
      xlabel="System size (L)",
      ylabel="Error (Mean Abs Diff)",
      yscale=log10
    )

    graph_types = sort(unique(df.graph_type))
    color_schemes = [:acton, :bamako]
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond, :star5]

    type_idx = 1
    for type in graph_types
      df_type = filter(row -> row.graph_type == type && row.model_h == h, df)
      if isempty(df_type)
        continue
      end

      maxdims = sort(unique(df_type.initial_state_initial_maxdim))

      D_to_val = if length(maxdims) > 1
        Dict(D => (i - 1) / (length(maxdims) - 1) * 0.6 + 0.2 for (i, D) in enumerate(sort(maxdims)))
      else
        Dict(D => 0.5 for D in maxdims)
      end

      for maxdim in maxdims
        L_values = sort(unique(filter(row -> row.initial_state_initial_maxdim == maxdim, df_type).graph_L))
        errors = []
        Ls_for_plot = []

        for L in L_values
          use_ed_benchmark = (L == 4)

          df_L = filter(row -> row.graph_L == L, df_type)
          if isempty(df_L)
            continue
          end
          maxdims_at_L = sort(unique(df_L.initial_state_initial_maxdim))
          benchmark_maxdim = maximum(maxdims_at_L)

          if !use_ed_benchmark && maxdim >= benchmark_maxdim
            continue # Cannot compute error against itself or a smaller maxdim
          end

          df_current = filter(row -> row.initial_state_initial_maxdim == maxdim && row.graph_L == L, df_type)
          if isempty(df_current)
            continue
          end

          local error
          if use_ed_benchmark
            error, _ = compute_ed_error_and_rows(df_current, L, h)
          else
            df_benchmark = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim && row.graph_L == L, df_type)
            error, _ = compute_benchmark_error_and_rows(df_current, df_benchmark)
          end
          isnothing(error) && continue

          push!(errors, error + 1e-16) # for log scale
          push!(Ls_for_plot, L)
        end

        if !isempty(Ls_for_plot)
          cscheme = color_schemes[mod1(type_idx, length(color_schemes))]
          if cscheme == :acton
            line_color = get(ColorSchemes.acton, D_to_val[maxdim])
          else
            line_color = get(ColorSchemes.bamako, D_to_val[maxdim])
          end
          marker = markers[mod1(type_idx, length(markers))]
          label = "$type, D=$maxdim"
          lines!(ax, Ls_for_plot, errors, label=label, color=line_color)
          scatter!(ax, Ls_for_plot, errors, color=line_color, label=label, marker=marker, markersize=15)
        end
      end
      type_idx += 1
    end

    try
      Legend(fig[1, 2], ax, merge=true)
    catch
    end
    plots_dir = joinpath("plots", dir)
    mkpath(plots_dir)
    filename = "error_vs_system_size_h$(h).pdf"
    save(joinpath(plots_dir, filename), fig)
    println("Saved plot to $(joinpath(plots_dir, filename))")
  end
end

function plot_mean_imbalance(df::DataFrame; L::Int, h, dir, maxdim=nothing)
  df_filtered = filter(row -> row.graph_L == L && row.model_h == h, df)
  if isempty(df_filtered)
    @warn "No data for L=$L, h=$h"
    return
  end
  # Keep desired χ range and non-empty time series; allow varying end times
  df_filtered = filter(row -> row.initial_state_initial_maxdim >= 128 && !isempty(row.times), df_filtered)

  fig = Figure(fontsize=11pt)
  ax = Axis(fig[1, 1], xlabel=L"t", ylabel="Mean Imbalance", title="L=$L, h=$h")

  # Determine common gridnums across graph types at their respective highest χ
  graph_types = sort(unique(df_filtered.graph_type))
  dmax_by_type = Dict{String,Int}()
  grids_by_type = Dict{String,Set{Int}}()
  for type in graph_types
    df_t = filter(row -> row.graph_type == type, df_filtered)
    isempty(df_t) && (grids_by_type[type] = Set{Int}(); continue)
    Ds_t = sort(unique(df_t.initial_state_initial_maxdim))
    isempty(Ds_t) && (grids_by_type[type] = Set{Int}(); continue)
    Dmax_t = Ds_t[end]
    dmax_by_type[type] = Dmax_t
    df_t_dmax = filter(row -> row.initial_state_initial_maxdim == Dmax_t, df_t)
    grids_by_type[type] = Set(df_t_dmax.graph_gridnum)
  end
  sets_for_intersection = collect(values(grids_by_type))
  common_grids = isempty(sets_for_intersection) ? Set{Int}() : reduce(intersect, sets_for_intersection)
  if isempty(common_grids)
    @warn "No common gridnums across methods at highest χ; using all available runs per method."
    df_common = df_filtered
  else
    println("Common gridnums used across methods: " * string(sort(collect(common_grids))))
    df_common = filter(row -> row.graph_gridnum in common_grids, df_filtered)
  end

  stats = compute_mean_imbalance_stats(df_common)

  # Only plot highest χ per method with distinct linestyles; add divergence markers
  linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
  plotted_any = false
  # Derive available methods from stats keys
  graph_types = sort(unique(first.(keys(stats))))
  # Color by graph type using magma colormap
  n_types = length(graph_types)
  # Avoid extreme ends when few types; keep colors in mid-range
  low, high = 0.2, 0.8
  fracs = if n_types == 1
    [0.55]
  elseif n_types == 2
    [0.35, 0.65]
  else
    [low + (high - low) * (j - 1) / (n_types - 1) for j in 1:n_types]
  end

  for (i, type) in enumerate(graph_types)
    Ds_type = sort([D for (t, D) in keys(stats) if t == type])
    isempty(Ds_type) && continue
    Dmax = Ds_type[end]
    stat_max = get(stats, (type, Dmax), nothing)
    stat_max === nothing && continue

    # Info: list simulations (gridnums) that contribute to this mean
    df_used = filter(row -> row.graph_type == type && row.initial_state_initial_maxdim == Dmax, df_common)
    used_info = [(r.graph_gridnum, maximum(r.times)) for r in eachrow(df_used)]
    println("Using $(length(used_info)) simulations for type='$type', D=$(Dmax): (grid, t_max)=" * string(used_info))

    times = stat_max.times
    mean_vals = stat_max.mean
    stderr_vals = stat_max.stderr

    color = get(ColorSchemes.magma, fracs[i])
    style = linestyles[(i-1)%length(linestyles)+1]
    label = "$type, D=$Dmax"
    lines!(ax, times, mean_vals; color=color, linestyle=style, label=label, linewidth=2)
    band!(ax, times, mean_vals .- stderr_vals, mean_vals .+ stderr_vals; color=(color, 0.25))
    plotted_any = true

    # Divergence vs the next-lower χ (mean over gridnums)
    if length(Ds_type) >= 2
      df_type = filter(row -> row.graph_type == type, df_common)
      threshold = 0.01 / 5
      divmean = earliest_divergence_mean_time_by_grid(df_type; threshold)
      if divmean !== nothing
        tdiv = divmean.tmean
        # Place marker on the plotted mean curve at nearest time to tdiv
        idx_near = argmin(abs.(times .- tdiv))
        y_on_curve = mean_vals[idx_near]
        vlines!(ax, [tdiv]; color=color, linestyle=style, linewidth=1.5)
        scatter!(ax, [tdiv], [y_on_curve]; color=color, marker=:star5, markersize=9)
        text!(ax, tdiv, y_on_curve; text=L"\Delta>%$(threshold)", align=(:left, :top), color=color, fontsize=7)
      end
    end
  end

  if plotted_any
    axislegend(ax, position=:rb)
  end

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)
  filename_pdf = "mean_imbalance_L$(L)_h$(h)_maxdim$(maxdim).pdf"
  filename_png = "mean_imbalance_L$(L)_h$(h)_maxdim$(maxdim).png"
  save(joinpath(plots_dir, filename_pdf), fig)
  save(joinpath(plots_dir, filename_png), fig)
  println("Saved plot to $(joinpath(plots_dir, filename_pdf))")

end

function main(df; dir, L_values=[4, 6, 8, 10])
  plot_individual_imbalance(df, L_values; dir)
  plot_individual_imbalance_error(df, L_values; dir)
  for L in L_values
    println("Generating plots for L=$L")
    for h in [0.0, 5.0, 10.0, 20.0, 30.0, 50.0]
      plot_mean_imbalance(df; L=L, h, dir)
    end
    plot_accuracy_convergence(df; L=L, dir)
    plot_accuracy_vs_parameters(df; L=L, dir)
    plot_runtime_vs_parameters(df; L=L, dir)
    plot_accuracy_vs_runtime(df; L=L, dir)
    plot_params_runtime_colored_by_runtime(df; L=L, dir)
  end
  plot_error_vs_disorder(df; dir)
  plot_error_vs_system_size(df; dir)
end

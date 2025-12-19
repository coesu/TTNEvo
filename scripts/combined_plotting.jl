using CairoMakie
using CairoMakie: hidexdecorations!, linkxaxes!, linkyaxes!
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
using LsqFit
using Roots
using NetworkLayout
using ColorSchemes

default_colorscheme() = ColorSchemes.Zissou1

function set_default_colorscheme!(scheme)
  return scheme
end

scheme_color(fraction) = get(default_colorscheme(), clamp(fraction, 0.0, 1.0))

function scheme_colors(n::Integer; low::Float64=0.15, high::Float64=0.85)
  cs = default_colorscheme()
  empty_palette = similar(cs.colors, 0)
  n <= 0 && return empty_palette
  low == high && return fill(get(cs, clamp(low, 0.0, 1.0)), n)
  n == 1 && return [get(cs, clamp((low + high) / 2, 0.0, 1.0))]
  step = (high - low) / max(n - 1, 1)
  return [get(cs, clamp(low + step * (i - 1), 0.0, 1.0)) for i in 1:n]
end

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
      xminortickalign=1,
      ytickalign=1,
      yminortickalign=1,
      spinewidth=1,
      xgridvisible=true, ygridvisible=true,
      xgridcolor=RGBAf(0, 0, 0, 0.12), ygridcolor=RGBAf(0, 0, 0, 0.12),
      xminorgridcolor=RGBAf(0, 0, 0, 0.12), yminorgridcolor=RGBAf(0, 0, 0, 0.12),
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

global labels = Dict("FreeGraph" => "TTN", "SnakeGraph" => "MPS", "HierarchicalTree" => "Hierch", "HilbertCurve" => "Hilbert")

dir_labels = Dict(
  "bench_baseline" => "1TDVP",
  "bench_krylov" => "Krylov",
  "bench_shrewd" => "CBE",
  "bench_twosite" => "2TDVP",
)

dir_label(dir::AbstractString) = begin
  base = basename(dir)
  for (pattern, lbl) in dir_labels
    occursin(pattern, base) && return lbl
  end
  return base
end

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
  columnar_imbalance_ed(sz_matrix, L)

Compute columnar imbalance from ED `sz_expectations` matrix shaped (time, sites).
"""
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
      # y = div(i - 1, L) + 1  # unused for columnar
      if iseven(x)
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

"""
  plot_simulation_with_ed(sim_dir::String; outfile::Union{Nothing,String}=nothing)

Load a single simulation from `sim_dir` (expects a JLD2 saved by run_simulation/save_simulation_data)
and plot ED − TN of the columnar imbalance for the same L, h, gridnum.
Also shows the maximal bond dimension χ_max and evolution step times.

Saves a PDF and PNG under `plots/single_with_ed/<basename>` unless `outfile` is provided.
Returns the Makie Figure.
"""
function plot_simulation_with_ed(sim_dir::String; outfile::Union{Nothing,String}=nothing)
  # Find a finished result if available, otherwise any JLD2 file
  @assert isdir(sim_dir) "Directory not found: $(sim_dir)"
  files = filter(f -> endswith(f, ".jld2"), readdir(sim_dir))
  @assert !isempty(files) "No .jld2 files found in $(sim_dir)"
  finished = filter(f -> occursin("finished_results", f), files)
  file = isempty(finished) ? files[1] : finished[1]
  fullpath = joinpath(sim_dir, file)

  cfg = nothing
  try
    cfg = jldopen(fullpath, "r") do io
      return io["config"]
    end
  catch e
    error("Failed to load config from $(fullpath): $(e)")
  end

  dir_labels = Dict(
    "bench_baseline" => "1TDVP",
    "bench_krylov" => "Krylov",
    "bench_shrewd" => "CBE",
    "bench_twosite" => "2TDVP",
  )
  # Extract simulation info
  L = cfg.graph.L
  h = cfg.model.h
  grid = cfg.graph.gridnum
  times = cfg.observer.times
  sim_imb = TTNEvo.columnar_imbalance_total(cfg.observer.sz)

  # Load ED series for same parameters
  ed_imb = nothing
  function _load_ed(L, h, grid)
    # Try the commonly used ED paths
    p1 = joinpath(@__DIR__, "../../heisenberg_ed/data/fun_results_columnar_Lx=$(L)_Ly=$(L)_hmax=$(h)_gridnum=$(grid).jld2")
    p2 = joinpath(@__DIR__, "../../heisenberg_ed/data/results_columnar_Lx=$(L)_Ly=$(L)_hmax=$(h)_gridnum=$(grid).jld2")
    if isfile(p1)
      d = load(p1)
      return columnar_imbalance_ed(d["sz_expectations"], L)
    elseif isfile(p2)
      d = load(p2)
      return columnar_imbalance_ed(d["sz_expectations"], L)
    else
      return nothing
    end
  end

  ed_imb = _load_ed(L, h, grid)
  if ed_imb === nothing
    @warn "No ED data found for L=$(L), h=$(h), grid=$(grid). Skipping ED overlay."
  end

  # Time grid for ED assumes 0.1 step
  ed_times = ed_imb === nothing ? Float64[] : collect(0.1:0.1:0.1*length(ed_imb))

  # Compute aligned absolute differences |ED − TN|
  T = Float64[]
  D = Float64[]
  if ed_imb !== nothing
    ed_map = Dict(round(tt, digits=8) => i for (i, tt) in pairs(ed_times))
    for (i, tt) in pairs(times)
      key = round(tt, digits=8)
      if haskey(ed_map, key)
        j = ed_map[key]
        diff = abs(ed_imb[j] - sim_imb[i])
        diff > 0 || continue
        push!(T, tt)
        push!(D, diff)
      end
    end
  end

  # Prepare auxiliary data for bond dimension and step time
  t_md = Float64[]
  d_md = Float64[]
  if !isempty(cfg.observer.maxdim)
    for pair in cfg.observer.maxdim
      time = first(pair)
      time > 0 || continue
      ld = last(pair)
      maxd = try
        maximum(ld[e] for e in edges(ld))
      catch
        try
          maximum(values(ld))
        catch
          NaN
        end
      end
      isfinite(maxd) && maxd > 0 || continue
      push!(t_md, time)
      push!(d_md, maxd)
    end
  end

  rt_times = Float64[]
  rt_vals = Float64[]
  if !isempty(cfg.observer.ex_times) && !isempty(times)
    n = min(length(times), length(cfg.observer.ex_times))
    for idx in 1:n
      val = cfg.observer.ex_times[idx]
      isfinite(val) && val > 0 || continue
      time_val = times[idx]
      time_val > 0 || continue
      push!(rt_times, time_val)
      push!(rt_vals, val)
    end
  end

  # Plot
  fig = Figure(size=(12cm, 13cm))
  ax = Axis(fig[1, 1],
    xlabel="t",
    ylabel="|ED − TN| imbalance",
    yscale=log10,
    xscale=log10,
  )
  ax_md = Axis(fig[2, 1],
    xlabel="t",
    ylabel=L"\chi_{\max}",
    xscale=log10,
  )
  ax_rt = Axis(fig[3, 1],
    xlabel="t",
    ylabel=L"\Delta t_{\mathrm{step}} = t_i - t_{i-1}\,[\mathrm{s}]",
    yscale=log10,
    xscale=log10,
  )

  linkxaxes!(ax, ax_md, ax_rt)

  series_colors = scheme_colors(3)

  if !isempty(T)
    lines!(ax, T, D; color=series_colors[1], label="ED − TN")
    ax.yticks = ([1e-2, 1e-5, 1e-8, 1e-11], [L"10^{-2}", L"10^{-5}", L"10^{-8}", L"10^{-11}"])
  else
    @warn "No common time grid with ED found; cannot plot ED − TN differences."
  end

  if !isempty(t_md)
    order = sortperm(t_md)
    lines!(ax_md, t_md[order], d_md[order]; color=series_colors[2])
    ax_md.yticks = ([32, 64, 128], ["32", "64", "128"])
  else
    @warn "No bond-dimension data available for plotting."
  end

  if !isempty(rt_times)
    lines!(ax_rt, rt_times, rt_vals; color=series_colors[3])
    kmin = floor(Int, log10(minimum(rt_vals)))
    kmax = ceil(Int, log10(maximum(rt_vals)))
    ax_rt.yticks = logticks1(kmin, kmax)
  else
    @warn "No positive step-time samples available for plotting."
  end

  xtick_positions = [0.1, 1.0, 10.0]
  xtick_labels = ["0.1", "1", "10"]
  ax.xticks = (xtick_positions, xtick_labels)
  ax_md.xticks = (xtick_positions, xtick_labels)
  ax_rt.xticks = (xtick_positions, xtick_labels)

  try
    axislegend(ax; position=:rb)
  catch
  end

  # Save
  base = isnothing(outfile) ? joinpath("plots", "single_with_ed", splitdir(sim_dir)[2]) : outfile
  mkpath(dirname(base))
  save(base * ".pdf", fig)
  save(base * ".png", fig)
  println("Saved plot to $(base).pdf and $(base).png")
  return fig
end

"""
  plot_simulations_with_ed(sim_dirs::Vector{String}; labels=nothing, outfile=nothing)

Overlay the ED − TN columnar imbalance differences for multiple simulations with
the same (L, h, gridnum). `sim_dirs` should be directories containing JLD2 outputs
saved by run_simulation/save_simulation_data.

Saves PDF and PNG to `plots/single_with_ed/comparison_<L>_<h>_g<grid>` unless
`outfile` is provided. Returns the Makie Figure.
"""
function plot_simulations_with_ed(sim_dirs::Vector{String}; labels=nothing, outfile=nothing)
  @assert !isempty(sim_dirs) "No simulation directories provided"

  # Helper to choose the best file inside a directory
  function _pick_file(dir)
    files = filter(f -> endswith(f, ".jld2"), readdir(dir))
    @assert !isempty(files) "No .jld2 files found in $(dir)"
    finished = filter(f -> occursin("finished_results", f), files)
    file = isempty(finished) ? files[1] : finished[1]
    return joinpath(dir, file)
  end

  # Load first config to determine (L,h,grid)
  first_file = _pick_file(sim_dirs[1])
  @show first_file
  cfg0 = jldopen(first_file, "r") do io
    io["config"]
  end
  L = cfg0.graph.L
  h = cfg0.model.h
  grid = cfg0.graph.gridnum

  # Load ED
  function _load_ed(L, h, grid)
    p1 = joinpath(@__DIR__, "../../heisenberg_ed/data/fun_results_columnar_Lx=$(L)_Ly=$(L)_hmax=$(h)_gridnum=$(grid).jld2")
    p2 = joinpath(@__DIR__, "../../heisenberg_ed/data/results_columnar_Lx=$(L)_Ly=$(L)_hmax=$(h)_gridnum=$(grid).jld2")
    if isfile(p1)
      d = load(p1)
      return columnar_imbalance_ed(d["sz_expectations"], L)
    elseif isfile(p2)
      d = load(p2)
      return columnar_imbalance_ed(d["sz_expectations"], L)
    else
      return nothing
    end
  end
  ed_imb = _load_ed(L, h, grid)
  ed_times = ed_imb === nothing ? Float64[] : collect(0.1:0.1:0.1*length(ed_imb))

  # Prepare plot with subplots for max bond dimension and step time
  fig = Figure(size=(400, 400), fontsize=8pt)
  ax = Axis(fig[1, 1],
    xlabel="t",
    ylabel=L"|I_{\mathrm{ED}} - I_{\mathrm{TN}}|",
    yscale=log10,
    xscale=log10,
  )
  ax_md = Axis(fig[2, 1],
    xlabel="t",
    ylabel=L"\chi_{\max}",
    xscale=log10,
  )
  ax_rt = Axis(fig[3, 1],
    xlabel="t",
    ylabel=L"t_{\mathrm{step}}",
    xscale=log10,
  )

  linkxaxes!(ax, ax_md, ax_rt)

  # remove redundant x-axis labels/ticks from top and middle plots
  ax.xlabelvisible = false
  ax.xticklabelsvisible = false
  ax_md.xlabelvisible = false
  ax_md.xticklabelsvisible = false


  xtick_positions = [0.1, 1.0, 10.0]
  xtick_labels = ["0.1", "1", "10"]
  ax.xticks = (xtick_positions, xtick_labels)
  ax_md.xticks = (xtick_positions, xtick_labels)
  ax_rt.xticks = (xtick_positions, xtick_labels)

  # Colors/labels
  default_labels = [dir_label(dir) for dir in sim_dirs]
  @show default_labels
  labels = isnothing(labels) ? default_labels : labels
  labels = ["1TDVP",
    "Krylov",
    "CBE",
    "2TDVP",
  ]
  @show labels

  all_rt_vals = Float64[]

  ncurves = length(sim_dirs)
  colors = scheme_colors(ncurves)

  for (i, dir) in enumerate(sim_dirs)
    file = _pick_file(dir)
    cfg = jldopen(file, "r") do io
      io["config"]
    end
    # sanity check
    @assert cfg.graph.L == L && cfg.model.h == h && cfg.graph.gridnum == grid "All simulations must share (L, h, gridnum)"
    times = cfg.observer.times
    sim_imb = TTNEvo.columnar_imbalance_total(cfg.observer.sz)
    # Compute aligned difference ED − TN
    if ed_imb === nothing
      @warn "No ED data found; skipping $(labels[i])"
      continue
    end
    ed_map = Dict(round(tt, digits=8) => j for (j, tt) in pairs(ed_times))
    T = Float64[]
    D = Float64[]
    for (k, tt) in pairs(times)
      tt <= 0 && continue
      key = round(tt, digits=8)
      if haskey(ed_map, key)
        j = ed_map[key]
        diff = abs(ed_imb[j] - sim_imb[k])
        diff > 0 || continue
        push!(T, tt)
        push!(D, diff)
      end
    end
    if !isempty(T)
      lines!(ax, T, D; color=colors[(i-1)%length(colors)+1], label=labels[i])
    else
      @warn "No common time grid with ED for $(labels[i]); skipping curve."
    end

    # Plot max bond dimension over time if available
    if !isempty(cfg.observer.maxdim)
      t_md = Float64[]
      d_md = Float64[]
      for pair in cfg.observer.maxdim
        time = first(pair)
        time <= 0 && continue
        ld = last(pair)
        # Try to compute max over edges; fallback to values
        maxd = try
          maximum(ld[e] for e in edges(ld))
        catch
          try
            maximum(values(ld))
          catch
            NaN
          end
        end
        isfinite(maxd) && maxd > 0 || continue
        push!(t_md, time)
        push!(d_md, maxd)
      end
      if !isempty(t_md)
        p = sortperm(t_md)
        lines!(ax_md, t_md[p], d_md[p]; color=colors[(i-1)%length(colors)+1], label=labels[i])
      end
    end

    # Plot execution time per step
    if !isempty(cfg.observer.ex_times) && !isempty(times)
      n = min(length(times), length(cfg.observer.ex_times))
      rt_times = Float64[]
      rt_vals = Float64[]
      for idx in 1:n
        tval = times[idx]
        ex = cfg.observer.ex_times[idx]
        (tval > 0 && isfinite(ex) && ex > 0) || continue
        push!(rt_times, tval)
        push!(rt_vals, ex)
      end
      if !isempty(rt_times)
        lines!(ax_rt, rt_times, rt_vals; color=colors[(i-1)%length(colors)+1])
        append!(all_rt_vals, rt_vals)
      end
    end
  end
  ax.yticks = ([1e-2, 1e-5, 1e-8, 1e-11], [L"10^{-2}", L"10^{-5}", L"10^{-8}", L"10^{-11}"])
  ax_md.yticks = ([32, 64, 128], ["32", "64", "128"])
  ax_rt.yticks = ([5, 10, 20])
  # if !isempty(all_rt_vals)
  #   kmin = floor(Int, log10(minimum(all_rt_vals)))
  #   kmax = ceil(Int, log10(maximum(all_rt_vals)))
  #   ax_rt.yticks = logticks1(kmin, kmax)
  # end

  Label(fig[1, 1, TopLeft()], "a)", fontsize=9, font=:bold, halign=:left, padding=(0, 0, -10, 0))
  Label(fig[2, 1, TopLeft()], "b)", fontsize=9, font=:bold, halign=:left, padding=(0, 0, -10, 0))
  Label(fig[3, 1, TopLeft()], "c)", fontsize=9, font=:bold, halign=:left, padding=(0, 0, -10, 0))

  Legend(fig[2, 2], ax_md)

  base = isnothing(outfile) ? joinpath("plots", "single_with_ed", @sprintf("comparison_L%d_h%.1f_g%d", L, h, grid)) : outfile
  mkpath(dirname(base))
  save(base * ".pdf", fig)
  save(base * ".png", fig)
  println("Saved comparison plot to $(base).pdf and $(base).png")
  return fig
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
  fit_power_law_beta(times, values; tmin=5.0, tmax=50.0)

Estimate the decay exponent β from an averaged imbalance curve by fitting
  values ≈ A * t^(-β) on a log–log window t ∈ [tmin, tmax].
Returns a NamedTuple `(beta, stderr, npts, r2)` or `nothing` if not enough points.
"""


function fit_power_law_beta(times::AbstractVector, values::AbstractVector;
  yerrors::Union{Nothing,AbstractVector}=nothing,
  tmin::Real=5.0, tmax::Real=50.0,
  known_variances::Bool=true)

  isempty(times) && return nothing
  isempty(values) && return nothing

  idx = findall(i -> times[i] >= tmin && times[i] <= tmax &&
                       isfinite(values[i]) && values[i] > 0 &&
                       isfinite(times[i]), eachindex(times))
  length(idx) < 3 && return nothing

  t = Float64[times[i] for i in idx]
  y = Float64[values[i] for i in idx]

  X = log.(t)
  Y = log.(y)
  n = length(X)

  # weights in log space
  w = if yerrors === nothing
    ones(n)
  else
    σy = Float64[yerrors[i] for i in idx]
    σY = σy ./ y               # propagate: σY = σy / y
    1.0 ./ (σY .^ 2)
  end

  # weighted sums
  Sw = sum(w)
  Sx = sum(w .* X)
  Sy = sum(w .* Y)
  Sxx = sum(w .* X .* X)
  Sxy = sum(w .* X .* Y)

  den = Sw * Sxx - Sx^2
  den == 0 && return nothing

  b = (Sw * Sxy - Sx * Sy) / den
  a = (Sy * Sxx - Sx * Sxy) / den

  # residuals
  Ŷ = a .+ b .* X
  resid = Y .- Ŷ

  RSS = sum(w .* resid .^ 2)
  s2 = known_variances ? 1.0 : RSS / max(n - 2, 1)

  # parameter SEs
  stderr_b = sqrt(s2 * Sw / den)
  stderr_a = sqrt(s2 * Sxx / den)

  # weighted R² in log space
  Ybar_w = Sy / Sw
  SS_tot = sum(w .* (Y .- Ybar_w) .^ 2)
  r2 = SS_tot ≈ 0 ? 1.0 : 1.0 - RSS / SS_tot

  return (beta=-b,
    stderr=stderr_b,
    A=exp(a),
    stderr_A=exp(a) * stderr_a,
    r2=r2,
    npts=n,
    rmse_log=sqrt(s2))  # optional: RMSE in log units
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
  compute_beta_over_gridmean(df; L, h, tmin=5.0, tmax=50.0)

For fixed (L, h), take the highest available χ per `graph_type`, fit β for each individual
simulation on [tmin, tmax], and then average the fitted parameters. Returns a Dict keyed by
`graph_type::String` with values `(beta, stderr, npts, r2, A, Dmax, nfits)`.
"""
function compute_beta_over_gridmean(df::DataFrame; L::Int, h, tmin::Real=50.0, tmax::Real=100.0, Dmax=nothing)
  df_lh = filter(row -> row.graph_L == L && row.model_h == h, df)
  isempty(df_lh) && return Dict{String,NamedTuple}()
  df_lh = filter(row -> !isempty(row.times), df_lh)

  out = Dict{String,NamedTuple}()
  methods = sort(unique(df_lh.graph_type))
  for method in methods
    df_method = filter(row -> row.graph_type == method, df_lh)
    isempty(df_method) && continue
    Ds = sort(unique(df_method.initial_state_initial_maxdim))
    isempty(Ds) && continue
    if isnothing(Dmax)
      Dmax = Ds[end]
    end
    df_dmax = filter(row -> row.initial_state_initial_maxdim == Dmax, df_method)
    isempty(df_dmax) && continue

    fits = NamedTuple[]
    for row in eachrow(df_dmax)
      fit = fit_power_law_beta(row.times, row.imbalance; tmin=tmin, tmax=tmax)
      fit === nothing && continue
      push!(fits, fit)
    end
    isempty(fits) && continue

    betas = [f.beta for f in fits]
    beta_mean = mean(betas)
    beta_std = length(betas) > 1 ? std(betas; corrected=true) : fits[1].stderr
    beta_se = length(betas) > 1 ? beta_std / sqrt(length(betas)) : fits[1].stderr

    npts_vals = [f.npts for f in fits]
    r2_vals = [f.r2 for f in fits]
    A_vals = [f.A for f in fits]

    finite_r2 = filter(isfinite, r2_vals)
    r2_mean = isempty(finite_r2) ? NaN : mean(finite_r2)

    out[method] = (
      beta=beta_mean,
      stderr=beta_se,
      npts=round(Int, mean(npts_vals)),
      r2=r2_mean,
      A=mean(A_vals),
      Dmax=Dmax,
      nfits=length(fits),
    )
  end
  return out
end

using DataFrames
using Logging


function compute_beta_with_mean(
  df::DataFrame;
  L::Int,
  h,
  tmin::Real=50.0,
  tmax::Real=100.0,
  Dmax=nothing,
)
  df_lh = filter(row -> row.graph_L == L && row.model_h == h && !isempty(row.times), df)
  isempty(df_lh) && return Dict{String,NamedTuple}()

  stats = compute_mean_imbalance_stats(df_lh)
  isempty(stats) && return Dict{String,NamedTuple}()

  out = Dict{String,NamedTuple}()
  methods = sort(unique(first.(keys(stats))))
  isempty(methods) && return Dict{String,NamedTuple}()

  for method in methods
    Ds_type = sort([D for (t, D) in keys(stats) if t == method])
    isempty(Ds_type) && continue

    Dsel = isnothing(Dmax) ? Ds_type[end] : Dmax
    haskey(stats, (method, Dsel)) || continue
    st = stats[(method, Dsel)]
    fit = fit_power_law_beta(st.times, st.mean; yerrors=st.stderr, tmin=tmin, tmax=tmax)
    fit === nothing && continue
    fit.beta === NaN && continue

    out[string(method)] = (
      beta=fit.beta,
      stderr=fit.stderr,
      npts=fit.npts,
      r2=fit.r2,
      A=fit.A,
      Dmax=Dsel,
      nfits=1,
    )
  end

  return out
end


"""
  plot_beta_vs_disorder(df; L_values, h_values, dir, tmin=5.0, tmax=50.0)

For each method (`graph_type`), plot β versus disorder strength for the available system sizes.
Uses the highest available χ per method and the grid-averaged imbalance; styling matches the
publication theme and draws hues from the default colormap.
"""
function plot_beta_vs_disorder(
  df::DataFrame;
  L_values=[4, 6, 8, 10],
  h_values=[2.5, 5.0, 7.5, 10.0, 20.0, 30.0, 50.0],
  dir::String,
  tmin::Real=50.0,
  tmax::Real=100.0,
  xlim::Tuple{<:Real,<:Real}=(NaN, NaN),
  ylim::Tuple{<:Real,<:Real}=(1e-3, 0.3),
  yscale=identity,
)
  df_sub = filter(row -> row.graph_L in L_values && row.model_h in h_values, df)
  methods = sort(unique(df_sub.graph_type))
  isempty(methods) && return

  Ls_sorted = sort(unique(L_values))
  hs_sorted = sort(float.(h_values))

  low, high = 0.2, 0.85
  nL = length(Ls_sorted)
  color_fracs = nL == 1 ? [0.5] : [low + (high - low) * (i - 1) / (nL - 1) for i in 1:nL]
  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]

  for (method_idx, method) in enumerate(methods)
    fig = Figure(size=(400, 250), fontsize=12pt)
    ax = Axis(fig[1, 1],
      xlabel=L"h",
      ylabel=L"\beta",
      yscale=log10,  # Assuming logarithmic scale
      yticks=LogTicks([-1, -2, -3]),
      yminorticks=IntervalsBetween(9),  # Automatic minor ticks (9 per decade)
      yminorticksvisible=true,  # Show minor ticks
      yminorgridwidth=1,
      yminorgridvisible=true,  # Show minor grid lines
    )
    xlo, xhi = xlim
    if isnan(xlo) || isnan(xhi)
      xlo, xhi = minimum(hs_sorted), maximum(hs_sorted)
    end
    xlims!(ax, xlo, xhi)
    ylims!(ax, first(ylim), last(ylim))

    for (iL, L) in enumerate(Ls_sorted)
      betas = Float64[]
      errs = Float64[]
      hs = Float64[]
      for h in hs_sorted
        vals = compute_beta_with_mean(df; L=L, h=h, tmin=tmin, tmax=tmax)
        haskey(vals, method) || continue
        push!(hs, h)
        push!(betas, vals[method].beta)
        push!(errs, vals[method].stderr)
      end
      isempty(hs) && continue

      order = sortperm(hs)
      hs_use = hs[order]
      betas_use = betas[order]
      errs_use = errs[order]

      color = scheme_color(color_fracs[iL])
      marker = marker_shapes[(iL-1)%length(marker_shapes)+1]

      # small horizontal offset for each method
      offset = 0.2 * (iL - (length(Ls_sorted) + 1) / 2)

      hs_use_shifted = hs_use .+ offset

      cutoff = 1e-3
      if yscale == log10
        clamp!(betas_use, cutoff, 2.0)
        clamp!(errs_use, 0.0, 2.0 - cutoff)  # Errors can't exceed the range of betas_use
        lower = min.(errs_use, betas_use .- cutoff)  # Lower error can't push value below cutoff
        upper = min.(errs_use, 2.0 .- betas_use)  # Upper error can't push value above 2.0
      end

      # Plotting
      scatter!(ax, hs_use_shifted, betas_use;
        color=color,
        marker=marker,
        markersize=9,
        strokecolor=:black,
        label="L=$L",
        strokewidth=1.0,
      )

      yvals = betas_use

      errorbars!(ax, hs_use_shifted, yvals, lower, upper;
        direction=:y,
        color=color,
        linewidth=1.0,
        whiskerwidth=6,
      )
      hlines!(ax, [0.0]; color=:black, linewidth=1.0, linestyle=:dash)
    end

    try
      axislegend(ax; position=:rt, framevisible=false)
    catch
    end

    plots_dir = joinpath("plots", dir)
    mkpath(plots_dir)
    fname = joinpath(plots_dir, @sprintf("beta_vs_h_%s_tmin%.1f_tmax%.1f.pdf", method, tmin, tmax))
    save(fname, fig)
    save(replace(fname, ".pdf" => ".png"), fig)
    println("Saved β vs h plot for method=$(method) to $(fname)")
    display(fig)
  end
end

# --- Shared model (p = [p1, p2, p3, p4]) ---
model(x, p) = @. p[1] * exp(-p[2] * x)

# --- Fit + hc + σ_hc ---
function calculate_hc(h::AbstractVector, beta::AbstractVector, err::AbstractVector, beta_crit=0.01)
  p0 = [1.0, 0.01]

  fit = curve_fit(model, h, beta, 1.0 ./ (err .^ 2), p0)
  p̂ = coef(fit)
  cov = estimate_covar(fit)

  # Root of f(x) = model(x,p̂) - beta_crit
  f̂(x) = model(x, p̂) - beta_crit
  hc = find_zero(f̂, 0.0)

  # Hand-derived derivatives for uncertainty
  exp_term = exp(-p̂[2] * hc)
  dfdx = -p̂[1] * p̂[2] * exp_term
  df_dp = [exp_term,
    -p̂[1] * hc * exp_term,
  ]
  J = -df_dp / dfdx
  σ_hc = sqrt(J' * cov * J)

  return hc, σ_hc, p̂
end

# --- Main: compute (h,β,σ) per L, fit, and plot everything on one figure ---
function get_beta_values(df::DataFrame; tmin=50.0, tmax=100.0, method::AbstractString="FreeGraph", beta_crit=0.01, Dmax=nothing)
  L_vals = [4, 6, 8, 10]

  beta_fig = Figure(size=(500, 300), fontsize=9)
  ax_beta = Axis(beta_fig[1, 1],
    xlabel="h",
    ylabel="β",
  )

  hlines!(ax_beta, [0.0], color=:gray, linestyle=:dash, label="β = $(beta_crit)")

  results = Dict{Int,NamedTuple}()
  palette = default_colorscheme().colors

  label_assigned = false
  for (i, L) in enumerate(L_vals)
    color = palette[1+(i-1)%length(palette)]

    betas, errs, hs = Float64[], Float64[], Float64[]
    h_vals = sort(unique(df[df.graph_L.==L, :].model_h))
    for h in h_vals
      vals = compute_beta_with_mean(df; L=L, h=h, tmin=tmin, tmax=tmax, Dmax)
      haskey(vals, method) || continue
      push!(hs, h)
      push!(betas, vals[method].beta)
      push!(errs, vals[method].stderr)
    end

    isempty(hs) && continue
    h = collect(hs)[2:end]
    beta = collect(betas)[2:end]
    err = collect(errs)[2:end]

    hc, σ_hc, p̂ = calculate_hc(h, beta, err, beta_crit)
    results[L] = (h=h, beta=beta, err=err, hc=hc, σ_hc=σ_hc, p̂=p̂)

    errorbars!(ax_beta, h, beta, err;
      color=color,
      whiskerwidth=10,
    )
    scatter!(ax_beta, h, beta; color=color, markersize=7, label=L"L=%$L")

    hgrid = range(minimum(h), maximum(h), length=300)
    lines!(ax_beta, hgrid, model(hgrid, p̂); color=color, linewidth=2)

    vlines!(ax_beta, [hc]; color=color, linestyle=:dot, linewidth=2)
    vspan!(ax_beta, hc - σ_hc, hc + σ_hc; color=(color, 0.18))
  end

  axislegend(ax_beta, position=:rb)

  hc_fig = Figure(size=(500, 300), fontsize=12pt)
  ax_hc = Axis(hc_fig[1, 1],
    xlabel=L"L",
    ylabel=L"h_c",
  )

  if !isempty(results)
    L_sorted = sort(collect(keys(results)))
    hc_vals = [results[L].hc for L in L_sorted]
    σhc_vals = [results[L].σ_hc for L in L_sorted]

    errorbars!(ax_hc, L_sorted, hc_vals, σhc_vals;
      color=:black,
      whiskerwidth=10,
      label=L"\chi=%$Dmax"
    )
    scatter!(ax_hc, L_sorted, hc_vals; color=:black, markersize=8)
    ll = 2:0.01:10
    # lines!(ax_hc, ll, avalanche_critical_disorder.(ll); label="Analytical")

    axislegend(ax_hc)
  end

  plots_dir = joinpath("plots", "beta")
  mkpath(plots_dir)

  beta_fname = joinpath(plots_dir, "beta_vs_h_with_fit.pdf")
  save(beta_fname, beta_fig)
  save(replace(beta_fname, ".pdf" => ".png"), beta_fig)
  println("Saved β(h) plot to $(beta_fname)")

  hc_fname = joinpath(plots_dir, "hc_vs_L.pdf")
  save(hc_fname, hc_fig)
  save(replace(hc_fname, ".pdf" => ".png"), hc_fig)
  println("Saved h_c vs L plot to $(hc_fname)")

  display(beta_fig)
  display(hc_fig)
  return ax_hc
end

function get_beta_values!(ax_beta, ax_hc, df;
  tmin=50.0,
  tmax=100.0,
  method="FreeGraph",
  beta_crit=0.001,
  Dmax::Union{Nothing,Int}=nothing,
  color=nothing,
  marker=nothing,
  L_vals=[4, 6, 8],
  offset=nothing,
  start=1
)
  results = Dict{Int,NamedTuple}()
  linestyles = (:solid, :dash, :dot, :dashdot)
  palette = default_colorscheme().colors
  default_markers = (:circle, :rect, :diamond)
  label_assigned = false

  for (i, L) in enumerate(L_vals)
    betas, errs, hs = Float64[], Float64[], Float64[]
    h_vals = sort(unique(df[df.graph_L.==L, :].model_h))
    for h in h_vals
      vals = compute_beta_with_mean(df; L=L, h=h, tmin=tmin, tmax=tmax, Dmax)
      haskey(vals, method) || continue
      push!(hs, h)
      push!(betas, vals[method].beta)
      push!(errs, vals[method].stderr)
    end
    if length(hs) <= 1
      continue
    end

    h = collect(hs)[start:end]
    beta = collect(betas)[start:end]
    err = collect(errs)[start:end]
    hc, σ_hc, p̂ = calculate_hc(h, beta, err, beta_crit)
    results[L] = (h=h, beta=beta, err=err, hc=hc, σ_hc=σ_hc, p̂=p̂)

    linestyle = linestyles[1+(i-1)%length(linestyles)]
    local_color = isnothing(color) ? palette[1+(i-1)%length(palette)] : color
    local_marker = isnothing(marker) ? default_markers[1+(i-1)%length(default_markers)] : marker
    legend_label = if isnothing(color)
      L"L=%$L"
    elseif label_assigned
      L"L=%$L"
    else
      L"\chi=%$Dmax"
    end

    errorbars!(ax_beta, h, beta, err;
      color=local_color,
      whiskerwidth=10,
    )
    scatter!(ax_beta, h, beta;
      color=local_color,
      marker=local_marker,
      label=legend_label,
    )
    hgrid = range(minimum(h), maximum(h), length=300)
    lines!(ax_beta, hgrid, model(hgrid, p̂);
      color=local_color,
      linestyle=linestyle,
      linewidth=2,
    )
    vlines!(ax_beta, [hc]; color=local_color, linestyle=:dot)
    if !isnothing(color) && legend_label !== nothing
      label_assigned = true
    end
  end


  if !isempty(results)
    L_sorted = sort(collect(keys(results)))
    hc_vals = [results[L].hc for L in L_sorted]
    σhc_vals = [results[L].σ_hc for L in L_sorted]

    if !isnothing(offset)
      L_sorted = L_sorted .+ offset
    end
    hc_color = isnothing(color) ? :black : color
    hc_marker = isnothing(marker) ? :circle : marker
    errorbars!(ax_hc, L_sorted, hc_vals, σhc_vals;
      color=hc_color,
      whiskerwidth=10,
    )
    scatter!(ax_hc, L_sorted, hc_vals;
      color=hc_color,
      marker=hc_marker,
      label=L"\chi=%$Dmax"
    )
  end
end

function all_beta_plots(df_all, df_beta)
  # beta_combined_plot(df_all; dir="beta_comp", start=2, beta_crit=0.01, use_color=false, D_values=[128], tmin=50.0)
  beta_combined_plot(df_all; dir="beta_chi_new", start=2, beta_crit=0.005, use_color=true, D_values=[32, 64, 128], tmin=25.0)
  plot_beta_vs_disorder(df_all; dir="plots_new", tmin=25.0, tmax=100.0, xlim=(2.5, 50), yscale=log10)
  plot_beta_vs_disorder(df_beta; dir="beta_32_new", tmin=25.0, tmax=100.0, yscale=log10, L_values=4:2:12, h_values=5:5:50)
  beta_combined_plot(df_beta; dir="beta_32_new", start=1, beta_crit=0.005, D_values=[32], use_color=false, tmin=25.0)
end

function beta_combined_plot(df; dir="beta", start=1, beta_crit=0.002, use_color=true, D_values, tmin)
  beta_fig = Figure(size=(400, 250))
  ax_beta = Axis(beta_fig[1, 1], xlabel="h", ylabel="β")
  hlines!(ax_beta, [beta_crit]; color=:gray, linestyle=:dash)

  hc_fig = Figure(size=(400, 250), fontsize=12pt)
  # hc_fig = Figure(size=(500, 300))
  ax_hc = Axis(hc_fig[1, 1], xlabel=L"L", ylabel=L"h_c", xticks=[4, 6, 8, 10, 12])

  palette = default_colorscheme().colors
  markers = (:circle, :rect, :diamond, :utriangle, :dtriangle)
  # D_values = [32, 64, 128, 196]

  for (i, D) in enumerate(D_values)
    color = palette[2+(i-1)%length(palette)]
    marker = markers[1+(i-1)%length(markers)]
    if !use_color
      color = :black
      color = nothing
    end
    get_beta_values!(ax_beta, ax_hc, df;
      Dmax=D,
      beta_crit=beta_crit,
      color=color,
      L_vals=[4, 6, 8, 10, 12],
      marker=marker,
      offset=(i - 2) * 0.05,
      start,
      tmin
    )
  end

  axislegend(ax_beta)
  axislegend(ax_hc, position=:rb)

  # ll = LinRange(4, 10, 400)
  # norm = 50 / avalanche_critical_disorder(8)
  # lines!(ax_hc, ll, norm * avalanche_critical_disorder.(ll);
  #   color=:black,
  #   linestyle=:dash,
  #   label="Avalanche",
  # )

  # axislegend(ax_beta, position=:rb)
  # axislegend(ax_hc, position=:rb)

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)

  beta_fname = joinpath(plots_dir, "beta_vs_h_comparison.pdf")
  save(beta_fname, beta_fig)
  save(replace(beta_fname, ".pdf" => ".png"), beta_fig)
  println("Saved β(h) comparison to $(beta_fname)")
  display(beta_fig)

  hc_fname = joinpath(plots_dir, "hc_vs_L_comparison.pdf")
  save(hc_fname, hc_fig)
  save(replace(hc_fname, ".pdf" => ".png"), hc_fig)
  display(hc_fig)
  println("Saved h_c(L) comparison to $(hc_fname)")

  return (beta_fig=beta_fig, hc_fig=hc_fig)
end

function avalanche_critical_disorder(L, c1=1.57)
  return exp(c1 * log(L^2)^(1 / 3))
end

"""
  plot_fit_vs_mean(df; L_values, h_values, dir, tmin=5.0, tmax=50.0, xlog=true, ylog=false)

For each pair (L, h), plot the averaged imbalance (mean over gridnums at highest χ per
method) and overlay the best-fit power-law A * t^{−β} within the fit window [tmin, tmax].
Saves one figure per (L, h). X and Y log scaling are controlled independently via `xlog`/`ylog`.
"""
function plot_fit_vs_mean(
  df::DataFrame;
  L_values=[4, 6, 8, 10],
  h_values=[0.0, 5.0, 10.0, 20.0, 30.0, 50.0],
  dir::String,
  tmin::Real=50.0,
  tmax::Real=100.0,
  xlog::Bool=false,
  ylog::Bool=false,
  xlim::Tuple{<:Real,<:Real}=(1e-1, 100.0),
  ylim::Tuple{<:Real,<:Real}=(1e-4, 1.0),
)
  for L in sort(unique(L_values))
    for h in sort(h_values)
      df_lh = filter(row -> row.graph_L == L && row.model_h == h && !isempty(row.times), df)
      isempty(df_lh) && continue

      stats = compute_mean_imbalance_stats(df_lh)
      isempty(stats) && continue

      # Determine methods present and pick highest χ for each
      methods = sort(unique(first.(keys(stats))))
      isempty(methods) && continue

      # Colors per method using the default colormap mid-range for readability
      n_types = length(methods)
      low, high = 0.2, 0.8
      fracs = if n_types == 1
        [0.55]
      elseif n_types == 2
        [0.35, 0.65]
      else
        [low + (high - low) * (j - 1) / (n_types - 1) for j in 1:n_types]
      end

      fig = Figure(fontsize=11pt, size=multiplot_size())
      ax = Axis(fig[1, 1], xlabel=L"t", ylabel="Averaged imbalance")
      # Apply fixed limits and scales; ensure positive if a log scale is used
      xlo, xhi = xlim
      ylo, yhi = ylim
      if xlog
        xlo = max(eps(), xlo)
        xhi = max(xhi, xlo * 10)
        xlims!(ax, xlo, xhi)
        ax.xscale = log10
      end
      if ylog
        ylo = max(eps(), ylo)
        yhi = max(yhi, ylo * 10)
        ylims!(ax, ylo, yhi)
        ax.yscale = log10
      end
      xlims!(ax, xlo, xhi)
      ylims!(ax, ylo, yhi)

      # Mark fit window
      vlines!(ax, [tmin, tmax]; color=:gray, linestyle=:dot)

      plotted = false
      for (i, method) in enumerate(methods)
        Ds_type = sort([D for (t, D) in keys(stats) if t == method])
        isempty(Ds_type) && continue
        Dmax = Ds_type[end]
        st = stats[(method, Dmax)]

        # Fit β and A on the specified window
        fit = fit_power_law_beta(st.times, st.mean; tmin=tmin, tmax=tmax)
        fit === nothing && continue

        # Evaluate fitted curve within [tmin, tmax]
        idxf = findall(t -> t >= tmin && t <= tmax, st.times)
        isempty(idxf) && continue
        tf = st.times[idxf]
        yfit = fit.A .* (tf .^ (-fit.beta))

        color = scheme_color(fracs[i])
        label_mean = "$(labels[method]), D=$(Dmax) mean"
        label_fit = @sprintf("%s fit (β=%.3f)", labels[method], fit.beta)

        # Mean ± stderr with filtering/clamping only if the corresponding axis is log
        # Build mask for valid points
        mask = trues(length(st.times))
        if xlog
          mask .&= st.times .> 0
        end
        if ylog
          mask .&= st.mean .> 0
        end
        if any(mask)
          tp = st.times[mask]
          mp = st.mean[mask]
          sp = st.stderr[mask]
          lines!(ax, tp, mp; color=color, linewidth=2, label=label_mean)
          if ylog
            lower = max.(mp .- sp, 1e-12)
            upper = max.(mp .+ sp, 1e-12)
            band!(ax, tp, lower, upper; color=(color, 0.25))
          else
            band!(ax, tp, mp .- sp, mp .+ sp; color=(color, 0.25))
          end
        end
        # Fitted within the window
        lines!(ax, tf, yfit; color=color, linestyle=:dash, linewidth=2, label=label_fit)

        plotted = true
      end

      !plotted && continue
      try
        axislegend(ax, position=:rb)
      catch
      end
      plots_dir = joinpath("plots", dir)
      mkpath(plots_dir)
      fname = joinpath(plots_dir, @sprintf("fit_vs_mean_L%d_h%.1f_tmin%.1f_tmax%.1f.pdf", L, h, tmin, tmax))
      save(fname, fig)
      save(replace(fname, ".pdf" => ".png"), fig)
      println("Saved fit-vs-mean plot to $(fname)")
    end
  end
end

function multiplot_fit_vs_mean(
  df::DataFrame;
  L::Int=12,
  h_values=[5.0, 10.0, 20.0, 30.0, 50.0],
  dir::String,
  tmin::Real=50.0,
  tmax::Real=100.0,
  xlog::Bool=false,
  ylog::Bool=false,
  xlim::Tuple{<:Real,<:Real}=(1e-1, 100.0),
  ylim::Tuple{<:Real,<:Real}=(1e-4, 1.0),
  Dmax=128,
)
  fig = Figure(fontsize=11, size=(500, 150 * length(h_values)))
  axes = Axis[]  # collect axes so we can link them later

  for (i, h) in enumerate(h_values)
    df_lh = filter(row -> row.graph_L == L && row.model_h == h && !isempty(row.times), df)
    isempty(df_lh) && continue

    stats = compute_mean_imbalance_stats(df_lh)
    isempty(stats) && continue

    # keep only "FreeGraph"
    methods = filter(m -> m == "FreeGraph", unique(first.(keys(stats))))
    isempty(methods) && continue

    ax = Axis(fig[i, 1], xlabel=L"t", ylabel="⟨I⟩")
    push!(axes, ax)

    # label h in bottom-left corner
    text!(ax, 0.05, 0.95,
      text="h = $h",
      align=(:left, :top),
      space=:relative,
      fontsize=12,
      color=:black,
    )

    # Axis scaling
    xlo, xhi = xlim
    ylo, yhi = ylim
    if xlog
      # xlo = max(eps(), xlo)
      # xhi = max(xhi, xlo * 10)
      xlims!(ax, xlo, xhi)
      ax.xscale = log10
    end
    if ylog
      ylo = max(eps(), ylo)
      yhi = max(yhi, ylo * 10)
      ax.yscale = log10
    end
    xlims!(ax, xlo, xhi)
    # ylims!(ax, ylo, yhi)

    vlines!(ax, [tmin, tmax]; color=:gray, linestyle=:dot)

    for method in methods
      Ds_type = sort([D for (t, D) in keys(stats) if t == method])
      isempty(Ds_type) && continue
      st = stats[(method, Dmax)]

      fit = fit_power_law_beta(st.times, st.mean; tmin=tmin, tmax=tmax)
      fit === nothing && continue

      idxf = findall(t -> t >= tmin && t <= tmax, st.times)
      isempty(idxf) && continue
      tf = st.times[idxf]
      yfit = fit.A .* (tf .^ (-fit.beta))

      color_mean = :dodgerblue
      color_fit = :red   # different color for the fit

      label_mean = "TTN"

      mask = trues(length(st.times))
      if xlog
        mask .&= st.times .> 0
      end
      if ylog
        mask .&= st.mean .> 0
      end
      if any(mask)
        tp = st.times[mask]
        mp = st.mean[mask]
        sp = st.stderr[mask]
        lines!(ax, tp, mp; color=color_mean, linewidth=2, label=label_mean)
        if ylog
          lower = max.(mp .- sp, 1e-12)
          upper = max.(mp .+ sp, 1e-12)
          band!(ax, tp, lower, upper; color=(color_mean, 0.25))
        else
          band!(ax, tp, mp .- sp, mp .+ sp; color=(color_mean, 0.25))
        end
      end

      # fit in a different color and dashed style
      lines!(ax, tf, yfit; color=color_fit, linestyle=:dash, linewidth=2, label="Fit")

      # add β value in top-right corner of plot
      text!(ax, 0.95, 0.95,
        text=@sprintf("β = %.3f", fit.beta),
        align=(:right, :top),
        space=:relative,
        fontsize=12,
        color=:black,
      )
    end

    if i == 1
      axislegend(ax, position=:rb)
    end
  end

  # Link all x-axes
  if !isempty(axes)
    linkxaxes!(axes...)
  end

  plots_dir = joinpath("plots", dir)
  mkpath(plots_dir)
  fname = joinpath(plots_dir, @sprintf("multiplot_fit_vs_mean_L%d.pdf", L))
  save(fname, fig)
  save(replace(fname, ".pdf" => ".png"), fig)
  println("Saved multiplot fit-vs-mean to $(fname)")
end


"""
  compute_ed_error_and_rows(df_current, L, h)

Compute mean absolute error of `df_current` trajectories against ED.
Returns (error::Union{Nothing,Float64}, df_current_common::DataFrame).
"""
function compute_ed_error_and_rows(df_current::DataFrame, L::Int, h)
  if isempty(df_current)
    return nothing, nothing, df_current
  end
  common_gridnums = Set(df_current.graph_gridnum)
  if isempty(common_gridnums)
    return nothing, nothing, df_current
  end
  df_current_common = filter(r -> r.graph_gridnum in common_gridnums, df_current)
  sort!(df_current_common, :graph_gridnum)
  current_imbalances = [r.imbalance for r in eachrow(df_current_common)]
  if isempty(current_imbalances)
    return nothing, nothing, df_current_common
  end
  min_len = minimum(length, current_imbalances)
  if min_len < 2
    return nothing, nothing, df_current_common
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
    return nothing, nothing, df_current
  end
  benchmark_gridnums = Set(df_benchmark.graph_gridnum)
  current_gridnums = Set(df_current.graph_gridnum)
  common_gridnums = intersect(benchmark_gridnums, current_gridnums)
  if isempty(common_gridnums)
    return nothing, nothing, df_current
  end
  df_benchmark_common = filter(r -> r.graph_gridnum in common_gridnums, df_benchmark)
  df_current_common = filter(r -> r.graph_gridnum in common_gridnums, df_current)
  sort!(df_benchmark_common, :graph_gridnum)
  sort!(df_current_common, :graph_gridnum)
  benchmark_imbalances = [r.imbalance for r in eachrow(df_benchmark_common)]
  current_imbalances = [r.imbalance for r in eachrow(df_current_common)]
  if isempty(benchmark_imbalances) || isempty(current_imbalances)
    return nothing, nothing, df_current_common
  end
  min_len_benchmark = minimum(length, benchmark_imbalances)
  min_len_current = minimum(length, current_imbalances)
  min_len = min(min_len_benchmark, min_len_current)
  if min_len == 0
    return nothing, nothing, df_current_common
  end
  benchmark_matrix = hcat([imb[1:min_len] for imb in benchmark_imbalances]...)
  current_matrix = hcat([imb[1:min_len] for imb in current_imbalances]...)
  error = abs.(benchmark_matrix .- current_matrix)
  error = mean(error, dims=1)
  mean_error = exp.(mean(log.(error)))
  mean_error = mean(error)
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

  failed_files = Vector{Any}()

  @threads for i in 1:length(jld2_files)
    filepath = jld2_files[i]
    try
      results[i] = load(filepath)["results"]
    catch
      push!(failed_files, filepath)
      @warn "Could not load file '$filepath' on thread $(threadid())"
    end
  end
  loaded_data = filter(!isnothing, results)
  println("Finished loading. Loaded $(length(loaded_data)) files successfully.")
  open("failed_files.txt", "w") do io
    for f in failed_files
      println(io, f)
    end
  end
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

function load_new(L)

  tree = "data/L$L-column-tree-new-pre-det"
  if L == 8
    snake = "data/L$L-column-snake"
  else
    snake = "data/proc/L$L-column-snake"
  end
  snake_new = "data/L$L-column-snake-new"

  return load_general_dirs([tree, snake, snake_new])
end

function load_L_new(L; old_snake=false)
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
    snake = "data/L$L-column-snake-new"
    tree_pre = "data/L$L-column-tree-new-pre-det"
  end
  if L == 6
    snake = "data/L$L-column-snake-new"
  end

  return load_general_dirs([tree_pre, snake])
end

using JSON

"""
    choose_dirs(;
        pattern::Union{Regex,String,Nothing}=nothing,
        save_as::Union{String,Nothing}=nothing,
        load_from::Union{String,Nothing}=nothing,
        interactive::Bool=false,
        base::String="data"
    )

Helper to choose simulation directories inside `base` (default: "data").
Includes both the subdirs of `base` and any inside `base/proc`.

# Options
- `pattern`       : Regex or string to filter directories.
- `save_as`       : Save the chosen dirs under this name in `dir_selections.json`.
- `load_from`     : Load a previously saved selection by name.
- `interactive`   : If true, lets you pick dirs interactively.
- `base`          : Path to the folder containing simulation dirs.

# Returns
A vector of full paths (relative to project dir).
"""
function choose_dirs(;
  pattern::Union{Regex,String,Nothing}=nothing,
  save_as::Union{String,Nothing}=nothing,
  load_from::Union{String,Nothing}=nothing,
  interactive::Bool=false,
  base::String="data"
)
  # Path to config file (kept at project root)
  configfile = "dir_selections.json"

  # Load previous selections if file exists
  saved = isfile(configfile) ? JSON.parsefile(configfile) : Dict{String,Any}()

  dirs = String[]

  if load_from !== nothing
    # Load saved selection
    if haskey(saved, load_from)
      dirs = saved[load_from]
    else
      error("No saved selection named '$load_from' in $configfile")
    end
  else
    # Direct subdirs of base
    direct_subdirs = filter(isdir, joinpath.(base, readdir(base)))

    # Subdirs inside base/proc (if it exists)
    proc_path = joinpath(base, "proc")
    proc_subdirs = isdir(proc_path) ? filter(isdir, joinpath.(proc_path, readdir(proc_path))) : String[]

    # Combine
    all_dirs = vcat(direct_subdirs, proc_subdirs)

    if pattern !== nothing
      pat = pattern isa Regex ? pattern : Regex(pattern)
      dirs = filter(x -> occursin(pat, basename(x)), all_dirs)
    elseif interactive
      println("Available directories in $base (and $base/proc if present):")
      for (i, d) in enumerate(all_dirs)
        println("[$i] $(basename(dirname(d)))/$(basename(d))")
      end
      print("Select directories (comma-separated indices): ")
      choice = readline()
      indices = parse.(Int, split(choice, ","))
      dirs = all_dirs[indices]
    else
      dirs = all_dirs
    end

    # Save selection if requested
    if save_as !== nothing
      saved[save_as] = dirs
      open(configfile, "w") do io
        JSON.print(io, saved)
      end
      println("Saved selection '$save_as' to $configfile")
    end
  end

  return dirs
end

function load_choose_dirs(; remove_small_time=false, kwargs...)
  return load_general_dirs(choose_dirs(; kwargs...); remove_small_time)
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
    df = remove_small_time_simulations(df; t_end=100.0)
  end
  if remove_dup
    df = remove_duplicates(df)
  end
  df.imbalance = TTNEvo.columnar_imbalance_total.(df.sz)

  df_selection = select!(df, Not([:maxdim, :sz]))

  return df_selection
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
        scatter!(ax, dims, errors, label=labels[type], color=line_color)
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
        maxdims = [32, 64, 128, 196]
      end

      if length(maxdims) < 2 && !use_ed_benchmark
        continue
      end

      benchmark_maxdim = maximum(maxdims)
      df_benchmark = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim, df_type)
      if isempty(df_benchmark)
        continue
      end

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
        scatter!(ax, params, errors; label=labels[type],
          color=line_color, marker=marker, markersize=6)
        errorbars!(ax, params, errors, stds; direction=:y, color=line_color)
      end
    end

    # put legend only once
    if h_idx == 1
      try
        axislegend(ax; position=:rt, orientation=:vertical, framevisible=false)
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

  palette = default_colorscheme().colors
  n_graph_types = max(length(graph_types), 1)
  if n_graph_types == 1
    colors = [palette[round(Int, length(palette) / 2)]]
  else
    palette_indices = range(10, length(palette) - 10; length=n_graph_types)
    colors = [palette[clamp(round(Int, idx), 1, length(palette))] for idx in palette_indices]
  end
  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]
  axes = Axis[]

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(grid[1, h_idx]; yscale=log10)
    push!(axes, ax)

    ax.xticks = WilkinsonTicks(7)
    ax.yticks = LogTicks(WilkinsonTicks(5))
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
        colormap=default_colorscheme(),
        colorscale=log10,
        colorrange=(global_rt_min + 1e-16, global_rt_max + 1e-16),
        marker=marker,
        markersize=4.5,
        label=labels[s.type],
      )
    end

    if h_idx == 1
      axislegend(ax; position=:rt, orientation=:vertical, framevisible=false)
    end
  end

  # Set decade ticks to avoid fractional exponents on the colorbar
  cb_exp_min = floor(Int, log10(global_rt_min + 1e-16))
  cb_exp_max = ceil(Int, log10(global_rt_max + 1e-16))
  cb_positions = 10.0 .^ collect(cb_exp_min:cb_exp_max)
  @show "testtest"
  @show cb_exp_min:cb_exp_max
  cb_labels = [L"10^{ %$e }" for e in cb_exp_min:cb_exp_max]

  Colorbar(fig[1, 3],
    colormap=default_colorscheme(),
    limits=(global_rt_min + 1e-16, global_rt_max + 1e-16),
    label="Execution Time (s)",
    scale=log10,
    ticks=(cb_positions, cb_labels),
  )

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
  @show h_values
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
      stds = Float64[]

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
        push!(stds, serr)

        if rt < global_rt_min
          global_rt_min = rt
        end
        if rt > global_rt_max
          global_rt_max = rt
        end
      end

      if !isempty(params)
        push!(series_list, (type=type, type_idx=type_idx, params=params, runtimes=runtimes, errors=errors, stds=stds))
      end
    end
    data_by_h[h] = series_list
  end

  if !isfinite(global_rt_min) || !isfinite(global_rt_max)
    @warn "No valid runtime range computed for L=$L"
    return
  end

  # Plot with standard size to align with other figures/LaTeX
  fig = Figure(size=multiplot_size(), fontsize=11, figure_padding=6)
  grid = fig[1, 2] = GridLayout()
  if L == 4
    fig[1, 1] = Label(fig, L"\langle |I_{\mathrm{ED}} - I_{\mathrm{TN}}| \rangle"; rotation=π / 2, tellheight=false)
  else
    fig[1, 1] = Label(fig, L"\langle |I_{\chi_{\mathrm{max}}} - I_{\chi}| \rangle"; rotation=π / 2, tellheight=false)
  end
  fig[2, 2] = Label(fig, L"\text{Number of Parameters} / 10^5"; tellwidth=false)

  palette = default_colorscheme().colors
  n_graph_types = max(length(graph_types), 1)
  if n_graph_types == 1
    colors = [palette[round(Int, length(palette) / 2)]]
  else
    palette_indices = range(10, length(palette) - 10; length=n_graph_types)
    colors = [palette[clamp(round(Int, idx), 1, length(palette))] for idx in palette_indices]
  end
  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]
  axes = Axis[]

  alphabet = 'a':'z'  # panel labels

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(
      grid[1, h_idx];
      yscale=log10,
      title=L"(%$(alphabet[h_idx])) $h=%$h$",
      titlealign=:left,
      titlesize=11,
    )
    push!(axes, ax)

    if L == 4
      ax.xticks = ([0, 0.5, 1])
    end
    if L == 12
      ax.yticks = ([10.0^(-x) for x in 1:1:12], [L"10^{-%$x}" for x in 1:1:12])
      ylims!(ax, (1e-3, 1e-2))
    else
      ax.yticks = ([10.0^(-x) for x in 2:2:12], [L"10^{-%$x}" for x in 2:2:12])
    end

    if h_idx != 1
      hideydecorations!(ax, grid=false)
    end

    for s in get(data_by_h, h, NamedTuple[])
      marker = marker_shapes[(s.type_idx-1)%length(marker_shapes)+1]
      if !isempty(s.params)
        order = sortperm(s.params)
        lines!(ax, s.params[order], s.errors[order]; color=:black, linewidth=1.0, transparency=true, alpha=0.5)
      end
      errorbars!(ax, s.params, s.errors, s.stds;
        color=:black,
        whiskerwidth=4
      )
      scatter!(ax, s.params, s.errors;
        color=s.runtimes,
        colormap=default_colorscheme(),
        colorrange=(global_rt_min + 1e-16, global_rt_max + 1e-16),
        colorscale=log10,
        marker=marker,
        markersize=8,
        strokecolor=:black,
        strokewidth=0.8,
        label=labels[s.type],
      )
    end
    if h_idx == 1
      if L == 12
        axislegend(ax; position=(0.05, 0.02), framevisible=false, patchlabelgap=-4)
      else
        axislegend(ax; position=(0.6, 0.02), framevisible=false, patchlabelgap=-4)
      end
    end
  end

  cb2_exp_min = floor(Int, log10(global_rt_min + 1e-16))
  cb2_exp_max = ceil(Int, log10(global_rt_max + 1e-16))
  cb2_positions = 10.0 .^ collect(cb2_exp_min:cb2_exp_max)
  cb2_labels = [L"10^{%$(e)}" for e in cb2_exp_min:cb2_exp_max]

  Colorbar(fig[1, 3],
    colormap=default_colorscheme(),
    scale=log10,
    limits=(10.0^cb2_exp_min, 10.0^cb2_exp_max),
    label="Execution Time (s)",
    ticks=(cb2_positions, cb2_labels),
  )

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
  return (600, 200)
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

  n_graph_types = max(length(graph_types), 1)
  colors = scheme_colors(n_graph_types)
  marker_shapes = [:circle, :rect, :utriangle, :dtriangle, :cross]
  axes = Axis[]

  for (h_idx, h) in enumerate(h_values)
    ax = Axis(
      grid[1, h_idx];
      # xscale=log10,
      # yscale=log10,
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

        @show labels[type]
        lines!(ax, params, runtimes; color=line_color, linewidth=1.5)
        scatter!(ax, params, runtimes; label=labels[type],
          color=line_color, marker=marker, markersize=6)
      end
    end

    if h_idx == 1
      Legend(fig[1, 3], ax)
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

function plot_individual_imbalance_single(df, L_values=[4, 6, 8]; dir)
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

        # single-column figure (imbalance only), linear x-axis
        local fig
        if L == 4
          fig = Figure(fontsize=8pt, size=(450, 450), figure_padding=5)
        else
          fig = Figure(fontsize=8pt, size=(450, 300), figure_padding=5)
        end

        axes_imbalance = Axis[]
        axis_labels = Any[]

        # ED benchmark (only for L==4); used just to overlay on imbalance panel
        ed_imbalance = nothing
        if L == 4
          ed_data, _ = get_ed_benchmark(L, h, [grid])
          if !isempty(ed_data)
            ed_imbalance = ed_data
          end
        end

        for (i, gtype) in enumerate(graph_types)
          ax_imb = Axis(fig[i, 1];
            xlabel=L"t",
            ylabel=L"I",
            # linear x-axis (removed xscale=log10)
            # no custom log ticks; let Makie choose
          )
          push!(axes_imbalance, ax_imb)

          label_txt = get(labels, gtype, string(gtype))
          push!(axis_labels, L"%$(label_txt) $h=%$h$, $L=%$L$")
        end

        if length(axes_imbalance) > 1
          linkxaxes!(axes_imbalance...)
          linkyaxes!(axes_imbalance...)
        end

        # color mapping across χ values
        Ds = unique(dfi.initial_state_initial_maxdim)
        Ds_sorted = sort(Ds, rev=true)
        D_to_val = if length(Ds_sorted) > 1
          Dict(D => (i - 1) / (length(Ds_sorted) - 1) for (i, D) in enumerate(Ds_sorted))
        else
          Dict(D => 0.5 for (i, D) in enumerate(Ds_sorted))
        end

        for (idx, gtype) in enumerate(graph_types)
          ax_imb = axes_imbalance[idx]
          df_graph = filter(row -> row.graph_type == gtype, dfi)

          for d in eachrow(df_graph)
            Dval = D_to_val[d.initial_state_initial_maxdim]
            color = scheme_color(Dval)
            lines!(ax_imb, d.times, d.imbalance;
              label=L"\chi=%$(d.initial_state_initial_maxdim)", color=color)
          end

          # Optional: overlay ED curve on imbalance (only for L==4)
          if ed_imbalance !== nothing
            max_T = maximum(maximum.(df_graph.times))
            max_T = min(max_T, 0.1 * length(ed_imbalance))
            if max_T >= 0.1
              time = 0.1:0.1:max_T
              lines!(ax_imb, time, ed_imbalance[1:length(time)]; label="ED", color=:black)
            end
          end
        end

        for (idx, ax) in enumerate(axes_imbalance)
          text!(ax, 0.98, 0.98;
            text=axis_labels[idx],
            align=(:right, :top),
            space=:relative)
          if idx != length(axes_imbalance)
            hidexdecorations!(ax, grid=false, label=true)
            ax.xlabelvisible = false
          end
        end

        # legend on the right
        try
          Legend(fig[1:length(graph_types), 2], axes_imbalance[1]; merge=true)
        catch
        end

        plot_dir = joinpath("plots", dir, "individual")
        mkpath(plot_dir)
        pdf_path = joinpath(plot_dir, "imbalance_L=$(L)_h=$(h)_grid=$(grid).pdf")
        png_path = joinpath(plot_dir, "imbalance_L=$(L)_h=$(h)_grid=$(grid).png")
        save(pdf_path, fig)
        save(png_path, fig)
        println("Saved imbalance-only plot to $(pdf_path) and $(png_path)")
      end
    end
  end
end

function plot_accuracy_vs_runtime(df::DataFrame; L::Int, dir)
  h_values = sort(unique(filter(row -> row.graph_L == L, df).model_h))
  graph_types = sort(unique(filter(row -> row.graph_L == L, df).graph_type))

  if isempty(h_values)
    @warn "No data for L=$L"
    return
  end

  use_ed_benchmark = L == 4

  fig = Figure(size=multiplot_size(), fontsize=8)

  # Outer layout: col 1 = shared ylabel, col 2 = plots grid
  grid = fig[1, 2] = GridLayout()
  fig[1, 1] = Label(fig, L"|I_{\mathrm{ED}} - I_{\mathrm{TN}}|"; rotation=π / 2, tellheight=false, padding=(0, 0, 0, 0))

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
        scatter!(ax, runtimes, errors; label=labels[type],
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

  colgap!(fig.layout, 4.0)
  rowgap!(fig.layout, 4.0)
  colgap!(grid, 2.0)
  rowgap!(grid, 0.0)
  linkyaxes!(axes...)   # shared y-axis

  fig[2, 2] = Label(fig, "Execution Time (s)"; tellwidth=false)

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

        local fig
        if L == 4
          fig = Figure(fontsize=8pt, size=(600, 450), figure_padding=5)
        else
          fig = Figure(fontsize=8pt, size=(600, 225), figure_padding=5)
        end
        axes_imbalance = Axis[]
        axes_error = Axis[]
        axis_labels = Any[]
        error_mins = Float64[]
        error_maxs = Float64[]

        ed_imbalance = nothing
        if L == 4
          ed_data, _ = get_ed_benchmark(L, h, [grid])
          if !isempty(ed_data)
            ed_imbalance = ed_data
          end
        end

        # Pre-compute benchmark (highest χ) per graph type in case ED is unavailable
        benchmark_series = Dict{String,NamedTuple}()

        for (i, gtype) in enumerate(graph_types)
          ax_imb = Axis(fig[i, 1],
            xlabel=L"t",
            ylabel=L"I",
            xscale=log10,
            # xticks=[0.1, 1, 10, 100],
            xticks=logticks1_with_minors(-1, 2),
            limits=(0.1, 100, nothing, nothing),
          )
          yl = L == 4 ? L"|I_{\mathrm{ED}} - I_{\chi}|" : L"|I_{\chi_{\mathrm{max}}} - I_{\chi}|"
          ax_err = Axis(fig[i, 2],
            xlabel=L"t",
            ylabel=yl,
            xscale=log10,
            yscale=log10,
            xticks=logticks1_with_minors(-1, 2),
          )

          xlims!(ax_err, (0.1, 100.0))
          if L == 4
            ylims!(ax_err, (1e-14, 1))
          else
            ylims!(ax_err, (1e-8, 1))
          end

          push!(axes_imbalance, ax_imb)
          push!(axes_error, ax_err)
          label_txt = get(labels, gtype, string(gtype))
          push!(axis_labels, L"%$(label_txt) $h=%$h$, $L=%$L$")
          push!(error_mins, Inf)
          push!(error_maxs, 0.0)

          if ed_imbalance === nothing
            df_graph = filter(row -> row.graph_type == gtype, dfi)
            if !isempty(df_graph)
              maxdims = sort(unique(df_graph.initial_state_initial_maxdim))
              if !isempty(maxdims)
                benchmark_maxdim = maximum(maxdims)
                benchmark_row_list = filter(row -> row.initial_state_initial_maxdim == benchmark_maxdim, df_graph)
                if !isempty(benchmark_row_list)
                  benchmark_row = first(benchmark_row_list)
                  benchmark_series[gtype] = (times=benchmark_row.times, values=benchmark_row.imbalance, maxdim=benchmark_maxdim)
                end
              end
            end
          end
        end

        if length(axes_error) > 1
          linkxaxes!(axes_imbalance...)
          linkyaxes!(axes_imbalance...)
          linkxaxes!(axes_error...)
          linkyaxes!(axes_error...)
        end

        Ds = unique(dfi.initial_state_initial_maxdim)
        Ds_sorted = sort(Ds, rev=true)
        D_to_val = if length(Ds_sorted) > 1
          Dict(D => (i - 1) / (length(Ds_sorted) - 1) for (i, D) in enumerate(Ds_sorted))
        else
          Dict(D => 0.5 for (i, D) in enumerate(Ds_sorted))
        end

        for (idx, gtype) in enumerate(graph_types)
          ax_imb = axes_imbalance[idx]
          ax_err = axes_error[idx]
          df_graph = filter(row -> row.graph_type == gtype, dfi)

          for d in eachrow(df_graph)
            Dval = D_to_val[d.initial_state_initial_maxdim]
            color = scheme_color(Dval)
            lines!(ax_imb, d.times, d.imbalance;
              label=L"\chi=%$(d.initial_state_initial_maxdim)",
              color=color)

            if ed_imbalance !== nothing
              series = compute_ed_error_series(d.times, d.imbalance, ed_imbalance)
              series === nothing && continue
              times_err, error = series
              err_vals = error .+ 1e-16
              lines!(ax_err, times_err, err_vals; color=color)
              error_mins[idx] = min(error_mins[idx], minimum(err_vals))
              error_maxs[idx] = max(error_maxs[idx], maximum(err_vals))
            else
              bench = get(benchmark_series, gtype, nothing)
              bench === nothing && continue
              if d.initial_state_initial_maxdim == bench.maxdim
                continue
              end
              series = compute_benchmark_error_series(d.times, d.imbalance, bench.times, bench.values)
              series === nothing && continue
              times_err, error = series
              err_vals = error .+ 1e-16
              lines!(ax_err, times_err, err_vals; color=color)
              error_mins[idx] = min(error_mins[idx], minimum(err_vals))
              error_maxs[idx] = max(error_maxs[idx], maximum(err_vals))
            end
          end

          if ed_imbalance !== nothing
            max_T = maximum(maximum.(df_graph.times))
            max_T = min(max_T, 0.1 * length(ed_imbalance))
            if max_T >= 0.1
              time = 0.1:0.1:max_T
              lines!(ax_imb, time, ed_imbalance[1:length(time)]; label="ED", color=:black)
            end
          end
        end

        for (idx, ax) in enumerate(axes_imbalance)
          text!(ax, 0.98, 0.98;
            text=axis_labels[idx],
            align=(:right, :top),
            space=:relative)
          if idx != length(axes_imbalance)
            hidexdecorations!(ax, grid=false, label=true)
            ax.xlabelvisible = false
          end
        end

        for (idx, ax) in enumerate(axes_error)
          # text!(ax, 0.02, 0.98;
          #   text=L"\text{Error}",
          #   align=(:left, :top),
          #   space=:relative)
          if isfinite(error_mins[idx]) && error_maxs[idx] > 0
            kmin_est = floor(Int, log10(error_mins[idx]))
            kmax_est = ceil(Int, log10(error_maxs[idx]))
            even_floor_exp = iseven(kmin_est) ? kmin_est : kmin_est - 1
            even_ceil_exp = iseven(kmax_est) ? kmax_est : kmax_est + 1
            start_exp = min(even_floor_exp, -2)
            end_exp = max(even_ceil_exp, start_exp)
            exps = collect(start_exp:2:end_exp)
            positions = Float64[10.0^k for k in exps]
            labels = [L"10^{%$k}" for k in exps]
            ax.yticks = (positions, labels)
          end
          if idx != length(axes_error)
            hidexdecorations!(ax, grid=false, label=true)
            ax.xlabelvisible = false
          end
        end

        plot_dir = joinpath("plots", dir, "individual")
        mkpath(plot_dir)

        try
          Legend(fig[1:length(graph_types), 3], axes_imbalance[1]; merge=true)
        catch
        end

        pdf_path = joinpath(plot_dir, "imbalance_L=$(L)_h=$(h)_grid=$(grid).pdf")
        png_path = joinpath(plot_dir, "imbalance_L=$(L)_h=$(h)_grid=$(grid).png")
        save(pdf_path, fig)
        save(png_path, fig)
        println("Saved combined imbalance plot to $(pdf_path) and $(png_path)")
      end
    end
  end
end

function plot_individual_imbalance_error(df, L_values=[4, 6, 8]; dir)
  @info "plot_individual_imbalance_error merged into plot_individual_imbalance; skipping separate output."
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
    linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond, :star5]

    for (type_idx, type) in enumerate(graph_types)
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
          type_fraction = length(graph_types) == 1 ? 0.5 : (type_idx - 1) / (length(graph_types) - 1)
          fraction = clamp(0.1 + 0.8 * (0.6 * D_to_val[maxdim] + 0.4 * type_fraction), 0.0, 1.0)
          line_color = scheme_color(fraction)
          linestyle = linestyles[mod1(type_idx, length(linestyles))]
          marker = markers[mod1(type_idx, length(markers))]
          label = "$type, D=$maxdim"
          lines!(ax, hs_for_plot, errors, label=label, color=line_color, linestyle=linestyle)
          scatter!(ax, hs_for_plot, errors, color=line_color, label=label, marker=marker, markersize=15)
        end
      end
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
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond, :star5]

    for (type_idx, type) in enumerate(graph_types)
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
          type_fraction = length(graph_types) == 1 ? 0.5 : (type_idx - 1) / (length(graph_types) - 1)
          fraction = clamp(0.1 + 0.8 * (0.6 * D_to_val[maxdim] + 0.4 * type_fraction), 0.0, 1.0)
          line_color = scheme_color(fraction)
          marker = markers[mod1(type_idx, length(markers))]
          label = "$(labels[type]), D=$maxdim"
          lines!(ax, Ls_for_plot, errors, label=label, color=line_color)
          scatter!(ax, Ls_for_plot, errors, color=line_color, label=label, marker=marker, markersize=15)
        end
      end
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
  df_filtered = filter(row -> row.initial_state_initial_maxdim >= 32 && !isempty(row.times), df_filtered)

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
  # Color by graph type using the default colormap
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

    # if type == "HilbertCurve"
    #   mean_vals = 1 .- 2 .* (1 .- mean_vals)
    # end

    color = scheme_color(fracs[i])
    style = linestyles[(i-1)%length(linestyles)+1]
    label = "$(labels[type]), D=$Dmax"
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

function plot_all_imbalance(f; dir)

end

function main(df; dir::Union{Nothing,String}=nothing, L_values=nothing, individual=false)
  if :graph_L ∉ propertynames(df)
    error("DataFrame must contain a :graph_L column to infer system sizes")
  end

  inferred_Ls = sort!(collect(Set(skipmissing(df.graph_L))))
  isempty(inferred_Ls) && error("No L values found in DataFrame; provide L_values explicitly")

  if isnothing(L_values)
    L_values = inferred_Ls
  else
    L_values = collect(L_values)
  end
  isempty(L_values) && error("L_values cannot be empty")

  auto_dir = isnothing(dir)
  base_dir = auto_dir ? (length(L_values) == 1 ? "L$(first(L_values))" : "combined") : dir

  if individual
    plot_individual_imbalance(df, L_values; dir=base_dir)
    plot_individual_imbalance_error(df, L_values; dir=base_dir)
  end

  for L in L_values
    current_dir = auto_dir && length(L_values) > 1 ? joinpath(base_dir, "L$(L)") : base_dir
    println("Generating plots for L=$L")
    for h in [0.0, 5.0, 10.0, 20.0, 30.0, 50.0]
      plot_mean_imbalance(df; L=L, h, dir=current_dir)
    end
    plot_accuracy_convergence(df; L=L, dir=current_dir)
    plot_accuracy_vs_parameters(df; L=L, dir=current_dir)
    plot_runtime_vs_parameters(df; L=L, dir=current_dir)
    plot_accuracy_vs_runtime(df; L=L, dir=current_dir)
    plot_params_runtime_colored_by_runtime(df; L=L, dir=current_dir)
  end

  plot_error_vs_disorder(df; dir=base_dir)
  plot_error_vs_system_size(df; dir=base_dir)
  # plot_beta_vs_disorder(df;
  #   L_values=L_values,
  #   h_values=[0.0, 2.5, 5.0, 7.5, 10.0, 20.0, 30.0, 50.0],
  #   dir=base_dir,
  #   tmin=50.0,
  #   tmax=100.0,
  #   xlim=(0.0, 50.0),
  #   ylim=(0.0, 1.0),
  # )
  # plot_fit_vs_mean(df;
  #   L_values=L_values,
  #   h_values=[0.0, 2.5, 5.0, 7.5, 10.0, 20.0, 30.0, 50.0],
  #   dir=base_dir,
  # )
  multiplot_fit_vs_mean(df; dir=base_dir)
end

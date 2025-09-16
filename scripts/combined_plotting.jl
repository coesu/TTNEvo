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

global labels = Dict("FreeGraph" => "TTN", "SnakeGraph" => "MPS", "HierarchicalTree" => "Hierch", "HilbertCurve" => "Hilbert")

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
        push!(T, tt)
        push!(D, abs(ed_imb[j] - sim_imb[i]))
      end
    end
  end

  # Plot
  fig = Figure(size=(10cm, 6cm))
  ax = Axis(fig[1, 1], xlabel="t", ylabel="|ED − TN| imbalance", title="L=$(L), h=$(h), grid=$(grid)")
  if !isempty(T)
    lines!(ax, T, D; color=:steelblue, label="ED − TN")
    hlines!(ax, [0.0]; color=:gray, linestyle=:dot)
  else
    @warn "No common time grid with ED found; cannot plot ED − TN differences."
  end
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
  fig = Figure(size=(12cm, 13cm))
  ax = Axis(fig[1, 1], xlabel="t", ylabel="|ED − TN| imbalance", title="L=$(L), h=$(h), grid=$(grid)", yscale=log10)
  ax_md = Axis(fig[2, 1], xlabel="t", ylabel="maxlinkdim")
  ax_rt = Axis(fig[3, 1], xlabel="t", ylabel="step time [s]")

  # Colors/labels
  default_labels = [basename(dir) for dir in sim_dirs]
  labels = isnothing(labels) ? default_labels : labels

  # Use magma colormap for consistency with other plots
  ncurves = length(sim_dirs)
  colors = [get(ColorSchemes.magma, (i - 0.5) / max(ncurves, 1)) for i in 1:ncurves]

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
      key = round(tt, digits=8)
      if haskey(ed_map, key)
        j = ed_map[key]
        push!(T, tt)
        push!(D, abs.(ed_imb[j] - sim_imb[k]))
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
        push!(t_md, first(pair))
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
        push!(d_md, maxd)
      end
      p = sortperm(t_md)
      lines!(ax_md, t_md[p], d_md[p]; color=colors[(i-1)%length(colors)+1])
    end

    # Plot execution time per step
    if !isempty(cfg.observer.ex_times) && !isempty(times)
      n = min(length(times), length(cfg.observer.ex_times))
      lines!(ax_rt, times[1:n], cfg.observer.ex_times[1:n]; color=colors[(i-1)%length(colors)+1])
    end
  end
  # Reference line at zero
  hlines!(ax, [0.0]; color=:gray, linestyle=:dot)

  try
    axislegend(ax; position=:rb)
  catch
  end

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
function fit_power_law_beta(times::AbstractVector, values::AbstractVector; tmin::Real=5.0, tmax::Real=50.0)
  isempty(times) && return nothing
  isempty(values) && return nothing
  # Select window and ensure positivity
  idx = findall(i -> times[i] >= tmin && times[i] <= tmax && isfinite(values[i]) && values[i] > 0 && isfinite(times[i]), eachindex(times))
  length(idx) < 3 && return nothing
  t = Float64[times[i] for i in idx]
  y = Float64[values[i] for i in idx]
  # Log–log linear regression: log y = a + b * log t, with β = -b and A = exp(a)
  X = log.(t)
  Y = log.(y)
  n = length(X)
  Sx = sum(X)
  Sy = sum(Y)
  Sxx = sum(abs2, X)
  Sxy = sum(X .* Y)
  den = n * Sxx - Sx^2
  den == 0 && return nothing
  b = (n * Sxy - Sx * Sy) / den
  a = (Sy - b * Sx) / n
  # Residuals and errors
  Ŷ = a .+ b .* X
  resid = Y .- Ŷ
  s2 = sum(abs2, resid) / max(n - 2, 1)
  stderr_b = sqrt(s2 * n / den)
  # R^2
  SS_tot = sum(abs2, Y .- mean(Y))
  SS_res = sum(abs2, resid)
  r2 = SS_tot ≈ 0 ? 1.0 : 1.0 - SS_res / SS_tot
  return (beta=-b, stderr=stderr_b, npts=n, r2=r2, A=exp(a))
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

For fixed (L, h), take the highest available χ per `graph_type`, compute the
mean imbalance across gridnums, and fit β from t^{−β} on [tmin, tmax].
Returns a Dict keyed by `graph_type::String` with values `(beta, stderr, npts, r2, Dmax)`.
"""
function compute_beta_over_gridmean(df::DataFrame; L::Int, h, tmin::Real=5.0, tmax::Real=50.0)
  df_lh = filter(row -> row.graph_L == L && row.model_h == h, df)
  isempty(df_lh) && return Dict{String,NamedTuple}()
  # Only keep non-empty times
  df_lh = filter(row -> !isempty(row.times), df_lh)
  stats = compute_mean_imbalance_stats(df_lh)
  out = Dict{String,NamedTuple}()
  for type in sort(unique(first.(keys(stats))))
    Ds_type = sort([D for (t, D) in keys(stats) if t == type])
    isempty(Ds_type) && continue
    Dmax = Ds_type[end]
    st = stats[(type, Dmax)]
    fit = fit_power_law_beta(st.times, st.mean; tmin=tmin, tmax=tmax)
    fit === nothing && continue
    out[type] = merge(fit, (Dmax=Dmax,))
  end
  return out
end

"""
  plot_beta_vs_disorder(df; L_values, h_values, dir, tmin=5.0, tmax=50.0)

For each method (`graph_type`), plot β versus disorder strength for multiple system sizes.
Uses the highest available χ per method and the grid-averaged imbalance.
"""
function plot_beta_vs_disorder(
  df::DataFrame;
  L_values=[4, 6, 8, 10],
  h_values=[0.0, 5.0, 10.0, 20.0, 30.0, 50.0],
  dir::String,
  tmin::Real=5.0,
  tmax::Real=50.0,
  xlim::Tuple{<:Real,<:Real}=(NaN, NaN),
  ylim::Tuple{<:Real,<:Real}=(0.0, 1.0),
)
  # Collect all methods from the filtered data
  df_sub = filter(row -> row.graph_L in L_values && row.model_h in h_values, df)
  methods = sort(unique(df_sub.graph_type))
  isempty(methods) && return

  # Color map across L values for visual consistency
  Ls_sorted = sort(unique(L_values))
  colors = wong_colors()
  color_map = Dict{Int,Any}()
  for (i, L) in enumerate(Ls_sorted)
    color_map[L] = colors[(i-1)%length(colors)+1]
  end

  for method in methods
    fig = Figure(fontsize=11pt, size=(10cm, 6cm))
    ax = Axis(fig[1, 1], xlabel=L"h", ylabel=L"\beta", title="$(labels[method])")
    # Apply fixed limits if provided
    xlo, xhi = xlim
    if isnan(xlo) || isnan(xhi)
      xlo, xhi = minimum(h_values), maximum(h_values)
    end
    xlims!(ax, xlo, xhi)
    ylims!(ax, first(ylim), last(ylim))

    for L in Ls_sorted
      betas = Float64[]
      errs = Float64[]
      hs = Float64[]
      for h in sort(h_values)
        vals = compute_beta_over_gridmean(df; L=L, h=h, tmin=tmin, tmax=tmax)
        haskey(vals, method) || continue
        push!(hs, float(h))
        push!(betas, vals[method].beta)
        push!(errs, vals[method].stderr)
      end
      isempty(hs) && continue
      c = color_map[L]
      label = "L=$L"
      lines!(ax, hs, betas; color=c, label=label)
      scatter!(ax, hs, betas; color=c, label=label)
      # Optional uncertainty in β as vertical range bars (robust Makie recipe)
      if !isempty(errs)
        lows = betas .- errs
        highs = betas .+ errs
        rangebars!(ax, hs, lows, highs; color=c)
      end
    end

    try
      axislegend(ax, position=:rt)
    catch
    end
    plots_dir = joinpath("plots", dir)
    mkpath(plots_dir)
    fname = joinpath(plots_dir, "beta_vs_h_$(method)_tmin$(tmin)_tmax$(tmax).pdf")
    save(fname, fig)
    # also PNG
    save(replace(fname, ".pdf" => ".png"), fig)
    println("Saved β vs h plot for method=$(method) to $(fname)")
  end
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

      # Colors per method using magma mid-range for readability
      n_types = length(methods)
      low, high = 0.2, 0.8
      fracs = if n_types == 1
        [0.55]
      elseif n_types == 2
        [0.35, 0.65]
      else
        [low + (high - low) * (j - 1) / (n_types - 1) for j in 1:n_types]
      end

      fig = Figure(fontsize=11pt, size=(12cm, 7cm))
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

        color = get(ColorSchemes.magma, fracs[i])
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
  @show min_len
  if min_len == 0
    return nothing, nothing, df_current_common
  end
  benchmark_matrix = hcat([imb[1:min_len] for imb in benchmark_imbalances]...)
  current_matrix = hcat([imb[1:min_len] for imb in current_imbalances]...)
  error = abs.(benchmark_matrix .- current_matrix)
  error = mean(error, dims=1)
  mean_error = exp.(mean(log.(error)))
  @show mean_error
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
        colorscale=log10,
        colorrange=(global_rt_min + 1e-16, global_rt_max + 1e-16),
        marker=marker,
        markersize=4.5,
        label=labels[s.type],
      )
    end

    if h_idx == 1
      axislegend(ax; position=:rt, orientation=:vertical, nbanks=1, framevisible=false)
    end
  end

  # Set decade ticks to avoid fractional exponents on the colorbar
  cb_exp_min = floor(Int, log10(global_rt_min + 1e-16))
  cb_exp_max = ceil(Int, log10(global_rt_max + 1e-16))
  cb_positions = 10.0 .^ collect(cb_exp_min:cb_exp_max)
  cb_labels = ["10^$(e)" for e in cb_exp_min:cb_exp_max]

  Colorbar(fig[1, 3],
    colormap=:viridis,
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
        colorscale=log10,
        marker=marker,
        markersize=8,
        strokecolor=:black,
        strokewidth=0.8,
        label=labels[s.type],
      )
    end
    if h_idx == 1
      axislegend(ax; position=(0.6, 0.12), framevisible=false, patchlabelgap=-4)
    end
  end

  # Set decade ticks to avoid fractional exponents on the colorbar
  cb2_exp_min = floor(Int, log10(global_rt_min + 1e-16))
  cb2_exp_max = ceil(Int, log10(global_rt_max + 1e-16))
  cb2_positions = 10.0 .^ collect(cb2_exp_min:cb2_exp_max)
  cb2_labels = [L"10^%$(e)" for e in cb2_exp_min:cb2_exp_max]

  Colorbar(fig[1, 3],
    colormap=:magma,
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
        scatter!(ax, params, runtimes; label=labels[type],
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
          # begin
          #   threshold = 0.01 / 2.0
          #   divpt = earliest_divergence(df_graph; threshold)
          #   if divpt !== nothing
          #     tdiv, ydiv = divpt
          #     # draw vertical marker and a star at the max-D curve
          #     vlines!(ax, [tdiv]; color=:black, linestyle=:dash, linewidth=1.5)
          #     scatter!(ax, [tdiv], [ydiv]; color=:black, marker=:star5, markersize=9)
          #     # Optional small annotation
          #     text!(ax, tdiv, ydiv; text=L"\Delta>%$(threshold)", align=(:left, :top), color=:black, fontsize=7)
          #   end
          # end
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
              window = 100
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
          label = "$(labels[type]), D=$maxdim"
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

function main(df; dir, L_values=[4, 6, 8, 10], individual=false)
  if individual
    plot_individual_imbalance(df, L_values; dir)
    plot_individual_imbalance_error(df, L_values; dir)
  end
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
  # New: β vs h plots using grid-averaged imbalance at highest χ per method
  plot_beta_vs_disorder(df;
    L_values=L_values,
    h_values=[0.0, 2.5, 5.0, 7.5, 10.0, 20.0, 30.0, 50.0],
    dir=dir,
    tmin=50.0,
    tmax=100.0,
    xlim=(0.0, 50.0),
    ylim=(0.0, 1.0),
  )
  # New: Overlay fitted power-law with averaged imbalance per (L, h)
  plot_fit_vs_mean(df;
    L_values=L_values,
    h_values=[0.0, 2.5, 5.0, 7.5, 10.0, 20.0, 30.0, 50.0],
    dir=dir,
  )
end

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

using Random, StatsBase, LsqFit, Statistics

const BETA_SYM = Symbol("β")
const BETA_MED_SYM = Symbol("β_med")

function powerlaw_model(t, p)
  A = p[1]
  β = p[2]
  C = length(p) >= 3 ? p[3] : zero(A)
  @. A * t^(-β) + C
end

function _row_value(row, candidates; default=nothing)
  names_row = propertynames(row)
  for name in candidates
    if name in names_row
      val = row[name]
      if !(val isa Missing)
        return val
      end
    end
  end
  return default
end


function calculate_beta(df::DataFrame; fitting_method=fit_power_law_beta, fit_window=(50.0, 100.0))
  h_values = sort(unique(df.model_h))
  L_values = sort(unique(df.graph_L))
  chis = sort(unique(df.initial_state_initial_maxdim))

  results = DataFrame(
    h=eltype(h_values)[],
    L=eltype(L_values)[],
    chi=eltype(chis)[],
    β=Float64[],
    β_err=Float64[],
    A=Float64[],
    A_err=Float64[],
  )

  for chi in chis
    for L in L_values
      for h in h_values
        df_filtered = filter(x -> x.model_h == h && x.graph_L == L && x.initial_state_initial_maxdim == chi, df)
        if nrow(df_filtered) == 0
          continue
        end

        res = fitting_method(df_filtered; fit_window)

        @info "Adding $chi, $L, $h to the result"
        push!(results, (
          h=h,
          L=L,
          chi=chi,
          β=res.beta,
          β_err=res.stderr,
          A=res.A,
          A_err=res.stderr_A,
        ))
      end
    end
  end

  return results
end

function calculate_beta_with_time_window_error(df_beta; starts=(20.0, 50.0))
  df1 = calculate_beta(df_beta; fit_window=(starts[1], 100.0))
  df2 = calculate_beta(df_beta; fit_window=(starts[2], 100.0))

  return DataFrame(
    h=df1.h,
    L=df1.L,
    chi=df1.chi,
    β=(df1.β .+ df2.β) ./ 2,
    β_err=abs.(df1.β .- df2.β) ./ 2 + df1.β_err,
    A=(df1.A .+ df2.A) ./ 2,
    A_err=abs.(df1.A .- df2.A) ./ 2
  )
end

function plot_hc_new_error(df::DataFrame; beta_crit=(0.005, 0.001))
  L_vals = Float64[]
  hc_vals = Float64[]
  err_vals = Float64[]

  for L in sort(unique(df.L))
    df_sub = filter(x -> x.L == L, df)
    if nrow(df_sub) < 2
      @warn "Skipping L=$(L) – not enough points to determine h_c"
      continue
    end

    try
      hc1, _ = calculate_hc(df_sub.h, df_sub.β, df_sub.β_err, beta_crit[1])
      hc2, _ = calculate_hc(df_sub.h, df_sub.β, df_sub.β_err, beta_crit[2])
      push!(L_vals, Float64(L))
      push!(hc_vals, Float64((hc1 + hc2) / 2))
      push!(err_vals, Float64(abs((hc1 - hc2) / 2)))
    catch err
      @warn "Failed to evaluate h_c for L=$(L)" exception = (err, catch_backtrace())
    end
  end

  isempty(L_vals) && return nothing

  fig = Figure()
  ax = Axis(fig[1, 1];
    xlabel=L"L",
    ylabel=L"h_c")

  scatter!(ax, L_vals, hc_vals; markersize=10, color=:royalblue)
  errorbars!(ax, L_vals, hc_vals, err_vals; direction=:y, color=:royalblue)
  lines!(ax, L_vals, hc_vals; color=:royalblue, linestyle=:dash)

  return fig
end


function plot_hc_new_error_inv_L(df::DataFrame; beta_crit=(0.005, 0.001))
  L_vals = Float64[]
  hc_vals = Float64[]
  err_vals = Float64[]

  for L in sort(unique(df.L))
    df_sub = filter(x -> x.L == L, df)
    if nrow(df_sub) < 2
      @warn "Skipping L=$(L) – not enough points to determine h_c"
      continue
    end

    try
      hc1, _ = calculate_hc(df_sub.h, df_sub.β, df_sub.β_err, beta_crit[1])
      hc2, _ = calculate_hc(df_sub.h, df_sub.β, df_sub.β_err, beta_crit[2])
      push!(L_vals, Float64(L))
      push!(hc_vals, Float64((hc1 + hc2) / 2))
      push!(err_vals, Float64(abs((hc1 - hc2) / 2)))
    catch err
      @warn "Failed to evaluate h_c for L=$(L)" exception = (err, catch_backtrace())
    end
  end

  isempty(L_vals) && return nothing

  fig = Figure()
  ax = Axis(fig[1, 1];
    xlabel=L"1/L",
    ylabel=L"h_c")
  xlims!(ax, (0.0, 0.26))

  scatter!(ax, 1 ./ L_vals, hc_vals; markersize=10, color=:royalblue)
  errorbars!(ax, 1 ./ L_vals, hc_vals, err_vals; direction=:y, color=:royalblue)
  lines!(ax, 1 ./ L_vals, hc_vals; color=:royalblue, linestyle=:dash)

  return fig
end

function plot_hc_inv_L(df::DataFrame; beta_crit::Real=0.02)
  L_vals = Float64[]
  hc_vals = Float64[]
  err_vals = Float64[]

  for L in sort(unique(df.L))
    df_sub = filter(x -> x.L == L, df)
    if nrow(df_sub) < 2
      @warn "Skipping L=$(L) – not enough points to determine h_c"
      continue
    end

    try
      hc, σ_hc, _ = calculate_hc(df_sub.h, df_sub.β, df_sub.β_err, beta_crit)
      push!(L_vals, Float64(L))
      push!(hc_vals, Float64(hc))
      push!(err_vals, Float64(σ_hc))
    catch err
      @warn "Failed to evaluate h_c for L=$(L)" exception = (err, catch_backtrace())
    end
  end

  isempty(L_vals) && return nothing

  fig = Figure()
  ax = Axis(fig[1, 1];
    xlabel=L"1/L",
    ylabel=L"h_c",)

  xlims!(ax, (0.0, 0.26))

  scatter!(ax, 1 ./ L_vals, hc_vals; markersize=10, color=:royalblue)
  errorbars!(ax, 1 ./ L_vals, hc_vals, err_vals; direction=:y, color=:royalblue)
  lines!(ax, 1 ./ L_vals, hc_vals; color=:royalblue, linestyle=:dash)

  return fig
end

function plot_hc(df::DataFrame; beta_crit::Real=0.02)
  L_vals = Float64[]
  hc_vals = Float64[]
  err_vals = Float64[]

  for L in sort(unique(df.L))
    df_sub = filter(x -> x.L == L, df)
    if nrow(df_sub) < 2
      @warn "Skipping L=$(L) – not enough points to determine h_c"
      continue
    end

    try
      hc, σ_hc, _ = calculate_hc(df_sub.h, df_sub.β, df_sub.β_err, beta_crit)
      push!(L_vals, Float64(L))
      push!(hc_vals, Float64(hc))
      push!(err_vals, Float64(σ_hc))
    catch err
      @warn "Failed to evaluate h_c for L=$(L)" exception = (err, catch_backtrace())
    end
  end

  isempty(L_vals) && return nothing

  fig = Figure()
  ax = Axis(fig[1, 1];
    xlabel="L",
    ylabel="h_c(L)",
  )

  scatter!(ax, L_vals, hc_vals; markersize=10, color=:royalblue)
  errorbars!(ax, L_vals, hc_vals, err_vals; direction=:y, color=:royalblue)
  lines!(ax, L_vals, hc_vals; color=:royalblue, linestyle=:dash)

  return fig
end

function plot_beta_vs_h(beta_df::DataFrame; chi=nothing, L=nothing)
  df_filtered = chi === nothing ? beta_df : filter(row -> row.chi == chi, beta_df)

  unique_L = sort(unique(df_filtered.L))
  if L !== nothing
    unique_L = intersect(unique_L, [L])
  end

  if isempty(unique_L)
    error("No entries left after filtering; check chi and L.")
  end

  fig = Figure()
  ax = Axis(fig[1, 1]; xlabel="h", ylabel="β", yscale=log10)

  min_beta = 1e-3

  for Lval in unique_L
    df_L = sort(filter(row -> row.L == Lval, df_filtered), :h)
    if nrow(df_L) == 0
      continue
    end

    h_vals = Float64[]
    β_vals = Float64[]
    β_err_low = Float64[]
    β_err_high = Float64[]

    for row in eachrow(df_L)
      β_val = _row_value(row, (BETA_MED_SYM, BETA_SYM))
      β_val === nothing && continue

      β_v = Float64(β_val)
      push!(h_vals, Float64(row.h))
      push!(β_vals, β_v)

      low_candidate = _row_value(row, (:β_low, Symbol("β_lower")))
      high_candidate = _row_value(row, (:β_high, Symbol("β_upper")))
      if low_candidate !== nothing && high_candidate !== nothing
        push!(β_err_low, abs(β_v - Float64(low_candidate)))
        push!(β_err_high, abs(Float64(high_candidate) - β_v))
      else
        err = _row_value(row, (:β_err, Symbol("β_error"), Symbol("β_stderr")))
        if err === nothing
          push!(β_err_low, NaN)
          push!(β_err_high, NaN)
        else
          err_val = abs(Float64(err))
          push!(β_err_low, err_val)
          push!(β_err_high, err_val)
        end
      end
    end

    if isempty(β_vals)
      @warn "No β values to plot for L=$(Lval); skipping."
      continue
    end

    β_vals_clamped = similar(β_vals)
    β_err_low_clamped = fill(NaN, length(β_vals))
    β_err_high_clamped = fill(NaN, length(β_vals))

    for i in eachindex(β_vals)
      val = β_vals[i]
      clamped_val = max(val, min_beta)
      β_vals_clamped[i] = clamped_val

      err_low = β_err_low[i]
      if isfinite(err_low)
        low_val = val - err_low
        low_clamped = max(low_val, min_beta)
        β_err_low_clamped[i] = clamped_val - low_clamped
      end

      err_high = β_err_high[i]
      if isfinite(err_high)
        high_val = val + err_high
        high_clamped = max(high_val, min_beta)
        β_err_high_clamped[i] = high_clamped - clamped_val
      end
    end

    err_mask = isfinite.(β_err_low_clamped) .& isfinite.(β_err_high_clamped)
    if any(err_mask)
      errorbars!(ax, h_vals[err_mask], β_vals_clamped[err_mask], β_err_low_clamped[err_mask], β_err_high_clamped[err_mask]; whiskerwidth=6)
    end
    scatter!(ax, h_vals, β_vals_clamped; markersize=12, label="L=$(Lval)")
  end

  axislegend(ax; position=:rb, framevisible=false)

  return fig, ax
end

function plot_beta_vs_h_with_hc_fit(beta_df::DataFrame; chi=nothing, L=nothing,
  beta_crit::Real=0.001, n_fit_points::Int=200)
  df_filtered = chi === nothing ? beta_df : filter(row -> row.chi == chi, beta_df)

  unique_L = sort(unique(df_filtered.L))
  if L !== nothing
    unique_L = intersect(unique_L, [L])
  end

  if isempty(unique_L)
    error("No entries left after filtering; check chi and L.")
  end

  fig = Figure()
  ax = Axis(fig[1, 1]; xlabel="h", ylabel="β", yscale=log10)

  min_beta = 1e-3
  colors = Makie.wong_colors()
  results = NamedTuple[]

  for (idx, Lval) in enumerate(unique_L)
    df_L = sort(filter(row -> row.L == Lval, df_filtered), :h)
    if nrow(df_L) == 0
      continue
    end

    h_vals = Float64[]
    β_vals = Float64[]
    β_err_low = Float64[]
    β_err_high = Float64[]
    β_err_fit = Float64[]

    for row in eachrow(df_L)
      β_val = _row_value(row, (BETA_MED_SYM, BETA_SYM))
      β_val === nothing && continue

      β_v = Float64(β_val)
      push!(h_vals, Float64(row.h))
      push!(β_vals, β_v)

      low_candidate = _row_value(row, (:β_low, Symbol("β_lower")))
      high_candidate = _row_value(row, (:β_high, Symbol("β_upper")))

      err_low = NaN
      err_high = NaN
      if low_candidate !== nothing
        err_low = abs(β_v - Float64(low_candidate))
      end
      if high_candidate !== nothing
        err_high = abs(Float64(high_candidate) - β_v)
      end

      if isfinite(err_low) && isfinite(err_high)
        push!(β_err_fit, max((err_low + err_high) / 2, eps()))
      else
        err = _row_value(row, (:β_err, Symbol("β_error"), Symbol("β_stderr")))
        if err === nothing
          push!(β_err_fit, 1.0)
        else
          push!(β_err_fit, max(abs(Float64(err)), eps()))
        end
      end

      push!(β_err_low, isfinite(err_low) ? err_low : NaN)
      push!(β_err_high, isfinite(err_high) ? err_high : NaN)
    end

    if isempty(β_vals)
      @warn "No β values to plot for L=$(Lval); skipping."
      continue
    end

    color = colors[mod1(idx, length(colors))]

    β_vals_clamped = similar(β_vals)
    β_err_low_clamped = fill(NaN, length(β_vals))
    β_err_high_clamped = fill(NaN, length(β_vals))

    for i in eachindex(β_vals)
      val = β_vals[i]
      clamped_val = max(val, min_beta)
      β_vals_clamped[i] = clamped_val

      err_low = β_err_low[i]
      if isfinite(err_low)
        low_val = val - err_low
        low_clamped = max(low_val, min_beta)
        β_err_low_clamped[i] = clamped_val - low_clamped
      end

      err_high = β_err_high[i]
      if isfinite(err_high)
        high_val = val + err_high
        high_clamped = max(high_val, min_beta)
        β_err_high_clamped[i] = high_clamped - clamped_val
      end
    end

    err_mask = isfinite.(β_err_low_clamped) .& isfinite.(β_err_high_clamped)
    if any(err_mask)
      errorbars!(ax, h_vals[err_mask], β_vals_clamped[err_mask], β_err_low_clamped[err_mask],
        β_err_high_clamped[err_mask]; whiskerwidth=6, color=color)
    end
    scatter!(ax, h_vals, β_vals_clamped; markersize=12, color=color, label="L=$(Lval)")

    try
      hc, σ_hc, p̂ = calculate_hc(h_vals, β_vals, β_err_fit, beta_crit)
      h_range = range(minimum(h_vals), stop=maximum(h_vals), length=n_fit_points)
      fit_vals = [p̂[1] * exp(-p̂[2] * h) for h in h_range]
      fit_vals_clamped = max.(fit_vals, 0.0)
      lines!(ax, h_range, fit_vals_clamped; color=color, linestyle=:dash, linewidth=2)
      vlines!(ax, [hc]; color=color, linestyle=:dot, linewidth=1.5)
      ylims!(ax, (1e-3, 0.2))
      push!(results, (L=Lval, hc=hc, σ_hc=σ_hc, p̂=p̂))
    catch err
      @warn "Failed to evaluate hc fit for L=$(Lval)" exception = (err, catch_backtrace())
    end
  end

  axislegend(ax; position=:rb, framevisible=false)

  return fig, ax, results
end

function plot_all_single(df::DataFrame; dir="plots/beta_new/single/")
  try
    mkdir(dir)
  catch
  end

  row = first(df)
  fig = Figure(size=(640, 480))

  ax = Axis(fig[1, 1]; xlabel="t", ylabel="I(t)", title="χ=$(row.initial_state_initial_maxdim), L=$(row.graph_L), h=$(row.model_h)")
  plt = lines!(ax, row.times, row.imbalance)
  for row in eachrow(df)
    plt[1] = row.times
    plt[2] = row.imbalance
    ax.title = "χ=$(row.initial_state_initial_maxdim), L=$(row.graph_L), h=$(row.model_h), grid=$(row.graph_gridnum)"
    save(joinpath(dir, "plot_χ=$(row.initial_state_initial_maxdim)_L=$(row.graph_L)_h=$(row.model_h)_grid=$(row.graph_gridnum).png"), fig)
  end

end

function plot_and_save_imbalance_fits(df::DataFrame, beta_df::DataFrame; chi=nothing,
  output_dir::AbstractString="beta_fit_plots", fit_window=(50.0, 100.0))
  chi_values = sort(unique(beta_df.chi))
  if chi === nothing
    if length(chi_values) != 1
      error("Multiple χ values detected; pass `chi` to select one.")
    end
    chi = first(chi_values)
  elseif chi ∉ chi_values
    error("Requested χ=$(chi) not found in beta DataFrame.")
  end

  tmin, tmax = fit_window

  mkpath(output_dir)

  df_chi = filter(row -> row.initial_state_initial_maxdim == chi, df)
  beta_chi = filter(row -> row.chi == chi, beta_df)

  saved_paths = String[]

  for L in sort(unique(beta_chi.L))
    df_L = filter(row -> row.graph_L == L, df_chi)
    beta_L = filter(row -> row.L == L, beta_chi)

    for h in sort(unique(beta_L.h))
      df_group = filter(row -> row.model_h == h, df_L)
      if nrow(df_group) == 0
        continue
      end

      params_row = filter(row -> row.h == h, beta_L)
      if nrow(params_row) == 0
        continue
      end
      params_row = first(params_row)

      std_label = "± std"
      mean_label = "mean imbalance"
      fit_label = "power-law fit"

      fig = Figure(resolution=(640, 480))
      ax = Axis(fig[1, 1]; xlabel="t", ylabel="I(t)", title="χ=$(chi), L=$(L), h=$(round(h, digits=3))")

      traces = collect(df_group.imbalance)
      @show length(traces)
      isempty(traces) && continue

      imbalance_mat = hcat(traces...)
      @show size(imbalance_mat)
      imbalance_mean = vec(mean(imbalance_mat; dims=2))
      imbalance_std = vec(std(imbalance_mat; dims=2, corrected=false)) ./ sqrt(length(traces))

      t = df_group.times[1]

      lower = imbalance_mean .- imbalance_std
      upper = imbalance_mean .+ imbalance_std

      band!(ax, t, lower, upper;
        color=(:black, 0.15), label=std_label)
      lines!(ax, t, imbalance_mean; color=:black, label=mean_label)

      A = params_row.A
      β = params_row.β
      C = 0.0

      if A === nothing || β === nothing
        @warn "Missing fit parameters for χ=$(chi), L=$(L), h=$(h); skipping fit overlay."
        continue
      end

      mask = (t .>= tmin) .& (t .<= tmax)
      if count(mask) < 2
        @warn "No time points within fit window [$(tmin), $(tmax)] for χ=$(chi), L=$(L), h=$(h); skipping fit overlay."
        continue
      end

      t_fit = t[mask]
      fit_params = [Float64(A), Float64(β), Float64(C)]
      fit_vals = powerlaw_model(t_fit, fit_params)
      lines!(ax, t_fit, fit_vals; color=:crimson, linewidth=2, label=fit_label)

      axislegend(ax; position=:rt, framevisible=false)

      h_str = replace(@sprintf("%0.3f", h), "." => "p")
      out_path = joinpath(output_dir, "chi$(chi)_L$(L)_h$(h_str).png")
      save(out_path, fig)
      push!(saved_paths, out_path)

    end
  end

  return saved_paths
end

function bootstrap_powerlaw(df::DataFrame; nboot::Int=500, fit_window=(50.0, 100.0))
  tmin, tmax = fit_window
  nreal = nrow(df)

  # assume identical time grid for all realizations
  t_all = df.times[1]
  mask = (t_all .>= tmin) .& (t_all .<= tmax)
  t = t_all[mask]

  params = Array{Float64}(undef, nboot, 3)

  for b in 1:nboot
    # resample disorder realizations with replacement
    inds = sample(1:nreal, nreal; replace=true)

    # elementwise average imbalance over sampled realizations
    I_mat = reduce(hcat, [df.imbalance[i] for i in inds])
    I_mean = mean(I_mat; dims=2)
    I_mean = vec(I_mean[mask])

    p0 = [1.0, 0.2, 1.0]

    try
      fit = curve_fit(powerlaw_model, t, I_mean, p0)
      params[b, :] = coef(fit)
    catch
      params[b, :] .= NaN
    end
  end

  # --- cleanly drop failed fits ---
  good = [!any(isnan, row) for row in eachrow(params)]  # Bool mask per row
  params = params[good, :]

  # --- summarize ---
  A_med = median(params[:, 1])
  β_med = median(params[:, 2])
  C_med = median(params[:, 3])
  β_CI = quantile(params[:, 2], [0.16, 0.84])

  return (; A_med, β_med, C_med, β_CI, params)
end

"""
  fit_power_law_beta(times, values; tmin=5.0, tmax=50.0)

Estimate the decay exponent β from an averaged imbalance curve by fitting
  values ≈ A * t^(-β) on a log–log window t ∈ [tmin, tmax].
Returns a NamedTuple `(beta, stderr, npts, r2)` or `nothing` if not enough points.
"""


function fit_power_law_beta(df::DataFrame; fit_window=(50.0, 100.0),
  known_variances::Bool=true)
  tmin, tmax = fit_window

  times = df.times[1]

  values = mean(df.imbalance)
  yerrors = std(df.imbalance) ./ sqrt(nrow(df))

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

function fit_power_law_beta_direct(df::DataFrame; fit_window=(50.0, 100.0))
  tmin, tmax = fit_window
  times = df.times[1]

  values = mean(df.imbalance)
  yerrors = std(df.imbalance) ./ sqrt(nrow(df))

  isempty(times) && return nothing
  isempty(values) && return nothing

  idx = findall(i -> times[i] >= tmin && times[i] <= tmax &&
                       isfinite(values[i]) && values[i] > 0 &&
                       isfinite(times[i]), eachindex(times))

  length(idx) < 3 && return nothing

  t = Float64[times[i] for i in idx]
  y = Float64[values[i] for i in idx]
  σ = isnothing(yerrors) ? nothing : Float64[yerrors[i] for i in idx]

  # Guard invalid uncertainties; fall back to unweighted fit if needed.
  weights = nothing
  if σ !== nothing
    valid = all(isfinite, σ) && all(>(0.0), σ)
    if valid
      weights = 1.0 ./ (σ .^ 2)
    end
  end

  # Initial guesses from end points in log space.
  β0 = (log(y[1]) - log(y[end])) / (log(t[end]) - log(t[1]))
  β0 = isnan(β0) || !isfinite(β0) ? 0.5 : max(β0, 0.0)
  A0 = y[1] * t[1]^β0
  A0 = isnan(A0) || !isfinite(A0) ? maximum(y) : max(A0, eps())
  p0 = [A0, β0]

  fit = try
    if weights === nothing
      curve_fit(powerlaw_model, t, y, p0)
    else
      curve_fit(powerlaw_model, t, y, weights, p0)
    end
  catch
    return nothing
  end

  params = coef(fit)
  if length(params) == 2
    Â, β̂ = params[1], params[2]
    C = 0.0
  else
    Â, β̂, C = params[1], params[2], params[3]
  end

  cov = try
    estimate_covar(fit)
  catch
    fill(NaN, length(params), length(params))
  end

  stderr_A = sqrt(cov[1, 1]) |> x -> isfinite(x) ? x : NaN
  stderr_β = sqrt(cov[2, 2]) |> x -> isfinite(x) ? x : NaN
  if length(params) == 3
    stderr_C = sqrt(cov[3, 3]) |> x -> isfinite(x) ? x : NaN
  end

  ŷ = powerlaw_model(t, params)
  resid = y .- ŷ
  n = length(t)

  if weights === nothing
    RSS = sum(resid .^ 2)
    μ = mean(y)
    TSS = sum((y .- μ) .^ 2)
  else
    RSS = sum(weights .* resid .^ 2)
    Sw = sum(weights)
    μ = sum(weights .* y) / Sw
    TSS = sum(weights .* (y .- μ) .^ 2)
  end

  r2 = TSS ≈ 0 ? 1.0 : max(0.0, 1.0 - RSS / TSS)
  rmse = sqrt(sum(resid .^ 2) / n)

  return (beta=β̂,
    stderr=stderr_β,
    A=Â,
    stderr_A=stderr_A,
    r2=r2,
    npts=n,
    rmse=rmse)

end

function plot_all(df::DataFrame; dir="plots/beta_new")
  try
    mkdir(dir)
  catch
  end
  res = calculate_beta_with_time_window_error(df; starts=(30.0, 50.0))

  fig = plot_hc_new_error(res)
  save(joinpath(dir, "hc_graph.pdf"), fig)
  save(joinpath(dir, "hc_graph.png"), fig)

  fig = plot_hc_new_error_inv_L(res)
  save(joinpath(dir, "hc_graph_inv_L.pdf"), fig)
  save(joinpath(dir, "hc_graph_inv_L.png"), fig)

  plot_and_save_imbalance_fits(df, res; output_dir=joinpath(dir, "fits"))

  fig, _ = plot_beta_vs_h(res)
  save(joinpath(dir, "beta_vs_h.pdf"), fig)
  save(joinpath(dir, "beta_vs_h.png"), fig)

  fig, _ = plot_beta_vs_h_with_hc_fit(res)
  save(joinpath(dir, "beta_vs_h_with_fit.pdf"), fig)
  save(joinpath(dir, "beta_vs_h_with_fit.png"), fig)

  return
end

using CairoMakie
using CairoMakie: hidexdecorations!, linkxaxes!, linkyaxes!, LinearTicks, LineElement, MarkerElement
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
using LaTeXStrings

function plot_L12_comparison(df_in::DataFrame; h_values::Tuple, gridnum=nothing)
  palette = ColorSchemes.Zissou1Continuous.colors
  palette = palette[[1, 4, 8, 11]]

  scientific_components(x::Real) = x == 0 ? (0.0, 0) : begin
    exp = floor(Int, log10(abs(x)))
    mant = x / 10.0^exp
    (mant, exp)
  end

  function single_plot(df_in, h, gridnum, fig, row, is_bottom, subplot_labels)
    df = filter(row -> row.model_h == h && row.graph_gridnum == gridnum, df_in)
    df_mps = filter(row -> row.graph_type == "SnakeGraph", df)
    df_ttn = filter(row -> row.graph_type == "FreeGraph", df)

    # ---------- Left plot: imbalance ----------
    dims = sort!(unique(vcat(df_mps.initial_state_initial_maxdim, df_ttn.initial_state_initial_maxdim)))
    if isempty(dims)
      error("No data available for h=$h and gridnum=$gridnum")
    end

    ax = Axis(fig[row, 1], ylabel=L"I(t)", xticks=LinearTicks(7), limits=((0, 100), nothing))
    if is_bottom
      ax.xlabel = L"t"
    else
      hidexdecorations!(ax; grid=false)
    end

    marker_mps = :circle
    marker_ttn = :diamond
    linestyle_mps = :solid
    linestyle_ttn = :solid

    color_mps = palette[1]
    color_ttn = palette[end]

    scale_factor = 1e5

    maxd = maximum(dims)
    @show maxd
    max_mps_matches = filter(r -> r.initial_state_initial_maxdim == maxd, df_mps)
    max_ttn_matches = filter(r -> r.initial_state_initial_maxdim == maxd, df_ttn)
    isempty(max_mps_matches) && error("Missing MPS data for max chi=$maxd")
    isempty(max_ttn_matches) && error("Missing TTN data for max chi=$maxd")
    max_mps = only(max_mps_matches)
    max_ttn = only(max_ttn_matches)

    function pick_run(df, target_dim)
      matches = filter(r -> r.initial_state_initial_maxdim == target_dim, df)
      if !isempty(matches)
        return only(matches), target_dim
      end
      dims_available = sort(unique(df.initial_state_initial_maxdim))
      isempty(dims_available) && return nothing
      nearest_idx = argmin(abs.(dims_available .- target_dim))
      nearest_dim = dims_available[nearest_idx]
      matches = filter(r -> r.initial_state_initial_maxdim == nearest_dim, df)
      isempty(matches) && return nothing
      return only(matches), nearest_dim
    end

    function legend_label(base::String, ref_run, run)
      accuracy = mean(abs.(ref_run.imbalance .- run.imbalance))
      npar = round(minimum(run.num_size))
      mantissa, exponent = scientific_components(accuracy)
      accuracy_mantissa = @sprintf("%.1f", mantissa)
      accuracy_str = L"%$accuracy_mantissa \cdot 10^{%$exponent}"
      npar_str = @sprintf("%.0f", npar)
      return L"\text{%$base},\; \varepsilon = %$accuracy_str,\; N_{\mathrm{par}} = %$npar_str"
    end

    target_mps = pick_run(df_mps, 64)
    target_ttn = pick_run(df_ttn, 32)

    if target_mps !== nothing
      mps, _ = target_mps
      label = legend_label("MPS", max_mps, mps)
      lines!(ax, mps.times, mps.imbalance;
        color=color_mps, linestyle=linestyle_mps, label=label)
    end
    if target_ttn !== nothing
      ttn, _ = target_ttn
      label = legend_label("TTN", max_ttn, ttn)
      lines!(ax, ttn.times, ttn.imbalance;
        color=color_ttn, linestyle=linestyle_ttn, label=label)
    end

    axislegend(ax; position=:rt, orientation=:vertical,
      framevisible=false, padding=(4, 4, 4, 4), patchsize=(14, 6))

    # ---------- Right plot: errors vs size/time ----------

    ax_nop = Axis(fig[row, 2],
      xreversed=true, xscale=log10,
      xlabel=L"\mathrm{accuracy}", ylabel=L"N_{\mathrm{par}} / 10^{5}")
    if !is_bottom
      hidexdecorations!(ax_nop; grid=false)
    end

    ax_time = Axis(fig[row, 2],
      xreversed=true, xscale=log10,
      xlabel=L"\mathrm{accuracy}", ylabel=L"t_{\mathrm{ex}}",
      yaxisposition=:right)
    if !is_bottom
      hidexdecorations!(ax_time; grid=false)
    end

    color_npar = color_mps
    color_tex = color_ttn

    ax_nop.ylabelcolor = color_npar
    ax_nop.yticklabelcolor = color_npar
    ax_nop.leftspinecolor[] = color_npar
    ax_time.ylabelcolor = color_tex
    ax_time.yticklabelcolor = color_tex
    ax_time.rightspinecolor[] = color_tex

    smaller_dims = filter(d -> d != maxd, dims)

    err_mps_nop = Float64[]
    nop_vals_mps = Float64[]
    err_ttn_nop = Float64[]
    nop_vals_ttn = Float64[]

    for maxdim in smaller_dims
      mps_matches = filter(r -> r.initial_state_initial_maxdim == maxdim, df_mps)
      ttn_matches = filter(r -> r.initial_state_initial_maxdim == maxdim, df_ttn)
      isempty(mps_matches) && continue
      isempty(ttn_matches) && continue
      mps = only(mps_matches)
      ttn = only(ttn_matches)

      err_mps = abs.(max_mps.imbalance .- mps.imbalance)
      err_ttn = abs.(max_ttn.imbalance .- ttn.imbalance)

      mean_err_mps = mean(err_mps)
      mean_err_ttn = mean(err_ttn)

      if mean_err_mps <= 0 && mean_err_ttn <= 0
        continue
      end

      nop_mps = minimum(mps.num_size) / scale_factor
      nop_ttn = minimum(ttn.num_size) / scale_factor

      if mean_err_mps > 0
        push!(err_mps_nop, mean_err_mps)
        push!(nop_vals_mps, nop_mps)
      end
      if mean_err_ttn > 0
        push!(err_ttn_nop, mean_err_ttn)
        push!(nop_vals_ttn, nop_ttn)
      end
    end

    nop_lines_mps = nothing
    nop_scatter_mps = nothing
    if !isempty(err_mps_nop)
      order = sortperm(err_mps_nop)
      x_sorted = err_mps_nop[order]
      y_sorted = nop_vals_mps[order]
      nop_scatter_mps = scatter!(ax_nop, x_sorted, y_sorted;
        color=color_npar, marker=marker_mps)
      nop_lines_mps = lines!(ax_nop, x_sorted, y_sorted;
        color=color_npar, linestyle=linestyle_mps)
    end
    nop_lines_ttn = nothing
    nop_scatter_ttn = nothing
    if !isempty(err_ttn_nop)
      order = sortperm(err_ttn_nop)
      x_sorted = err_ttn_nop[order]
      y_sorted = nop_vals_ttn[order]
      nop_scatter_ttn = scatter!(ax_nop, x_sorted, y_sorted;
        color=color_npar, marker=marker_ttn)
      nop_lines_ttn = lines!(ax_nop, x_sorted, y_sorted;
        color=color_npar, linestyle=linestyle_ttn)
    end

    err_mps_time = Float64[]
    time_vals_mps = Float64[]
    err_ttn_time = Float64[]
    time_vals_ttn = Float64[]

    for maxdim in dims
      mps_matches = filter(r -> r.initial_state_initial_maxdim == maxdim, df_mps)
      ttn_matches = filter(r -> r.initial_state_initial_maxdim == maxdim, df_ttn)
      isempty(mps_matches) && continue
      isempty(ttn_matches) && continue
      mps = only(mps_matches)
      ttn = only(ttn_matches)

      err_mps = abs.(max_mps.imbalance .- mps.imbalance)
      err_ttn = abs.(max_ttn.imbalance .- ttn.imbalance)

      mean_err_mps = mean(err_mps)
      mean_err_ttn = mean(err_ttn)

      start = 10
      time_mps = minimum(mps.ex_times[start:end])
      time_ttn = minimum(ttn.ex_times[start:end])

      if mean_err_mps > 0
        push!(err_mps_time, mean_err_mps)
        push!(time_vals_mps, time_mps)
      end
      if mean_err_ttn > 0
        push!(err_ttn_time, mean_err_ttn)
        push!(time_vals_ttn, time_ttn)
      end
    end

    time_lines_mps = nothing
    time_scatter_mps = nothing
    if !isempty(err_mps_time)
      order = sortperm(err_mps_time)
      x_sorted = err_mps_time[order]
      y_sorted = time_vals_mps[order]
      time_scatter_mps = scatter!(ax_time, x_sorted, y_sorted;
        color=color_tex, marker=marker_mps)
      time_lines_mps = lines!(ax_time, x_sorted, y_sorted;
        color=color_tex, linestyle=linestyle_mps)
    end
    time_lines_ttn = nothing
    time_scatter_ttn = nothing
    if !isempty(err_ttn_time)
      order = sortperm(err_ttn_time)
      x_sorted = err_ttn_time[order]
      y_sorted = time_vals_ttn[order]
      time_scatter_ttn = scatter!(ax_time, x_sorted, y_sorted;
        color=color_tex, marker=marker_ttn)
      time_lines_ttn = lines!(ax_time, x_sorted, y_sorted;
        color=color_tex, linestyle=linestyle_ttn)
    end

    marker_handles = AbstractPlot[]
    marker_labels = String[]
    if (nop_scatter_mps !== nothing) || (time_scatter_mps !== nothing)
      handle = scatterlines!(ax_nop, [NaN], [NaN]; color=:black,
        linestyle=linestyle_mps, marker=marker_mps, markersize=10,
        markercolor=:black, visible=false)
      push!(marker_handles, handle)
      push!(marker_labels, "MPS")
    end
    if (nop_scatter_ttn !== nothing) || (time_scatter_ttn !== nothing)
      handle = scatterlines!(ax_nop, [NaN], [NaN]; color=:black,
        linestyle=linestyle_ttn, marker=marker_ttn, markersize=10,
        markercolor=:black, visible=false)
      push!(marker_handles, handle)
      push!(marker_labels, "TTN")
    end
    if !isempty(marker_handles)
      axislegend(ax_nop, marker_handles, marker_labels;
        position=:rt, orientation=:horizontal, framevisible=false,
        patchsize=(14, 6))
    end

    # Add text label inside each subplot row (top left)
    # Subplot annotations
    if subplot_labels !== nothing
      text!(ax, 0.04, 0.94, text=subplot_labels[1], space=:relative,
        align=(:left, :top), color=:black)
      text!(ax_nop, 0.04, 0.94, text=subplot_labels[2], space=:relative,
        align=(:left, :top), color=:black)
    end

    text!(ax, 0.04, 0.06, text=L"h=%$h", space=:relative,
      align=(:left, :bottom), color=:black)

    return ax
  end

  # ---------------- Master Figure ----------------
  fig = Figure(size=(600, 400), fontsize=10)

  left_axes = Axis[]

  labels = [("(a)", "(b)"), ("(c)", "(d)")]
  push!(left_axes, single_plot(df_in, h_values[1], gridnum, fig, 1, false, labels[1]))
  push!(left_axes, single_plot(df_in, h_values[2], gridnum, fig, 2, true, labels[2]))

  linkxaxes!(left_axes...)

  display(fig)
  save(joinpath("plots", "paper", "L12_plot.pdf"), fig)
end

function plot_L12_all_imbalances(df_in::DataFrame; h_values::AbstractVector, gridnum=nothing)
  palette = ColorSchemes.Zissou1Continuous.colors
  palette = palette[[1, 4, 8, 11]]

  fig_height = 300 * length(h_values)
  fig = Figure(size=(1000, fig_height))

  axes = Axis[]

  for (row_idx, h) in enumerate(h_values)
    df = filter(row -> row.model_h == h && (gridnum === nothing || row.graph_gridnum == gridnum), df_in)
    df_mps = filter(row -> row.graph_type == "SnakeGraph", df)
    df_ttn = filter(row -> row.graph_type == "FreeGraph", df)

    dims = sort!(unique(vcat(df_mps.initial_state_initial_maxdim, df_ttn.initial_state_initial_maxdim)))
    isempty(dims) && error("No data available for h=$h and gridnum=$gridnum")

    ax = Axis(fig[row_idx, 1], ylabel=L"I(t)", xticks=LinearTicks(7))
    if row_idx == length(h_values)
      ax.xlabel = L"t"
    else
      hidexdecorations!(ax; grid=false)
    end

    for (i, maxdim) in enumerate(dims)
      color = palette[mod1(i, length(palette))]
      mps_matches = filter(r -> r.initial_state_initial_maxdim == maxdim, df_mps)
      ttn_matches = filter(r -> r.initial_state_initial_maxdim == maxdim, df_ttn)
      isempty(mps_matches) && continue
      isempty(ttn_matches) && continue

      mps = only(mps_matches)
      ttn = only(ttn_matches)

      lines!(ax, mps.times, mps.imbalance,
        label=L"\mathrm{MPS},\; \chi = %$maxdim", color=color, linestyle=:dot)
      lines!(ax, ttn.times, ttn.imbalance,
        label=L"\mathrm{TTN},\; \chi = %$maxdim", color=color)
    end

    axislegend(ax;
      position=:rt, orientation=:vertical, framevisible=false,
      padding=(6, 6, 6, 6))

    text!(ax, 0.05, 0.95, text=L"h = %$h", space=:relative,
      align=(:left, :top), color=:black, fontsize=14)

    push!(axes, ax)
  end

  linkxaxes!(axes...)

  display(fig)
end

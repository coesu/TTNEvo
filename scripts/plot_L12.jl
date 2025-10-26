using CairoMakie
using CairoMakie: hidexdecorations!, linkxaxes!, linkyaxes!, LinearTicks, LineElement, MarkerElement, IntervalsBetween
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

function plot_L12_comparison(df::DataFrame; h_values, grid)
  palette = ColorSchemes.Zissou1Continuous.colors[[1, 4, 8]]

  mps = filter(x -> x.graph_type == "SnakeGraph" && x.graph_gridnum == grid, df)
  ttn = filter(x -> x.graph_type == "FreeGraph" && x.graph_gridnum == grid, df)

  fig = Figure(size=(600, length(h_values) * 200))
  panel_letters = [Char('a' + mod(idx - 1, 26)) for idx in 1:(2*length(h_values))]
  panel_idx = 1
  axes_time = Axis[]
  axes_error = Axis[]

  for (row_idx, h) in enumerate(h_values)
    ax = Axis(fig[row_idx, 1])
    push!(axes_time, ax)
    if row_idx == length(h_values)
      ax.xlabel = L"t"
    else
      hidexdecorations!(ax, grid=false)
    end
    text!(ax, 0.05, 0.95; text="($(panel_letters[panel_idx]))", space=:relative,
      align=(:left, :top), color=:black, fontsize=14)
    text!(ax, 0.95, 0.95,
      text=L"h=%$h",
      align=(:right, :top),
      fontsize=14,
      space=:relative,
    )
    panel_idx += 1
    h_mps = sort(filter(x -> x.model_h == h, mps), :initial_state_initial_maxdim)
    h_ttn = sort(filter(x -> x.model_h == h, ttn), :initial_state_initial_maxdim)
    for (series_idx, (m, t)) in enumerate(zip(eachrow(h_mps), eachrow(h_ttn)))
      mps_label = (row_idx == 1 && series_idx == 1) ? "MPS" : nothing
      ttn_label = (row_idx == 1 && series_idx == 1) ? "TTN" : nothing
      lines!(ax, m.times, m.imbalance, color=palette[series_idx], linestyle=:dash, label=mps_label)
      lines!(ax, t.times, t.imbalance, color=palette[series_idx], label=ttn_label)
    end
    if row_idx == 1
      axislegend(ax, position=:lt)
    end

    ref_mps = last(h_mps)
    ref_ttn = last(h_ttn)

    rest_mps = h_mps[1:(end-1), :]
    rest_ttn = h_ttn[1:(end-1), :]

    error_mps = []
    error_ttn = []

    mean_num_size_mps = []
    mean_num_size_ttn = []

    runtimes_mps = []
    runtimes_ttn = []

    for (rest_idx, (m, t)) in enumerate(zip(eachrow(rest_mps), eachrow(rest_ttn)))
      min_len_mps = min(length(ref_mps.imbalance), length(m.imbalance))
      @show min_len_mps
      push!(error_mps, mean(abs.(ref_mps.imbalance[1:min_len_mps] .- m.imbalance[1:min_len_mps])))
      min_len_ttn = min(length(ref_ttn.imbalance), length(t.imbalance))
      push!(error_ttn, mean(abs.(ref_ttn.imbalance[1:min_len_ttn] .- t.imbalance[1:min_len_ttn])))
      push!(mean_num_size_mps, mean(m.num_size))
      push!(mean_num_size_ttn, mean(t.num_size))
      push!(runtimes_mps, mean(m.ex_times))
      push!(runtimes_ttn, mean(t.ex_times))
    end

    ax_err = Axis(fig[row_idx, 2], yscale=log10)
    push!(axes_error, ax_err)
    if row_idx == length(h_values)
      ax_err.xlabel = L"N_{\mathrm{par}}"
      ax_err.ylabel = L"\langle |I_{\chi_{\max}} - I_{\chi}| \rangle"
    else
      hidexdecorations!(ax_err, grid=false)
    end
    text!(ax_err, 0.05, 0.95; text="($(panel_letters[panel_idx]))", space=:relative,
      align=(:left, :top), color=:black, fontsize=14)
    panel_idx += 1

    lines!(ax_err, mean_num_size_mps ./ 1e5, error_mps; color=:black, linewidth=1.0, transparency=true, alpha=0.5)
    lines!(ax_err, mean_num_size_ttn ./ 1e5, error_ttn; color=:black, linewidth=1.0, transparency=true, alpha=0.5)
    # scatter!(ax_err, mean_num_size_mps ./ 1e5, error_mps)
    # scatter!(ax_err, mean_num_size_ttn ./ 1e5, error_ttn)
    @show maximum(runtimes_mps)
    @show maximum(runtimes_ttn)

    @show runtimes_mps
    scatter!(ax_err, mean_num_size_ttn ./ 1e5, error_ttn;
      color=round.(runtimes_ttn),
      colormap=default_colorscheme(),
      colorrange=(20, 640),
      # colorscale=log10,
      marker=:circle,
      markersize=8,
      strokecolor=:black,
      strokewidth=0.8,
      label=row_idx == 1 ? "TTN" : nothing,
    )
    scatter!(ax_err, mean_num_size_mps ./ 1e5, error_mps;
      color=round.(runtimes_mps),
      colormap=default_colorscheme(),
      colorrange=(20, 640),
      # colorscale=log10,
      marker=:rect,
      markersize=8,
      strokecolor=:black,
      strokewidth=0.8,
      label=row_idx == 1 ? "MPS" : nothing,
    )
    Colorbar(fig[row_idx, 3],
      colormap=default_colorscheme(),
      # scale=log10,
      limits=(20, 640),
      label=L"t_{\mathrm{ex}}",
    )
    if row_idx == 1
      axislegend(ax_err, position=:lt)
    end

    combined_errors = filter(!iszero, vcat(error_mps, error_ttn))
    if !isempty(combined_errors)
      positive_errors = filter(>(0), combined_errors)
      @show positive_errors
      if !isempty(positive_errors)
        if row_idx == 1
          min_exp, max_exp = -3, -1
        else
          min_exp, max_exp = -4, -2
        end
        major_ticks = 10.0 .^ (min_exp:max_exp)
        @show major_ticks
        major_labels = [L"10^{%$k}" for k in min_exp:max_exp]
        ylims!(ax_err, major_ticks[1], major_ticks[end] * 1.5)
        ax_err.yticks = (major_ticks, major_labels)
        ax_err.yminorticks = IntervalsBetween(9)
        ax_err.yminorticksvisible = true
      end
    end
  end

  !isempty(axes_time) && linkxaxes!(axes_time...)
  !isempty(axes_error) && linkxaxes!(axes_error...)

  display(fig)
  plots_dir = joinpath("plots", "paper")
  mkpath(plots_dir)
  save(joinpath(plots_dir, "L12_plot.pdf"), fig)
end

function plot_L12_all_imbalances(df_in::DataFrame; h_values::AbstractVector, gridnum=nothing)
  palette = ColorSchemes.Zissou1Continuous.colors
  palette = palette[[1, 4, 8]]

  fig_height = 400 * length(h_values)
  fig = Figure(size=(600, fig_height))

  axes = Axis[]

  for (row_idx, h) in enumerate(h_values)
    df = filter(row -> row.model_h == h && (gridnum === nothing || row.graph_gridnum == gridnum), df_in)
    df_mps = filter(row -> row.graph_type == "SnakeGraph", df)
    df_ttn = filter(row -> row.graph_type == "FreeGraph", df)
    display(df_ttn)

    dims = sort!(unique(vcat(df_mps.initial_state_initial_maxdim, df_ttn.initial_state_initial_maxdim)))
    isempty(dims) && error("No data available for h=$h and gridnum=$gridnum")

    ax = Axis(fig[row_idx, 1], ylabel=L"I(t)", xticks=LinearTicks(6))
    xlims!(ax, (0, 100))
    if row_idx == length(h_values)
      ax.xlabel = L"t"
    else
      hidexdecorations!(ax; grid=false)
    end

    for (i, maxdim) in enumerate(dims)
      color = palette[mod1(i, length(palette))]
      mps_matches = filter(r -> r.initial_state_initial_maxdim == maxdim, df_mps)
      ttn_matches = filter(r -> r.initial_state_initial_maxdim == maxdim, df_ttn)

      isempty(ttn_matches) && continue
      ttn = only(ttn_matches)
      lines!(ax, ttn.times, ttn.imbalance,
        label=L"\mathrm{TTN},\; \chi = %$maxdim", color=color)
      isempty(mps_matches) && continue
      mps = only(mps_matches)
      lines!(ax, mps.times, mps.imbalance,
        label=L"\mathrm{MPS},\; \chi = %$maxdim", color=color, linestyle=:dash)
    end

    axislegend(ax;
      position=:rt, orientation=:vertical, framevisible=false,
      padding=(6, 6, 6, 6))

    text!(ax, 0.05, 0.95, text=L"$L = 12$, $h = %$h$", space=:relative,
      align=(:left, :top), color=:black, fontsize=14)

    push!(axes, ax)
  end

  linkxaxes!(axes...)

  save(joinpath("plots", "paper", "L12_plot_all_imb.pdf"), fig)
  display(fig)
end

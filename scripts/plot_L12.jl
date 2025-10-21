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

function plot_L12_comparison(
    df::DataFrame;
    h_values::NTuple{2, Real},
    gridnum=nothing,
    graph_types::Tuple{String,String}=("MPS", "TTN"),
    target_dims::Dict{String,Int},
    reference_dims::Dict{String,Int},
    scale_factor::Real=1e5,
)
  palette = ColorSchemes.Zissou1Continuous.colors[[2, 9]]
  label_map = Dict(
    "MPS" => "MPS",
    "TTN" => "TTN",
  )
  graph_alias = Dict(
    "MPS" => "SnakeGraph",
    "SnakeGraph" => "SnakeGraph",
    "TTN" => "FreeGraph",
    "FreeGraph" => "FreeGraph",
  )

  label_for(name) = get(label_map, name, name)

  function resolve_dim(dict::Dict{String,Int}, key::String, resolved::String)
    if haskey(dict, key)
      return dict[key]
    elseif haskey(dict, resolved)
      return dict[resolved]
    else
      error("Missing χ entry for $(key) (graph type $(resolved))")
    end
  end

  function select_run(rows::DataFrame, χ::Int, context::String)
    matches = filter(row -> row.initial_state_initial_maxdim == χ, rows)
    isempty(matches) && error("Missing $(context) with χ = $(χ)")
    return only(matches)
  end

  function accuracy_against_reference(run::DataFrameRow, reference::DataFrameRow)
    return mean(abs.(reference.imbalance .- run.imbalance))
  end

  fig = Figure(size=(900, 450), fontsize=10)
  imbalance_axes = Axis[]

  for (row_idx, h) in enumerate(h_values)
    df_h = filter(row -> row.model_h == h && (gridnum === nothing || row.graph_gridnum == gridnum), df)
    isempty(df_h) && error("No rows for h = $(h), gridnum = $(gridnum)")

    for (col_idx, alias) in enumerate(graph_types)
      graph_type = get(graph_alias, alias) do
        error("Unknown graph identifier $(alias)")
      end
      df_type = filter(row -> row.graph_type == graph_type, df_h)
      isempty(df_type) && error("No rows for graph_type = $(graph_type), h = $(h)")

      target_dim = resolve_dim(target_dims, alias, graph_type)
      reference_dim = resolve_dim(reference_dims, alias, graph_type)

      target_run = select_run(df_type, target_dim, "$(alias) target")
      reference_run = select_run(df_type, reference_dim, "$(alias) reference")

      ax = Axis(fig[row_idx, col_idx],
        ylabel=row_idx == 1 ? L"I(t)" : "",
        xticks=LinearTicks(7),
        limits=((0, 100), nothing))
      row_idx == length(h_values) || hidexdecorations!(ax; grid=false)
      ax.xlabel = row_idx == length(h_values) ? L"t" : ""

      friendly = label_for(alias)
      lines!(ax, target_run.times, target_run.imbalance;
        color=palette[col_idx],
        linewidth=1.8,
        label="$(friendly), χ = $(target_run.initial_state_initial_maxdim)")

      text!(ax, 0.05, 0.90, text=L"h = %$h", space=:relative,
        align=(:left, :top))

      if row_idx == 1
        Label(fig[row_idx, col_idx], friendly; tellwidth=false, tellheight=false, padding=(0, 0, 4, 0))
      end

      push!(imbalance_axes, ax)

      if row_idx == 1 && col_idx == length(graph_types)
        axislegend(ax; position=:rt, framevisible=false)
      end
    end

    acc_col = length(graph_types) + 1
    ax_acc = Axis(fig[row_idx, acc_col],
      xscale=log10,
      yscale=log10,
      xlabel=row_idx == length(h_values) ? L"N_{\mathrm{par}} / 10^{5}" : "",
      ylabel=row_idx == 1 ? L"\varepsilon" : "")
    row_idx == length(h_values) || hidexdecorations!(ax_acc; grid=false)

    for (col_idx, alias) in enumerate(graph_types)
      graph_type = graph_alias[alias]
      df_type = filter(row -> row.graph_type == graph_type, df_h)
      reference_dim = resolve_dim(reference_dims, alias, graph_type)
      reference_run = select_run(df_type, reference_dim, "$(alias) reference")
      dims = sort(unique(df_type.initial_state_initial_maxdim))
      dims = filter(χ -> χ != reference_dim, dims)

      errors = Float64[]
      params = Float64[]

      for χ in dims
        run = select_run(df_type, χ, "$(alias) χ=$(χ)")
        push!(errors, accuracy_against_reference(run, reference_run) + 1e-16)
        push!(params, minimum(run.num_size) / scale_factor)
      end

      isempty(params) && continue
      order = sortperm(params)
      color = palette[col_idx]
      friendly = label_for(alias)
      lines!(ax_acc, params[order], errors[order]; color=color, linewidth=1.5)
      marker = col_idx == 1 ? :circle : :diamond
      scatter!(ax_acc, params[order], errors[order]; color=color, marker=marker,
        label=friendly)
    end

    if row_idx == 1
      axislegend(ax_acc; position=:rt, framevisible=false)
    end
  end

  linkxaxes!(imbalance_axes...)
  display(fig)
  plots_dir = joinpath("plots", "paper")
  mkpath(plots_dir)
  save(joinpath(plots_dir, "L12_plot.pdf"), fig)
end

function plot_L12_all_imbalances(df_in::DataFrame; h_values::AbstractVector, gridnum=nothing)
  palette = ColorSchemes.Zissou1Continuous.colors
  palette = palette[[1, 4, 8, 11]]

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

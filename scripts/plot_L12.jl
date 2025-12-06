using CairoMakie
using CairoMakie: hidexdecorations!, linkxaxes!, linkyaxes!, LinearTicks, LineElement, MarkerElement, IntervalsBetween, Legend
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
  palette = ColorSchemes.Zissou1Continuous.colors[[1, 4, 8, 11]]

  mps = filter(x -> x.graph_type == "SnakeGraph" && x.graph_gridnum == grid, df)
  ttn = filter(x -> x.graph_type == "FreeGraph" && x.graph_gridnum == grid, df)

  fig = Figure(size=(600, length(h_values) * 150), fontsize=9)
  panel_letters = [Char('a' + mod(idx - 1, 26)) for idx in 1:(2*length(h_values))]
  panel_idx = 1
  axes_time = Axis[]
  axes_error = Axis[]
  labeled_dims = Set{Int}()
  maxdim_elements = LineElement[]
  maxdim_labels = LaTeXString[]

  for (row_idx, h) in enumerate(h_values)
    h = Int(h)
    panel_letter = panel_letters[panel_idx]
    panel_idx += 1
    ax = Axis(fig[row_idx, 1];
      title=L"(%$(panel_letter))\enspace h=%$h",
      titlealign=:left,
      titlesize=9,
    )
    push!(axes_time, ax)
    ax.ylabel = L"I(t)"
    if row_idx == length(h_values)
      ax.xlabel = L"t"
    else
      hidexdecorations!(ax, grid=false)
    end
    h_mps = sort(filter(x -> x.model_h == h, mps), :initial_state_initial_maxdim)
    h_ttn = sort(filter(x -> x.model_h == h, ttn), :initial_state_initial_maxdim)
    for (series_idx, (m, t)) in enumerate(zip(eachrow(h_mps), eachrow(h_ttn)))
      dim = m.initial_state_initial_maxdim
      color = palette[series_idx]
      dim_label = (row_idx == 1 && !(dim in labeled_dims)) ? L"\chi = %$dim" : nothing
      if dim_label !== nothing
        push!(labeled_dims, dim)
        push!(maxdim_elements, LineElement(color=color, linestyle=:solid, linewidth=1))
        push!(maxdim_labels, dim_label)
      end
      lines!(ax, m.times, m.imbalance, color=color, linewidth=1, linestyle=:dash)
      lines!(ax, t.times, t.imbalance, color=color, linewidth=1, label=dim_label)
    end
    if row_idx == 1
      legend_grid = GridLayout(tellwidth=false, tellheight=true)
      fig[row_idx, 1, Top()] = legend_grid

      Legend(legend_grid[1, 1],
        [LineElement(color=:black, linestyle=:solid, linewidth=1),
          LineElement(color=:black, linestyle=:dash, linewidth=1)],
        ["TTN", "MPS"];
        orientation=:vertical, framevisible=false, padding=(0, 0, -40, 0),
        labelsize=8, patchsize=(12, 8), halign=:center)

      if !isempty(maxdim_elements)
        Legend(legend_grid[1, 2],
          maxdim_elements, maxdim_labels;
          orientation=:vertical, framevisible=false, padding=(0, 0, -60, 0),
          labelsize=8, patchsize=(12, 8), halign=:center)
      end
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

    panel_letter = panel_letters[panel_idx]
    panel_idx += 1
    ax_err = Axis(fig[row_idx, 2];
      yscale=log10,
      title=L"(%$(panel_letter))\enspace h=%$h",
      titlealign=:left,
      titlesize=9,
    )
    push!(axes_error, ax_err)
    ax_err.ylabel = L"|I_{\chi_{\max}} - I_{\chi}|"
    if row_idx == length(h_values)
      ax_err.xlabel = L"Number of parameters/$10^5$"
    else
      hidexdecorations!(ax_err, grid=false)
    end

    lines!(ax_err, mean_num_size_mps ./ 1e5, error_mps; color=:black, linewidth=1.0, transparency=true, alpha=0.5)
    lines!(ax_err, mean_num_size_ttn ./ 1e5, error_ttn; color=:black, linewidth=1.0, transparency=true, alpha=0.5)

    colorbarmax = 400
    scatter!(ax_err, mean_num_size_ttn ./ 1e5, error_ttn;
      color=round.(runtimes_ttn),
      colormap=default_colorscheme(),
      colorrange=(20, colorbarmax),
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
      colorrange=(20, colorbarmax),
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
      limits=(20, colorbarmax),
      label="Execution time (s)",
    )
    if row_idx == 1
      legend_grid_err = GridLayout(tellwidth=false, tellheight=true)
      fig[row_idx, 2, Top()] = legend_grid_err

      Legend(legend_grid_err[1, 1],
        [MarkerElement(marker=:circle, markersize=6, color=:black, strokecolor=:black, strokewidth=0.8),
          MarkerElement(marker=:rect, markersize=6, color=:black, strokecolor=:black, strokewidth=0.8)],
        ["TTN", "MPS"];
        orientation=:vertical, framevisible=false, padding=(0, 0, -40, 0),
        labelsize=8, patchsize=(12, 8), halign=:center)
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

  colsize!(fig.layout, 1, Relative(0.6))
  colsize!(fig.layout, 2, Relative(0.4))

  display(fig)
  plots_dir = joinpath("plots", "paper")
  mkpath(plots_dir)
  save(joinpath(plots_dir, "L12_plot.pdf"), fig)
end

function plot_L12_all_imbalances(df_in::DataFrame; h_values::AbstractVector, gridnum=nothing)
  palette = ColorSchemes.Zissou1Continuous.colors
  palette = palette[[1, 4, 8, 11]]

  fig_height = 400 * length(h_values)
  fig = Figure(size=(700, fig_height))

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
      align=(:left, :top), color=:black, fontsize=9)

    push!(axes, ax)
  end

  linkxaxes!(axes...)

  save(joinpath("plots", "paper", "L12_plot_all_imb.pdf"), fig)
  display(fig)
end

function load_ttn_32(
  dir::AbstractString;
  gridnums::Union{AbstractVector{<:Integer},Integer},
  h_fields::Union{AbstractVector{<:Real},Real},
  L::Int=12,
  D::Union{Nothing,AbstractVector{<:Integer},Integer}=32,
)
  isdir(dir) || error("Directory not found: $(dir)")

  files = filter(f -> endswith(f, ".jld2"), readdir(dir))
  isempty(files) && return DataFrame()

  grid_vec = gridnums isa AbstractVector ? collect(gridnums) : [gridnums]
  h_vec = h_fields isa AbstractVector ? collect(h_fields) : [h_fields]
  D_vec = D === nothing ? nothing : (D isa AbstractVector ? collect(D) : [D])

  matched_files = String[]
  for gridnum in grid_vec
    for h_field in h_vec
      h_string = @sprintf("%.2f", h_field)
      prefix = "free_L=$(L)_gridnum=$(gridnum)_h=$(h_string)"
      subset = filter(files) do file
        occursin(prefix, file) &&
          (D_vec === nothing || any(d -> occursin("_D=$(d)_", file), D_vec))
      end
      append!(matched_files, subset)
    end
  end

  unique!(matched_files)

  if isempty(matched_files)
    @warn "No files matched requested parameters in $(dir)"
    return DataFrame()
  end

  loaded_results = Vector{Any}()
  for file in matched_files
    filepath = joinpath(dir, file)
    try
      push!(loaded_results, load(filepath)["results"])
    catch err
      @warn "Failed to load file $(filepath)" exception = (err, catch_backtrace())
    end
  end

  isempty(loaded_results) && return DataFrame()

  df = DataFrame()
  for result in loaded_results
    flat = merge(result["parameters"], result["observables"])
    push!(df, flat, cols=:union)
  end

  df.imbalance = TTNEvo.columnar_imbalance_total.(df.sz)
  select!(df, Not([:maxdim, :sz]))
  return df
end

function load_L12_new()
  df12 = load_choose_dirs(; remove_small_time=false, load_from="L12-new")
  df = load_ttn_32("data/L12-column-tree-beta-pre-det/"; gridnums=[2, 3], h_fields=[10.0, 20.0, 30.0])
  return vcat(df, df12)
end

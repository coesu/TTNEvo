
using DataFrames
using JLD2
using TTNEvo

function load_beta_data(; remove_small_time=true, kwargs...)
  dirs = choose_dirs(; kwargs...)
  df = DataFrame()

  failed_files = []

  wanted = [
    "times",
    "graph_L",
    "ex_times",
    "model_h",
    "initial_state_initial_maxdim",
    "graph_type",
    "num_size",
    "graph_gridnum",
  ]

  for dir in dirs
    for file in readdir(dir)
      filepath = joinpath(dir, file)
      try
        jldopen(filepath) do f
          d = f["results"]
          flat_dict = merge(d["parameters"], d["observables"])
          selected = Dict(k => get(flat_dict, k, missing) for k in wanted)

          imbalance = try
            if haskey(flat_dict, "sz")
              TTNEvo.columnar_imbalance_total(flat_dict["sz"])
            else
              missing
            end
          catch err
            @warn "Failed to compute columnar imbalance" exception = (err, catch_backtrace()) filepath
            missing
          end

          selected["imbalance"] = imbalance

          push!(df, selected, cols=:union)
        end
      catch e
        if isa(e, JLD2.InvalidDataException)
          push!(failed_files, filepath)
          @warn "Could not load file '$filepath'"
        else
          rethrow(e)
        end
      end
    end
  end

  return remove_small_time_simulations(df; t_end=100.0)
end

function save_df(df)
  df = df[:, [
    "times",
    "imbalance",
    "ex_times",
    "num_size",
    "model_h",
    "initial_state_initial_maxdim",
    "graph_type",
    "graph_gridnum",
    "graph_L",
  ]]
  return df

end

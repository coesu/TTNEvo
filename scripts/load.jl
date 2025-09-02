using JLD2
using Dates # Import the Dates library
using Base.Threads # For multi-threading

"""
    load_dir(dir; time_after=nothing)

Loads JLD2 files from a directory in parallel, optionally filtering for files created after a specific date and time.

The function assumes filenames contain a timestamp in the format `YYYYMMDD-H-M-S`,
like in `..._20250626-9-43-34_results.jld2`.

# Arguments
- `dir::String`: The path to the directory to scan.

# Keyword Arguments
- `time_after::Union{DateTime, Nothing}`: If provided, only files with a timestamp
  in their name later than this `DateTime` will be loaded. Defaults to `nothing`,
  which loads all files.

# Returns
- `Vector{TreeConfig}`: A vector containing the "results" object from each loaded file.
"""
function load_dir(dir; time_after::Union{DateTime,Nothing}=nothing, starts_with=nothing, contain=nothing)
  entries = readdir(dir)

  regex = r"_(\d{8}-\d+-\d+-\d+)_"
  date_format = dateformat"yyyymmdd-H-M-S"

  jld2_files = String[]
  println("Scanning directory '$dir' for .jld2 files...")
  if !isnothing(time_after)
    println("Filtering for files with a timestamp after: $time_after")
  end

  for entry in entries
    if !isnothing(contain)
      if !contains(entry, contain)
        continue
      end
    end
    if !isnothing(starts_with)
      if !startswith(entry, starts_with)
        continue
      end
    end
    full_path = joinpath(dir, entry)
    if isfile(full_path) && endswith(entry, ".jld2")
      # 2. FILTERING LOGIC
      if isnothing(time_after)
        # If no filter is set, add all .jld2 files
        push!(jld2_files, full_path)
      else
        # If a filter is set, try to match the date in the filename
        datestring_match = match(regex, entry)

        if !isnothing(datestring_match)
          try
            # Extract the captured string (e.g., "20250626-9-43-34")
            datestring = first(datestring_match.captures)
            # Parse it into a DateTime object
            file_datetime = DateTime(datestring, date_format)

            # Compare it with the filter time
            if file_datetime > time_after
              push!(jld2_files, full_path)
            end
          catch e
            @warn "Could not parse date from filename '$entry'. Skipping. Error: $e"
          end
        end
      end
    end
  end

  println("\nFound $(length(jld2_files)) files to load.")

  # Pre-allocate a vector to store results from each thread.
  # It can hold either a TreeConfig object or `nothing` if loading fails.
  # results = Vector{Union{TreeConfig,Nothing}}(nothing, length(jld2_files))
  results = Vector{Any}(nothing, length(jld2_files))

  # Use multi-threading to load files in parallel.
  # Each thread will work on an element of the jld2_files array.
  @threads for i in 1:length(jld2_files)
    filepath = jld2_files[i]
    results[i] = load(filepath)["results"]
    # try
    #   # Load the file and store its "results" in the pre-allocated vector.
    #   results[i] = load(filepath)["results"]
    #   println("- Loaded: $(basename(filepath)) on thread $(threadid())")
    # catch e
    #   @warn "Could not load file '$filepath' on thread $(threadid()): $e"
    #   # In case of an error, results[i] remains `nothing`.
    # end
  end

  # Filter out the `nothing` values from failed loads.
  # The resulting vector will be of type Vector{TreeConfig}.
  loaded_data = filter(!isnothing, results)

  println("\nFinished loading. Loaded $(length(loaded_data)) files successfully.")
  return loaded_data
end

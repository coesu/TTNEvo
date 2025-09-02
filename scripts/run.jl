using TTNEvo
using Random
try
  using MKL
  using LinearAlgebra
  @show BLAS.get_num_threads()
  @show BLAS.get_config()
  println("Using MKL")
catch
  println("Loading MKL was not successful")
end

try
  config_file_path = joinpath(@__DIR__, "../configs", ARGS[1], "config.jl")
  include(config_file_path)
catch
  config_file_path = joinpath(@__DIR__, "../src", "config.jl")
  include(config_file_path)
end

runs_per_job = try
  parse(Int, ARGS[2])
catch
  1
end

array_id = try
  id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
  start = runs_per_job * (id - 1) + 1
  e = start + runs_per_job - 1
  collect(start:e)
catch
  [i for i in 1:total_combinations]
end

@show array_id

for idd in array_id
  current_config = parameter_sets[idd]
  @show idd
  @show current_config
  println("Starting simulation for configuration ID: $idd")
  TTNEvo.run_simulation(current_config; check_for_previous_run=true)
end

println("All simulations complete.")

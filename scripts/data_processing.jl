using TTNEvo
using NamedGraphs
using ITensorNetworks
using Statistics
using HDF5
using Graphs
using JLD2

function remove_tree_struct_and_sz_array(datas; save_dir=nothing)
  for data in datas
    for config in data
      # maxdim = config.observer.maxdim
      # @show maxdim
      # config.observer.maxdim = [maxdim[i] for i in 1:50:length(maxdim)]
      keep_every_nth!(config.observer.maxdim, 50)
      for (i, sz) in enumerate(config.observer.sz)
        config.observer.sz[i] = sz_dict_to_array(sz)
      end
      if !isnothing(save_dir)
        TTNEvo.save_simulation_data(joinpath("data/preprocessed/", save_dir), config)
      else
        TTNEvo.save_simulation_data(joinpath("data/preprocessed/", config.save_dir), config)
      end
    end
  end
end

function keep_every_nth!(v::Vector, n)
  j = 1
  for i in 1:n:length(v)
    v[j] = v[i]
    j += 1
  end
  resize!(v, j - 1)
end

function sz_dict_to_array(dict_sz)
  L = Int(sqrt(length(dict_sz)))
  sz = zeros(Float64, L, L)
  if length(first(keys(dict_sz))) == 3
    for k in keys(dict_sz)
      sz[k[2:3]...] = real(dict_sz[k][2])
    end
  else
    for k in keys(dict_sz)
      sz[k[1], k[2]] = real(dict_sz[k])
    end
  end
  return sz
end

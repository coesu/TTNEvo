using TensorOperations
using OMEinsumContractionOrders
using ITensors
using ITensorNetworks

function get_peak_memory(tensors::Vector{ITensor}, sequence)
  memo = Dict{Any,Any}()
  for i in 1:length(tensors)
    memo[i] = (inds(tensors[i]), dim(tensors[i]))
  end

  peak_mem = Ref(0)
  for i in 1:length(tensors)
    if dim(tensors[i]) > peak_mem[]
      peak_mem[] = dim(tensors[i])
    end
  end

  function recurse(sub_sequence)
    if haskey(memo, sub_sequence)
      return memo[sub_sequence]
    end

    if isa(sub_sequence, Int)
      return memo[sub_sequence]
    end

    left_inds, _ = recurse(sub_sequence[1])
    right_inds, _ = recurse(sub_sequence[2])

    result_inds = Tuple(symdiff(collect(left_inds), collect(right_inds)))

    size_of_result = isempty(result_inds) ? 1 : dim(result_inds)

    if size_of_result > peak_mem[]
      peak_mem[] = size_of_result
    end

    memo[sub_sequence] = (result_inds, size_of_result)
    return result_inds, size_of_result
  end

  recurse(sequence)

  return peak_mem[]
end


function bench_contractions()

  N = 10000

  kwargs = [
    (; alg="greedy"),
    # (; alg="kahypar_bipartite", sc_target=30, max_group_size=4),
    # (; alg="kahypar_bipartite", sc_target=28, max_group_size=4, sub_optimizer=TreeSA(sc_weight=0.0)),
    (; alg="optimal"),
    (; alg="tree_sa", sc_weight=2.0, sc_target=45)
  ]

  memory = zeros((N, length(kwargs)))
  flops = zeros((N, length(kwargs)))

  for i in 1:N
    l_low = 100
    l_high = 300
    h_low = 10
    h_high = 50

    l1 = Index(rand(l_low:l_high), "l1")
    l2 = Index(rand(l_low:l_high), "l2")
    l3 = Index(rand(l_low:l_high), "l3")

    lh1 = Index(rand(h_low:h_high), "lh1")
    lh2 = Index(rand(h_low:h_high), "lh2")
    lh3 = Index(rand(h_low:h_high), "lh3")

    h1 = Index(1, "h1")

    A = random_itensor(l1, l2, l3, h1)
    H = random_itensor(lh1, lh2, lh3, h1, prime(h1))

    Ap = prime(A)

    env1 = random_itensor(l1, prime(l1), lh1)
    env2 = random_itensor(l2, prime(l2), lh2)


    for (j, kw) in enumerate(kwargs)
      # println("_"^20)
      # println(kw[1])
      temp = [A, H, Ap, env1, env2]
      @time cs = ITensorNetworks.contraction_sequence(temp; kw...)
      # @show cs
      # @time log(sum(ITensors.contraction_cost(temp; sequence=cs)))
      # @show log(sum(ITensors.contraction_cost(temp; sequence=cs)))
      # @show get_peak_memory(temp, cs) * 8 / 1e9
      flops[i, j] = log(sum(ITensors.contraction_cost(temp; sequence=cs)))
      memory[i, j] = get_peak_memory(temp, cs) * 8 / 1e9
      # @time t = contract(temp; sequence=cs)
    end
  end
  return memory, flops
end




# println("_"^20)
# temp = [A, H, Ap, env1, env2]
# cs = [[[5, 2], [4, 1]], 3]
# cs = Any[1, Any[2, Any[4, Any[3, 5]]]]
# @show log(sum(ITensors.contraction_cost(temp; sequence=cs)))
# @show get_peak_memory(temp, cs)
# @show get_peak_memory(temp, cs) * 8 / 1e9



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

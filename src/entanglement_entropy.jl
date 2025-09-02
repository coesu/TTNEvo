using ITensorNetworks: uniqueinds, ortho_region
using ITensors: prime, replaceind!
using LinearAlgebra: NumberArray

export entanglement_entropy_ttn
export entanglement_entropy_sequence

function entanglement_entropy_sequence(ψ::AbstractTreeTensorNetwork; subsystem=nothing)
  L = Int(sqrt(nv(ψ)))

  if isnothing(subsystem)
    subsystem = if iseven(L)
      l = L ÷ 2
      Tuple.(CartesianIndices((l:(l+1), l:(l+1))))
    else
      l = L ÷ 2
      Tuple.(CartesianIndices((l:(l+2), l:(l+2))))
    end
  end
  subsystem = reduce(vcat, subsystem)

  ϕ = prime(deepcopy(ψ))
  to_contract = setdiff(vertices(ψ), subsystem)
  for v in to_contract
    i = uniqueind(ϕ, v)
    replaceind!(ϕ[v], i, noprime(i))
  end
  seq = contraction_sequence(ϕ ⊗ ψ; alg="tree_sa")

  return seq

end

function entanglement_entropy(ψ::AbstractTreeTensorNetwork; subsystem=nothing, sequence)
  return 0.0
  try
    L = Int(sqrt(nv(ψ)))
  catch
    @disable_warn_order return entanglement_entropy_free(ψ; subsystem, sequence)
  end
  L = Int(sqrt(nv(ψ)))

  if isnothing(subsystem)
    subsystem = if iseven(L)
      l = L ÷ 2
      Tuple.(CartesianIndices((l:(l+1), l:(l+1))))
    else
      l = L ÷ 2
      Tuple.(CartesianIndices((l:(l+2), l:(l+2))))
    end
  end
  subsystem = reduce(vcat, subsystem)

  ϕ = prime(deepcopy(ψ))
  to_contract = setdiff(vertices(ψ), subsystem)
  for v in to_contract
    i = uniqueind(ϕ, v)
    replaceind!(ϕ[v], i, noprime(i))
  end

  # @disable_warn_order ρ = contract(ϕ ⊗ ψ; sequence)
  # @time @disable_warn_order ρ = contract(ϕ ⊗ ψ; sequence=contraction_sequence(ϕ ⊗ ψ; alg="tree_sa"))
  @disable_warn_order ρ = contract(ϕ ⊗ ψ; sequence=contraction_sequence(ϕ ⊗ ψ; alg="greedy"))

  _, S, _ = svd(ρ, filter(i -> plev(i) == 1, inds(ρ)))
  Sv = diag(S)
  Sv = Sv[Sv.>1e-14] / sum(Sv)
  return -sum(p * log(p) for p in Sv)
end

function entanglement_entropy_free(ψ::AbstractTreeTensorNetwork; subsystem=nothing, sequence)
  L = Int(sqrt((nv(ψ) + 2) ÷ 2))

  if isnothing(subsystem)
    subsystem = if iseven(L)
      l = L ÷ 2
      [(1, v...) for v in Tuple.(CartesianIndices((l:(l+1), l:(l+1))))]
    else
      l = L ÷ 2
      [(1, v...) for v in Tuple.(CartesianIndices((l:(l+2), l:(l+2))))]
    end
  end
  subsystem = reduce(vcat, subsystem)

  ϕ = prime(deepcopy(ψ))
  to_contract = setdiff(vertices(ψ), subsystem)
  for v in to_contract
    i = uniqueind(ϕ, v)
    replaceind!(ϕ[v], i, noprime(i))
  end

  # @disable_warn_order ρ = contract(ϕ ⊗ ψ; sequence)
  # @disable_warn_order ρ = contract(ϕ ⊗ ψ; sequence=contraction_sequence(ϕ ⊗ ψ; alg="optimal"))
  # @time @disable_warn_order ρ = contract(ϕ ⊗ ψ; sequence=contraction_sequence(ϕ ⊗ ψ; alg="tree_sa"))
  @disable_warn_order ρ = contract(ϕ ⊗ ψ; sequence=contraction_sequence(ϕ ⊗ ψ; alg="greedy"))

  _, S, _ = svd(ρ, filter(i -> plev(i) == 1, inds(ρ)))
  Sv = diag(S)
  Sv = Sv[Sv.>1e-14] / sum(Sv)
  return -sum(p * log(p) for p in Sv)
end

function entanglement_entropy(S::Vector{Float64})
  S = S[S.>1e-14] / sqrt(sum(S .* S))
  return -sum(p^2 * log(p^2) for p in S)
end

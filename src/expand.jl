using ITensorNetworks: AbstractTTN, tree_orthogonalize, linkinds, tebd, random_ttn, siteinds
using ITensorNetworks
using ITensorNetworks.ITensorsExtensions: group_terms
using ITensors: Algorithm, unwrap_array_type, δ, scalartype, Trotter, commontags, svd
using ITensors: AlgorithmSelection
using Adapt: adapt

export expand

macro Algorithm_str(s)
  return :(Algorithm{$(Expr(:quote, Symbol(s)))})
end

function expand(state, reference; alg, kwargs...)
  return expand(Algorithm(alg), state, reference; kwargs...)
end
function expand(state, reference, g; alg, kwargs...)
  return expand(Algorithm(alg), state, reference, g; kwargs...)
end

function expand(
  ::Algorithm"orthogonalize",
  state::AbstractTTN,
  references::Vector{AbstractTTN};
  cutoff=1e-12,
)
  maxbond = max_linkdim(state)

  s = siteinds(state)
  L = sqrt(length(vertices(state)) / 2)

  sweeps = [
    pair for (pair,) in
    ITensorNetworks.forward_sweep(Base.Forward, state; region_kwargs=(), nsites=2)
  ]

  state = ITensorNetworks.tree_orthogonalize(state, sweeps[1, 1])
  references = map(reference -> tree_orthogonalize(reference, sweeps[1, 1]), references)

  for edge in sweeps
    (j, i) = edge

    linds = [s[i]; linkinds(state, i => j)]
    linds = [linkinds(state, i => j)]
    _, λⱼ, basisⱼ = svd(state[j], linds; righttags=tags(s[j]...))
    _, λⱼ, basisⱼ = svd(state[j], linds)
    rinds = uniqueinds(basisⱼ, λⱼ)

    idⱼ = prod(rinds) do r
      return adapt(unwrap_array_type(basisⱼ), denseblocks(δ(scalartype(state), r', dag(r))))
    end

    projectorⱼ = idⱼ - prime(basisⱼ, rinds) * dag(basisⱼ)

    ρⱼ = sum(reference -> prime(reference[j], rinds) * dag(reference[j]), references)
    ρⱼ /= tr(ρⱼ)
    # Apply projectorⱼ
    ρⱼ_projected = apply(apply(projectorⱼ, ρⱼ), projectorⱼ)
    expanded_basisⱼ = basisⱼ

    if norm(ρⱼ_projected) > 10^3 * eps(real(scalartype(state)))
      # Diagonalize projected density matrix ρⱼ_projected
      # to compute reference_basisⱼ, which spans part of right basis
      # of references which is orthogonal to right basis of state
      dⱼ, reference_basisⱼ = eigen(
        ρⱼ_projected; cutoff, ishermitian=true, righttags="bϕ_$j,Link"
      )
      state_indⱼ = only(commoninds(basisⱼ, λⱼ))
      reference_indⱼ = only(commoninds(reference_basisⱼ, dⱼ))
      expanded_basisⱼ, expanded_indⱼ = directsum(
        basisⱼ => state_indⱼ,
        reference_basisⱼ => reference_indⱼ;
        tags=tags(linkinds(state, i => j)...),
      )
    end

    state[i] = state[i] * (state[j] * dag(expanded_basisⱼ))
    state[j] = expanded_basisⱼ
    for reference in references
      reference[i] = reference[i] * (reference[j] * dag(expanded_basisⱼ))
      reference[j] = expanded_basisⱼ
    end
  end
  return state
end

function expand(
  ::Algorithm"global_krylov",
  state::AbstractTTN,
  operator::AbstractTTN,
  g;
  krylovdim=2,
  cutoff=1e-12,
  apply_kwargs=(),
)
  references = Vector{AbstractTTN}(undef, krylovdim)
  for k in 1:krylovdim
    previous_reference = get(references, k - 1, state)
    # @time references[k] = apply(
    #   operator, previous_reference; init=previous_reference, maxdim=16, alg=:fit
    # )
    @time references[k] = apply_tto(operator, previous_reference, g; maxdim=16)
    maxbond = max_linkdim(references[k])
  end
  return expand(state, references; alg="orthogonalize", cutoff)
end

function contract_tto(A::AbstractTTN, ψ::AbstractTTN, g; truncation=true, kwargs...)
  N = length(A)
  if N != length(ψ)
    @warn (DimensionMismatch("lengths of MPO ($N) and MPS ($(length(ψ))) do not match"))
  end
  ψ_out = ttn(g)

  for j in vertices(g)
    ψ_out[j] = A[j] * ψ[j]
  end

  for b in edges(g)
    Al = commoninds(A[src(b)], A[dst(b)])
    tt = tags(Al[1])
    ψl = commoninds(ψ[src(b)], ψ[dst(b)])
    l = [Al..., ψl...]
    if !isempty(l)
      C = combiner(l; tags=tt)
      ψ_out[src(b)] *= C
      ψ_out[dst(b)] *= dag(C)
    end
  end
  if truncation
    ψ_out = truncate(ψ_out; kwargs...)
  end
  return ψ_out
end

apply_tto(A, B, g; kwargs...) = replaceprime(contract_tto(A, B, g; kwargs...), 1 => 0)

function trivial_expand(state; expansion_dim=2, cutoff=10e-12, apply_kwargs=())
  s = siteinds(state)
  L = sqrt(length(vertices(state)) / 2)

  sweeps = [
    pair for (pair,) in
    ITensorNetworks.forward_sweep(Base.Forward, state; region_kwargs=(), nsites=2)
  ]

  state = ITensorNetworks.tree_orthogonalize(state, sweeps[1, 1])
  references = map(reference -> tree_orthogonalize(reference, sweeps[1, 1]), references)

  for edge in sweeps
    (j, i) = edge

    dⱼ, reference_basisⱼ = eigen(r; cutoff, ishermitian=true, righttags="bϕ_$j,Link")
    state_indⱼ = only(commoninds(basisⱼ, λⱼ))
    reference_indⱼ = only(commoninds(reference_basisⱼ, dⱼ))
    expanded_basisⱼ, expanded_indⱼ = directsum(
      basisⱼ => state_indⱼ,
      reference_basisⱼ => reference_indⱼ;
      tags=tags(linkinds(state, i => j)...),
    )

    state[i] = state[i] * (state[j] * dag(expanded_basisⱼ))
    state[j] = expanded_basisⱼ
    for reference in references
      reference[i] = reference[i] * (reference[j] * dag(expanded_basisⱼ))
      reference[j] = expanded_basisⱼ
    end
  end
  return state
end

function rand_ttn(sites, state; maxdim=50)
  println("starting rand_ttn")
  reference = copy(state)

  L = sqrt(nv(state) / 2)
  sweeps = [
    pair for (pair,) in ITensorNetworks.forward_sweep(
      Base.Forward, reference; root_vertex=(2, 2), region_kwargs=(), nsites=2
    )
  ]
  while max_linkdim(reference) < maxdim
    for edge in sweeps
      (j, i) = edge

      si = sites[i]
      sj = sites[j]
      G = randomU(si, sj)

      T = noprime(G * reference[i] * reference[j])
      rinds = uniqueinds(reference[i], reference[j])
      U, S, V = svd(T, rinds)
      u = commonind(U, S)
      l = tags(commonind(state[i], state[j]))

      new = settags(u, l)
      replaceind!(U, u, new)
      replaceind!(S, u, new)

      reference[i] = U
      reference[j] = S * V
      reference[j] /= norm(reference[j])
    end
    @show max_linkdim(reference)
  end
  return truncate(reference; maxdim)
end

function randomU(s1, s2)
  M = random_itensor(QN(), s1', s2', dag(s1), dag(s2))
  U, S, V = svd(M, (s1', s2'))
  u = commonind(U, S)
  v = commonind(S, V)
  replaceind!(U, u, v)
  G = U * V
  return G
end

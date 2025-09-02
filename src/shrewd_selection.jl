using ITensors
using Graphs
using NamedGraphs.GraphsExtensions: root_vertex

using NamedGraphs
using NamedGraphs.GraphsExtensions: incident_edges
using Graphs: grid
using ITensorNetworks:
  siteinds,
  tdvp,
  op,
  ttn,
  apply,
  inner,
  vertices,
  forward_sweep,
  direction,
  sub_time_steps,
  ProjTTN,
  position,
  set_ortho_region,
  projected_operator_tensors,
  TreeTensorNetwork,
  environment,
  on_edge,
  contract, linkdim
using ITensors
using KrylovKit: exponentiate

function get_rorth(R, W, B, Λ)
  Rtemp = noprime(contract([Λ, B, W, R...]; sequence=contraction_sequence([Λ, B, W, R...]; alg="optimal")))
  Rorth = Rtemp - B * (dag(B) * Rtemp)
  return Rorth
end

function get_lorth(L, W, A, U, S)
  Ltemp = noprime(
    contract([S, A, U, W, L...]; sequence=contraction_sequence([S, A, U, W, L...]; alg="optimal"))
  )
  Lorth = Ltemp - A * (dag(A) * Ltemp)
  return Lorth
end

export shrewd_selection

function check_left_orth(A, ref)
  return isapprox(
    A * prime(dag(A), commonind(A, ref)), delta(inds(A * prime(dag(A), commonind(A, ref))))
  )
end
function check_right_orth(B, ref)
  return isapprox(
    B * prime(dag(B), commonind(B, ref)), delta(inds(B * prime(dag(B), commonind(B, ref))))
  )
end

function get_corth(L, Wa, Wb, R, A, Λ, B, Apr)
  if isempty(L)
    to_contract = [prime(dag(Apr)), Wa, A]
  else
    to_contract = [prime(dag(Apr)), L..., Wa, A]
  end

  Lpr = noprime(contract(to_contract; sequence=contraction_sequence(to_contract; alg="optimal")))
  if isempty(R)
    to_contract = [Lpr, Λ, B, Wb]
  else
    to_contract = [Lpr, Λ, B, Wb, R...]
  end
  Ctemp = noprime(contract(to_contract; sequence=contraction_sequence(to_contract; alg="optimal")))
  Corth = Ctemp - (Ctemp * dag(B)) * B
  return Corth
end

function shrewd_selection(state, P, edge; D=100)
  state = orthogonalize(state, src(edge))
  A = state[dst(edge)]
  C = state[src(edge)]
  D = max(20, linkdim(state, edge))

  U, Λ, V = svd(
    C, commoninds(A, C); lefttags=tags(commonind(A, C)), righttags=tags(commonind(A, C))
  )

  A = A * U
  B = V

  # P = ProjTTN(operator)
  P = position(P, state, [src(edge), dst(edge)])

  envs, local_H = get_envs_and_local_ham(P)
  L = []
  R = []
  Wa = ITensor
  Wb = ITensor
  for env in envs
    if hascommoninds(A, env)
      push!(L, env)
    end
    if hascommoninds(B, env)
      push!(R, env)
    end
  end
  for H in local_H
    if hascommoninds(A, H)
      Wa = H
    end
    if hascommoninds(B, H)
      Wb = H
    end
  end

  Rorth = get_rorth(R, Wb, B, Λ)

  U, S, V = svd(Rorth, commonind(A, Rorth))

  Lorth = get_lorth(L, Wa, A, U, S)

  Dprime = round(0.5 * D)
  u, s, _ = svd(Lorth, uniqueinds(Lorth, commonind(S, V)); maxdim=Dprime)

  Uh, _, _ = svd(u * s, commoninds(u * s, A); cutoff=1e-14)

  Corth = get_corth(L, Wa, Wb, R, A, Λ, B, Uh)

  Ut, St, _ = svd(Corth, commoninds(Corth, Uh); maxdim=round(D * 0.2))

  Atr = Uh * Ut

  Aex, _ = directsum(
    A => commonind(A, Λ), Atr => commonind(Ut, St); tags=tags(commonind(A, Λ))
  )

  test = state[src(edge)] * state[dst(edge)]
  if isapprox(((state[src(edge)] * state[dst(edge)]) * dag(Aex)) * Aex, test)
    state[src(edge)] = (state[src(edge)] * state[dst(edge)]) * dag(Aex)
    state[dst(edge)] = Aex
    # else
    #   println("false")
  end

  return state
end

using ITensorNetworks:
  ttn,
  commoninds,
  tags,
  replacetags,
  svd,
  AbstractTTN,
  _TreeTensorNetwork,
  random_tensornetwork,
  contraction_sequence,
  ⊗,
  op,
  random_ttn,
  dmrg
using ITensors
using Graphs
using OMEinsumContractionOrders: OMEinsumContractionOrders
using NamedGraphs.GraphsExtensions: leaf_vertices, is_leaf_vertex, default_root_vertex
using Dictionaries
using Random

export infinite_temperature_state

function infinite_temperature_state(L, graph, sites)
  ψ = ttn(sites)

  for v in vertices(graph)
    if v[1] <= L
      w = v .+ (L, L)
      sv = sites[v...][1]
      sw = sites[w...][1]

      vind = inds(ψ[v...])
      wind = inds(ψ[w...])
      all_inds = union(vind, wind)

      com_ind = commoninds(ψ[v...], ψ[w...])[1]

      A_ind = filter(x -> x != com_ind, all_inds)

      a = 1 / sqrt(2)
      b = -1 / sqrt(2)
      A = ITensor(ComplexF64, A_ind...)

      A[[v => v == sv ? 2 : 1 for v in A_ind]...] = a
      A[[v => v == sw ? 2 : 1 for v in A_ind]...] = b

      U, S, V = svd(A, (filter(x -> x != com_ind, vind)..., sv); cutoff=1e-15)
      ψ[v...] = replacetags(U, "Link,u", tags(com_ind))
      ψ[w...] = replacetags(S * V, "Link,u", tags(com_ind))
    end
  end
  return ψ
end

export get_contraction_sequence

function get_contraction_sequence(sites, maxdim; ancilla=false)
  if ancilla
    psi = random_ttn(sites; link_space_physical=maxdim, link_space_ancilla=2)
  else
    psi = random_ttn(sites; link_space=maxdim)
  end

  println("Z")
  psidag = sim(dag(psi); sites=[])
  Z = psidag ⊗ psi
  return contraction_sequence(Z; alg="tree_sa")
end

export expanded_infinite_temperature_state

function expanded_infinite_temperature_state(
  L, graph, graph_grid, sites, contract_sq; maxdim, qns
)
  psi = normalize(ttn(v -> v[1] > L ? "↑" : "↓", sites); contract_sq)
  # psi = normalize(ttn(v -> rand(["↑", "↓"]), sites); contract_sq)

  nsites = 1
  if qns
    # psi = rand_ttn(sites, psi; maxdim)
    nsites = 2
  else
    psi = random_ttn(sites; link_space_physical=maxdim, link_space_ancilla=2)
  end

  @show max_linkdim(psi)

  nsweeps = 10
  cutoff = 1e-15
  for (i, a) in enumerate([0.1, 0.01, 1e-12])
    # H = truncate(inf_temp_hamiltionian(graph, sites, a))
    H = inf_temp_hamiltionian(graph, graph_grid, sites, a)
    @show a
    @show maxdim
    println("starting $nsites site dmrg")
    flush(stdout)
    e, psi = dmrg(
      H,
      psi;
      nsweeps,
      # cutoff,
      maxdim,
      nsites,
      outputlevel=1,
      updater_kwargs=(; krylovdim=3, maxiter=1),
    )
    @show e
    @show max_linkdim(psi)
  end
  return psi
end

using NamedGraphs.GraphsExtensions: GraphsExtensions, post_order_dfs_vertices

function inf_temp_hamiltionian(graph, graph_grid, sites, a)
  os = OpSum()
  L = sqrt(nv(graph) / 2)

  for e in edges(graph)
    if !check_phys_edge(e, L)
      # os -= 1 / 4, "Id", src(e), "Id", dst(e)
      # os -= "Sz", src(e), "Sz", dst(e)
      os += "S+", src(e), "S-", dst(e)
      os += "S-", src(e), "S+", dst(e)
    end
    os += a, "Sz", src(e), "Sz", dst(e)
    os += a, "S+", src(e), "S-", dst(e)
    os += a, "S-", src(e), "S+", dst(e)
  end
  for e in edges(graph_grid)
    os += a, "Sz", src(e), "Sz", dst(e)
    os += a, "S+", src(e), "S-", dst(e)
    os += a, "S-", src(e), "S+", dst(e)
  end
  return ttn(os, sites)
end

function product_state_hamiltionian(graph, graph_grid, sites, state_string, a)
  os = OpSum()

  for e in edges(graph_grid)
    os += a, "Sz", src(e), "Sz", dst(e)
    os += a, "S+", src(e), "S-", dst(e)
    os += a, "S-", src(e), "S+", dst(e)
  end
  for v in vertices(graph)
    if state_string[v] == "↑"
      os -= "Sz", v
    else
      os += "Sz", v
    end
  end
  return ttn(os, sites)
end

function product_state_hamiltionian(graph, graph_grid, graph_type::FreeGraph, sites, state_string, a)
  os = OpSum()

  for e in edges(graph_grid)
    os += a, "Sz", src(e), "Sz", dst(e)
    os += a, "S+", src(e), "S-", dst(e)
    os += a, "S-", src(e), "S+", dst(e)
  end
  for v in vertices(graph_grid)
    if state_string[v] == "↑"
      os -= "Sz", v
    else
      os += "Sz", v
    end
  end
  return ttn(os, sites)
end

function bell_state_hamiltionian(sites)
  os = OpSum()

  os -= "Sz", (1, 1, 1)
  os -= "Sz", (1, 1, 2), "Sz", (1, 2, 1)
  os -= "Sx", (1, 1, 2), "Sx", (1, 2, 1)
  os += "Sz", (1, 2, 2)

  return ttn(os, sites)
end

export expanded_random_product_state
function expanded_random_product_state(
  L, graph, graph_grid, sites, contract_sq, state_string; maxdim, qns, nsweeps=3
)
  psi = normalize(ttn(v -> state_string[v], sites); contract_sq)
  println("expanded_random_product_state")
  @show state_string

  nsites = 1
  if qns
    nsites = 2
  else
    psi = random_ttn(sites; link_space=maxdim)
  end

  @show max_linkdim(psi)

  cutoff = 1e-15
  for (i, a) in enumerate([0.1, 0.01, 1e-12])
    H = product_state_hamiltionian(graph, graph_grid, sites, state_string, a)
    @show a
    @show maxdim
    println("starting $nsites site dmrg")
    flush(stdout)
    e, psi = dmrg(
      H,
      psi;
      nsweeps,
      # cutoff,
      maxdim,
      nsites,
      outputlevel=1,
      updater_kwargs=(; krylovdim=3, maxiter=1),
    )
    @show e
    @show max_linkdim(psi)
  end
  return psi
end

function expanded_random_product_state(
  L, graph, graph_grid, graph_type::FreeGraph, sites, contract_sq, state_string; maxdim, qns, nsweeps=3
)
  println("expanded_random_product_state")
  @show state_string

  nsites = 1
  if qns
    nsites = 2
  else
    psi = random_ttn(sites; link_space=maxdim)
  end

  @show max_linkdim(psi)

  cutoff = 1e-15
  for (i, a) in enumerate([0.1, 0.01, 1e-12])
    H = product_state_hamiltionian(graph, graph_grid, graph_type, sites, state_string, a)
    @show a
    @show maxdim
    println("starting $nsites site dmrg")
    flush(stdout)
    e, psi = dmrg(
      H,
      psi;
      nsweeps,
      # cutoff,
      maxdim,
      nsites,
      outputlevel=1,
      updater_kwargs=(; krylovdim=3, maxiter=1),
    )
    @show e
    @show max_linkdim(psi)
  end
  return psi
end

function L2_bell_state(sites; maxdim=2, qns=false)
  nsites = 1
  if qns
    nsites = 2
  else
    psi = random_ttn(sites; link_space=maxdim)
  end

  cutoff = 1e-15
  H = bell_state_hamiltionian(sites)
  e, psi = dmrg(
    H,
    psi;
    nsweeps=5,
    # cutoff,
    maxdim,
    nsites,
    outputlevel=1,
    updater_kwargs=(; krylovdim=3, maxiter=1),
  )
  return psi

end

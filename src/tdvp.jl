using ITensors
using Graphs
using NamedGraphs.GraphsExtensions: root_vertex

using NamedGraphs
using NamedGraphs.GraphsExtensions: incident_edges, post_order_dfs_edges
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
  orthogonalize, _truncate_edge
using ITensors
using LinearAlgebra: qr
using KrylovKit: exponentiate

function inserter(
  state::AbstractTTN,
  phi::ITensor,
  region;
  normalize=false,
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  internal_kwargs=nothing,
)
  state = copy(state)
  spec = nothing
  if length(region) == 2
    v = last(region)
    e = edgetype(state)(first(region), last(region))
    indsTe = inds(state[first(region)])
    L, phi, spec = factorize(phi, indsTe; tags=tags(state, e), maxdim, mindim, cutoff)
    state[first(region)] = L
  else
    v = only(region)
  end
  state[v] = phi
  state = set_ortho_region(state, [v])
  normalize && (state[v] /= norm(state[v]))
  return state, spec
end

function inserter(
  state::AbstractTTN,
  phi::ITensor,
  region::NamedEdge;
  cutoff=nothing,
  maxdim=nothing,
  mindim=nothing,
  normalize=false,
  internal_kwargs=nothing,
)
  state[dst(region)] *= phi
  state = set_ortho_region(state, [dst(region)])
  return state, nothing
end

function extracter(
  state, projected_operator, region; internal_kwargs=nothing, cutoff=nothing
)
  if isa(region, AbstractEdge)
    # TODO: add functionality for orthogonalizing onto a bond so that can be called instead
    vsrc, vdst = src(region), dst(region)
    state = orthogonalize(state, vsrc)
    left_inds = uniqueinds(state[vsrc], state[vdst])
    U, S, V = svd(
      state[vsrc],
      left_inds;
      lefttags=tags(state, region),
      righttags=tags(state, region),
      cutoff,
    )
    state[vsrc] = U
    local_tensor = S * V
  else
    state = orthogonalize(state, region)
    local_tensor = prod(state[v] for v in region)
  end
  projected_operator = position(projected_operator, state, region)
  return state, projected_operator, local_tensor
end

function exponentiate_updater(
  init;
  state!,
  projected_operator!,
  outputlevel=0,
  time_step,
  krylovdim=30,
  maxiter=100,
  verbosity=0,
  tol=1E-12,
  ishermitian=true,
  issymmetric=true,
  eager=true,
)
  result, exp_info = exponentiate(
    projected_operator![],
    time_step,
    init;
    krylovdim,
    maxiter,
    verbosity,
    tol,
    ishermitian,
    issymmetric,
    eager,
  )
  return result, (; info=exp_info)
end

export my_tdvp

function my_tdvp(
  operator,
  time_step,
  state;
  order=2,
  nsites=1,
  outputlevel=0,
  use_expansion=false,
  cutoff=nothing,
  maxdim=Inf,
)
  sweeps = []
  for (substep, fac) in enumerate(sub_time_steps(order))
    sub_time_step = time_step * fac
    append!(
      sweeps,
      forward_sweep(
        direction(substep),
        state;
        root_vertex=GraphsExtensions.default_root_vertex(state),
        nsites,
        reverse_step=true,
        region_kwargs=(sub_time_step),
        reverse_kwargs=(-sub_time_step),
      ),
    )
  end

  projected_operator = ProjTTN(operator)

  if use_expansion && max_linkdim(state) < maxdim
    expansion_sweep = forward_sweep(
      Base.Forward,
      state;
      root_vertex=GraphsExtensions.default_root_vertex(state),
      nsites=2,
      reverse_step=false,
      region_kwargs=(),
    )
    for (edge, region_kwargs) in expansion_sweep
      state = shrewd_selection(
        state, projected_operator, edgetype(state)(first(edge), last(edge))
      )
    end
    if max_linkdim(state) > maxdim
      state = truncate(state; maxdim)
    end
  end

  for s in sweeps
    region = s[1]


    state, projected_operator, phi = extracter(state, projected_operator, region)
    state! = Ref(state)
    projected_operator! = Ref(projected_operator)
    phi, info = exponentiate_updater(
      phi; state!, projected_operator!, verbosity=outputlevel, time_step=s[2]
    )
    state = state![]
    projected_operator = projected_operator![]
    state, spec = inserter(state, phi, region; maxdim, cutoff)
  end
  return state
end

function get_envs_and_local_ham(P::ProjTTN)
  environments = ITensor[environment(P, edge) for edge in incident_edges(P)]
  if on_edge(P)
    return environments
  else
    site_tensors = ITensor[]
    for s in P.pos
      push!(site_tensors, P.operator[s])
    end
    return environments, site_tensors
  end
end

function cominds(tensors::Vector{ITensor})
  inds = []
  for t in tensors
    for z in tensors
      if t != z
        union(inds, commoninds(t, z))
      end
    end
  end
  return inds
end

function init_omega(envs, p)
  is = vcat(
    filter(
      i -> plev(i) == 1,
      collect(Base.Iterators.flatten((inds(e) for e in envs if hascommoninds(e, p)))),
    ),
    filter(i -> plev(i) == 1, inds(p)),
  )
  k = Index(1; tags="k")
  # rng = Xoshiro(2)
  # return random_itensor(rng, k, is...)
  return random_itensor(k, is...)
end

export shrewd_selection

export adapt_bond_dimension

function adapt_bond_dimension(ψ; maxdim, maxtensorsize)
  total_error = 0.0
  for v in post_order_dfs_vertices(ψ, (1, 1, 1))
    if dim(ψ[v]) > maxtensorsize
      ψ = orthogonalize(ψ, v)
      ss = Vector{ITensor}()
      for n in neighbors(ψ, v)
        left_inds = uniqueinds(ψ, edgetype(ψ)(v => n))
        ltags = tags(ψ, v => n)
        U, S, V = svd(ψ[v], left_inds; lefttags=ltags)
        push!(ss, S)
      end
      x_opt = find_min_x(ss, maxtensorsize; x_low=1e-14)

      for n in neighbors(ψ, v)
        left_inds = uniqueinds(ψ, edgetype(ψ)(v => n))
        ltags = tags(ψ, v => n)
        # println()
        # U, S, V = svd(ψ[v], left_inds; lefttags=ltags)
        # @show x_opt
        # @show minimum(diag(S))
        # @show diag(S)[diag(S).>x_opt]
        U, S, V, spec = svd(ψ[v], left_inds; lefttags=ltags, cutoff=x_opt^2, use_absolute_cutoff=true)
        total_error += truncerror(spec)
        # @show diag(S) .> x_opt
        # @show minimum(diag(S))
        ψ[v] = U
        ψ[n] *= (S * V)

        # move back the orthoregion
        left_inds = uniqueinds(ψ, edgetype(ψ)(n => v))
        ltags = tags(ψ, n => v)
        U, S, V = svd(ψ[n], left_inds; lefttags=ltags)
        # Q, R = factorize(ψ[n], left_inds; ortho="left")
        # ψ[n] = Q
        # ψ[v] *= R

        ψ[n] = U
        ψ[v] *= (S * V)
      end
      # @show dim(ψ[v])
    end
  end
  println("adapt bond dim")
  @show total_error
  return ψ, total_error
end

function find_min_x(ss, maxtensorsize; x_low=1e-12, x_high=1.0, tol=1e-14)
  while x_high - x_low > tol
    x_mid = (x_low + x_high) / 2
    res = foldl(*, sum.(map(arr -> arr .> x_mid, array.(diag.(ss)))))
    if res < maxtensorsize
      x_high = x_mid
    else
      x_low = x_mid
    end
  end
  return x_high
end

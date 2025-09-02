using ITensorNetworks: AbstractTreeTensorNetwork

using TensorOperations: VectorInterface
abstract type AbstractObserver end
abstract type AbstractObservable end

Base.@kwdef mutable struct Observer <: AbstractObserver
  times::Vector{Float64} = Vector{Float64}()
  sz::Vector{Any} = Vector{Any}()
  ee::Vector{Float64} = Vector{Float64}()
  ex_times::Vector{Float64} = Vector{Float64}()
  maxdim::Vector{Pair{Float64,Any}} = Vector{Pair{Float64,Any}}()
  memory::Vector{Int} = Vector{Int}()
  num_size::Vector{Int} = Vector{Int}()
  errors::Vector{Pair{Float64,Any}} = Vector{Pair{Float64,Any}}()
end

function measure!(obs::Observer, t, ψ, H, ee_sequence, ex_time; errors=nothing, save_linkdims=false)
  ee = entanglement_entropy(ψ; sequence=ee_sequence)

  if length(first(vertices(ψ))) == 3
    sz = expect("Sz", ψ; vertices=[v for v in vertices(ψ) if v[1] == 1])
  else
    sz = expect("Sz", ψ)
  end

  L = maximum(v[2] for v in keys(sz))
  sz_array = zeros(Float64, L, L)
  for (k, v) in pairs(sz)
    sz_array[k[end-1], k[end]] = real(v)
  end

  push!(obs.sz, sz_array)
  push!(obs.ee, ee)
  push!(obs.times, t)
  push!(obs.ex_times, ex_time)
  if save_linkdims
    push!(obs.maxdim, t => linkdims(ψ))
  end
  push!(obs.memory, Base.summarysize(ψ))
  push!(obs.num_size, number_size(ψ))
  if !isnothing(errors)
    push!(obs.errors, t => errors)
  end
end

export imbalance

function imbalance(sz)
  diff = 0
  for v in keys(sz)
    if iseven(v[end] + v[end-1])
      diff += sz[v][2]
    else
      diff -= sz[v][2]
    end
  end
  return real(diff) * 2 / length(sz)
end

function columnar_imbalance(sz)
  diff = 0
  for v in keys(sz)
    if iseven(v[end-1])
      diff += sz[v][2]
    else
      diff -= sz[v][2]
    end
  end
  return real(diff) * 2 / length(sz)
end

export imbalance_snake
function imbalance_snake(sz)
  diff = 0
  for v in keys(sz)
    if iseven(v[end] + v[end-1])
      diff += sz[v]
    else
      diff -= sz[v]
    end
  end
  return real(diff) * 2 / length(sz)
end

export columnar_imbalance_snake
function columnar_imbalance_snake(sz)
  diff = 0
  for v in keys(sz)
    if iseven(v[end-1])
      diff += sz[v]
    else
      diff -= sz[v]
    end
  end
  return real(diff) * 2 / length(sz)
end

export columnar_imbalance
function columnar_imbalance(sz::Matrix{Float64})
  s = size(sz)
  diff = 0.0
  for i in 1:s[1]
    if iseven(i)
      diff += sum(sz[i, :])
    else
      diff -= sum(sz[i, :])
    end
  end
  return real(diff) * 2 / length(sz)
end

function columnar_imbalance_total(sz::Vector{Any})
  imb = Float64[]
  for szi in sz
    push!(imb, columnar_imbalance(szi))
  end
  return imb
end

function columnar_imbalance(sz::Union{Vector{ComplexF64},Vector{Float64}})
  diff = 0
  for (i, szi) in enumerate(sz)
    if iseven(i)
      diff += szi
    else
      diff -= szi
    end
  end
  return real(diff) * 2 / length(sz)
end

function columnar_imbalance_ed(sz_matrix, L)
  imbalances = Float64[]
  for t in 1:size(sz_matrix, 1)
    sz_t = sz_matrix[t, :]

    N = L * L
    @assert length(sz_t) == N

    sublattice_A = 0.0
    sublattice_B = 0.0

    for i in 1:N
      x = (i - 1) % L + 1
      y = div(i - 1, L) + 1

      if x % 2 == 0
        sublattice_A += sz_t[i]
      else
        sublattice_B += sz_t[i]
      end
    end

    imbalance = (sublattice_A - sublattice_B) / (0.5 * N)
    push!(imbalances, imbalance)
  end
  return imbalances
end


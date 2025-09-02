using ITensors.Ops: OpSum
using ITensorNetworks
using Dictionaries: AbstractDictionary
using Graphs

export HeisenbergModel
export XXModel

to_callable(value::Type) = value
to_callable(value::Function) = value
to_callable(value::AbstractDict) = Base.Fix1(getindex, value)
to_callable(value::AbstractDictionary) = Base.Fix1(getindex, value)
function to_callable(value::AbstractArray{<:Any,N}) where {N}
  getindex_value(x::Integer) = value[x]
  getindex_value(x::Tuple{Vararg{Integer,N}}) = value[x...]
  getindex_value(x::CartesianIndex{N}) = value[x]
  return getindex_value
end
to_callable(value) = Returns(value)

function nth_nearest_neighbors(g, v, n::Int)
  isone(n) && return neighborhood(g, v, 1)
  return setdiff(neighborhood(g, v, n), neighborhood(g, v, n - 1))
end

next_nearest_neighbors(g, v) = nth_nearest_neighbors(g, v, 2)

function check_phys_edge(edge, L)
  return src(edge)[1] <= L && dst(edge)[1] <= L
end
function check_phys_vertex(vertex, L)
  return vertex[1] <= L
end

abstract type AbstractModel end
export AbstractModel

Base.@kwdef struct HeisenbergModel <: AbstractModel
  J1::Union{Number,Function,AbstractDict,AbstractDictionary,AbstractArray} = 1.0
  JZ::Union{Number,Function,AbstractDict,AbstractDictionary,AbstractArray} = 1.0
  h::Union{Number,Function,AbstractDict,AbstractDictionary,AbstractArray} = 0.0
end

# Define the Heisenberg model with ancilla
Base.@kwdef struct HeisenbergAncillaModel <: AbstractModel
  J1::Union{Number,Function,AbstractDict,AbstractDictionary,AbstractArray} = 1.0
  J2::Union{Number,Function,AbstractDict,AbstractDictionary,AbstractArray} = 0.0
  h::Union{Number,Function,AbstractDict,AbstractDictionary,AbstractArray} = 0.0
  scale_J1::Bool = false
end

# XX interaction model
Base.@kwdef struct XXModel <: AbstractModel
  J::Union{Number,Function,AbstractDict,AbstractDictionary,AbstractArray} = 1.0
end

export build_hamiltonian

function build_hamiltonian(model::HeisenbergAncillaModel, g::AbstractGraph, field)
  (; J1, J2, h, scale_J1) = model
  h = field .* h * 2 .- h
  (; J1, J2, h) = map(to_callable, (; J1, J2, h))
  L = sqrt(nv(g) ÷ 2)
  ℋ = OpSum()
  for e in edges(g)
    if check_phys_edge(e, L)
      ℋ += J1(e) / 2, "S+", src(e), "S-", dst(e)
      ℋ += J1(e) / 2, "S-", src(e), "S+", dst(e)
      ℋ += J1(e), "Sz", src(e), "Sz", dst(e)
    end
  end
  for v in vertices(g)
    if check_phys_vertex(v, L)
      for nn in next_nearest_neighbors(g, v)
        e = edgetype(g)(v, nn)
        if check_phys_edge(e, L)
          ℋ += J2(e) / 2, "S+", src(e), "S-", dst(e)
          ℋ += J2(e) / 2, "S-", src(e), "S+", dst(e)
          ℋ += J2(e), "Sz", src(e), "Sz", dst(e)
        end
      end
    end
  end
  for v in vertices(g)
    if check_phys_vertex(v, L)
      ℋ += h(v), "Sz", v
    end
  end
  return ℋ
end

function build_hamiltonian(model::HeisenbergModel, g::AbstractGraph, field)
  (; J1, JZ, h) = model
  h = field .* h * 2 .- h
  (; J1, JZ, h) = map(to_callable, (; J1, JZ, h))
  ℋ = OpSum()
  for e in edges(g)
    ℋ += J1(e) / 2, "S+", src(e), "S-", dst(e)
    ℋ += J1(e) / 2, "S-", src(e), "S+", dst(e)
    ℋ += JZ(e), "Sz", src(e), "Sz", dst(e)
  end

  for v in vertices(g)
    if length(v) == 3
      ℋ += h(v[2:3]), "Sz", v
    else
      ℋ += h(v), "Sz", v
    end
  end
  return ℋ
end

function build_hamiltonian(model::XXModel, g::AbstractGraph)
  (; J) = model
  J_call = to_callable(J)

  ℋ = OpSum()
  for e in edges(g)
    ℋ += J_call(e) / 2, "S+", src(e), "S-", dst(e)
    ℋ += J_call(e) / 2, "S-", src(e), "S+", dst(e)
  end
  return ℋ
end

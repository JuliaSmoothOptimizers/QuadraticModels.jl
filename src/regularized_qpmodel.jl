mutable struct RegularizedQuadraticModel{T, S, M1, M2} <: AbstractQuadraticModel{T, S}
  model::QuadraticModel{T, S, M1, M2}
  σ::T
end

function Base.getproperty(obj::RegularizedQuadraticModel, sym::Symbol)
  if sym == :meta || sym == :counters || sym == :data
    return getproperty(obj.model, sym)
  else
    return getfield(obj, sym)
  end
end

# Constructors
function RegularizedQuadraticModel(
  c::S,
  H::Union{AbstractMatrix{T}, AbstractLinearOperator{T}};
  σ::T = zero(T),
  kwargs...,
) where {T, S}
  model = QuadraticModel(c, H; kwargs...)
  return RegularizedQuadraticModel(model, σ)
end

function RegularizedQuadraticModel(
  c::S,
  Hrows::AbstractVector{<:Integer},
  Hcols::AbstractVector{<:Integer},
  Hvals::S;
  σ::T = zero(T),
  kwargs...,
) where {T, S}
  model = QuadraticModel(c, Hrows, Hcols, Hvals; kwargs...)
  return RegularizedQuadraticModel(model, σ)
end

function RegularizedQuadraticModel(
  model::AbstractNLPModel{T, S}, 
  x::AbstractVector; 
  σ::T = zero(T),
  kwargs...
) where {T, S}
  model = QuadraticModel(model, x; kwargs...)
  return RegularizedQuadraticModel(model, σ)
end

# NLPModels API
function NLPModels.objgrad!(qp::RegularizedQuadraticModel, x::AbstractVector, g::AbstractVector)
  f, g = objgrad!(qp.model, x, g)
  iszero(qp.σ) && return f, g
  f += qp.σ*dot(x, x)/2
  @. g += qp.σ * x 
  return f, g
end

function NLPModels.obj(qp::RegularizedQuadraticModel, x::AbstractVector)
  f = obj(qp.model, x)
  iszero(qp.σ) && return f
  return f + qp.σ*dot(x, x)/2
end

function NLPModels.grad!(qp::RegularizedQuadraticModel, x::AbstractVector, g::AbstractVector)
  grad!(qp.model, x, g)
  iszero(qp.σ) && return g
  @. g += qp.σ * x
  return g
end
export RegularizedQuadraticModel

mutable struct RegularizedQuadraticModel{T, S, M1, M2} <: AbstractQuadraticModel{T, S}
  model::QuadraticModel{T, S, M1, M2}
  meta::NLPModelMeta{T, S}
  σ::T
  _diag_idx::Vector{Int}
end

function Base.getproperty(obj::RegularizedQuadraticModel, sym::Symbol)
  if sym == :counters || sym == :data
    return getproperty(obj.model, sym)
  else
    return getfield(obj, sym)
  end
end

# Constructors

function RegularizedQuadraticModel(
  model::QuadraticModel{T, S},
  σ::T
) where{T, S}

  isa(model.data.H, AbstractLinearOperator) && 
    return RegularizedQuadraticModel(model, model.meta, σ, Int[])

  # Update nnzh: reg_qp.meta.nnzh ≠ reg_qp.model.meta.nnzh
  nz_diag = 0
  @inbounds for i = 1:model.meta.nvar
    if model.data.H[i, i] == zero(T)
      nz_diag += 1
    end
  end
  meta = NLPModelMeta(model.meta, nnzh = model.meta.nnzh + nz_diag)
  return RegularizedQuadraticModel(model, meta, σ, zeros(Int, meta.nvar - nz_diag))
end

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

function NLPModels.hess_structure!(
  qp::RegularizedQuadraticModel{T, S, M1},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where{T, S, M1 <: AbstractMatrix{T}}
  nnz_H = qp.model.meta.nnzh
  @views hess_structure!(qp.model, rows[1:nnz_H], cols[1:nnz_H])

  k = nnz_H
  @inbounds for i = 1:qp.meta.nvar
    if qp.data.H[i,i] == zero(T) # Else, this entry has already been added by the previous hess_structure! call
      k += 1
      rows[k] = i
      cols[k] = i
    end
  end

  if qp._diag_idx[1] == 0 # This field has not been initialized yet
    k = 0
    @inbounds for i = 1:nnz_H
      if rows[i] == cols[i]
        k += 1
        qp._diag_idx[k] = i
      end
    end
  end

  return rows, cols
end

function NLPModels.hess_coord!(
  qp::RegularizedQuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where{T, S, M1 <: AbstractMatrix{T}}
  nnz_H = qp.model.meta.nnzh
  @views hess_coord!(qp.model, x, vals[1:nnz_H]; obj_weight = obj_weight)
  σ = obj_weight*qp.σ
  vals[nnz_H + 1:end] .= σ
  vals[qp._diag_idx] .+= σ

  return vals
end

function NLPModels.hprod!(
  qp::RegularizedQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  hprod!(qp.model, x, v, Hv, obj_weight = obj_weight)
  @. Hv += qp.σ.obj_weight*v
  return Hv
end

for fname in (
  :jac_lin_structure!,
  :jac_lin_coord!,
  :jac_lin,
  :cons_lin!,
  :jprod_lin!,
  :jtprod!,
  :jtprod_lin!
)
  @eval begin
    NLPModels.$fname(
      qp::RegularizedQuadraticModel,
      args...
    ) = $fname(qp.model, args...)
  end
end
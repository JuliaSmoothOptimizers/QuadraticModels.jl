export RegularizedQuadraticModel

"""
    reg_qp = RegularizedQuadraticModel(model, σ; selected = 1:model.meta.nvar)

    reg_qp = RegularizedQuadraticModel(c, Hrows, Hcols, Hvals; σ = σ, selected = 1:n, Arows = Arows, Acols = Acols, Avals = Avals, 
                        lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar, sortcols = false)

    reg_qp = RegularizedQuadraticModel(c, H; σ = σ, selected = 1:n, A = A, lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar)

Create a regularized quadratic model ``min ~\\tfrac{1}{2} x^T H x + tfrac{1}{2}σ‖x_{1:n}‖² + c^T x + c_0`` with optional bounds
`lvar ≦ x ≦ uvar` and optional linear constraints `lcon ≦ Ax ≦ ucon`.

The instance of `RegularizedQuadraticModel` created contains the fields: 
-  `model` of type `QuadraticModel`, which represents the unregularized quadratic model.
-  `meta` of type [`NLPModels.NLPModelMeta`](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/models/#NLPModels.NLPModelMeta) 
   from [`NLPModels.jl`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl),
-  `σ` the regularization parameter,
-  `selected` the set of variable indices to which the quadratic regularization is applied.
The `counters` and `data` fields of the instance are shared with those of its `model`, that is, `qp.data == qp.model.data` and `qp.counters = qp.model.counters`.
"""
mutable struct RegularizedQuadraticModel{T, S, M1, M2, I} <: AbstractQuadraticModel{T, S}
  model::QuadraticModel{T, S, M1, M2}
  meta::NLPModelMeta{T, S}
  σ::T
  selected::I # Number of slack variables, see SlackModel
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
  σ::T;
  selected = 1:model.meta.nvar
) where{T, S}

  isa(model.data.H, AbstractLinearOperator) && 
    return RegularizedQuadraticModel(model, model.meta, σ, selected)

  # Update nnzh: reg_qp.meta.nnzh ≠ reg_qp.model.meta.nnzh
  nz_diag = 0
  @inbounds for i in selected
    if model.data.H[i, i] == zero(T)
      nz_diag += 1
    end
  end
  meta = NLPModelMeta(model.meta, nnzh = model.meta.nnzh + nz_diag)
  return RegularizedQuadraticModel(model, meta, σ, selected)
end

function RegularizedQuadraticModel(
  model::QuadraticModel{T, S, M1},
  σ::T;
  selected = 1:model.meta.nvar
) where{T, S, M1 <: SparseMatrixCOO{T}}
  # Update nnzh: reg_qp.meta.nnzh ≠ reg_qp.model.meta.nnzh
  n_diag = 0

  # What if i is in H.rows, H.cols even though H[i, i] is zero ? 
  # For SparseMatricesCOO, we only need to check whether diagonal elements are structurally present.
  @inbounds for k in eachindex(model.data.H.rows)
    if model.data.H.rows[k] == model.data.H.cols[k]
      if model.data.H.rows[k] in selected
        n_diag += 1
      end
    end
  end
  nz_diag = length(selected) - n_diag
  meta = NLPModelMeta(model.meta, nnzh = model.meta.nnzh + nz_diag)
  return RegularizedQuadraticModel(model, meta, σ, selected)
end

function RegularizedQuadraticModel(
  c::S,
  H::Union{AbstractMatrix{T}, AbstractLinearOperator{T}};
  σ::T = zero(T),
  selected = 1:length(c),
  kwargs...,
) where {T, S}
  model = QuadraticModel(c, H; kwargs...)
  return RegularizedQuadraticModel(model, σ, selected = selected)
end

function RegularizedQuadraticModel(
  c::S,
  Hrows::AbstractVector{<:Integer},
  Hcols::AbstractVector{<:Integer},
  Hvals::S;
  σ::T = zero(T),
  selected = 1:length(c),
  kwargs...,
) where {T, S}
  model = QuadraticModel(c, Hrows, Hcols, Hvals; kwargs...)
  return RegularizedQuadraticModel(model, σ, selected = selected)
end

function RegularizedQuadraticModel(
  model::AbstractNLPModel{T, S}, 
  x::AbstractVector; 
  σ::T = zero(T),
  selected = 1:length(x),
  kwargs...
) where {T, S}
  model = QuadraticModel(model, x; kwargs...)
  return RegularizedQuadraticModel(model, σ, selected = selected)
end

# NLPModels API
function NLPModels.objgrad!(qp::RegularizedQuadraticModel, x::AbstractVector, g::AbstractVector)
  f, g = objgrad!(qp.model, x, g)
  iszero(qp.σ) && return f, g
  @views f += qp.σ*dot(x[qp.selected], x[qp.selected])/2
  @. @views g[qp.selected] += qp.σ * x[qp.selected]
  return f, g
end

function NLPModels.obj(qp::RegularizedQuadraticModel, x::AbstractVector)
  f = obj(qp.model, x)
  iszero(qp.σ) && return f
  @views f += qp.σ*dot(x[qp.selected], x[qp.selected])/2
  return f
end

function NLPModels.grad!(qp::RegularizedQuadraticModel, x::AbstractVector, g::AbstractVector)
  grad!(qp.model, x, g)
  iszero(qp.σ) && return g
  @. @views g[qp.selected] += qp.σ * x[qp.selected]
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
  @inbounds for i in qp.selected
    if qp.model.data.H[i,i] == zero(T)
      if !any(j -> qp.data.H.rows[j] == i && qp.data.H.cols[j] == i,
          eachindex(qp.data.H.rows)) # Need to check if i is not already structurally present in H
          k += 1
          rows[k] = i
          cols[k] = i
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
) where{T, S, M1 <: Matrix{T}}
  NLPModels.increment!(qp.model, :neval_hess)
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = j:(qp.meta.nvar)
      vals[count] = obj_weight * qp.model.data.H[i, j]
      if i == j 
        vals[count] += obj_weight * qp.σ
      end
      count += 1
    end
  end
  return vals
end

function NLPModels.hess_coord!(
  qp::RegularizedQuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where{T, S, M1 <: SparseMatrixCSC{T}}
  NLPModels.increment!(qp.model, :neval_hess)
  fill_coord!(qp.model.data.H, vals, obj_weight; σ = qp.σ)
  return vals
end

function NLPModels.hess_coord!(
  qp::RegularizedQuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where{T, S, M1 <: SparseMatrixCOO{T}}
  NLPModels.increment!(qp.model, :neval_hess)
  @inbounds for i = 1:qp.model.meta.nnzh
    vals[i] = obj_weight * qp.model.data.H.vals[i]
    if qp.model.data.H.rows[i] == qp.model.data.H.cols[i]
      vals[i] += obj_weight * qp.σ
    end
  end
  vals[qp.model.meta.nnzh+1:end] .= obj_weight * qp.σ

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
  @. @views Hv[qp.selected] += qp.σ*obj_weight*v[qp.selected]
  return Hv
end

for fname in (
  :jprod_lin!,
  :jtprod!,
  :jtprod_lin!
)
  @eval begin
    NLPModels.$fname(
      qp::RegularizedQuadraticModel,
      x::AbstractVector,
      v::AbstractVector,
      Atv::AbstractVector
    ) = $fname(qp.model, x, v, Atv)
  end
end

for fname in (
  :jac_lin_structure!,
  :jac_lin_coord!,
  :cons_lin!,
)
  @eval begin
    NLPModels.$fname(
      qp::RegularizedQuadraticModel,
      x::AbstractVector,
      v::AbstractVector,
    ) = $fname(qp.model, x, v)
  end
end

NLPModels.jac_lin(
  qp::RegularizedQuadraticModel,
  x::AbstractVector
) = jac_lin(qp.model, x)

function NLPModelsModifiers.SlackModel(
  qp::RegularizedQuadraticModel{T, S},
  name = qp.meta.name * "-slack",
) where {T, S}
  model = SlackModel(qp.model)
  ns = qp.meta.ncon - length(qp.meta.jfix)
  return RegularizedQuadraticModel(model, qp.σ, selected = 1:qp.meta.nvar)
end

function SlackModel!(qp::RegularizedQuadraticModel{T, S, M1, M2}) where {T, S, M1, M2 <: SparseMatrixCOO}
  ns = qp.meta.ncon - length(qp.meta.jfix)
  SlackModel!(qp.model)
end
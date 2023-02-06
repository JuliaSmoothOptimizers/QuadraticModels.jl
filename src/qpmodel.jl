export jac_structure!, hess_structure!, jac_coord!, hess_coord!, SlackModel!

mutable struct QPData{
  T,
  S,
  M1 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M2 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
}
  c0::T         # constant term in objective
  c::S          # linear term
  H::M1
  A::M2
end

isdense(data::QPData{T, S, M1, M2}) where {T, S, M1, M2} = M1 <: DenseMatrix || M2 <: DenseMatrix

function Base.convert(
  ::Type{QPData{T, S, MCOO, MCOO}},
  data::QPData{T, S, M1, M2},
) where {T, S, M1 <: AbstractMatrix, M2 <: AbstractMatrix, MCOO <: SparseMatrixCOO{T}}
  HCOO = (M1 <: SparseMatrixCOO) ? data.H : SparseMatrixCOO(data.H)
  ACOO = (M2 <: SparseMatrixCOO) ? data.A : SparseMatrixCOO(data.A)
  return QPData(data.c0, data.c, HCOO, ACOO)
end
Base.convert(
  ::Type{QPData{T, S, MCOO, MCOO}},
  data::QPData{T, S, M1, M2},
) where {T, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO, MCOO <: SparseMatrixCOO{T}} = data

abstract type AbstractQuadraticModel{T, S} <: AbstractNLPModel{T, S} end

"""
    qp = QuadraticModel(c, Hrows, Hcols, Hvals; Arows = Arows, Acols = Acols, Avals = Avals, 
                        lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar, sortcols = false)

    qp = QuadraticModel(c, H; A = A, lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar)

Create a Quadratic model ``min ~\\tfrac{1}{2} x^T H x + c^T x + c_0`` with optional bounds
`lvar ≦ x ≦ uvar` and optional linear constraints `lcon ≦ Ax ≦ ucon`.
The user should only give the lower triangle of `H` to the `QuadraticModel` constructor.

With the first constructor, if `sortcols = true`, then `Hcols` and `Acols` are sorted in ascending order 
(`Hrows`, `Hvals` and `Arows`, `Avals` are then sorted accordingly).

You can also use [`QPSReader.jl`](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) to
create a Quadratic model from a QPS file:

    using QPSReader
    qps = readqps("QAFIRO.SIF")
    qp = QuadraticModel(qps)

The instance of `QuadraticModel{T, S, D}` created contains the fields:
- `meta` of type [`NLPModels.NLPModelMeta`](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/models/#NLPModels.NLPModelMeta) 
  from [`NLPModels.jl`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl),
- `data`, of type `QuadraticModels.QPData` depending on the input types
  of the `A` and `H` matrices.
- `counters` of type [`NLPModels.Counters`](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/reference/#NLPModels.Counters)
  from [`NLPModels.jl`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
  
Using [`NLPModelsModifiers.SlackModel`](https://juliasmoothoptimizers.github.io/NLPModelsModifiers.jl/stable/reference/#NLPModelsModifiers.SlackModel)
from [`NLPModelsModifiers.jl`](https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl) with a `QuadraticModel` 
based on a `QPData` with dense matrices will convert the field `data` to a `QPData` with SparseMatricesCOO.  

Its in-place variant `SlackModel!` specific to QuadraticModels will only work with a `QuadraticModel` based on
a `QPData` with SparseMatricesCOO.
"""
mutable struct QuadraticModel{T, S, M1, M2} <: AbstractQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::QPData{T, S, M1, M2}
end

function Base.convert(
  ::Type{QuadraticModel{T, S, Mconv, Mconv}},
  qm::QuadraticModel{T, S, M1, M2},
) where {T, S, M1 <: AbstractMatrix, M2 <: AbstractMatrix, Mconv}
  data_conv = convert(QPData{T, S, Mconv, Mconv}, qm.data)
  return QuadraticModel(qm.meta, qm.counters, data_conv)
end

function QuadraticModel(
  c::S,
  Hrows::AbstractVector{<:Integer},
  Hcols::AbstractVector{<:Integer},
  Hvals::S;
  Arows::AbstractVector{<:Integer} = Int[],
  Acols::AbstractVector{<:Integer} = Int[],
  Avals::S = S(undef, 0),
  lcon::S = S(undef, 0),
  ucon::S = S(undef, 0),
  lvar::S = fill!(S(undef, length(c)), eltype(c)(-Inf)),
  uvar::S = fill!(S(undef, length(c)), eltype(c)(Inf)),
  c0::T = zero(eltype(c)),
  sortcols::Bool = false,
  kwargs...,
) where {T, S}
  @assert all(lvar .≤ uvar)
  @assert all(lcon .≤ ucon)
  nnzh = length(Hvals)
  if !(nnzh == length(Hrows) == length(Hcols))
    error("The length of Hrows, Hcols and Hvals must be the same")
  end
  nnzj = length(Avals)
  if !(nnzj == length(Arows) == length(Acols))
    error("The length of Arows, Acols and Avals must be the same")
  end
  ncon = length(lcon)
  if ncon != length(ucon)
    error("The length of lcon and ucon must be the same")
  end
  nvar = length(c)
  if !(nvar == length(lvar) == length(uvar))
    error("The length of c, lvar and uvar must be the same")
  end
  if sortcols
    pH = sortperm(Hcols)
    permute!(Hrows, pH)
    permute!(Hcols, pH)
    permute!(Hvals, pH)
    pA = sortperm(Acols)
    permute!(Arows, pA)
    permute!(Acols, pA)
    permute!(Avals, pA)
  end
  QuadraticModel(
    NLPModelMeta{T, S}(
      length(c),
      lvar = lvar,
      uvar = uvar,
      ncon = ncon,
      lcon = lcon,
      ucon = ucon,
      nnzj = nnzj,
      lin_nnzj = nnzj,
      nln_nnzj = 0,
      nnzh = nnzh,
      lin = 1:ncon,
      islp = (ncon == 0);
      kwargs...,
    ),
    Counters(),
    QPData(
      c0,
      c,
      SparseMatrixCOO(nvar, nvar, Hrows, Hcols, Hvals),
      SparseMatrixCOO(ncon, nvar, Arows, Acols, Avals),
    ),
  )
end

similar_empty_matrix(H::AbstractMatrix{T}, n::Integer) where {T} = similar(H, 0, n)
similar_empty_matrix(::SparseMatrixCOO{T, I}, n::Integer) where {T, I} =
  SparseMatrixCOO(0, n, I[], I[], T[])
similar_empty_matrix(::AbstractLinearOperator{T}, n::Integer) where {T} = opZeros(T, 0, n)

function QuadraticModel(
  c::S,
  H::Union{AbstractMatrix{T}, AbstractLinearOperator{T}};
  A::Union{AbstractMatrix, AbstractLinearOperator} = similar_empty_matrix(H, length(c)),
  lcon::S = S(undef, 0),
  ucon::S = S(undef, 0),
  lvar::S = fill!(S(undef, length(c)), T(-Inf)),
  uvar::S = fill!(S(undef, length(c)), T(Inf)),
  c0::T = zero(T),
  kwargs...,
) where {T, S}
  @assert all(lvar .≤ uvar)
  @assert all(lcon .≤ ucon)
  ncon, nvar = size(A)
  if typeof(H) <: AbstractLinearOperator # convert A to a LinOp if A is a Matrix?
    nnzh = 0
    nnzj = 0
    data = QPData(c0, c, H, A)
  else
    nnzh = typeof(H) <: DenseMatrix ? nvar * (nvar + 1) / 2 : nnz(H)
    nnzj = nnz(A)
    data = typeof(H) <: Symmetric ? QPData(c0, c, H.data, A) : QPData(c0, c, H, A)
  end

  QuadraticModel(
    NLPModelMeta{T, S}(
      nvar,
      lvar = lvar,
      uvar = uvar,
      ncon = size(A, 1),
      lcon = lcon,
      ucon = ucon,
      nnzj = nnzj,
      lin_nnzj = nnzj,
      nln_nnzj = 0,
      nnzh = nnzh,
      lin = 1:ncon,
      islp = (ncon == 0);
      kwargs...,
    ),
    Counters(),
    data,
  )
end

"""
    QuadraticModel(nlp, x)

Creates a quadratic Taylor model of `nlp` around `x`.
"""
function QuadraticModel(model::AbstractNLPModel{T, S}, x::AbstractVector; kwargs...) where {T, S}
  nvar = model.meta.nvar
  ncon = model.meta.ncon
  c0 = obj(model, x)
  g = grad(model, x)
  Hrows, Hcols = hess_structure(model)
  Hvals = hess_coord(model, x)
  if model.meta.ncon > 0
    c = cons(model, x)
    Arows, Acols = jac_structure(model)
    Avals = jac_coord(model, x)
    QuadraticModel(
      g,
      Hrows,
      Hcols,
      Hvals,
      c0 = c0,
      Arows = Arows,
      Acols = Acols,
      Avals = Avals,
      lcon = model.meta.lcon .- c,
      ucon = model.meta.ucon .- c,
      lvar = model.meta.lvar .- x,
      uvar = model.meta.uvar .- x,
      x0 = fill!(S(undef, model.meta.nvar), zero(T)),
    )
  else
    QuadraticModel(
      g,
      Hrows,
      Hcols,
      Hvals,
      c0 = c0,
      lvar = model.meta.lvar .- x,
      uvar = model.meta.uvar .- x,
      x0 = fill!(S(undef, model.meta.nvar), zero(T)),
    )
  end
end

linobj(qp::AbstractQuadraticModel, args...) = qp.data.c

function NLPModels.objgrad!(qp::AbstractQuadraticModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(qp, :neval_obj)
  NLPModels.increment!(qp, :neval_grad)
  mul!(g, Symmetric(qp.data.H, :L), x)
  f = qp.data.c0 + dot(qp.data.c, x) + dot(g, x) / 2
  @. g .+= qp.data.c
  return f, g
end

function NLPModels.obj(qp::AbstractQuadraticModel{T, S}, x::AbstractVector) where {T, S}
  NLPModels.increment!(qp, :neval_obj)
  Hx = fill!(S(undef, qp.meta.nvar), zero(T))
  mul!(Hx, Symmetric(qp.data.H, :L), x)
  return qp.data.c0 + dot(qp.data.c, x) + dot(Hx, x) / 2
end

function NLPModels.grad!(qp::AbstractQuadraticModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(qp, :neval_grad)
  mul!(g, Symmetric(qp.data.H, :L), x)
  g .+= qp.data.c
  return g
end

# TODO: Better hess_op

function NLPModels.hess_structure!(
  qp::QuadraticModel{T, S, M1},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: SparseMatrixCOO}
  rows .= qp.data.H.rows
  cols .= qp.data.H.cols
  return rows, cols
end

function fill_structure!(S::SparseMatrixCSC, rows, cols)
  count = 1
  @inbounds for col = 1:size(S, 2), k = S.colptr[col]:(S.colptr[col + 1] - 1)
    rows[count] = S.rowval[k]
    cols[count] = col
    count += 1
  end
end

function fill_coord!(S::SparseMatrixCSC, vals, obj_weight)
  count = 1
  @inbounds for col = 1:size(S, 2), k = S.colptr[col]:(S.colptr[col + 1] - 1)
    vals[count] = obj_weight * S.nzval[k]
    count += 1
  end
end

function NLPModels.hess_structure!(
  qp::QuadraticModel{T, S, M1},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: SparseMatrixCSC}
  fill_structure!(qp.data.H, rows, cols)
  return rows, cols
end

function NLPModels.hess_structure!(
  qp::QuadraticModel{T, S, M1},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: Matrix}
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = j:(qp.meta.nvar)
      rows[count] = i
      cols[count] = j
      count += 1
    end
  end
  return rows, cols
end

function NLPModels.hess_coord!(
  qp::QuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: SparseMatrixCOO}
  NLPModels.increment!(qp, :neval_hess)
  vals .= obj_weight .* qp.data.H.vals
  return vals
end

function NLPModels.hess_coord!(
  qp::QuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: SparseMatrixCSC}
  NLPModels.increment!(qp, :neval_hess)
  fill_coord!(qp.data.H, vals, obj_weight)
  return vals
end

function NLPModels.hess_coord!(
  qp::QuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: Matrix}
  NLPModels.increment!(qp, :neval_hess)
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = j:(qp.meta.nvar)
      vals[count] = obj_weight * qp.data.H[i, j]
      count += 1
    end
  end
  return vals
end

NLPModels.hess_coord!(
  qp::QuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) = hess_coord!(qp, x, vals, obj_weight = obj_weight)

function NLPModels.jac_lin_structure!(
  qp::QuadraticModel{T, S, M1, M2},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: SparseMatrixCOO}
  @lencheck qp.meta.lin_nnzj rows cols
  rows .= qp.data.A.rows
  cols .= qp.data.A.cols
  return rows, cols
end

function NLPModels.jac_lin_structure!(
  qp::QuadraticModel{T, S, M1, M2},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: SparseMatrixCSC}
  @lencheck qp.meta.lin_nnzj rows cols
  fill_structure!(qp.data.A, rows, cols)
  return rows, cols
end

function NLPModels.jac_lin_structure!(
  qp::QuadraticModel{T, S, M1, M2},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: Matrix}
  @lencheck qp.meta.lin_nnzj rows cols
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = 1:(qp.meta.ncon)
      rows[count] = i
      cols[count] = j
      count += 1
    end
  end
  return rows, cols
end

function NLPModels.jac_lin_coord!(
  qp::QuadraticModel{T, S, M1, M2},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, M1, M2 <: SparseMatrixCOO}
  @lencheck qp.meta.nvar x
  @lencheck qp.meta.lin_nnzj vals
  NLPModels.increment!(qp, :neval_jac_lin)
  vals .= qp.data.A.vals
  return vals
end

function NLPModels.jac_lin_coord!(
  qp::QuadraticModel{T, S, M1, M2},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, M1, M2 <: SparseMatrixCSC}
  @lencheck qp.meta.nvar x
  @lencheck qp.meta.lin_nnzj vals
  NLPModels.increment!(qp, :neval_jac_lin)
  fill_coord!(qp.data.A, vals, one(T))
  return vals
end

function NLPModels.jac_lin_coord!(
  qp::QuadraticModel{T, S, M1, M2},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, M1, M2 <: Matrix}
  @lencheck qp.meta.nvar x
  @lencheck qp.meta.lin_nnzj vals
  NLPModels.increment!(qp, :neval_jac_lin)
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = 1:(qp.meta.ncon)
      vals[count] = qp.data.A[i, j]
      count += 1
    end
  end
  return vals
end

function NLPModels.jac_lin(
  qp::QuadraticModel{T, S, M1, M2},
  x::AbstractVector,
) where {T, S, M1 <: AbstractLinearOperator, M2 <: AbstractLinearOperator}
  @lencheck qp.meta.nvar x
  increment!(qp, :neval_jac_lin)
  return qp.data.A
end

function NLPModels.cons_lin!(qp::AbstractQuadraticModel, x::AbstractVector, c::AbstractVector)
  @lencheck qp.meta.nvar x
  @lencheck qp.meta.nlin c
  NLPModels.increment!(qp, :neval_cons_lin)
  mul!(c, qp.data.A, x)
  return c
end

function NLPModels.hprod!(
  qp::AbstractQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  NLPModels.increment!(qp, :neval_hprod)
  mul!(Hv, Symmetric(qp.data.H, :L), v)
  if obj_weight != 1
    Hv .*= obj_weight
  end
  return Hv
end

NLPModels.hprod!(
  qp::AbstractQuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) = hprod!(qp, x, v, Hv, obj_weight = obj_weight)

function NLPModels.jprod_lin!(
  qp::AbstractQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Av::AbstractVector,
)
  @lencheck qp.meta.nvar x v
  @lencheck qp.meta.nlin Av
  NLPModels.increment!(qp, :neval_jprod_lin)
  mul!(Av, qp.data.A, v)
  return Av
end

function NLPModels.jtprod!(
  qp::AbstractQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Atv::AbstractVector,
)
  @lencheck qp.meta.nvar x Atv
  @lencheck qp.meta.ncon v
  NLPModels.increment!(qp, :neval_jtprod)
  mul!(Atv, transpose(qp.data.A), v)
  return Atv
end

function NLPModels.jtprod_lin!(
  qp::AbstractQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Atv::AbstractVector,
)
  @lencheck qp.meta.nvar x Atv
  @lencheck qp.meta.nlin v
  NLPModels.increment!(qp, :neval_jtprod_lin)
  mul!(Atv, transpose(qp.data.A), v)
  return Atv
end

function SlackModel!(qp::QuadraticModel{T, S, M1, M2}) where {T, S, M1, M2 <: SparseMatrixCOO}
  qp.meta.ncon == length(qp.meta.jfix) && return qp

  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix
  append!(qp.data.A.rows, qp.meta.jlow)
  append!(qp.data.A.rows, qp.meta.jupp)
  append!(qp.data.A.rows, qp.meta.jrng)
  append!(qp.data.A.cols, (qp.meta.nvar + 1):(qp.meta.nvar + ns))
  append!(qp.data.A.vals, (-one(T) for _ = 1:ns))
  append!(qp.data.c, (zero(T) for _ = 1:ns))

  qp.meta = NLPModelsModifiers.slack_meta(qp.meta, name = qp.meta.name)
  return qp
end

function slackdata(
  data::QPData{T, S, M1, M2},
  meta::NLPModelMeta{T},
  ns::Int,
) where {T, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO}
  nvar_slack = meta.nvar + ns
  return QPData(
    copy(data.c0),
    [data.c; fill!(similar(data.c, ns), zero(T))],
    SparseMatrixCOO(
      nvar_slack,
      nvar_slack,
      copy(data.H.rows),
      copy(data.H.cols),
      copy(data.H.vals),
    ),
    SparseMatrixCOO(
      meta.ncon,
      nvar_slack,
      [data.A.rows; meta.jlow; meta.jupp; meta.jrng],
      [data.A.cols; (meta.nvar + 1):(meta.nvar + ns)],
      [data.A.vals; fill!(similar(data.c, ns), -one(T))],
    ),
  )
end

function prodPermutedMinusOnes!(res, v, α, β::T, p::Vector{Int}) where {T}
  res .= β == zero(T) ? zero(T) : β .* res
  res[p] .-= α .* v
  return res
end
function tprodPermutedMinusOnes!(res, v, α, β::T, p::Vector{Int}) where {T}
  res .= β == zero(T) ? zero(T) : β .* res
  res .-= @views α .* v[p]
  return res
end

function opPermutedMinusOnes(T::DataType, ncon::Int, ns::Int, p::Vector{Int})
  prod! = (res, v, α, β) -> prodPermutedMinusOnes!(res, v, α, β, p)
  tprod! = (res, v, α, β) -> tprodPermutedMinusOnes!(res, v, α, β, p)
  return LinearOperator(T, ncon, ns, false, false, prod!, tprod!)
end

function slackdata(data::QPData{T}, meta::NLPModelMeta{T}, ns::Int) where {T}
  return QPData(
    copy(data.c0),
    [data.c; fill!(similar(data.c, ns), zero(T))],
    BlockDiagonalOperator(data.H, opZeros(T, ns, ns)),
    [data.A opPermutedMinusOnes(T, meta.ncon, ns, [meta.jlow; meta.jupp; meta.jrng])],
  )
end

function NLPModelsModifiers.SlackModel(
  qp::AbstractQuadraticModel{T, S},
  name = qp.meta.name * "-slack",
) where {T, S}
  qp.meta.ncon == length(qp.meta.jfix) && return qp
  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix

  if isdense(qp.data) # convert to QPDataCOO first
    dataCOO = convert(QPData{T, S, SparseMatrixCOO{T, Int}, SparseMatrixCOO{T, Int}}, qp.data)
    data = slackdata(dataCOO, qp.meta, ns)
  else
    data = slackdata(qp.data, qp.meta, ns)
  end

  meta = NLPModelsModifiers.slack_meta(qp.meta, name = qp.meta.name)

  return QuadraticModel(meta, Counters(), data)
end

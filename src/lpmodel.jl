"""
    lp = LinearModel(c; Arows = Arows, Acols = Acols, Avals = Avals,
                     lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar, c0=c0, sortcols = false)

    lp = LinearModel(c; A = A, lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar, c0 = c0)

Create a Linear model ``min ~c^T x + c_0`` with optional bounds
`lvar ≦ x ≦ uvar` and optional linear constraints `lcon ≦ Ax ≦ ucon`.

With the first constructor, if `sortcols = true`, then `Acols` is sorted in ascending order 
(`Arows` and `Avals` are then sorted accordingly).

You can also use [`QPSReader.jl`](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) to
create a Linear model from a QPS file:

    using QPSReader
    qps = readqps("QAFIRO.SIF")
    lp = LinearModel(qps)
"""
mutable struct LinearModel{T, S, M1, M2} <: AbstractQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::QPData{T, S, M1, M2}
end

function LinearModel(
  c::S;
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
  nnzh = 0
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
    pA = sortperm(Acols)
    permute!(Arows, pA)
    permute!(Acols, pA)
    permute!(Avals, pA)
  end
  LinearModel(
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
      islp = true;
      kwargs...,
    ),
    Counters(),
    QPData(
      c0,
      c,
      SparseMatrixCOO(0, nvar, similar(Arows, 0), similar(Acols, 0), similar(Avals, 0)),
      SparseMatrixCOO(ncon, nvar, Arows, Acols, Avals),
    ),
  )
end

function LinearModel(
  c::S;
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}} = SparseMatrixCOO(0, length(c), Int[], Int[], T[]),
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
  nnzh = 0
  nnzj = nnz(A)
  H = similar_empty_matrix(A, length(c))
  data = QPData(c0, c, H, A)

  LinearModel(
    NLPModelMeta{T, S}(
      nvar,
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
    data,
  )
end

"""
    LinearModel(nlp, x)

Creates a linear Taylor model of `nlp` around `x`.
"""
function LinearModel(model::AbstractNLPModel{T, S}, x::AbstractVector; kwargs...) where {T, S}
  nvar = model.meta.nvar
  ncon = model.meta.ncon
  c0 = obj(model, x)
  g = grad(model, x)
  if model.meta.ncon > 0
    c = cons(model, x)
    Arows, Acols = jac_structure(model)
    Avals = jac_coord(model, x)
    LinearModel(
      g,
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
    LinearModel(
      g,
      c0 = c0,
      lvar = model.meta.lvar .- x,
      uvar = model.meta.uvar .- x,
      x0 = fill!(S(undef, model.meta.nvar), zero(T)),
    )
  end
end

function NLPModels.objgrad!(qp::LinearModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(qp, :neval_obj)
  NLPModels.increment!(qp, :neval_grad)
  f = qp.data.c0 + dot(qp.data.c, x)
  g .+= qp.data.c
  return f, g
end

function NLPModels.obj(qp::LinearModel, x::AbstractVector)
  NLPModels.increment!(qp, :neval_obj)
  return qp.data.c0 + dot(qp.data.c, x)
end

function NLPModels.grad!(qp::LinearModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(qp, :neval_grad)
  g .= qp.data.c
  return g
end

function NLPModels.hess_structure!(
  lp::LinearModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  return rows, cols
end

function NLPModels.hess_coord!(
  lp::LinearModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  NLPModels.increment!(lp, :neval_hess)
  fill!(vals, zero(eltype(x)))
  return vals
end

function NLPModels.hprod!(
  lp::LinearModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  NLPModels.increment!(lp, :neval_hprod)
  fill!(Hv, zero(eltype(x)))
  return Hv
end

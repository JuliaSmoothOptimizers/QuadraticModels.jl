export jac_structure!, hess_structure!, jac_coord!, hess_coord!, SlackModel!

abstract type AbstractQPData{T, S} end

mutable struct QPDataCOO{T, S} <: AbstractQPData{T, S}
  c0::T          # constant term in objective
  c::S          # linear term
  Hrows::Vector{Int}      # quadratic term
  Hcols::Vector{Int}
  Hvals::S
  Arows::Vector{Int}      # constraints matrix
  Acols::Vector{Int}
  Avals::S
end

mutable struct QPDataDense{T, S, M1 <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}} <: AbstractQPData{T, S}
  c0::T          # constant term in objective
  c::S          # linear term
  H::M1
  A::M2
end

abstract type AbstractQuadraticModel{T, S} <: AbstractNLPModel{T, S} end

"""
    qp = QuadraticModel(c, Hrows, Hcols, Hvals; Arows = Arows, Acols = Acols, Avals = Avals, 
                        lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar)

    qp = QuadraticModel(c, H; A = A, lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar)

Create a Quadratic model ``min ~\\tfrac{1}{2} x^T Q x + c^T x + c_0`` with optional bounds
`lvar ≦ x ≦ uvar` and optional linear constraints `lcon ≦ Ax ≦ ucon`.

You can also use [`QPSReader.jl`](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) to
create a Quadratic model from a QPS file:

    using QPSReader
    qps = readqps("QAFIRO.SIF")
    qp = QuadraticModel(qps)
"""
mutable struct QuadraticModel{T, S, D <: AbstractQPData{T, S}} <: AbstractQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::D
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
    NLPModelMeta(
      length(c),
      lvar = lvar,
      uvar = uvar,
      ncon = ncon,
      lcon = lcon,
      ucon = ucon,
      nnzj = nnzj,
      nnzh = nnzh,
      lin = 1:ncon,
      nln = Int[],
      islp = (ncon == 0);
      kwargs...,
    ),
    Counters(),
    QPDataCOO(c0, c, Hrows, Hcols, Hvals, Arows, Acols, Avals),
  )
end

function QuadraticModel(
  c::S,
  H::AbstractMatrix{T};
  A::AbstractMatrix = similar(c, 0, length(c)),
  lcon::S = S(undef, 0),
  ucon::S = S(undef, 0),
  lvar::S = fill!(S(undef, length(c)), T(-Inf)),
  uvar::S = fill!(S(undef, length(c)), T(Inf)),
  c0::T = zero(T),
  kwargs...,
) where {T, S}
  ncon, nvar = size(A)
  if typeof(H) <: SparseMatrixCSC
    tril!(H)
    nnzh, Hrows, Hcols, Hvals = nnz(H), findnz(H)...
    nnzj, Arows, Acols, Avals = if ncon == 0
      0, Int[], Int[], similar(c, 0)
    elseif issparse(A)
      nnz(A), findnz(A)...
    else
      I = ((i, j, A[i, j]) for i = 1:ncon, j = 1:nvar)
      nvar * ncon, getindex.(I, 1)[:], getindex.(I, 2)[:], getindex.(I, 3)[:]
    end
    data = QPDataCOO(c0, c, Hrows, Hcols, Hvals, Arows, Acols, Avals)
  else
    nnzh, nnzj = nvar^2, nvar*ncon
    data = QPDataDense(c0, c, H, A)
  end
  QuadraticModel(
    NLPModelMeta(
      nvar,
      lvar = lvar,
      uvar = uvar,
      ncon = size(A, 1),
      lcon = lcon,
      ucon = ucon,
      nnzj = nnzj,
      nnzh = nnzh,
      lin = 1:ncon,
      nln = Int[],
      islp = (ncon == 0);
      kwargs...,
    ),
    Counters(),
    data,
  )
end

QuadraticModel(c::S, H::AbstractMatrix; args...) where {S} = QuadraticModel(c, sparse(H); args...)

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
  if typeof(qp.data) <: QPDataCOO
    coo_sym_prod!(qp.data.Hrows, qp.data.Hcols, qp.data.Hvals, x, g)
  else
    mul!(g, Symmetric(qp.data.H, :L), x)
  end
  f = qp.data.c0 + dot(qp.data.c, x) + dot(g, x) / 2
  @. g .+= qp.data.c
  return f, g
end

function NLPModels.obj(qp::AbstractQuadraticModel{T, S}, x::AbstractVector) where {T, S}
  NLPModels.increment!(qp, :neval_obj)
  Hx = fill!(S(undef, qp.meta.nvar), zero(T))
  if typeof(qp.data) <: QPDataCOO
    coo_sym_prod!(qp.data.Hrows, qp.data.Hcols, qp.data.Hvals, x, Hx)
  else
    mul!(Hx, Symmetric(qp.data.H, :L), x)
  end
  return qp.data.c0 + dot(qp.data.c, x) + dot(Hx, x) / 2
end

function NLPModels.grad!(qp::AbstractQuadraticModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(qp, :neval_grad)
  if typeof(qp.data) <: QPDataCOO
    coo_sym_prod!(qp.data.Hrows, qp.data.Hcols, qp.data.Hvals, x, g)
  else
    mul!(g, Symmetric(qp.data.H, :L), x)
  end
  g .+= qp.data.c
  return g
end

# TODO: Better hess_op

function NLPModels.hess_structure!(
  qp::QuadraticModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if typeof(qp.data) <: QPDataCOO
    rows .= qp.data.Hrows
    cols .= qp.data.Hcols
  else
    nvar = qp.meta.nvar
    for j in 1:nvar
      for i in 1:nvar
        rows[i + (j-1) * nvar] = i
        cols[i + (j-1) * nvar] = j
      end
    end
  end
  return rows, cols
end

function NLPModels.hess_coord!(
  qp::QuadraticModel{T},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T}
  NLPModels.increment!(qp, :neval_hess)
  if typeof(qp.data) <: QPDataCOO
    vals .= obj_weight * qp.data.Hvals
  else
    nvar = qp.meta.nvar
    for j in 1:nvar
      for i in 1:nvar
        vals[i + (j-1) * nvar] = (i ≥ j) ? obj_weight * qp.data.H[i, j] : zero(T)
      end
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

function NLPModels.jac_structure!(
  qp::QuadraticModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if typeof(qp.data) <: QPDataCOO
    rows .= qp.data.Arows
    cols .= qp.data.Acols
  else
    nvar, ncon = qp.meta.nvar, qp.meta.ncon
    for j in 1:nvar
      for i in 1:ncon
        rows[i + (j-1) * ncon] = i
        cols[i + (j-1) * ncon] = j
      end
    end
  end
  return rows, cols
end

function NLPModels.jac_coord!(qp::QuadraticModel, x::AbstractVector, vals::AbstractVector)
  NLPModels.increment!(qp, :neval_jac)
  if typeof(qp.data) <: QPDataCOO
    vals .= qp.data.Avals
  else
    vals .= @views qp.data.A[:]
  end
  return vals
end

function NLPModels.cons!(qp::AbstractQuadraticModel, x::AbstractVector, c::AbstractVector)
  NLPModels.increment!(qp, :neval_cons)
  if typeof(qp.data) <: QPDataCOO
    coo_prod!(qp.data.Arows, qp.data.Acols, qp.data.Avals, x, c)
  else
    mul!(c, qp.data.A, x)
  end
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
  if typeof(qp.data) <: QPDataCOO
    coo_sym_prod!(qp.data.Hrows, qp.data.Hcols, qp.data.Hvals, v, Hv)
  else
    mul!(Hv, Symmetric(qp.data.H, :L), v)
  end
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

function NLPModels.jprod!(
  qp::AbstractQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Av::AbstractVector,
)
  NLPModels.increment!(qp, :neval_jprod)
  if typeof(qp.data) <: QPDataCOO
    coo_prod!(qp.data.Arows, qp.data.Acols, qp.data.Avals, v, Av)
  else
    mul!(Av, qp.data.A, v)
  end
  return Av
end

function NLPModels.jtprod!(
  qp::AbstractQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Atv::AbstractVector,
)
  NLPModels.increment!(qp, :neval_jtprod)
  if typeof(qp.data) <: QPDataCOO
    coo_prod!(qp.data.Acols, qp.data.Arows, qp.data.Avals, v, Atv)
  else
    mul!(Atv, transpose(qp.data.A), v)
  end
  return Atv
end

function SlackModel!(qp::AbstractQuadraticModel)
  qp.meta.ncon == length(qp.meta.jfix) && return qp

  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix
  T = eltype(qp.data.c)
  append!(qp.data.Arows, qp.meta.jlow)
  append!(qp.data.Arows, qp.meta.jupp)
  append!(qp.data.Arows, qp.meta.jrng)
  append!(qp.data.Acols, (qp.meta.nvar + 1):(qp.meta.nvar + ns))
  append!(qp.data.Avals, (-one(T) for _ = 1:ns))
  append!(qp.data.c, (zero(T) for _ = 1:ns))

  qp.meta = NLPModelsModifiers.slack_meta(qp.meta, name = qp.meta.name)
  return qp
end

function NLPModelsModifiers.SlackModel(qp::AbstractQuadraticModel, name = qp.meta.name * "-slack")
  qp.meta.ncon == length(qp.meta.jfix) && return qp
  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix
  T = eltype(qp.data.c)

  if typeof(qp.data) <: QPDataCOO
    data = QPDataCOO(
      copy(qp.data.c0),
      [qp.data.c; fill!(similar(qp.data.c, ns), zero(T))],
      copy(qp.data.Hrows),
      copy(qp.data.Hcols),
      copy(qp.data.Hvals),
      [qp.data.Arows; qp.meta.jlow; qp.meta.jupp; qp.meta.jrng],
      [qp.data.Acols; (qp.meta.nvar + 1):(qp.meta.nvar + ns)],
      [qp.data.Avals; fill!(similar(qp.data.c, ns), -one(T))],
    )
  elseif typeof(qp.data) <: QPDataDense
    error("convert to COO")
  end

  meta = NLPModelsModifiers.slack_meta(qp.meta, name = qp.meta.name)

  return QuadraticModel(meta, Counters(), data)
end

export jac_structure!, hess_structure!, jac_coord!, hess_coord!, SlackModel!

mutable struct QPData{T}
  c0    :: T          # constant term in objective
  c     :: Vector{T}           # linear term
  Hrows :: Vector{Int}      # quadratic term
  Hcols :: Vector{Int}
  Hvals :: Vector{T}
  Arows :: Vector{Int}      # constraints matrix
  Acols :: Vector{Int}
  Avals :: Vector{T}
end

abstract type AbstractQuadraticModel <: AbstractNLPModel end

mutable struct QuadraticModel <: AbstractQuadraticModel
  meta     :: NLPModelMeta
  counters :: Counters
  data     :: QPData
end

function QuadraticModel(c :: AbstractVector{T},
                        Hrows :: AbstractVector{<: Integer}, Hcols :: AbstractVector{<: Integer}, Hvals :: AbstractVector;
                        Arows :: AbstractVector{<: Integer} = Int[], Acols :: AbstractVector{<: Integer} = Int[], Avals :: AbstractVector = T[],
                        lcon :: AbstractVector = T[], ucon :: AbstractVector = T[],
                        lvar :: AbstractVector = fill(T(-Inf), length(c)), uvar :: AbstractVector = fill(T(Inf), length(c)),
                        c0 :: T = zero(T), kwargs...) where T
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
  QuadraticModel(NLPModelMeta(length(c), lvar=lvar, uvar=uvar,
                              ncon=ncon, lcon=lcon, ucon=ucon,
                              nnzj=nnzj, nnzh=nnzh,
                              lin=1:ncon, nln=Int[], islp=(ncon == 0); kwargs...),
                 Counters(),
                 QPData(c0, c, Hrows, Hcols, Hvals, Arows, Acols, Avals))
end

function QuadraticModel(c :: AbstractVector{T}, H :: SparseMatrixCSC{T, Int};
                        A :: AbstractMatrix = zeros(T, 0, length(c)),
                        lcon :: AbstractVector = zeros(T, 0), ucon :: AbstractVector = zeros(T, 0),
                        lvar :: AbstractVector = fill(T(-Inf), length(c)), uvar :: AbstractVector = fill(T(Inf), length(c)),
                        c0 :: T = zero(T), kwargs...) where T
  ncon, nvar = size(A)
  tril!(H)
  nnzh, Hrows, Hcols, Hvals = nnz(H), findnz(H)...
  nnzj, Arows, Acols, Avals = if ncon == 0
    0, Int[], Int[], T[]
  elseif issparse(A)
    nnz(A), findnz(A)...
  else
    I = ((i,j,A[i,j]) for i = 1:ncon, j = 1:nvar)
    nvar * ncon, getindex.(I, 1)[:], getindex.(I, 2)[:], getindex.(I, 3)[:]
  end
  QuadraticModel(NLPModelMeta(nvar, lvar=lvar, uvar=uvar,
                              ncon=size(A,1), lcon=lcon, ucon=ucon,
                              nnzj=nnzj, nnzh=nnzh,
                              lin=1:ncon, nln=Int[], islp=(ncon == 0); kwargs...),
                 Counters(),
                 QPData(c0, c, Hrows, Hcols, Hvals, Arows, Acols, Avals))
end

QuadraticModel(c :: AbstractVector{T}, H :: AbstractMatrix; args...) where T = QuadraticModel(c, sparse(H); args...)

"""
    QuadraticModel(nlp, x)

Creates a quadratic Taylor model of `nlp` around `x`.
"""
function QuadraticModel(model :: AbstractNLPModel, x :: AbstractVector; kwargs...)
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
    QuadraticModel(g, Hrows, Hcols, Hvals, c0=c0,
                   Arows=Arows, Acols=Acols, Avals=Avals, lcon=model.meta.lcon .- c, ucon=model.meta.ucon .- c,
                   lvar=model.meta.lvar .- x, uvar=model.meta.uvar .- x, x0=zeros(model.meta.nvar))
  else
    QuadraticModel(g, Hrows, Hcols, Hvals, c0=c0,
                   lvar=model.meta.lvar .- x, uvar=model.meta.uvar .- x, x0=zeros(model.meta.nvar))
  end
end

linobj(qp::AbstractQuadraticModel, args...) = qp.data.c

function NLPModels.objgrad!(qp :: AbstractQuadraticModel, x :: AbstractVector, g :: AbstractVector)
  NLPModels.increment!(qp, :neval_obj)
  NLPModels.increment!(qp, :neval_grad)
  coo_sym_prod!(qp.data.Hrows, qp.data.Hcols, qp.data.Hvals, x, g)
  f = qp.data.c0 + dot(qp.data.c, x) + dot(g, x) / 2
  @. g .+= qp.data.c
  return f, g
end

function NLPModels.obj(qp :: AbstractQuadraticModel, x :: AbstractVector)
  NLPModels.increment!(qp, :neval_obj)
  Hx = zeros(qp.meta.nvar)
  coo_sym_prod!(qp.data.Hrows, qp.data.Hcols, qp.data.Hvals, x, Hx)
  return qp.data.c0 + dot(qp.data.c, x) + dot(Hx, x) / 2
end

function NLPModels.grad!(qp :: AbstractQuadraticModel, x :: AbstractVector, g :: AbstractVector)
  NLPModels.increment!(qp, :neval_grad)
  coo_sym_prod!(qp.data.Hrows, qp.data.Hcols, qp.data.Hvals, x, g)
  g .+= qp.data.c
  return g
end

# TODO: Better hess_op

function NLPModels.hess_structure!(qp :: QuadraticModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows .= qp.data.Hrows
  cols .= qp.data.Hcols
  return rows, cols
end

function NLPModels.hess_coord!(qp :: QuadraticModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real=one(eltype(x)))
  NLPModels.increment!(qp, :neval_hess)
  vals .= obj_weight * qp.data.Hvals
  return vals
end

NLPModels.hess_coord!(qp :: QuadraticModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real=one(eltype(x))) = hess_coord!(qp, x, vals, obj_weight=obj_weight)

function NLPModels.jac_structure!(qp :: QuadraticModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows .= qp.data.Arows
  cols .= qp.data.Acols
  return rows, cols
end

function NLPModels.jac_coord!(qp :: QuadraticModel, x :: AbstractVector, vals :: AbstractVector)
  NLPModels.increment!(qp, :neval_jac)
  vals .= qp.data.Avals
  return vals
end

function NLPModels.cons!(qp :: AbstractQuadraticModel, x :: AbstractVector, c :: AbstractVector)
  NLPModels.increment!(qp, :neval_cons)
  coo_prod!(qp.data.Arows, qp.data.Acols, qp.data.Avals, x, c)
  return c
end

function NLPModels.hprod!(qp :: AbstractQuadraticModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  NLPModels.increment!(qp, :neval_hprod)
  coo_sym_prod!(qp.data.Hrows, qp.data.Hcols, qp.data.Hvals, v, Hv)
  if obj_weight != 1
    Hv .*= obj_weight
  end
  return Hv
end

NLPModels.hprod!(qp :: AbstractQuadraticModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x))) = hprod!(qp, x, v, Hv, obj_weight=obj_weight)

function NLPModels.jprod!(qp :: AbstractQuadraticModel, x :: AbstractVector, v :: AbstractVector, Av :: AbstractVector)
  NLPModels.increment!(qp, :neval_jprod)
  coo_prod!(qp.data.Arows, qp.data.Acols, qp.data.Avals, v, Av)
  return Av
end

function NLPModels.jtprod!(qp :: AbstractQuadraticModel, x :: AbstractVector, v :: AbstractVector, Atv :: AbstractVector)
  NLPModels.increment!(qp, :neval_jtprod)
  coo_prod!(qp.data.Acols, qp.data.Arows, qp.data.Avals, v, Atv)
  return Atv
end

function SlackModel!(qp :: AbstractQuadraticModel)
  qp.meta.ncon == length(qp.meta.jfix) && return qp
  
  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix
  T = eltype(qp.data.c)
  append!(qp.data.Arows, qp.meta.jlow)
  append!(qp.data.Arows, qp.meta.jupp)
  append!(qp.data.Arows, qp.meta.jrng)
  append!(qp.data.Acols, qp.meta.nvar+1:qp.meta.nvar+ns)
  append!(qp.data.Avals, .-ones(T, ns))
  append!(qp.data.c, zeros(T, ns))

  qp.meta = NLPModels.slack_meta(qp.meta, name=qp.meta.name)
  return qp
end

function NLPModels.SlackModel(qp :: AbstractQuadraticModel, name=qp.meta.name * "-slack")
  qp.meta.ncon == length(qp.meta.jfix) && return qp
  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix
  T = eltype(qp.data.c)

  data = QPData(copy(qp.data.c0),
                [qp.data.c; zeros(T, ns)],
                copy(qp.data.Hrows),
                copy(qp.data.Hcols),
                copy(qp.data.Hvals),
                [qp.data.Arows; qp.meta.jlow; qp.meta.jupp; qp.meta.jrng],
                [qp.data.Acols; qp.meta.nvar+1:qp.meta.nvar+ns],
                [qp.data.Avals; .-ones(T, ns)],
                )
  meta = NLPModels.slack_meta(qp.meta, name=qp.meta.name)
  
  return QuadraticModel( meta, Counters(), data)
end
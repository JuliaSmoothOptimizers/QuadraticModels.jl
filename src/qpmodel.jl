export jac_structure!, hess_structure!, jac_coord!, hess_coord!

mutable struct QPData
    c0  :: Float64                  # constant term in objective
    c   :: AbstractVector{Float64}  # linear term
    H   :: AbstractMatrix{Float64}  # quadratic term
    opH :: AbstractLinearOperator   # assumed with preallocation!
    A   :: AbstractMatrix{Float64}  # constraint matrix
end

abstract type AbstractQuadraticModel <: AbstractNLPModel end

mutable struct QuadraticModel <: AbstractQuadraticModel
    meta     :: NLPModelMeta
    counters :: Counters
    data     :: QPData

  function QuadraticModel(c :: AbstractVector{Float64}, H :: AbstractMatrix{Float64},
                          opH :: AbstractLinearOperator,
                          A :: AbstractMatrix{Float64},
                          lcon :: AbstractVector{Float64}, ucon :: AbstractVector{Float64},
                          lvar :: AbstractVector{Float64}, uvar :: AbstractVector{Float64};
                          c0 :: Float64=0.0, kwargs...)
    ncon, nvar = size(A)
    nnzh = issparse(H) ? nnz(H) : (nvar * (nvar + 1) / 2)
    nnzj = issparse(A) ? nnz(A) : (nvar * ncon)
    new(NLPModelMeta(nvar,
                     lvar=lvar, uvar=uvar,
                     ncon=size(A,1), lcon=lcon, ucon=ucon,
                     nnzj=nnzj,
                     nnzh=nnzh,
                     lin=1:ncon, nln=Int[], islp=(ncon == 0); kwargs...),
        Counters(),
        QPData(c0, c, H, opH, A))
  end
end

function QuadraticModel(model :: AbstractNLPModel)
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    z = zeros(nvar)
    c = grad(model, z)
    H = hess(model, model.meta.x0)
    A = jac(model, z)
    QuadraticModel(c, H, opHermitian(H), A,
                   model.meta.lcon, model.meta.ucon,
                   model.meta.lvar, model.meta.uvar)
end

linobj(qp::AbstractQuadraticModel, args...) = qp.data.c

function NLPModels.obj(qp :: AbstractQuadraticModel, x :: AbstractVector)
    v = qp.data.opH * x
    f = qp.data.c0 + dot(qp.data.c, x) + 0.5 * dot(v, x)
    qp.counters.neval_hprod += 1
    f
end

function NLPModels.grad!(qp :: AbstractQuadraticModel, x :: AbstractVector, g :: AbstractVector)
    v = qp.data.opH * x
    @. g = qp.data.c + v
    qp.counters.neval_hprod += 1
    g
end

hess(qp :: AbstractQuadraticModel, ::AbstractVector; kwargs...) = qp.data.H

hess_op(qp :: AbstractQuadraticModel, ::AbstractVector; kwargs...) = qp.data.opH

"""
Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
function NLPModels.hess_structure!(qp :: QuadraticModel, rows :: Vector{<: Integer}, cols :: Vector{<: Integer})
    rows .= qp.data.H.rowval
    cols .= findnz(qp.data.H)[2]
end

"""
Evaluate the Lagrangian Hessian at `x` in sparse coordinate format. Only the lower triangle is returned.
"""
function NLPModels.hess_coord!(qp :: QuadraticModel, :: AbstractVector, rows :: AbstractVector{<: Integer},
                               cols :: AbstractVector{<: Integer}, vals :: Vector{<: AbstractFloat}; kwargs...)
    vals .= findnz(qp.data.H)[3]
end

jac(qp :: AbstractQuadraticModel, ::AbstractVector) = qp.data.A

jac_op(qp :: AbstractQuadraticModel, ::AbstractVector) = LinearOperator(qp.data.A)

jac_coord(qp :: AbstractQuadraticModel, ::AbstractVector) = findnz(qp.data.A)

"""
Return the structure of the constraints Jacobian in sparse coordinate format in place.
"""
function NLPModels.jac_structure!(qp :: QuadraticModel, rows :: Vector{<: Integer}, cols :: Vector{<: Integer})
    rows .= qp.data.A.rowval
    cols .= findnz(qp.data.A)[2]
end

"""
Return the structure of the constraints Jacobian in sparse coordinate format in place.
"""
function NLPModels.jac_coord!(qp :: QuadraticModel, x :: AbstractVector, rows :: Vector{<: Integer},
                              cols :: Vector{<: Integer}, vals :: Vector{<: AbstractFloat})
    vals .= findnz(qp.data.A)[3]
end

function NLPModels.cons!(qp :: AbstractQuadraticModel, x :: AbstractVector, c :: AbstractVector)
    mul!(c, qp.data.A, x)
    qp.counters.neval_jprod += 1
    c
end

function NLPModels.hprod!(qp :: AbstractQuadraticModel, x :: AbstractVector, v :: AbstractVector, Av :: AbstractVector)
    qp.counters.neval_hprod += 1
    Av .= qp.data.opH * v
    return Av
end

function NLPModels.jprod!(qp :: AbstractQuadraticModel, x :: AbstractVector, v :: AbstractVector, Av :: AbstractVector)
    qp.counters.neval_jprod += 1
    mul!(Av, qp.data.A, v)
    return Av
end

function NLPModels.jtprod!(qp :: AbstractQuadraticModel, x :: AbstractVector, v :: AbstractVector, Atv :: AbstractVector)
    qp.counters.neval_jtprod += 1
    mul!(Atv, qp.data.A', v)
    return Atv
end

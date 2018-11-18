mutable struct QPData
    c0::Float64                 # constant term in objective
    c::AbstractVector{Float64}  # linear term
    H::AbstractMatrix{Float64}  # quadratic term
    opH::AbstractLinearOperator # assumed with preallocation!
    A::AbstractMatrix{Float64}  # constraint matrix
end

mutable struct QPCounters
    neval_hprod::Int  # number of products with H
    neval_jprod::Int  # number of products with A
    neval_jtprod::Int # number of products with A'
    QPCounters() = new(0, 0, 0)
end

abstract type AbstractQuadraticModel <: AbstractNLPModel end

mutable struct QuadraticModel <: AbstractQuadraticModel
  meta::NLPModelMeta
  counters::QPCounters
  data::QPData

  function QuadraticModel(c::AbstractVector{Float64}, H::AbstractMatrix{Float64},
                          opH::AbstractLinearOperator,
                          A::AbstractMatrix{Float64},
                          lcon::AbstractVector{Float64}, ucon::AbstractVector{Float64},
                          lvar::AbstractVector{Float64}, uvar::AbstractVector{Float64};
                          c0::Float64=0.0, kwargs...)
    ncon, nvar = size(A)
    nnzh = issparse(H) ? nnz(H) : (nvar * (nvar + 1) / 2)
    nnzj = issparse(A) ? nnz(A) : (nvar * ncon)
    new(NLPModelMeta(nvar,
                     lvar=lvar, uvar=uvar,
                     ncon=size(A,1), lcon=lcon, ucon=ucon,
                     nnzj=nnzj,
                     nnzh=nnzh,
                     lin=1:ncon, nln=Int[], islp=(ncon == 0); kwargs...),
        QPCounters(),
        QPData(c0, c, H, opH, A))
  end
end

function QuadraticModel(model::AbstractNLPModel)
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

function objgrad(qp::AbstractQuadraticModel, x::AbstractVector)
    g = Vector{eltype(x)}(length(x))
    objgrad!(qp, x, g)
end

function objgrad!(qp::AbstractQuadraticModel, x::AbstractVector, g::AbstractVector)
    v = qp.data.opH * x
    @. g = qp.data.c + v
    f = qp.data.c0 + dot(qp.data.c, x) + 0.5 * dot(v, x)
    qp.counters.neval_hprod += 1
    (f, g)
end

function obj(qp::AbstractQuadraticModel, x::AbstractVector)
    v = qp.data.opH * x
    f = qp.data.c0 + dot(qp.data.c, x) + 0.5 * dot(v, x)
    qp.counters.neval_hprod += 1
    f
end

function grad(qp::AbstractQuadraticModel, x::AbstractVector)
    g = Vector{eltype(x)}(undef, qp.meta.nvar)
    grad!(qp, x, g)
end

function grad!(qp::AbstractQuadraticModel, x::AbstractVector, g::AbstractVector)
    v = qp.data.opH * x
    @. g = qp.data.c + v
    qp.counters.neval_hprod += 1
    g
end

hess_coord(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...) = findnz(qp.data.H)

hess(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...) = qp.data.H

hess_op(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...) = qp.data.opH

function cons(qp::AbstractQuadraticModel, x::AbstractVector)
    c = Vector{eltype(x)}(undef, qp.meta.ncon)
    cons!(qp, x, c)
end

function cons!(qp::AbstractQuadraticModel, x::AbstractVector, c::AbstractVector)
    mul!(c, qp.data.A, x)
    qp.counters.neval_jprod += 1
    c
end

jac_coord(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...) = findnz(qp.data.A)

jac(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...) = qp.data.A

jac_op(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...) = LinearOperator(qp.data.A)

function hprod(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...)
    @closure v -> begin
        qp.counters.neval_hprod += 1
        qp.data.opH * v
    end
end

function jprod(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...)
    @closure v -> begin
        qp.counters.neval_jprod += 1
        qp.data.A * v
    end
end

function jtprod(qp::AbstractQuadraticModel, ::AbstractVector; kwargs...)
    @closure v -> begin
        qp.counters.neval_jtprod += 1
        qp.data.A' * v
    end
end

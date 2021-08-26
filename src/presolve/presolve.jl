include("remove_ifix.jl")

mutable struct PresolvedQuadraticModel{T, S} <: AbstractQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::QPData{T, S}
  xrm::S
end

function copy_data(data::QPData, meta::NLPModelMeta)
  psdata = QPData(
    copy(data.c0),
    copy(data.c),
    copy(data.Hrows),
    copy(data.Hcols),
    copy(data.Hvals),
    copy(data.Arows),
    copy(data.Acols),
    copy(data.Avals),
  )
  lcon, ucon = copy(meta.lcon), copy(meta.ucon)
  lvar, uvar = copy(meta.lvar), copy(meta.uvar)
  nvar, ncon = copy(meta.nvar), copy(meta.ncon)

  return psdata, lcon, ucon, lvar, uvar, nvar, ncon 
end

"""
    psqm = presolve(qm::QuadraticModel{T, S}; kwargs...)

Apply a presolve routine to `qm` and returns a `PresolvedQuadraticModel{T, S} <: AbstractQuadraticModel{T, S}`.
The presolve operations currently implemented are:

- [`remove_ifix!`](@ref)

"""
function presolve(qm::QuadraticModel{T, S}; kwargs...) where {T <: Real, S}

  psdata, lcon, ucon, lvar, uvar, nvar, ncon = copy_data(qm.data, qm.meta)

  ifix = qm.meta.ifix
  if length(ifix) > 0
    xrm, psdata.c0, nvarps = remove_ifix!(
      ifix,
      psdata.Hrows,
      psdata.Hcols,
      psdata.Hvals,
      nvar,
      psdata.Arows,
      psdata.Acols,
      psdata.Avals,
      psdata.c,
      psdata.c0,
      lvar,
      uvar,
      lcon,
      ucon,
    )
  else
    nvarps = nvar
    xrm = S(undef, 0)
  end

  # form meta
  nnzh = length(psdata.Hvals)
  if !(nnzh == length(psdata.Hrows) == length(psdata.Hcols))
    error("The length of Hrows, Hcols and Hvals must be the same")
  end
  nnzj = length(psdata.Avals)
  if !(nnzj == length(psdata.Arows) == length(psdata.Acols))
    error("The length of Arows, Acols and Avals must be the same")
  end
  psmeta = NLPModelMeta{T, S}(
    nvarps,
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
  )
  ps = PresolvedQuadraticModel(psmeta, Counters(), psdata, xrm)

  return ps
end

"""
    postsolve!(qm::QuadraticModel{T, S}, psqm::PresolvedQuadraticModel{T, S}, 
               x_in::S, x_out::S) where {T, S}

Retrieve the solution `x_out` of the original QP `qm` given the solution of the presolved QP (`psqm`)
`x_in`.
"""
function postsolve!(qm::QuadraticModel{T, S}, psqm::PresolvedQuadraticModel{T, S}, x_in::S, x_out::S) where {T, S}
  if length(qm.meta.ifix) > 0
    restore_ifix!(qm.meta.ifix, psqm.xrm, x_in, x_out)
  end
end

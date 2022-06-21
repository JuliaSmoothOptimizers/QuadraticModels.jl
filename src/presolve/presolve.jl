include("remove_ifix.jl")
include("empty_rows.jl")
include("singleton_rows.jl")
include("postsolve_utils.jl")

mutable struct PresolvedData{T, S}
  ifix::Vector{Int}
  xrm::S
  row_cnt::Vector{Int}
  nconps::Int
end

mutable struct PresolvedQuadraticModel{T, S, M1, M2} <: AbstractQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::QPData{T, S, M1, M2}
  psd::PresolvedData{T, S}
end

"""
    stats_ps = presolve(qm::QuadraticModel{T, S}; kwargs...)

Apply a presolve routine to `qm` and returns a 
[`GenericExecutionStats`](https://juliasmoothoptimizers.github.io/SolverCore.jl/stable/reference/#SolverCore.GenericExecutionStats)
from the package [`SolverCore.jl`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl).
The presolve operations currently implemented are:

- [`remove_ifix!`](@ref) : remove fixed variables

The `PresolvedQuadraticModel{T, S} <: AbstractQuadraticModel{T, S}` is located in the `solver_specific` field:

    psqm = stats_ps.solver_specific[:presolvedQM]

and should be used to call [`postsolve!`](@ref).

If the presolved problem has 0 variables, `stats_ps.solution` contains a solution of the primal problem,
`stats_ps.multipliers` is a zero `SparseVector`, and, if we define

    s = qm.data.c + qm.data.H * stats_ps.solution

`stats_ps.multipliers_L` is the positive part of `s` and `stats_ps.multipliers_U` is the opposite of the negative part of `s`. 
"""
function presolve(
  qm::QuadraticModel{T, S, M1, M2};
  kwargs...,
) where {T <: Real, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO}
  start_time = time()
  psqm = deepcopy(qm)
  psdata = psqm.data
  lvar, uvar = psqm.meta.lvar, psqm.meta.uvar
  lcon, ucon = psqm.meta.lcon, psqm.meta.ucon
  nvar, ncon = psqm.meta.nvar, psqm.meta.ncon

  # empty rows
  row_cnt = zeros(Int, ncon)
  row_cnt!(psdata.A.rows, row_cnt) # number of coefficients per row
  rows_rm = removed_empty_rows(row_cnt) # indices of the empty rows
  if length(rows_rm) > 0
    Arows_sortperm = sortperm(psdata.A.rows) # permute rows 
    Arows_s = @views psdata.A.rows[Arows_sortperm]
    nconps = empty_rows!(psdata.A.rows, lcon, ucon, ncon, row_cnt, rows_rm, Arows_s)
  else
    nconps = ncon
  end

  # remove singleton rows
  if nconps != ncon
    row_cnt2 = Vector{Int}(undef, nconps)
  else
    row_cnt2 = row_cnt
  end
  row_cnt2 .= 0
  row_cnt!(psdata.A.rows, row_cnt2) # number of coefficients per rows
  singl_rows = removed_singleton_rows(row_cnt2) # indices of the empty rows
  if length(singl_rows) > 0
    nconps = singleton_rows!(psdata.A.rows, psdata.A.cols, psdata.A.vals, lcon, ucon, lvar, uvar, nvar, nconps, row_cnt2, singl_rows)
  else
    nconps = nconps
  end

  # remove fixed variables
  ifix = findall(lvar .== uvar)
  if length(ifix) > 0
    xrm, psdata.c0, nvarps = remove_ifix!(
      ifix,
      psdata.H.rows,
      psdata.H.cols,
      psdata.H.vals,
      nvar,
      psdata.A.rows,
      psdata.A.cols,
      psdata.A.vals,
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
  nnzh = length(psdata.H.vals)
  if !(nnzh == length(psdata.H.rows) == length(psdata.H.cols))
    error("The length of Hrows, Hcols and Hvals must be the same")
  end
  nnzj = length(psdata.A.vals)
  if !(nnzj == length(psdata.A.rows) == length(psdata.A.cols))
    error("The length of Arows, Acols and Avals must be the same")
  end

  if nvarps == 0
    feasible = all(qm.meta.lcon .<= qm.data.A * xrm .<= qm.meta.ucon)
    s = qm.data.c .+ Symmetric(qm.data.H, :L) * xrm
    i_l = findall(s .> zero(T))
    s_l = sparsevec(i_l, s[i_l])
    i_u = findall(s .< zero(T))
    s_u = sparsevec(i_u, .-s[i_u])
    return GenericExecutionStats(
      feasible ? :first_order : :infeasible,
      qm,
      solution = xrm,
      objective = obj(qm, xrm),
      multipliers = zeros(T, nconps),
      multipliers_L = s_l,
      multipliers_U = s_u,
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => nothing),
    )
  else
    psmeta = NLPModelMeta{T, S}(
      nvarps,
      lvar = lvar,
      uvar = uvar,
      ncon = nconps,
      lcon = lcon,
      ucon = ucon,
      nnzj = nnzj,
      nnzh = nnzh,
      lin = 1:nconps,
      islp = (nnzh == 0);
      minimize = qm.meta.minimize,
      kwargs...,
    )
    psd = PresolvedData{T, S}(ifix, xrm, row_cnt, nconps)
    ps = PresolvedQuadraticModel(psmeta, Counters(), psdata, psd)
    return GenericExecutionStats(
      :unknown,
      ps,
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => ps),
    )
  end
end

"""
    postsolve!(qm::QuadraticModel{T, S}, psqm::PresolvedQuadraticModel{T, S}, 
               x_in::S, x_out::S) where {T, S}

Retrieve the solution `x_out` of the original QP `qm` given the solution of the presolved QP (`psqm`)
`x_in`.
"""
function postsolve!(
  qm::QuadraticModel{T, S},
  psqm::PresolvedQuadraticModel{T, S},
  x_in::S,
  x_out::S,
  y_in::S,
  y_out::S,
  s_l::SparseVector{T, Int},
  s_u::SparseVector{T, Int},
) where {T, S}
  ifix = psqm.psd.ifix
  if length(ifix) > 0
    restore_ifix!(ifix, psqm.psd.xrm, x_in, x_out)
  else
    x_out .= @views x_in[1:(qm.meta.nvar)]
  end
  ncon = length(y_out)
  restore_y!(y_in, y_out, psqm.psd.row_cnt, ncon)

  ilow, iupp = s_l.nzind, s_u.nzind
  restore_ilow_iupp!(ilow, iupp, ifix)
  s_l_out = SparseVector(qm.meta.nvar, ilow, s_l.nzval)
  s_u_out = SparseVector(qm.meta.nvar, iupp, s_u.nzval)
  return s_l_out, s_u_out
end

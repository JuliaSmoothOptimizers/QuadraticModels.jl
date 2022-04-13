include("remove_ifix.jl")

mutable struct PresolvedQuadraticModel{T, S, M1, M2} <: AbstractQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::QPData{T, S, M1, M2}
  xrm::S
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

If the presolved problem has 0 variables, `stats_ps` contains a solution such that `stats_ps.solution` minimizes the primal problem,
`stats_ps.multipliers` is a `SparseVector` full of zeros, and, with

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

  ifix = qm.meta.ifix
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
        feasible ? :acceptable : :infeasible,
        qm,
        solution = xrm,
        objective = obj(qm, xrm),
        multipliers = zeros(T, qm.meta.nvar),
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
      ncon = ncon,
      lcon = lcon,
      ucon = ucon,
      nnzj = nnzj,
      nnzh = nnzh,
      lin = 1:ncon,
      islp = (ncon == 0);
      minimize = qm.meta.minimize,
      kwargs...,
    )
    ps = PresolvedQuadraticModel(psmeta, Counters(), psdata, xrm)
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
) where {T, S}
  if length(qm.meta.ifix) > 0
    restore_ifix!(qm.meta.ifix, psqm.xrm, x_in, x_out)
  else
    x_out .= @views x_in[1:(qm.meta.nvar)]
  end
end

abstract type PresolveOperation{T, S} end

"""
Type used to define a solution point when using [`postsolve`](@ref).

    sol = QMSolution(x, y, s_l, s_u)
"""
mutable struct QMSolution{S, V}
  x::S
  y::S
  s_l::V
  s_u::V
end

mutable struct Row{T}
  nzind::Vector{Int}
  nzval::Vector{T}
end

mutable struct Col{T}
  nzind::Vector{Int}
  nzval::Vector{T}
end

# struct for working data during presolve
mutable struct QuadraticModelPresolveData{T, S}
  # presolved x
  xps::S

  # problem data
  c::S
  c0::T
  arows::Vector{Row{T}}
  acols::Vector{Col{T}}
  hcols::Vector{Col{T}}
  lvar::S
  uvar::S
  lcon::S
  ucon::S
  nvar::Int
  ncon::Int

  # current kept rows/cols by presolve
  kept_rows::Vector{Bool}
  kept_cols::Vector{Bool}

  # current number of elements per row/col (-1 if not kept)
  row_cnt::Vector{Int}
  col_cnt::Vector{Int}

  # pass information for presolve reductions
  nb_pass::Int # number of passes in the presolve while loop
  empty_row_pass::Bool
  singl_row_pass::Bool
  free_lsc_pass::Bool
  free_row_pass::Bool
  ifix_pass::Bool

  # unbounded problem
  unbounded::Bool

  # infeasible problem
  infeasible_cst::Bool
  infeasible_bnd::Bool
end

function get_arows_acols(A::SparseMatrixCOO{T}, row_cnt, col_cnt, nvar, ncon) where {T}
  Arows, Acols, Avals = A.rows, A.cols, A.vals
  cnt_vec_rows = ones(Int, ncon)
  arows = [Row{T}(zeros(Int, row_cnt[i]), fill(T(Inf), row_cnt[i])) for i = 1:ncon]
  for k = 1:length(Arows)
    i, j, Ax = Arows[k], Acols[k], Avals[k]
    arows[i].nzind[cnt_vec_rows[i]] = j
    arows[i].nzval[cnt_vec_rows[i]] = Ax
    cnt_vec_rows[i] += 1
  end

  cnt_vec_cols = ones(Int, nvar)
  acols = [Col{T}(zeros(Int, col_cnt[j]), fill(T(Inf), col_cnt[j])) for j = 1:nvar]
  for k = 1:length(Arows)
    i, j, Ax = Arows[k], Acols[k], Avals[k]
    acols[j].nzind[cnt_vec_cols[j]] = i
    acols[j].nzval[cnt_vec_cols[j]] = Ax
    cnt_vec_cols[j] += 1
  end
  return arows, acols
end

function get_hcols(H::SparseMatrixCOO{T}, nvar) where {T}
  Hrows, Hcols, Hvals = H.rows, H.cols, H.vals
  hcol_cnt = zeros(Int, nvar)
  for k = 1:length(Hrows)
    i, j = Hrows[k], Hcols[k]
    hcol_cnt[i] += 1
    (i != j) && (hcol_cnt[j] += 1)
  end
  hcols = [Col{T}(zeros(Int, hcol_cnt[i]), fill(T(Inf), hcol_cnt[i])) for i = 1:nvar]

  cnt_vec_cols = ones(Int, nvar)
  for k = 1:length(Hrows)
    i, j, Hx = Hrows[k], Hcols[k], Hvals[k]
    hcols[j].nzind[cnt_vec_cols[j]] = i
    hcols[j].nzval[cnt_vec_cols[j]] = Hx
    cnt_vec_cols[j] += 1
    if i != j
      hcols[i].nzind[cnt_vec_cols[i]] = j
      hcols[i].nzval[cnt_vec_cols[i]] = Hx
      cnt_vec_cols[i] += 1
    end
  end
  return hcols
end

mutable struct PresolvedData{T, S}
  xps::S
  arows::Vector{Row{T}}
  acols::Vector{Col{T}}
  hcols::Vector{Col{T}}
  kept_rows::Vector{Bool}
  kept_cols::Vector{Bool}
  nvarps::Int
  nconps::Int
  nvar::Int
  ncon::Int
  operations::Vector{PresolveOperation{T, S}}
end

include("presolve_utils.jl")
include("remove_ifix.jl")
include("empty_rows.jl")
include("singleton_rows.jl")
include("unconstrained_reductions.jl")
include("linear_singleton_columns.jl")
include("primal_constraints.jl")
include("free_rows.jl")
include("postsolve_utils.jl")

mutable struct PresolvedQuadraticModel{T, S, M1, M2} <: AbstractQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::QPData{T, S, M1, M2}
  psd::PresolvedData{T, S}
end

function check_bounds!(qmp::QuadraticModelPresolveData)
  lvar, uvar, lcon, ucon = qmp.lvar, qmp.uvar, qmp.lcon, qmp.ucon
  kept_rows, kept_cols = qmp.kept_rows, qmp.kept_cols
  for i = 1:(qmp.ncon)
    if kept_rows[i] && lcon[i] > ucon[i]
      @warn "row $i primal infeasible"
      qmp.infeasible_bnd = true
      return nothing
    end
  end
  for j = 1:(qmp.nvar)
    if kept_cols[j] && lvar[j] > uvar[j]
      @warn "col $j primal infeasible"
      qmp.infeasible_bnd = true
      return nothing
    end
  end
end

"""
    stats_ps = presolve(qm::QuadraticModel{T, S}; fixed_vars_only = false, kwargs...)

Apply a presolve routine to `qm` and returns a 
[`GenericExecutionStats`](https://juliasmoothoptimizers.github.io/SolverCore.jl/stable/reference/#SolverCore.GenericExecutionStats)
from the package [`SolverCore.jl`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl).
The presolve operations currently implemented are:

- remove empty rows
- remove singleton rows
- fix linearly unconstrained variables (lps)
- remove free linear singleton columns whose associated variable does not appear in the hessian
- remove fixed variables

The `PresolvedQuadraticModel{T, S} <: AbstractQuadraticModel{T, S}` is located in the `solver_specific` field:

    psqm = stats_ps.solver_specific[:presolvedQM]

and should be used to call [`postsolve`](@ref).
Use `fixed_vars_only = true` if you only want to remove fixed variables.
Maximization problems are transformed to minimization problems.
If you need the objective of a presolved maximization problem, make sure to take the opposite of the objective of the presolved problem.

If the presolved problem has 0 variables, `stats_ps.solution` contains a solution of the primal problem,
`stats_ps.multipliers` is a zero `SparseVector`, and, if we define

    s = qm.data.c + qm.data.H * stats_ps.solution

`stats_ps.multipliers_L` is the positive part of `s` and `stats_ps.multipliers_U` is the opposite of the negative part of `s`.
The presolve operations are inspired from [`MathOptPresolve.jl`](https://github.com/mtanneau/MathOptPresolve.jl), and from:

* Gould, N., Toint, P. [*Preprocessing for quadratic programming*](https://doi.org/10.1007/s10107-003-0487-2), Math. Program., Ser. B 100, 95–132 (2004). 
"""
function presolve(
  qm::QuadraticModel{T, S, M1, M2};
  fixed_vars_only::Bool = false,
  kwargs...,
) where {T <: Real, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO}
  start_time = time()
  psqm = copy_qm(qm)
  psdata = psqm.data
  c = psdata.c
  lvar, uvar = psqm.meta.lvar, psqm.meta.uvar
  lcon, ucon = psqm.meta.lcon, psqm.meta.ucon
  nvar, ncon = psqm.meta.nvar, psqm.meta.ncon
  # copy if same vector
  lcon === ucon && (lcon = copy(lcon))
  lvar === uvar && (lvar = copy(lvar))
  row_cnt = zeros(Int, ncon)
  col_cnt = zeros(Int, nvar)
  kept_rows = fill(true, ncon)
  kept_cols = fill(true, nvar)
  nconps = ncon
  nvarps = nvar
  xps = S(undef, nvar)

  dropzeros!(psdata.A)
  dropzeros!(psdata.H)

  # number of coefficients per row
  vec_cnt!(row_cnt, psdata.A.rows)
  # number of coefficients per col
  vec_cnt!(col_cnt, psdata.A.cols)

  # get list of rows and list of columns of A
  arows, acols = get_arows_acols(psdata.A, row_cnt, col_cnt, nvar, ncon)
  # get list of columns of H
  hcols = get_hcols(psdata.H, nvar)

  qmp = QuadraticModelPresolveData(
    xps,
    c,
    psdata.c0,
    arows,
    acols,
    hcols,
    lvar,
    uvar,
    lcon,
    ucon,
    nvar,
    ncon,
    kept_rows,
    kept_cols,
    row_cnt,
    col_cnt,
    0,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
  )
  operations = PresolveOperation{T, S}[]
  keep_iterating = true
  infeasible = false

  if fixed_vars_only
    remove_ifix!(qmp, operations)
    keep_iterating = false
  end

  while keep_iterating
    empty_rows!(qmp, operations)
    singleton_rows!(qmp, operations)
    unconstrained_reductions!(qmp, operations)
    free_linear_singleton_columns!(qmp, operations)
    primal_constraints!(qmp, operations)
    free_rows!(qmp, operations)
    remove_ifix!(qmp, operations)
    check_bounds!(qmp)

    infeasible = qmp.infeasible_bnd || qmp.infeasible_cst
    # check if some presolve operations have been applied:
    reduction_pass = check_reductions(qmp)
    keep_iterating = reduction_pass && !qmp.unbounded && !infeasible
    keep_iterating && (qmp.nb_pass += 1)
  end

  if !isempty(operations)
    remove_rowscols_A_H!(psdata.A, psdata.H, qmp)
    nconps, nvarps = update_vectors!(qmp)
    psdata.c0 = qmp.c0
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

  if qmp.unbounded
    return GenericExecutionStats(
      :unbounded,
      qm,
      solution = qmp.xps,
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => nothing),
    )
  elseif infeasible
    return GenericExecutionStats(
      :infeasible,
      qm,
      solution = qmp.xps,
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => nothing),
    )
  elseif nvarps == 0
    feasible = all(qm.meta.lcon .<= qm.data.A * qmp.xps .<= qm.meta.ucon)
    s = qm.data.c .+ Symmetric(qm.data.H, :L) * qmp.xps
    return GenericExecutionStats(
      feasible ? :first_order : :infeasible,
      qm,
      solution = qmp.xps,
      objective = obj(qm, qmp.xps),
      multipliers = zeros(T, ncon),
      multipliers_L = max.(s, zero(T)),
      multipliers_U = max.(.-s, zero(T)),
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => nothing, :psoperations => operations),
    )
  else
    psmeta = NLPModelMeta{T, S}(
      nvarps,
      lvar = qmp.lvar,
      uvar = qmp.uvar,
      ncon = nconps,
      lcon = qmp.lcon,
      ucon = qmp.ucon,
      nnzj = nnzj,
      lin_nnzj = nnzj,
      nln_nnzj = 0,
      nnzh = nnzh,
      lin = 1:nconps,
      islp = (nnzh == 0);
      minimize = qm.meta.minimize,
      kwargs...,
    )
    psd = PresolvedData{T, S}(
      qmp.xps,
      qmp.arows,
      qmp.acols,
      qmp.hcols,
      qmp.kept_rows,
      qmp.kept_cols,
      nvarps,
      nconps,
      nvar,
      ncon,
      operations,
    )
    ps = PresolvedQuadraticModel(psmeta, Counters(), psdata, psd)
    return GenericExecutionStats(
      :unknown,
      ps,
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => ps, :psoperations => operations),
    )
  end
end

function postsolve!(
  qm::QuadraticModel{T, S},
  psqm::PresolvedQuadraticModel{T, S},
  sol::QMSolution{S},
  sol_in::QMSolution{S},
) where {T, S}
  x_in, y_in, s_l_in, s_u_in = sol_in.x, sol_in.y, sol_in.s_l, sol_in.s_u
  psd = psqm.psd
  n_operations = length(psd.operations)
  nvar = psd.nvar
  @assert nvar == length(sol.x)
  restore_x!(psd.kept_cols, x_in, sol.x, nvar)
  ncon = psd.ncon
  @assert ncon == length(sol.y)
  restore_y!(psd.kept_rows, y_in, sol.y, ncon)
  restore_s!(sol.s_l, sol.s_u, s_l_in, s_u_in, psd.kept_cols)

  for i = n_operations:-1:1
    operation_i = psd.operations[i]
    postsolve!(sol, operation_i, psd)
  end
end

"""
    sol = postsolve(qm::QuadraticModel{T, S}, psqm::PresolvedQuadraticModel{T, S}, 
                    sol_in::QMSolution{S}) where {T, S}

Retrieve the solution `sol = (x, y, s_l, s_u)` of the original QP `qm` given the solution of the presolved QP (`psqm`)
`sol_in` of type [`QMSolution`](@ref).
`sol_in.s_l` and `sol_in.s_u` can be sparse or dense vectors, but the output `sol.s_l` and `sol.s_u` are dense vectors. 
"""
function postsolve(
  qm::QuadraticModel{T, S},
  psqm::PresolvedQuadraticModel{T, S},
  sol_in::QMSolution{S},
) where {T, S}
  x = fill!(S(undef, psqm.psd.nvar), zero(T))
  y = fill!(S(undef, psqm.psd.ncon), zero(T))
  s_l = fill!(S(undef, psqm.psd.nvar), zero(T))
  s_u = fill!(S(undef, psqm.psd.nvar), zero(T))

  sol = QMSolution(x, y, s_l, s_u)
  postsolve!(qm, psqm, sol, sol_in)
  return sol
end

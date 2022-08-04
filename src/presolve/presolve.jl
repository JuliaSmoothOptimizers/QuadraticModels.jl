abstract type PresolveOperation{T, S} end

"""
Type used to define a solution point when using [`postsolve`](@ref).

    sol = QMSolution(x, y, s_l, s_u)

where `s_l` and `s_u` should be of type `SparseVector`.
"""
mutable struct QMSolution{T, S}
  x::S
  y::S
  s_l::SparseVector{T, Int}
  s_u::SparseVector{T, Int}
end

mutable struct Row{T}
  nzind::Vector{Int}
  nzval::Vector{T}
end

mutable struct Col{T}
  nzind::Vector{Int}
  nzval::Vector{T}
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
  nconps::Int
  nvarps::Int
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

function check_bounds(lvar, uvar, lcon, ucon, nvar, ncon, kept_rows, kept_cols)
  for i = 1:ncon
    if kept_rows[i] && lcon[i] > ucon[i]
      @warn "row $i primal infeasible"
      return true
    end
  end
  for j = 1:nvar
    if kept_cols[j] && lvar[j] > uvar[j]
      @warn "col $j primal infeasible"
      return true
    end
  end
  return false
end

"""
    stats_ps = presolve(qm::QuadraticModel{T, S}; kwargs...)

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
  nb_pass = 1
  keep_iterating = true
  unbounded = false
  infeasible = false
  operations = PresolveOperation{T, S}[]

  # number of coefficients per row
  vec_cnt!(row_cnt, psdata.A.rows)
  # number of coefficients per col
  vec_cnt!(col_cnt, psdata.A.cols)

  # get list of rows and list of columns of A
  arows, acols = get_arows_acols(psdata.A, row_cnt, col_cnt, nvar, ncon)
  # get list of columns of H
  hcols = get_hcols(psdata.H, nvar)

  while keep_iterating
    empty_row_pass = empty_rows!(operations, lcon, ucon, ncon, row_cnt, kept_rows)

    singl_row_pass = singleton_rows!(
      operations,
      arows,
      lcon,
      ucon,
      lvar,
      uvar,
      ncon,
      row_cnt,
      col_cnt,
      kept_rows,
      kept_cols,
    )

    unbounded =
      unconstrained_reductions!(operations, c, hcols, lvar, uvar, xps, nvar, col_cnt, kept_cols)

    free_lsc_pass, psdata.c0 = free_linear_singleton_columns!(
      operations,
      hcols,
      arows,
      acols,
      c,
      psdata.c0,
      lcon,
      ucon,
      lvar,
      uvar,
      nvar,
      row_cnt,
      col_cnt,
      kept_rows,
      kept_cols,
    )

    infeasible_cst = primal_constraints!(
      operations,
      arows,
      lcon,
      ucon,
      lvar,
      uvar,
      nvar,
      ncon,
      kept_rows,
      kept_cols,
      row_cnt,
      col_cnt,
    )

    free_rows_pass = free_rows!(operations, lcon, ucon, ncon, row_cnt, kept_rows)

    psdata.c0, ifix_pass = remove_ifix!(
      operations,
      hcols,
      acols,
      c,
      psdata.c0,
      lvar,
      uvar,
      lcon,
      ucon,
      nvar,
      row_cnt,
      col_cnt,
      kept_rows,
      kept_cols,
      xps,
    )

    infeasible_bnd = check_bounds(lvar, uvar, lcon, ucon, nvar, ncon, kept_rows, kept_cols)

    infeasible = infeasible_bnd || infeasible_cst
    reduction_pass =
      empty_row_pass || singl_row_pass || ifix_pass || free_lsc_pass || free_rows_pass
    keep_iterating = reduction_pass && !unbounded && !infeasible
    keep_iterating && (nb_pass += 1)
  end

  if !isempty(operations)
    remove_rowscols_A!(
      psdata.A.rows,
      psdata.A.cols,
      psdata.A.vals,
      kept_rows,
      kept_cols,
      nvar,
      ncon,
    )
    remove_rowscols_H!(psdata.H.rows, psdata.H.cols, psdata.H.vals, kept_cols, nvar)
    nconps, nvarps = update_vectors!(lcon, ucon, c, lvar, uvar, kept_rows, kept_cols, ncon, nvar)
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

  if unbounded
    return GenericExecutionStats(
      :unbounded,
      qm,
      solution = xps,
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => nothing),
    )
  elseif infeasible
    return GenericExecutionStats(
      :infeasible,
      qm,
      solution = xps,
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => nothing),
    )
  elseif nvarps == 0
    feasible = all(qm.meta.lcon .<= qm.data.A * xps .<= qm.meta.ucon)
    s = qm.data.c .+ Symmetric(qm.data.H, :L) * xps
    i_l = findall(s .> zero(T))
    s_l = sparsevec(i_l, s[i_l])
    i_u = findall(s .< zero(T))
    s_u = sparsevec(i_u, .-s[i_u])
    return GenericExecutionStats(
      feasible ? :first_order : :infeasible,
      qm,
      solution = xps,
      objective = obj(qm, xps),
      multipliers = zeros(T, ncon),
      multipliers_L = s_l,
      multipliers_U = s_u,
      iter = 0,
      elapsed_time = time() - start_time,
      solver_specific = Dict(:presolvedQM => nothing, :psoperations => operations),
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
      lin_nnzj = nnzj,
      nln_nnzj = 0,
      nnzh = nnzh,
      lin = 1:nconps,
      islp = (nnzh == 0);
      minimize = qm.meta.minimize,
      kwargs...,
    )
    psd = PresolvedData{T, S}(
      xps,
      arows,
      acols,
      hcols,
      kept_rows,
      kept_cols,
      nconps,
      nvarps,
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
  sol::QMSolution{T, S},
  sol_in::QMSolution{T, S},
) where {T, S}
  x_in, y_in = sol_in.x, sol_in.y
  psd = psqm.psd
  n_operations = length(psd.operations)
  nvar = psd.nvar
  @assert nvar == length(sol.x)
  restore_x!(psd.kept_cols, x_in, sol.x, nvar)
  ncon = psd.ncon
  @assert ncon == length(sol.y)
  restore_y!(psd.kept_rows, y_in, sol.y, ncon)
  for i = n_operations:-1:1
    operation_i = psd.operations[i]
    postsolve!(sol, operation_i, psd)
  end
end

"""
    sol = postsolve(qm::QuadraticModel{T, S}, psqm::PresolvedQuadraticModel{T, S}, 
                    sol_in::QMSolution{T, S}) where {T, S}

Retrieve the solution `sol = (x, y, s_l, s_u)` of the original QP `qm` given the solution of the presolved QP (`psqm`)
`sol_in` of type [`QMSolution`](@ref).
"""
function postsolve(
  qm::QuadraticModel{T, S},
  psqm::PresolvedQuadraticModel{T, S},
  sol_in::QMSolution{T, S},
) where {T, S}
  x = fill!(S(undef, psqm.psd.nvar), zero(T))
  y = fill!(S(undef, psqm.psd.ncon), zero(T))
  s_l = sol_in.s_l
  s_u = sol_in.s_u

  ilow, iupp = s_l.nzind, s_u.nzind
  restore_ilow_iupp!(ilow, iupp, psqm.psd.kept_cols)
  sol = QMSolution(
    x,
    y,
    SparseVector(qm.meta.nvar, ilow, s_l.nzval),
    SparseVector(qm.meta.nvar, iupp, s_u.nzval),
  )
  postsolve!(qm, psqm, sol, sol_in)
  return sol
end

include("presolve_utils.jl")
include("remove_ifix.jl")
include("empty_rows.jl")
include("singleton_rows.jl")
include("unconstrained_reductions.jl")
include("postsolve_utils.jl")

mutable struct PresolvedData{T, S}
  xps::S
  kept_rows::Vector{Bool}
  kept_cols::Vector{Bool}
  nconps::Int
  nvarps::Int
  operations_history::Vector{Int}
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

- [`empty_rows!`](@ref) : remove empty rows
- [`singleton_rows!`](@ref) : remove singleton rows
- [`unconstrained_reductions!`](@ref) : fix linearly unconstrained variables (lps)
- [`remove_ifix!`](@ref) : remove fixed variables

The `PresolvedQuadraticModel{T, S} <: AbstractQuadraticModel{T, S}` is located in the `solver_specific` field:

    psqm = stats_ps.solver_specific[:presolvedQM]

and should be used to call [`postsolve`](@ref).

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
  @assert qm.meta.minimize
  psqm = deepcopy(qm)
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
  # history:
  # 1 empty rows
  # 2 singleton rows
  # 3 linearly unconstrained variables
  # 4 fixed variables
  operations_history = Int[]

  # number of coefficients per row
  vec_cnt!(row_cnt, psdata.A.rows)
  # number of coefficients per col
  vec_cnt!(col_cnt, psdata.A.cols)

  while keep_iterating

    # empty rows
    empty_rows = find_empty_rowscols(row_cnt) # indices of the empty rows
    if !isempty(empty_rows)
      empty_row_pass = true
      empty_rows!(lcon, ucon, empty_rows, row_cnt, kept_rows)
      push!(operations_history, 1)
    else
      empty_row_pass = false
    end

    singl_rows = find_singleton_rowscols(row_cnt) # indices of the singleton rows
    if !isempty(singl_rows)
      singl_row_pass = true
      singleton_rows!(
        psdata.A.rows,
        psdata.A.cols,
        psdata.A.vals,
        lcon,
        ucon,
        lvar,
        uvar,
        singl_rows,
        row_cnt,
        col_cnt,
        kept_rows,
      )
      push!(operations_history, 2)
    else
      singl_row_pass = false
    end

    # unconstrained reductions
    lin_unconstr_vars = find_empty_rowscols(col_cnt) # indices of the singleton cols
    if !isempty(lin_unconstr_vars)
      unbounded = unconstrained_reductions!(
        c,
        psdata.H.rows,
        psdata.H.cols,
        psdata.H.vals,
        lvar,
        uvar,
        xps,
        lin_unconstr_vars,
      )
      push!(operations_history, 3)
    end

    # remove fixed variables
    ifix = find_fixed_variables(lvar, uvar, kept_cols)
    if !isempty(ifix)
      ifix_pass = true
      psdata.c0 = remove_ifix!(
        ifix,
        psdata.H.rows,
        psdata.H.cols,
        psdata.H.vals,
        nvarps,
        psdata.A.rows,
        psdata.A.cols,
        psdata.A.vals,
        c,
        psdata.c0,
        lvar,
        lcon,
        ucon,
        row_cnt,
        col_cnt,
        kept_cols,
        xps,
      )
      push!(operations_history, 4)
    else
      ifix_pass = false
    end

    keep_iterating = (empty_row_pass || singl_row_pass || ifix_pass) && !unbounded
    keep_iterating && (nb_pass += 1)
  end

  if !isempty(operations_history)
    remove_rowscols_A!(psdata.A.rows, psdata.A.cols, psdata.A.vals, kept_rows, kept_cols, nvar, ncon)
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
    psd = PresolvedData{T, S}(xps, kept_rows, kept_cols, nconps, nvarps, operations_history)
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
  restore_ifix!(psqm.psd.kept_cols, psqm.psd.xps, x_in, x_out)
  ncon = length(y_out)
  restore_y!(y_in, y_out, psqm.psd.kept_rows, ncon)

  ilow, iupp = s_l.nzind, s_u.nzind
  restore_ilow_iupp!(ilow, iupp, psqm.psd.kept_cols)
  s_l_out = SparseVector(qm.meta.nvar, ilow, s_l.nzval)
  s_u_out = SparseVector(qm.meta.nvar, iupp, s_u.nzval)
  return s_l_out, s_u_out
end

"""
    x, y, s_l, s_u = postsolve(qm::QuadraticModel{T, S}, psqm::PresolvedQuadraticModel{T, S}, 
                               x_in::S, y_in::S,
                               s_l_in::SparseVector{T, Int},
                               s_u_in::SparseVector{T, Int}) where {T, S}

Retrieve the solution `x, y, s_l, s_u` of the original QP `qm` given the solution of the presolved QP (`psqm`)
`x_in, y_in, s_l_in, s_u_in`.
"""
function postsolve(
  qm::QuadraticModel{T, S},
  psqm::PresolvedQuadraticModel{T, S},
  x_in::S,
  y_in::S,
  s_l::SparseVector{T, Int},
  s_u::SparseVector{T, Int},
) where {T, S}
  x_out = similar(qm.meta.x0)
  y_out = similar(qm.meta.y0)
  s_l_out, s_u_out = postsolve!(qm, psqm, x_in, x_out, y_in, y_out, s_l, s_u)
  return x_out, y_out, s_l_out, s_u_out
end

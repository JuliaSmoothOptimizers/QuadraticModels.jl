struct SingletonRow{T, S} <: PresolveOperation{T, S}
  i::Int # idx of singleton row
  j::Int
  aij::T
  tightened_lvar::Bool
  tightened_uvar::Bool
end

function singleton_rows!(
  qmp::QuadraticModelPresolveData{T, S},
  operations::Vector{PresolveOperation{T, S}},
) where {T, S}
  # assume Acols is sorted
  qmp.singl_row_pass = false
  arows, lcon, ucon, lvar, uvar = qmp.arows, qmp.lcon, qmp.ucon, qmp.lvar, qmp.uvar
  row_cnt, col_cnt = qmp.row_cnt, qmp.col_cnt
  kept_rows, kept_cols = qmp.kept_rows, qmp.kept_cols

  for i = 1:(qmp.ncon)
    (kept_rows[i] && (row_cnt[i] == 1)) || continue
    qmp.singl_row_pass = true
    tightened_lvar = false
    tightened_uvar = false
    aij = T(Inf)
    rowi = arows[i]
    k = 1
    j = rowi.nzind[k]
    while !(kept_cols[j])
      k += 1
      j = rowi.nzind[k]
    end
    aij = rowi.nzval[k]

    if aij > zero(T)
      lvar2 = lcon[i] / aij
      uvar2 = ucon[i] / aij
    elseif aij < zero(T)
      uvar2 = lcon[i] / aij
      lvar2 = ucon[i] / aij
    else
      error("remove explicit zeros in A")
    end
    if lvar[j] ≤ lvar2
      lvar[j] = lvar2
      tightened_lvar = true
    end
    if uvar[j] ≥ uvar2
      uvar[j] = uvar2
      tightened_uvar = true
    end

    row_cnt[i] = -1
    kept_rows[i] = false
    col_cnt[j] -= 1
    push!(operations, SingletonRow{T, S}(i, j, aij, tightened_lvar, tightened_uvar))
  end
end

function postsolve!(
  sol::QMSolution,
  operation::SingletonRow{T, S},
  psd::PresolvedData{T, S},
) where {T, S}
  i, j = operation.i, operation.j
  psd.kept_rows[i] = true
  aij = operation.aij
  sol.y[i] = zero(T)
  if operation.tightened_lvar
    sol.y[i] += sol.s_l[j] / aij
    sol.s_l[j] = zero(T)
  end
  if operation.tightened_uvar
    sol.y[i] -= sol.s_u[j] / aij
    sol.s_u[j] = zero(T)
  end
  if !operation.tightened_lvar && !operation.tightened_uvar
    sol.y[i] = zero(T)
  end
end

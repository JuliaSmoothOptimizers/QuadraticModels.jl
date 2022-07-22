struct SingletonRow{T, S} <: PresolveOperation{T, S}
  i::Int # idx of singleton row
  j::Int
  Aij::T
  tightened_lvar::Bool
  tightened_uvar::Bool
end

"""
    singleton_rows!(Arows, Acols, Avals, lcon, ucon,
                    lvar, uvar, nvar, ncon, row_cnt, singl_rows,
                    row_cnt, col_cnt, kept_rows)

Presolve procedure for singleton rows of A in `singl_rows`.
"""
function singleton_rows!(
  operations::Vector{PresolveOperation{T, S}},
  Arows,
  Acols,
  Avals,
  lcon::S,
  ucon::S,
  lvar,
  uvar,
  ncon,
  row_cnt,
  col_cnt,
  kept_rows,
  kept_cols,
) where {T, S}
  # assume Acols is sorted
  Annz = length(Arows)
  singl_row_pass = false

  for i=1:ncon
    (kept_rows[i] && (row_cnt[i] == 1)) || continue
    singl_row_pass = true
    k = 1
    tightened_lvar = false
    tightened_uvar = false
    j = 0
    Ax = T(Inf)
    found_singl = false
    while k <= Annz && !found_singl
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Ai == i && kept_cols[Aj]
        found_singl = true
        j = Aj
        col_cnt[Aj] -= 1
        if Ax > zero(T)
          lvar2 = lcon[i] / Ax
          uvar2 = ucon[i] / Ax
        elseif Ax < zero(T)
          uvar2 = lcon[i] / Ax
          lvar2 = ucon[i] / Ax
        else
          error("remove explicit zeros in A")
        end
        if lvar[Aj] < lvar2
          lvar[Aj] = lvar2
          tightened_lvar = true
        end
        if uvar[Aj] > uvar2
          uvar[Aj] = uvar2
          tightened_uvar = true
        end
      end
      k += 1
    end
    row_cnt[i] = -1
    kept_rows[i] = false
    push!(operations, SingletonRow{T, S}(i, j, Ax, tightened_lvar, tightened_uvar))
  end
  return singl_row_pass
end

function postsolve!(pt::OutputPoint{T, S}, operation::SingletonRow{T, S}) where {T, S}
  i, j = operation.i, operation.j
  dual_slack = -pt.s_l[j] + pt.s_u[j]
  Aij = operation.Aij
  if operation.tightened_lvar
    pt.y[i] = dual_slack / Aij
    pt.s_l[j] = zero(T)
  end
  if operation.tightened_uvar
    pt.y[i] = dual_slack / Aij
    pt.s_u[j] = zero(T)
  end
  if !operation.tightened_lvar && !operation.tightened_uvar
    pt.y[i] = zero(T)
  end
end
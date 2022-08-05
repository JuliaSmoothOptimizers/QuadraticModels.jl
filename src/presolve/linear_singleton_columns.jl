struct FreeLinearSingletonColumn{T, S} <: PresolveOperation{T, S}
  i::Int
  j::Int
  aij::T
  yi::T
  conival::T
end

function free_linear_singleton_columns!(
  operations::Vector{PresolveOperation{T, S}},
  hcols::Vector{Col{T}},
  arows::Vector{Row{T}},
  acols::Vector{Col{T}},
  c::AbstractVector{T},
  c0::T,
  lcon::AbstractVector{T},
  ucon::AbstractVector{T},
  lvar::AbstractVector{T},
  uvar::AbstractVector{T},
  nvar,
  row_cnt,
  col_cnt,
  kept_rows,
  kept_cols,
) where {T, S}
  free_lsc_pass = false
  c0_offset = zero(T)

  for j = 1:nvar
    (kept_cols[j] && (col_cnt[j] == 1)) || continue
    # check infinity bounds and no hessian contribution
    if lvar[j] == -T(Inf) && uvar[j] == T(Inf) && isempty(hcols[j].nzind)
      # find i the row index of the singleton column j, and aij
      aij = T(Inf)
      colj = acols[j]
      k = 1
      i = colj.nzind[k]
      while !(kept_rows[i])
        k += 1
        i = colj.nzind[k]
      end
      aij = colj.nzval[k]

      yi = c[j] / aij
      nzcj = c[j] != zero(T)
      if yi < zero(T)
        lcon[i] = ucon[i]
      elseif yi > zero(T)
        ucon[i] = lcon[i]
      end
      if abs(aij) > sqrt(eps(T))
        nzcj && (c0_offset += yi * ucon[i]) # update c0
        rowi = arows[i] # i-th row
        # new row to store for postsolve:
        for (l, ail) in zip(rowi.nzind, rowi.nzval)
          # add all col elements to rowi2 except the col j2 == j
          (kept_cols[l] && l != j) || continue
          nzcj && (c[l] -= yi * ail) # update c if c[j] != 0
          col_cnt[l] -= 1
        end
        conival = nzcj ? ucon[i] : (lcon[i] + ucon[i]) / 2 # constant for postsolve
        push!(operations, FreeLinearSingletonColumn{T, S}(i, j, aij, yi, conival))
        kept_cols[j] = false
        col_cnt[j] = -1
        kept_rows[i] = false
        row_cnt[i] = -1
        free_lsc_pass = true
      end
    end
  end
  return free_lsc_pass, c0 + c0_offset
end

function postsolve!(
  sol::QMSolution,
  operation::FreeLinearSingletonColumn{T, S},
  psd::PresolvedData{T, S},
) where {T, S}
  x = sol.x
  kept_rows, kept_cols = psd.kept_rows, psd.kept_cols
  i, j = operation.i, operation.j
  arowi = psd.arows[i]
  kept_rows[i] = true
  kept_cols[j] = true
  # x[j] = (coival - Σₖ Aik x[k]) / aij , where k ≂̸ j
  x[j] = operation.conival
  for (l, ail) in zip(arowi.nzind, arowi.nzval)
    (l != j && kept_cols[l]) || continue 
    x[j] -= ail * x[l]
  end
  x[j] /= operation.aij
  sol.s_l[j] = zero(T)
  sol.s_u[j] = zero(T)
  sol.y[operation.i] = operation.yi
end

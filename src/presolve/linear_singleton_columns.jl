struct FreeLinearSingletonColumn{T, S} <: PresolveOperation{T, S}
  i::Int
  j::Int
  aij::T
  arowi::Row{T}
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
      # find i the row index of the singleton column j, and Aij
      Aij = T(Inf)
      colj = acols[j]
      k = 1
      i = colj.nzind[k]
      while !(kept_rows[i])
        k += 1
        i = colj.nzind[k]
      end
      Aij = colj.nzval[k]

      yi = c[j] / Aij
      nzcj = c[j] != zero(T)
      if yi < zero(T)
        lcon[i] = ucon[i]
      elseif yi > zero(T)
        ucon[i] = lcon[i]
      end
      if abs(Aij) > sqrt(eps(T))
        nzcj && (c0_offset += yi * ucon[i]) # update c0
        nb_elem_i = row_cnt[i] - 1
        rowi = arows[i] # i-th row
        # new row to store for postsolve:
        rowi2 = Row(zeros(Int, nb_elem_i), zeros(T, nb_elem_i))
        c_i = 1
        for k = 1:length(rowi.nzind)
          j2 = rowi.nzind[k]
          # add all col elements to rowi2 except the col j2 == j
          if kept_cols[j2] && j2 != j
            rowi2.nzind[c_i] = j2
            Aij2 = rowi.nzval[k]
            rowi2.nzval[c_i] = Aij2
            nzcj && (c[j2] -= yi * Aij2) # update c if c[j] != 0
            col_cnt[j2] -= 1
            c_i += 1
          end
        end
        conival = nzcj ? ucon[i] : (lcon[i] + ucon[i]) / 2 # constant for postsolve
        push!(operations, FreeLinearSingletonColumn{T, S}(i, j, Aij, rowi2, yi, conival))
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

function postsolve!(pt::OutputPoint{T, S}, operation::FreeLinearSingletonColumn{T, S}) where {T, S}
  x = pt.x
  j = operation.j
  # x[j] = (coival - Σₖ Aik x[k]) / Aij , where k ≂̸ j
  x[j] = operation.conival
  for (i, Aij) in zip(operation.arowi.nzind, operation.arowi.nzval)
    x[j] -= Aij * x[i]
  end
  x[j] /= operation.aij
  pt.s_l[j] = zero(T)
  pt.s_u[j] = zero(T)
  pt.y[operation.i] = operation.yi
end

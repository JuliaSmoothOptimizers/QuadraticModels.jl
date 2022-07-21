# ̃xᵀ̃Hx̃ + ̃ĉᵀx̃ + lⱼ²Hⱼⱼ + cⱼxⱼ + c₀
# ̂c = ̃c + 2lⱼΣₖHⱼₖxₖ , k ≂̸ j

"""
    c0ps = remove_ifix!(ifix, Hrows, Hcols, Hvals, nvar, 
                        Arows, Acols, Avals, c, c0, 
                        lvar, lcon, ucon, row_cnt, col_cnt,
                        kept_cols, xps)

Presolve procedure to remove fixed variables to `lvar` at indices `ifix`.
"""
function remove_ifix!(
  ifix,
  Hrows,
  Hcols,
  Hvals,
  nvarps,
  Arows,
  Acols,
  Avals,
  c::AbstractVector{T},
  c0,
  lvar,
  lcon,
  ucon,
  row_cnt,
  col_cnt,
  kept_cols,
  xps,
) where {T}

  # assume Hcols is sorted
  c0_offset = zero(T)
  Hnnz = length(Hrows)
  Annz = length(Arows)
  # assume ifix is sorted and length(ifix) > 0
  nfix = length(ifix)

  # remove ifix 1 by 1 in H and A and update QP data
  for idxfix = 1:nfix
    currentifix = ifix[idxfix]
    xifix = lvar[currentifix]
    k = 1
    while k <= Hnnz && Hcols[k] <= (nvarps - idxfix + 1)
      Hi, Hj, Hx = Hrows[k], Hcols[k], Hvals[k] # Hj sorted 
      if Hi == Hj == currentifix
        c0_offset += xifix^2 * Hx / 2
      elseif Hi == currentifix
        c[Hj] += xifix * Hx
      elseif Hj == currentifix
        c[Hi] += xifix * Hx
      end
      k += 1
    end

    # remove ifix in A cols
    k = 1
    while k <= Annz
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Aj == currentifix
        row_cnt[Ai] -= 1
        con_offset = Ax * xifix
        lcon[Ai] -= con_offset
        ucon[Ai] -= con_offset
      end
      k += 1
    end

    # update c0 with c[currentifix] coeff
    c0_offset += c[currentifix] * xifix
  end

  # store removed x values
  xps[ifix] .= @views lvar[ifix]

  # update c0
  c0ps = c0 + c0_offset

  kept_cols[ifix] .= false
  col_cnt[ifix] .= -1

  return c0ps
end

function find_fixed_variables(lvar, uvar, kept_cols)
  out = Int[]
  for i=1:length(lvar)
    if (lvar[i] == uvar[i]) && kept_cols[i]
      push!(out, i)
    end 
  end
  return out
end

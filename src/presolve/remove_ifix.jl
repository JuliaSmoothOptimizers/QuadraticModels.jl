struct RemoveIfix{T, S} <: PresolveOperation{T, S}
  j::Int
  xj::T
end

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
  operations::Vector{PresolveOperation{T, S}},
  Hrows,
  Hcols,
  Hvals::S,
  Arows,
  Acols,
  Avals::S,
  c::S,
  c0::T,
  lvar::S,
  uvar::S,
  lcon::S,
  ucon::S,
  nvar,
  row_cnt,
  col_cnt,
  kept_cols,
  xps,
) where {T, S}

  ifix_pass = false
  # assume Hcols is sorted
  c0_offset = zero(T)
  Hnnz = length(Hrows)
  Annz = length(Arows)

  for j = 1:nvar
    (kept_cols[j] && (lvar[j] == uvar[j])) || continue
    ifix_pass = true
    xj = lvar[j]
    k = 1
    while k <= Hnnz
      Hi, Hj, Hx = Hrows[k], Hcols[k], Hvals[k] # Hj sorted 
      if Hi == Hj == j
        c0_offset += xj^2 * Hx / 2
      elseif Hi == j
        c[Hj] += xj * Hx
      elseif Hj == j
        c[Hi] += xj * Hx
      end
      k += 1
    end

    # remove ifix in A cols
    k = 1
    while k <= Annz
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Aj == j
        row_cnt[Ai] -= 1
        con_offset = Ax * xj
        lcon[Ai] -= con_offset
        ucon[Ai] -= con_offset
      end
      k += 1
    end

    # update c0 with c[currentifix] coeff
    c0_offset += c[j] * xj
    xps[j] = xj
    kept_cols[j] = false
    col_cnt[j] = -1

    push!(operations, RemoveIfix{T, S}(j, xj))
  end
  # update c0
  c0ps = c0 + c0_offset

  return c0ps, ifix_pass
end

function postsolve!(pt::OutputPoint{T, S}, operation::RemoveIfix{T, S}) where {T, S}
  pt.x[operation.j] = operation.xj
end
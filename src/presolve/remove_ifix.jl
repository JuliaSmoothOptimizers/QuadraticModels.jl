struct RemoveIfix{T, S} <: PresolveOperation{T, S}
  j::Int
  xj::T
end

# ̃xᵀ̃Hx̃ + ̃ĉᵀx̃ + lⱼ²Hⱼⱼ + cⱼxⱼ + c₀
# ̂c = ̃c + 2lⱼΣₖHⱼₖxₖ , k ≂̸ j

function remove_ifix!(
  operations::Vector{PresolveOperation{T, S}},
  hcols::Vector{Col{T}},
  acols::Vector{Col{T}},
  c::S,
  c0::T,
  lvar::S,
  uvar::S,
  lcon::S,
  ucon::S,
  nvar,
  row_cnt,
  col_cnt,
  kept_rows,
  kept_cols,
  xps,
) where {T, S}
  ifix_pass = false
  c0_offset = zero(T)

  for j = 1:nvar
    (kept_cols[j] && (lvar[j] == uvar[j])) || continue
    ifix_pass = true
    xj = lvar[j]
    for k = 1:length(hcols[j].nzind)
      i = hcols[j].nzind[k]
      Hx = hcols[j].nzval[k]
      if i == j
        c0_offset += xj^2 * Hx / 2
      else
        kept_cols[i] && (c[i] += xj * Hx)
      end
    end

    # remove ifix in A cols
    for k = 1:length(acols[j].nzind)
      i = acols[j].nzind[k]
      if kept_rows[i]
        row_cnt[i] -= 1
        con_offset = acols[j].nzval[k] * xj
        lcon[i] -= con_offset
        ucon[i] -= con_offset
      end
    end

    # update c0 with c[j] coeff
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

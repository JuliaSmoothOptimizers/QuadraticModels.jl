struct RemoveIfix{T, S} <: PresolveOperation{T, S}
  j::Int
  xj::T
  cj::T
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
    c_acolj = 0
    for (i, aij) in zip(acols[j].nzind, acols[j].nzval)
      if kept_rows[i]
        row_cnt[i] -= 1
        con_offset = aij * xj
        lcon[i] -= con_offset
        ucon[i] -= con_offset
        c_acolj += 1 # count number of rows in acols[j]
      end
    end

    # update c0 with c[j] coeff
    c0_offset += c[j] * xj
    xps[j] = xj
    kept_cols[j] = false
    col_cnt[j] = -1

    push!(operations, RemoveIfix{T, S}(j, xj, c[j]))
  end
  # update c0
  c0ps = c0 + c0_offset

  return c0ps, ifix_pass
end

function postsolve!(
  sol::QMSolution{T, S},
  operation::RemoveIfix{T, S},
  psd::PresolvedData{T, S},
) where {T, S}
  j = operation.j
  psd.kept_cols[j] = true
  acolj = psd.acols[j]
  hcolj = psd.hcols[j]
  x = sol.x
  x[j] = operation.xj
  y = sol.y

  ATyj = zero(T)
  for (i, aij) in zip(acolj.nzind, acolj.nzval)
    psd.kept_rows[i] || continue
    ATyj += aij * y[i]
  end

  Hxj = zero(T)
  for (i, hij) in zip(hcolj.nzind, hcolj.nzval)
    psd.kept_cols[i] || continue
    Hxj += hij * x[i]
  end
  
  s = operation.cj + Hxj - ATyj
  if s > zero(T)
    sol.s_l[j] = s
  else
    sol.s_u[j] = -s
  end
end

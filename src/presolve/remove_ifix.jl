struct RemoveIfix{T, S} <: PresolveOperation{T, S}
  j::Int
  xj::T
  cj::T
  acolj::Col{T}
  hcolj::Col{T}
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

    # store acolj for postsolve
    acolj = Col(zeros(Int, c_acolj), zeros(T, c_acolj))
    c_acolj = 1
    for k = 1:length(acols[j].nzind)
      i = acols[j].nzind[k]
      kept_rows[i] || continue
      aij = acols[j].nzval[k]
      acolj.nzind[c_acolj] = i
      acolj.nzval[c_acolj] = aij
      c_acolj += 1
    end

    # store hcolj for postsolve
    hcolj = Col{T}(
      [i for i in hcols[j].nzind if kept_cols[i]],
      [hij for (i, hij) in zip(hcols[j].nzind, hcols[j].nzval) if kept_cols[i]],
    )

    # update c0 with c[j] coeff
    c0_offset += c[j] * xj
    xps[j] = xj
    kept_cols[j] = false
    col_cnt[j] = -1

    push!(operations, RemoveIfix{T, S}(j, xj, c[j], acolj, hcolj))
  end
  # update c0
  c0ps = c0 + c0_offset

  return c0ps, ifix_pass
end

function postsolve!(sol::QMSolution{T, S}, operation::RemoveIfix{T, S}) where {T, S}
  j = operation.j
  sol.x[j] = operation.xj
  ATyj = @views dot(operation.acolj.nzval, sol.y[operation.acolj.nzind])
  Hxj = @views dot(operation.hcolj.nzval, sol.x[operation.hcolj.nzind])
  s = operation.cj + Hxj - ATyj
  if s > zero(T)
    sol.s_l[j] = s
  else
    sol.s_u[j] = -s
  end
end

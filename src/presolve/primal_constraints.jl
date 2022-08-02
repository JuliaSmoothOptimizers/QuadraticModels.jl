function primal_constraints!(
  operations::Vector{PresolveOperation{T, S}},
  arows::Vector{Row{T}},
  lcon::S,
  ucon::S,
  lvar::S,
  uvar::S,
  nvar,
  ncon,
  kept_rows,
  kept_cols,
  row_cnt,
  col_cnt,
) where {T, S}
  for i = 1:ncon
    (kept_rows[i] && !(lcon[i] == -T(Inf) && ucon[i] == T(Inf))) || continue
    rowi = arows[i]
    uconi2 = zero(T)
    lconi2 = zero(T)
    for (j, aij) in zip(rowi.nzind, rowi.nzval)
      kept_cols[j] || continue
      uconi2 += (aij > zero(T)) ? aij * uvar[j] : aij * lvar[j]
      lconi2 += (aij > zero(T)) ? aij * lvar[j] : aij * uvar[j]
    end
    if uconi2 < lcon[i] || lconi2 > ucon[i]
      @warn "implied bounds for row $i lead to a primal infeasible constraint"
      return true
    elseif uconi2 == lcon[i]
      for (j, aij) in zip(rowi.nzind, rowi.nzval)
        kept_cols[j] || continue
        if aij > zero(T)
          lvar[j] = uvar[j]
        else
          uvar[j] = lvar[j]
        end
      end
    elseif lconi2 == ucon[i]
      for (j, aij) in zip(rowi.nzind, rowi.nzval)
        kept_cols[j] || continue
        if aij > zero(T)
          uvar[j] = lvar[j]
        else
          lvar[j] = uvar[j]
        end
      end
    elseif lcon[i] < lconi2 && uconi2 < ucon[i]
      kept_rows[i] = false
      row_cnt[i] = -1
      for j in rowi.nzind
        kept_cols[j] || continue
        col_cnt[j] -= 1
      end
      push!(operations, EmptyRow{T, S}(i))
    elseif lcon[i] < lconi2
      lcon[i] = -T(Inf)
    elseif uconi2 < ucon[i]
      ucon[i] = T(Inf)
    end
  end
  return false
end

struct PrimalConstraint{T, S} <: PresolveOperation{T, S}
  i::Int
  forced_lcon::Bool # row i forced to lcon[i]
end

function primal_constraints!(
  qmp::QuadraticModelPresolveData{T, S},
  operations::Vector{PresolveOperation{T, S}},
) where {T, S}
  arows, lcon, ucon, lvar, uvar = qmp.arows, qmp.lcon, qmp.ucon, qmp.lvar, qmp.uvar
  kept_rows, kept_cols = qmp.kept_rows, qmp.kept_cols
  row_cnt, col_cnt = qmp.row_cnt, qmp.col_cnt
  for i = 1:(qmp.ncon)
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
      qmp.infeasible_cst = true
      return nothing
    elseif uconi2 == lcon[i]
      for (j, aij) in zip(rowi.nzind, rowi.nzval)
        kept_cols[j] || continue
        if aij > zero(T)
          lvar[j] = uvar[j]
        else
          uvar[j] = lvar[j]
        end
      end
      push!(operations, PrimalConstraint{T, S}(i, true))
    elseif lconi2 == ucon[i]
      for (j, aij) in zip(rowi.nzind, rowi.nzval)
        kept_cols[j] || continue
        if aij > zero(T)
          uvar[j] = lvar[j]
        else
          lvar[j] = uvar[j]
        end
      end
      push!(operations, PrimalConstraint{T, S}(i, false))
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
end

function postsolve!(
  sol::QMSolution,
  operation::PrimalConstraint{T, S},
  psd::PresolvedData{T, S},
) where {T, S}
  i = operation.i
  forced_lcon = operation.forced_lcon
  x = sol.x
  y = sol.y
  s_l = sol.s_l
  s_u = sol.s_u
  arowi = psd.arows[i]
  acols = psd.acols
  kept_cols = psd.kept_cols
  z = psd.z
  z .= psd.c
  n = length(z)

  add_Hx!(z, psd.hcols, kept_cols, x) # z = c + Hx
  for l = 1:n
    kept_cols[l] || continue
    for (k, akl) in zip(acols[l].nzind, acols[l].nzval)
      (psd.kept_rows[i] && k != i) || continue
      z[l] -= akl * y[k]
    end
  end

  if forced_lcon
    yi = T(-Inf)
    for (l, ail) in zip(arowi.nzind, arowi.nzval)
      (kept_cols[l]) || continue
      trial = z[l] / ail
      (trial > yi) && (yi = trial)
    end
  else
    yi = T(Inf)
    for (l, ail) in zip(arowi.nzind, arowi.nzval)
      (kept_cols[l]) || continue
      trial = z[l] / ail
      (trial < yi) && (yi = trial)
    end
  end
  y[i] = yi

  for (l, ail) in zip(arowi.nzind, arowi.nzval)
    (kept_cols[l]) || continue
    s = z[l] - ail * yi
    s_l[l] = s > 0 ? s : zero(T)
    s_u[l] = s < 0 ? -s : zero(T)
  end
end

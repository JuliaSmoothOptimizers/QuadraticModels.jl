struct UnconstrainedReduction{T, S} <: PresolveOperation{T, S}
  j::Int
end
# apply before remove ifix
function unconstrained_reductions!(
  qmp::QuadraticModelPresolveData{T, S},
  operations::Vector{PresolveOperation{T, S}},
) where {T, S}
  qmp.unbounded = false
  c, hcols, lvar, uvar = qmp.c, qmp.hcols, qmp.lvar, qmp.uvar
  xps = qmp.xps
  col_cnt, kept_cols = qmp.col_cnt, qmp.kept_cols

  # assume Hcols sorted
  for j = 1:(qmp.nvar)
    (kept_cols[j] && (col_cnt[j] == 0)) || continue
    # check empty rows/col j in H
    if isempty(hcols[j].nzind)
      if c[j] < zero(T)
        xps[j] = uvar[j]
        lvar[j] = uvar[j]
      else
        xps[j] = lvar[j]
        uvar[j] = lvar[j]
      end
      xps[j] == -T(Inf) && c[j] != zero(T) && (qmp.unbounded = true)
      if xps[j] == -T(Inf) && c[j] == zero(T)
        lvar[j] = zero(T)
        uvar[j] = zero(T)
        xps[j] = zero(T)
      end
      push!(operations, UnconstrainedReduction{T, S}(j))
    else
      continue
      # todo : QP 
      # Hrows[idx_deb] != var && continue
      # Hcols[idx_deb+1] == var && continue
      # nb_rowvar = count(isequal(var), Hrows)
      # nb_rowvar ≥ 2 && continue
    end
  end
end

function postsolve!(sol::QMSolution, operation::UnconstrainedReduction, psd::PresolvedData)
  psd.kept_cols[operation.j] = true
end

struct UnconstrainedReduction{T, S} <: PresolveOperation{T, S}
  j::Int
end
# apply before remove ifix
function unconstrained_reductions!(
  operations::Vector{PresolveOperation{T, S}},
  c::S,
  hcols::Vector{Col{T}},
  lvar::S,
  uvar::S,
  xps::S,
  nvar,
  col_cnt,
  kept_cols,
) where {T, S}
  unbounded = false
  # assume Hcols sorted
  for j = 1:nvar
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
      xps[j] == -T(Inf) && (unbounded = true)
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

  return unbounded
end

postsolve!(pt::OutputPoint, operation::UnconstrainedReduction) = nothing

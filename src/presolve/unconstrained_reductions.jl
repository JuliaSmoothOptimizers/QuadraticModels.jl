# apply before remove ifix
"""
    unbounded = unconstrained_reductions!(c, Hrows, Hcols, Hvals, lvar, uvar,
                                          xps, lin_unconstr_vars)

Fix linearly unconstrained variables, updating the corresponding elements of `xps`.
This only works for linear problems for now, but can be extended to some specific quadratic problems.
"""
function unconstrained_reductions!(
  c::AbstractVector{T},
  Hrows,
  Hcols,
  Hvals::AbstractVector{T},
  lvar::AbstractVector{T},
  uvar::AbstractVector{T},
  xps::AbstractVector{T},
  lin_unconstr_vars,
) where {T}
  unbounded = false
  # assume Hcols sorted
  for var in lin_unconstr_vars
    # check diagonal H or 
    idx_deb = findfirst(isequal(var), Hcols)
    if idx_deb === nothing
      if c[var] < zero(T)
        xps[var] = uvar[var]
        lvar[var] = uvar[var]
      else
        xps[var] = lvar[var]
        uvar[var] = lvar[var]
      end
      xps[var] == -T(Inf) && (unbounded = true)
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

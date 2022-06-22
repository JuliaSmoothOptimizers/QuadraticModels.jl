find_singleton_rows(row_cnt::Vector{Int}) = findall(isequal(1), row_cnt)

"""
    new_ncon = singleton_rows!(Arows, Acols, Avals, lcon, ucon,
                               lvar, uvar, nvar, ncon, row_cnt, singl_rows)

Removes the singleton rows of A, and the corresponding elements in lcon and ucon that are in `singl_rows`.
`row_cnt` is a vector of the number of elements per row.

Returns the new number of constraints `new_ncon` and updates in-place `Arows`, `Acols`, `Avals`, `lcon`, `ucon`, `lvar`, `uvar`.
"""
function singleton_rows!(
  Arows,
  Acols,
  Avals,
  lcon::Vector{T},
  ucon::Vector{T},
  lvar,
  uvar,
  nvar,
  ncon,
  row_cnt::Vector{Int},
  singl_rows::Vector{Int},
) where {T}

  # assume Acols is sorted
  Annz = length(Arows)
  nsingl = length(singl_rows)

  # remove ifix 1 by 1 in H and A and update QP data
  for idxsingl = 1:nsingl
    currentisingl = singl_rows[idxsingl]
    # index of the current singleton row that takes the number of 
    # already removed variables into account:
    newcurrentisingl = currentisingl - idxsingl + 1

    # remove singleton rows in A rows
    Awritepos = 1
    k = 1
    while k <= Annz - idxsingl + 1
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Ai == newcurrentisingl
        if Ax > zero(T)
          lvar[Aj] = max(lvar[Aj], lcon[currentisingl] / Ax)
          uvar[Aj] = min(uvar[Aj], ucon[currentisingl] / Ax)
        elseif Ax < zero(T)
          lvar[Aj] = max(lvar[Aj], ucon[currentisingl] / Ax)
          uvar[Aj] = min(uvar[Aj], lcon[currentisingl] / Ax)
        else
          error("remove explicit zeros in A")
        end
      else
        Arows[Awritepos] = (Ai < newcurrentisingl) ? Ai : Ai - 1
        Acols[Awritepos] = Aj
        Avals[Awritepos] = Ax
        Awritepos += 1
      end
      k += 1
    end
  end

  if nsingl > 0
    Annz -= nsingl
    resize!(Arows, Annz)
    resize!(Acols, Annz)
    resize!(Avals, Annz)
  end

  deleteat!(lcon, singl_rows)
  deleteat!(ucon, singl_rows)
  return ncon - nsingl
end

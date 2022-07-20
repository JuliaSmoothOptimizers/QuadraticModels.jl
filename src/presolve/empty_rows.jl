"""
    new_ncon = empty_rows!(Arows, lcon, ucon, ncon, row_cnt, empty_rows)

Removes the empty rows of A, and the corresponding elements in lcon and ucon that are in `empty_rows`.
`row_cnt` is a vector of the number of elements per row.

Returns the new number of constraints `new_ncon` and updates in-place `Arows`, `lcon`, `ucon`.
"""
function empty_rows!(
  Arows,
  lcon::Vector{T},
  ucon::Vector{T},
  ncon,
  row_cnt::Vector{Int},
  empty_rows::Vector{Int},
) where {T}
  new_ncon = 0
  for i = 1:ncon
    if row_cnt[i] == 0
      @assert lcon[i] ≤ zero(T) ≤ ucon[i]
    else
      new_ncon += 1
      lcon[new_ncon] = lcon[i]
      ucon[new_ncon] = ucon[i]
    end
  end
  resize!(lcon, new_ncon)
  resize!(ucon, new_ncon)

  # assume Acols is sorted
  Annz = length(Arows)
  nempty = length(empty_rows)

  for idxempty = 1:nempty
    currentiempty = empty_rows[idxempty]
    # index of the current singleton row that takes the number of 
    # already removed variables into account:
    newcurrentiempty = currentiempty - idxempty + 1

    # remove singleton rows in A rows
    Awritepos = 1
    k = 1
    while k <= Annz
      Ai = Arows[k]
      Arows[Awritepos] = (Ai < newcurrentiempty) ? Ai : Ai - 1
      Awritepos += 1
      k += 1
    end
  end
  return new_ncon
end

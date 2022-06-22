function row_cnt!(Arows, row_cnt::Vector{Int})
  for k = 1:length(Arows)
    i = Arows[k]
    row_cnt[i] += 1
  end
end

find_empty_rows(row_cnt::Vector{Int}) = findall(isequal(0), row_cnt)

"""
    new_ncon = empty_rows!(Arows, lcon, ucon, ncon, row_cnt, empty_rows, Arows_s)

Removes the empty rows of A, and the corresponding elements in lcon and ucon that are in `empty_rows`.
`Arows_s` is a view of `Arows` sorted in ascending order.
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
  Arows_s,
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

  c_rm = 1
  nrm = length(empty_rows)
  for k = 1:length(Arows)
    while c_rm ≤ nrm && Arows_s[k] ≥ empty_rows[c_rm]
      c_rm += 1
    end
    Arows_s[k] -= c_rm - 1
  end
  return new_ncon
end

"""
    new_ncon = empty_rows!(lcon, ucon, empty_rows, row_cnt, kept_rows)

Presolve operation for empty rows of A.
`row_cnt` is a vector of the number of elements per row.
`kept_rows` is a a vector of `Bool` indicating if a row is already considered as removed.
"""
function empty_rows!(
  lcon::Vector{T},
  ucon::Vector{T},
  empty_rows::Vector{Int},
  row_cnt,
  kept_rows::Vector{Bool},
) where {T}
  kept_rows[empty_rows] .= false
  for i in empty_rows
    (!kept_rows[i]) && @assert (lcon[i] ≤ zero(T) ≤ ucon[i])
  end
  row_cnt[empty_rows] .= -1
end

struct EmptyRow{T, S} <: PresolveOperation{T, S}
  i::Int # idx of empty row
end

"""
    new_ncon = empty_rows!(lcon, ucon, empty_rows, row_cnt, kept_rows)

Presolve operation for empty rows of A.
`row_cnt` is a vector of the number of elements per row.
`kept_rows` is a a vector of `Bool` indicating if a row is already considered as removed.
"""
function empty_rows!(
  operations::Vector{PresolveOperation{T, S}},
  lcon::S,
  ucon::S,
  ncon,
  row_cnt,
  kept_rows::Vector{Bool},
) where {T, S}
  empty_row_pass = false
  for i=1:ncon
    (kept_rows[i] && (row_cnt[i] == 0)) || continue
    empty_row_pass = true
    @assert (lcon[i] ≤ zero(T) ≤ ucon[i])
    row_cnt[i] = -1
    kept_rows[i] = false
    push!(operations, EmptyRow{T, S}(i))
  end
  return empty_row_pass
end

postsolve!(pt::OutputPoint, operation::EmptyRow) = nothing
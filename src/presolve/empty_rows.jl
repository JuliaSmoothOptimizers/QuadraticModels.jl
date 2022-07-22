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
  empty_rows::Vector{Int},
  row_cnt,
  kept_rows::Vector{Bool},
) where {T, S}
  kept_rows[empty_rows] .= false
  for i in empty_rows
    (!kept_rows[i]) && @assert (lcon[i] ≤ zero(T) ≤ ucon[i])
    push!(operations, EmptyRow{T, S}(i))
  end
  row_cnt[empty_rows] .= -1
end

postsolve!(pt::OutputPoint, operation::EmptyRow) = nothing
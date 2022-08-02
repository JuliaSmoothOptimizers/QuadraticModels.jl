struct FreeRow{T, S} <: PresolveOperation{T, S}
  i::Int # idx of free row
end

function free_rows!(
  operations::Vector{PresolveOperation{T, S}},
  lcon::S,
  ucon::S,
  ncon,
  row_cnt,
  kept_rows::Vector{Bool},
) where {T, S}
  free_row_pass = false
  for i = 1:ncon
    (kept_rows[i] && lcon[i] == -T(Inf) && ucon[i] == T(Inf)) || continue
    free_row_pass = true
    row_cnt[i] = -1
    kept_rows[i] = false
    push!(operations, FreeRow{T, S}(i))
  end
  return free_row_pass
end

postsolve!(pt::OutputPoint, operation::FreeRow) = nothing

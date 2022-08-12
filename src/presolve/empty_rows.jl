struct EmptyRow{T, S} <: PresolveOperation{T, S}
  i::Int # idx of empty row
end

function empty_rows!(
  qmp::QuadraticModelPresolveData{T, S},
  operations::Vector{PresolveOperation{T, S}},
) where {T, S}
  qmp.empty_row_pass = false
  lcon, ucon, row_cnt, kept_rows = qmp.lcon, qmp.ucon, qmp.row_cnt, qmp.kept_rows
  for i = 1:(qmp.ncon)
    (kept_rows[i] && (row_cnt[i] == 0)) || continue
    qmp.empty_row_pass = true
    @assert (lcon[i] ≤ zero(T) ≤ ucon[i])
    row_cnt[i] = -1
    kept_rows[i] = false
    push!(operations, EmptyRow{T, S}(i))
  end
end

function postsolve!(sol::QMSolution, operation::EmptyRow, psd::PresolvedData)
  psd.kept_rows[operation.i] = true
end

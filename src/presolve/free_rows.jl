struct FreeRow{T, S} <: PresolveOperation{T, S}
  i::Int # idx of free row
end

function free_rows!(
  qmp::QuadraticModelPresolveData{T, S},
  operations::Vector{PresolveOperation{T, S}},
) where {T, S}
  lcon, ucon = qmp.lcon, qmp.ucon
  row_cnt, kept_rows = qmp.row_cnt, qmp.kept_rows
  qmp.free_row_pass = false
  for i = 1:(qmp.ncon)
    (kept_rows[i] && lcon[i] == -T(Inf) && ucon[i] == T(Inf)) || continue
    qmp.free_row_pass = true
    row_cnt[i] = -1
    kept_rows[i] = false
    push!(operations, FreeRow{T, S}(i))
  end
end

function postsolve!(sol::QMSolution, operation::FreeRow, psd::PresolvedData)
  psd.kept_rows[operation.i] = true
end

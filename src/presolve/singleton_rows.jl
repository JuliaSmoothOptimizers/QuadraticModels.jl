"""
    singleton_rows!(Arows, Acols, Avals, lcon, ucon,
                    lvar, uvar, nvar, ncon, row_cnt, singl_rows,
                    row_cnt, col_cnt, kept_rows)

Presolve procedure for singleton rows of A in `singl_rows`.
"""
function singleton_rows!(
  Arows,
  Acols,
  Avals,
  lcon::Vector{T},
  ucon::Vector{T},
  lvar,
  uvar,
  singl_rows::Vector{Int},
  row_cnt,
  col_cnt,
  kept_rows,
) where {T}

  # assume Acols is sorted
  Annz = length(Arows)
  nsingl = length(singl_rows)

  # remove ifix 1 by 1 in H and A and update QP data
  for idxsingl = 1:nsingl
    currentisingl = singl_rows[idxsingl]
    k = 1
    while k <= Annz - idxsingl + 1
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Ai == currentisingl
        col_cnt[Aj] -= 1
        if Ax > zero(T)
          lvar[Aj] = max(lvar[Aj], lcon[currentisingl] / Ax)
          uvar[Aj] = min(uvar[Aj], ucon[currentisingl] / Ax)
        elseif Ax < zero(T)
          lvar[Aj] = max(lvar[Aj], ucon[currentisingl] / Ax)
          uvar[Aj] = min(uvar[Aj], lcon[currentisingl] / Ax)
        else
          error("remove explicit zeros in A")
        end
      end
      k += 1
    end
  end
  row_cnt[singl_rows] .= -1
  kept_rows[singl_rows] .= false
end

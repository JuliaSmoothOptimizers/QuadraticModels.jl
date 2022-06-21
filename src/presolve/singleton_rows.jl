removed_singleton_rows(row_cnt::Vector{Int}) = findall(isequal(1), row_cnt)

function singleton_rows!(Arows, Acols, Avals, lcon::Vector{T}, ucon::Vector{T}, lvar, uvar,
                         nvar, ncon, row_cnt::Vector{Int}, 
                         singl_rows::Vector{Int}) where {T}

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
        # oldAi = Ai + idxsingl - 1
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

  new_ncon = 0
  for i=1:ncon
    if row_cnt[i] != 1
      new_ncon += 1
      lcon[new_ncon] = lcon[i]
      ucon[new_ncon] = ucon[i]
    end
  end
  resize!(lcon, new_ncon)
  resize!(ucon, new_ncon)
  return new_ncon
end
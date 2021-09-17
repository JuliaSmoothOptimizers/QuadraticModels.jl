# ̃xᵀ̃Hx̃ + ̃ĉᵀx̃ + lⱼ²Hⱼⱼ + cⱼxⱼ + c₀
# ̂c = ̃c + 2lⱼΣₖHⱼₖxₖ , k ≂̸ j

"""
    xrm, c0ps, nvarrm = remove_ifix!(ifix, Hrows, Hcols, Hvals, nvar, 
                                     Arows, Acols, Avals, c, c0, 
                                     lvar, uvar, lcon, ucon)

Remove rows and columns in H, columns in A, and elements in lcon and ucon
corresponding to fixed variables, that are in `ifix`
(They should be the indices `i` where `lvar[i] == uvar[i]`).

Returns the removed elements of `lvar` (or `uvar`), the constant term in the QP
objective `c0ps` resulting from the fixed variables, and the new number of variables `nvarrm`.
`Hrows`, `Hcols`, `Hvals`, `Arows`, `Acols`, `Avals`, `c`, `lvar`, `uvar`,
`lcon` and `ucon` are updated in-place.
"""
function remove_ifix!(
  ifix,
  Hrows,
  Hcols,
  Hvals,
  nvar,
  Arows,
  Acols,
  Avals,
  c::AbstractVector{T},
  c0,
  lvar,
  uvar,
  lcon,
  ucon,
) where {T}

  # assume Hcols is sorted
  c0_offset = zero(T)
  Hnnz = length(Hrows)
  Hrm = 0
  Annz = length(Arows)
  Arm = 0
  # assume ifix is sorted and length(ifix) > 0
  nfix = length(ifix)

  # remove ifix 1 by 1 in H and A and update QP data
  for idxfix = 1:nfix
    currentifix = ifix[idxfix]
    xifix = lvar[currentifix]
    # index of the current fixed variable that takes the number of 
    # already removed variables into account:
    newcurrentifix = currentifix - idxfix + 1

    Hwritepos = 1
    # shift corresponding to the already removed fixed variables to update c:
    shiftHj = 1
    if Hnnz > 0
      oldHj = Hrows[1]
    end
    # remove ifix in H and update data
    k = 1
    while k <= Hnnz && Hcols[k] <= (nvar - idxfix + 1)
      Hi, Hj, Hx = Hrows[k], Hcols[k], Hvals[k] # Hj sorted 

      while (Hj == oldHj) && shiftHj <= idxfix - 1 && Hj + shiftHj - 1 >= ifix[shiftHj]
        shiftHj += 1
      end
      shiftHi = 1
      while shiftHi <= idxfix - 1 && Hi + shiftHi - 1 >= ifix[shiftHi]
        shiftHi += 1
      end
      if Hi == Hj == newcurrentifix
        Hrm += 1
        c0_offset += xifix^2 * Hx / 2
      elseif Hi == newcurrentifix
        Hrm += 1
        c[Hj + shiftHj - 1] += xifix * Hx
      elseif Hj == newcurrentifix
        Hrm += 1
        c[Hi + shiftHi - 1] += xifix * Hx
      else # keep Hi, Hj, Hx
        Hrows[Hwritepos] = (Hi < newcurrentifix) ? Hi : Hi - 1
        Hcols[Hwritepos] = (Hj < newcurrentifix) ? Hj : Hj - 1
        Hvals[Hwritepos] = Hx
        Hwritepos += 1
      end
      k += 1
    end

    # remove ifix in A cols
    Awritepos = 1
    currentAn = nvar - idxfix + 1  # remove rows if uplo == :U 
    k = 1
    while k <= Annz && Acols[k] <= currentAn
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Aj == newcurrentifix
        Arm += 1
        lcon[Ai] -= Ax * xifix
        ucon[Ai] -= Ax * xifix
      else
        if Awritepos != k
          Arows[Awritepos] = Ai
          Acols[Awritepos] = (Aj < newcurrentifix) ? Aj : Aj - 1
          Avals[Awritepos] = Ax
        end
        Awritepos += 1
      end
      k += 1
    end

    # update c0 with c[currentifix] coeff
    c0_offset += c[currentifix] * xifix
  end

  # resize Q and A
  if nfix > 0
    Hnnz -= Hrm
    Annz -= Arm
    resize!(Hrows, Hnnz)
    resize!(Hcols, Hnnz)
    resize!(Hvals, Hnnz)
    resize!(Arows, Annz)
    resize!(Acols, Annz)
    resize!(Avals, Annz)
  end

  # store removed x values
  xrm = lvar[ifix]

  # remove coefs in lvar, uvar, c
  deleteat!(c, ifix)
  deleteat!(lvar, ifix)
  deleteat!(uvar, ifix)

  # update c0
  c0ps = c0 + c0_offset

  nvarrm = nvar - nfix

  return xrm, c0ps, nvarrm
end

function restore_ifix!(ifix, xrm, x, xout)
  # put x and xrm inside xout
  cfix, cx = 1, 1
  nfix = length(ifix)
  for i = 1:length(xout)
    if cfix <= nfix && i == ifix[cfix]
      xout[i] = xrm[cfix]
      cfix += 1
    else
      xout[i] = x[cx]
      cx += 1
    end
  end
end

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
  nvarps,
  Arows,
  Acols,
  Avals,
  c::AbstractVector{T},
  c0,
  lvar,
  uvar,
  lcon,
  ucon,
  kept_cols,
  xps,
  nvar,
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
    while k <= Hnnz && Hcols[k] <= (nvarps - idxfix + 1)
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
    Hrows[Hwritepos:end] .= 0
    Hcols[Hwritepos:end] .= 0

    # remove ifix in A cols
    Awritepos = 1
    k = 1
    Arm_tmp = 0
    while k <= Annz - Arm
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Aj == newcurrentifix
        Arm_tmp += 1
        con_offset = Ax * xifix
        lcon[Ai] -= con_offset
        ucon[Ai] -= con_offset
      else
        Arows[Awritepos] = Ai
        Acols[Awritepos] = (Aj < newcurrentifix) ? Aj : Aj - 1
        Avals[Awritepos] = Ax
        Awritepos += 1
      end
      k += 1
    end
    Arm += Arm_tmp

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
  offset = 0
  c_cols = 1
  for i = 1:nvar
    if !kept_cols[i]
      offset += 1
    else
      if c_cols ≤ nfix && ifix[c_cols] + offset == i
        kept_cols[i] = false
        xps[i] = lvar[ifix[c_cols]]
        c_cols += 1
      end
    end
  end

  # remove coefs in lvar, uvar, c
  deleteat!(c, ifix)
  deleteat!(lvar, ifix)
  deleteat!(uvar, ifix)

  # update c0
  c0ps = c0 + c0_offset

  nvarrm = nvarps - nfix

  return c0ps, nvarrm
end

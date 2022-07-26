function vec_cnt!(v_cnt::Vector{Int}, v)
  for k = 1:length(v)
    i = v[k]
    v_cnt[i] += 1
  end
end

find_empty_rowscols(v_cnt::Vector{Int}) = findall(isequal(0), v_cnt)
find_singleton_rowscols(v_cnt::Vector{Int}) = findall(isequal(1), v_cnt)

function remove_rowscols_A!(Arows, Acols, Avals, kept_rows, kept_cols, nvar, ncon)
  Annz = length(Arows)
  Arm = 0
  nb_rm_rows = 0
  # remove rows
  for i0 = 1:ncon
    kept_rows[i0] && continue
    i = i0 - nb_rm_rows # up idx according to already removed rows
    nb_rm_rows += 1
    Awritepos = 1
    k = 1
    Arm_tmp = 0
    while k <= Annz - Arm
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Ai == i
        Arm_tmp += 1
      else
        Arows[Awritepos] = (Ai < i) ? Ai : Ai - 1
        Acols[Awritepos] = Aj
        Avals[Awritepos] = Ax
        Awritepos += 1
      end
      k += 1
    end
    Arm += Arm_tmp
  end

  # remove cols
  nb_rm_cols = 0
  for j0 = 1:nvar
    kept_cols[j0] && continue
    j = j0 - nb_rm_cols # up idx according to already removed cols
    nb_rm_cols += 1
    Awritepos = 1
    k = 1
    Arm_tmp = 0
    while k <= Annz - Arm
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Aj == j
        Arm_tmp += 1
      else
        Arows[Awritepos] = Ai
        Acols[Awritepos] = (Aj < j) ? Aj : Aj - 1
        Avals[Awritepos] = Ax
        Awritepos += 1
      end
      k += 1
    end
    Arm += Arm_tmp
  end

  if Arm > 0
    Annz -= Arm
    resize!(Arows, Annz)
    resize!(Acols, Annz)
    resize!(Avals, Annz)
  end
end

function remove_rowscols_H!(Hrows, Hcols, Hvals, kept_cols, nvar)
  Hnnz = length(Hrows)
  Hrm = 0
  nb_rm = 0
  # remove rows and cols
  for j0 = 1:nvar
    kept_cols[j0] && continue
    j = j0 - nb_rm # up idx according to already removed cols
    nb_rm += 1
    Hwritepos = 1
    k = 1
    Hrm_tmp = 0
    while k <= Hnnz - Hrm
      Hi, Hj, Hx = Hrows[k], Hcols[k], Hvals[k]
      if Hj == j
        Hrm_tmp += 1
      elseif Hi == j
        Hrm_tmp += 1
      else
        Hrows[Hwritepos] = (Hi < j) ? Hi : Hi - 1
        Hcols[Hwritepos] = (Hj < j) ? Hj : Hj - 1
        Hvals[Hwritepos] = Hx
        Hwritepos += 1
      end
      k += 1
    end
    Hrm += Hrm_tmp
  end

  if Hrm > 0
    Hnnz -= Hrm
    resize!(Hrows, Hnnz)
    resize!(Hcols, Hnnz)
    resize!(Hvals, Hnnz)
  end
end

function update_vectors!(lcon, ucon, c, lvar, uvar, kept_rows, kept_cols, ncon, nvar)
  nconps = 0
  for i = 1:ncon
    if kept_rows[i]
      nconps += 1
      lcon[nconps] = lcon[i]
      ucon[nconps] = ucon[i]
    end
  end
  nvarps = 0
  for j = 1:nvar
    if kept_cols[j]
      nvarps += 1
      lvar[nvarps] = lvar[j]
      uvar[nvarps] = uvar[j]
      c[nvarps] = c[j]
    end
  end

  resize!(lcon, nconps)
  resize!(ucon, nconps)
  resize!(lvar, nvarps)
  resize!(uvar, nvarps)
  resize!(c, nvarps)
  return nconps, nvarps
end

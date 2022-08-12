function switch_H_to_max!(
  data::QPData{T, S, M1, M2},
) where {T, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO}
  data.H.vals .= .-data.H.vals
end

function switch_H_to_max!(data::QPData)
  data.H = -data.H
end

check_reductions(qmp::QuadraticModelPresolveData) =
  qmp.empty_row_pass ||
  qmp.singl_row_pass ||
  qmp.ifix_pass ||
  qmp.free_lsc_pass ||
  qmp.free_row_pass

function copy_qm(qm::QuadraticModel{T, S}) where {T, S}
  data = deepcopy(qm.data)
  if !qm.meta.minimize
    switch_H_to_max!(data)
    data.c .= .-data.c
    data.c0 = -data.c0
    meta = NLPModelMeta{T, S}(
      qm.meta.nvar,
      lvar = qm.meta.lvar,
      uvar = qm.meta.uvar,
      ncon = qm.meta.ncon,
      lcon = qm.meta.lcon,
      ucon = qm.meta.ucon,
      nnzj = qm.meta.nnzj,
      nnzh = qm.meta.nnzh,
      lin = copy(qm.meta.lin),
      lin_nnzj = qm.meta.lin_nnzj,
      nln_nnzj = 0,
      islp = qm.meta.islp,
      x0 = qm.meta.x0,
      y0 = qm.meta.y0,
      minimize = true,
    )
  else
    meta = deepcopy(qm.meta)
  end
  return QuadraticModel(meta, qm.counters, data)
end

function vec_cnt!(v_cnt::Vector{Int}, v)
  for k = 1:length(v)
    i = v[k]
    v_cnt[i] += 1
  end
end

find_empty_rowscols(v_cnt::Vector{Int}) = findall(isequal(0), v_cnt)
find_singleton_rowscols(v_cnt::Vector{Int}) = findall(isequal(1), v_cnt)

function remove_rowscols_A_H!(A, H, qmp::QuadraticModelPresolveData)
  Arows, Acols, Avals = A.rows, A.cols, A.vals
  Hrows, Hcols, Hvals = H.rows, H.cols, H.vals
  kept_rows, kept_cols = qmp.kept_rows, qmp.kept_cols
  ps_rows_idx = zeros(Int, qmp.ncon)
  ci = 0
  for i = 1:(qmp.ncon)
    if kept_rows[i]
      ci += 1
      ps_rows_idx[i] = ci
    end
  end
  ps_cols_idx = zeros(Int, qmp.nvar)
  cj = 0
  for j = 1:(qmp.nvar)
    if kept_cols[j]
      cj += 1
      ps_cols_idx[j] = cj
    end
  end

  # erase old A values
  Annz = length(Arows)
  Arm = 0
  Awritepos = 0
  for k = 1:Annz
    i, j = Arows[k], Acols[k]
    if (kept_rows[i] && kept_cols[j])
      Awritepos += 1
      Arows[Awritepos] = ps_rows_idx[i]
      Acols[Awritepos] = ps_cols_idx[j]
      Avals[Awritepos] = Avals[k]
    else
      Arm += 1
    end
  end
  if Arm > 0
    Annz -= Arm
    resize!(Arows, Annz)
    resize!(Acols, Annz)
    resize!(Avals, Annz)
  end

  # erase old H values
  Hnnz = length(Hrows)
  Hrm = 0
  Hwritepos = 0
  for k = 1:Hnnz
    i, j = Hrows[k], Hcols[k]
    if (kept_cols[i] && kept_cols[j])
      Hwritepos += 1
      Hrows[Hwritepos] = ps_cols_idx[i]
      Hcols[Hwritepos] = ps_cols_idx[j]
      Hvals[Hwritepos] = Hvals[k]
    else
      Hrm += 1
    end
  end
  if Hrm > 0
    Hnnz -= Hrm
    resize!(Hrows, Hnnz)
    resize!(Hcols, Hnnz)
    resize!(Hvals, Hnnz)
  end
end

function update_vectors!(qmp::QuadraticModelPresolveData)
  c = qmp.c
  lcon, ucon, lvar, uvar = qmp.lcon, qmp.ucon, qmp.lvar, qmp.uvar
  kept_rows, kept_cols = qmp.kept_rows, qmp.kept_cols
  nconps = 0
  for i = 1:(qmp.ncon)
    if kept_rows[i]
      nconps += 1
      lcon[nconps] = lcon[i]
      ucon[nconps] = ucon[i]
    end
  end
  nvarps = 0
  for j = 1:(qmp.nvar)
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

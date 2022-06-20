function row_cnt!(Arows, row_cnt::Vector{Int})
  for k=1:length(Arows)
    i = Arows[k]
    row_cnt[i] += 1
  end
end

function removed_rows(row_cnt::Vector{Int})
  return findall(isequal(0), row_cnt)
end

function empty_rows!(Arows, lcon::Vector{T}, ucon::Vector{T}, ncon, row_cnt::Vector{Int}, 
                     rows_rm::Vector{Int}, Arows_sortperm::Vector{Int}) where {T}
  new_ncon = 0
  for i=1:ncon
    if row_cnt[i] == 0
      @assert lcon[i] ≤ zero(T) ≤ ucon[i]
    else
      new_ncon += 1
      lcon[new_ncon] = lcon[i]
      ucon[new_ncon] = ucon[i]
    end
  end
  resize!(lcon, new_ncon)
  resize!(ucon, new_ncon)

  Arows_s = @views Arows[Arows_sortperm]
  c_rm = 1
  nrm = length(rows_rm)
  for k=1:length(Arows)
    while c_rm ≤ nrm && Arows_s[k] ≥ rows_rm[c_rm]
      c_rm += 1 
    end
    Arows_s[k] -= c_rm - 1
  end
  return new_ncon
end

function restore_y!(y::Vector{T}, yout::Vector{T}, row_cnt, ncon) where {T}
  c_y = 0
  for i = 1:ncon
    if row_cnt[i] == 0
      yout[i] = zero(T)
    else
      c_y += 1
      yout[i] = y[c_y]
    end
  end
end
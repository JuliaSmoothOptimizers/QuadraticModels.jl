function vec_cnt!(v, v_cnt::Vector{Int})
  for k = 1:length(v)
    i = v[k]
    v_cnt[i] += 1
  end
end

find_empty_rowscols(v_cnt::Vector{Int}) = findall(isequal(0), v_cnt)
find_singleton_rowscols(v_cnt::Vector{Int}) = findall(isequal(1), v_cnt)

function update_kept_v!(kept_v::Vector{Bool}, vec_to_rm::Vector{Int}, n::Int)
  # update kept_v, shift indices if there are already some removed rows (kept_v[i] = false)
  # assume vec_to_rm sorted
  # if there are some already removed rows, vec_to_rm must be shifted
  n_rm = length(vec_to_rm)
  offset = 0
  c_v = 1
  for i = 1:n
    if !kept_v[i]
      offset += 1
    else
      if c_v ≤ n_rm && vec_to_rm[c_v] + offset == i
        kept_v[i] = false
        c_v += 1
      end
    end
  end
end

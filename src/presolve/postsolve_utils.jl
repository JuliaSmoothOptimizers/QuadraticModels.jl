function restore_ifix!(kept_cols, xps, x, xout)
  # put x and xps inside xout according to kept_cols
  nvar = length(xout)
  cx = 0
  for i = 1:nvar
    if kept_cols[i]
      cx += 1
      xout[i] = x[cx]
    else
      xout[i] = xps[i]
    end
  end
end

function restore_y!(y::Vector{T}, yout::Vector{T}, kept_rows::Vector{Bool}, ncon) where {T}
  c_y = 0
  for i = 1:ncon
    if !kept_rows[i]
      yout[i] = zero(T)
    else
      c_y += 1
      yout[i] = y[c_y]
    end
  end
end

function restore_ilow_iupp!(ilow, iupp, kept_cols)
  offset = 0
  nvar = length(kept_cols)
  nlow = length(ilow)
  nupp = length(iupp)
  c_low, c_upp = 1, 1
  for i = 1:nvar
    if kept_cols[i] == false
      offset += 1
    end
    if c_low ≤ nlow && ilow[c_low] + offset == i
      ilow[c_low] += 1
      c_low += 1
    end
    if c_upp ≤ nupp && iupp[c_upp] + offset == i
      iupp[c_upp] += 1
      c_upp += 1
    end
  end
end

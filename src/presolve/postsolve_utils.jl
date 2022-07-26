function restore_x!(kept_cols, xin::S, xout::S, nvar) where {S}
  # put x and xps inside xout according to kept_cols
  cx = 0
  for i = 1:nvar
    if kept_cols[i]
      cx += 1
      xout[i] = xin[cx]
    end
  end
end

function restore_y!(kept_rows::Vector{Bool}, y::Vector{T}, yout::Vector{T}, ncon) where {T}
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

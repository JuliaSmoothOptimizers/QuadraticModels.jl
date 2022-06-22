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

function restore_ilow_iupp!(ilow, iupp, ifix)
  c_fix = 1
  nfix = length(ifix)

  nlow = length(ilow)
  for i = 1:nlow
    while c_fix ≤ nfix && ilow[i] ≤ ifix[c_fix]
      c_fix += 1
    end
    ilow[i] += c_fix - 1
  end

  c_fix = 1
  nupp = length(iupp)
  for i = 1:nupp
    while c_fix ≤ nfix && iupp[i] ≤ ifix[c_fix]
      c_fix += 1
    end
    iupp[i] += c_fix - 1
  end
end

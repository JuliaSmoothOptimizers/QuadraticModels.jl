function restore_x!(kept_cols, x_in::S, x::S, nvar) where {S}
  # put x and xps inside xout according to kept_cols
  cx = 0
  for i = 1:nvar
    if kept_cols[i]
      cx += 1
      x[i] = x_in[cx]
    end
  end
end

function restore_y!(kept_rows::Vector{Bool}, y_in::Vector{T}, y::Vector{T}, ncon) where {T}
  c_y = 0
  for i = 1:ncon
    if !kept_rows[i]
      y[i] = zero(T)
    else
      c_y += 1
      y[i] = y_in[c_y]
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
      ilow[c_low] += offset
      c_low += 1
    end
    if c_upp ≤ nupp && iupp[c_upp] + offset == i
      iupp[c_upp] += offset
      c_upp += 1
    end
  end
end

function restore_s!(
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  s_l_in::SparseVector{T, Int},
  s_u_in::SparseVector{T, Int},
  kept_cols::Vector{Bool},
) where {T}
  ilow, iupp = copy(s_l_in.nzind), copy(s_u_in.nzind)
  restore_ilow_iupp!(ilow, iupp, kept_cols)
  s_l[ilow] .= s_l_in.nzval
  s_u[iupp] .= s_u_in.nzval
end

function restore_s!(
  s_l::S,
  s_u::S,
  s_l_in::S,
  s_u_in::S,
  kept_cols::Vector{Bool},
) where {S <: DenseVector}
  nvar = length(s_l)
  restore_x!(kept_cols, s_l_in, s_l, nvar)
  restore_x!(kept_cols, s_u_in, s_u, nvar)
end
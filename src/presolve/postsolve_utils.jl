function restore_ilow_iupp!(ilow, iupp, ifix)
  c_fix = 1
  nfix = length(ifix)

  nlow = length(ilow)
  for i in 1:nlow
    while c_fix ≤ nfix && ilow[i] ≤ ifix[c_fix]
      c_fix += 1
    end
    ilow[i] += c_fix - 1
  end

  c_fix = 1
  nupp = length(iupp)
  for i in 1:nupp
    while c_fix ≤ nfix && iupp[i] ≤ ifix[c_fix]
      c_fix += 1
    end
    iupp[i] += c_fix - 1
  end
end
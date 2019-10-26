
function simpleqp_autodiff()

  x0   = [-1.; 1.]
  f(x) = 4*x[1]^2 + 5*x[2]^2 + x[1]*x[2] + 1.5*x[1] - 2*x[2]
  c(x) = [2*x[1] + x[2]; -x[1] + 2*x[2]]
  lcon = [2.0; -Inf]
  ucon = [Inf; 6.0]
  uvar = [20; Inf]
  lvar = [0.0; 0.0]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon, lvar=lvar, uvar=uvar)

end

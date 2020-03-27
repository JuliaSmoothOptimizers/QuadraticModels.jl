
function simpleqp_autodiff()

  x0   = [-1.; 1.]
  f(x) = 4*x[1]^2 + 5*x[2]^2 + x[1]*x[2] + 1.5*x[1] - 2*x[2] + 1.0
  c(x) = [2*x[1] + x[2]; -x[1] + 2*x[2]]
  lcon = [2.0; -Inf]
  ucon = [Inf; 6.0]
  uvar = [20; Inf]
  lvar = [0.0; 0.0]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon, lvar=lvar, uvar=uvar)
end

function simpleqp_QP()
  c     = [1.5; -2.0]
  H     = [8.0 0.0; 1.0 10.0]
  A     = [2.0 1.0; -1.0 2.0]
  c0    = 1.0
  lcon  = [2.0; -Inf]
  ucon  = [Inf; 6.0]
  uvar  = [20; Inf]
  lvar  = [0.0; 0.0]
  x0    = [-1.0; 1.0]

  return QuadraticModel(c, H, A=A, lcon=lcon, ucon=ucon, lvar=lvar, uvar=uvar, c0=c0, x0=x0)
end

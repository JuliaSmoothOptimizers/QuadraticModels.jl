
function bndqp_autodiff()

  x0   = [0.5; 0.5]
  f(x) = -x[1]^2 + 2x[2]^2 + 3x[1] * x[2] + x[1] + x[2]
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]

  return ADNLPModel(f, x0, lvar=lvar, uvar=uvar, name="bndqp_autodiff")
end

function bndqp_QP()
  c    = [1.0; 1.0]
  H    = [-2.0 0.0; 3.0 4.0]
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0   = [0.5; 0.5]

  return QuadraticModel(c, H, lvar=lvar, uvar=uvar, x0=x0, name="bndqp_QP")
end

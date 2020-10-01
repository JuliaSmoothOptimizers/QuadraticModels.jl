
function bndqp_autodiff()

  x0   = [0.5; 0.5]
  f(x) = -x[1]^2 + 2x[2]^2 + 3x[1] * x[2] + x[1] + x[2]
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]

  return ADNLPModel(f, x0, lvar, uvar, name="bndqp_autodiff")
end

function bndqp_QP_dense()
  c    = [1.0; 1.0]
  H    = [-2.0 3.0; 3.0 4.0]
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0   = [0.5; 0.5]

  return QuadraticModel(c, H, lvar=lvar, uvar=uvar, x0=x0, name="bndqp_QP")
end

function bndqp_QP_sparse()
  c    = [1.0; 1.0]
  H    = sparse([-2.0 0.0; 3.0 4.0])
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0   = [0.5; 0.5]

  return QuadraticModel(c, H, lvar=lvar, uvar=uvar, x0=x0, name="bndqp_QP")
end

function bndqp_QP_symmetric()
  c    = [1.0; 1.0]
  H    = Symmetric([-2.0 0.0; 3.0 4.0], :L)
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0   = [0.5; 0.5]

  return QuadraticModel(c, H, lvar=lvar, uvar=uvar, x0=x0, name="bndqp_QP")
end

function bndqp_QPSData()
  c    = [1.0; 1.0]
  H    = [-2.0 0.0; 3.0 4.0]
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0   = [0.5; 0.5]
  qps = QPSData()
  qps.c = c
  qps.qrows, qps.qcols, qps.qvals = findnz(sparse(H))
  qps.lvar, qps.uvar = lvar, uvar
  qps.nvar = length(x0)
  return QuadraticModel(qps, x0)
end

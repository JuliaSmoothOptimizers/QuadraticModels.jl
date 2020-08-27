
function ineqconqp_autodiff()

  x0   = ones(2)
  f(x) = (x[1] - 1)^2 / 2 + (x[2] - 1)^2 / 2
  c(x) = [x[1] - x[2]; x[2] - x[1]; x[1] + x[2]]
  lcon = [0.0; -Inf; -1.0]
  ucon = [Inf;  0.0;  1.0]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon, name="ineqconqp_autodiff")
end

function ineqconqp_QP()
  c     = -ones(2)
  Hrows = [1, 2]
  Hcols = [1, 2]
  Hvals = ones(2)
  Arows = [1, 1, 2, 2, 3, 3]
  Acols = [1, 2, 1, 2, 1, 2]
  Avals = [1.0; -1.0; -1.0; 1.0; 1.0; 1.0]
  c0    = 1.0
  lcon  = [0.0; -Inf; -1.0]
  ucon  = [Inf;  0.0;  1.0]
  x0    = ones(2)

  return QuadraticModel(c, Hrows, Hcols, Hvals, Arows=Arows, Acols=Acols, Avals=Avals, lcon=lcon, ucon=ucon, c0=c0, x0=x0, name="ineqconqp_QP")
end

function ineqconqp_QPSData()
  c     = -ones(2)
  Hrows = [1, 2]
  Hcols = [1, 2]
  Hvals = ones(2)
  Arows = [1, 1, 2, 2, 3, 3]
  Acols = [1, 2, 1, 2, 1, 2]
  Avals = [1.0; -1.0; -1.0; 1.0; 1.0; 1.0]
  c0    = 1.0
  lcon  = [0.0; -Inf; -1.0]
  ucon  = [Inf;  0.0;  1.0]
  x0    = ones(2)
  lvar = [-Inf; -Inf]
  uvar = [Inf; Inf]
  qps = QPSData()
  qps.c = c
  qps.c0 = c0
  qps.qrows, qps.qcols, qps.qvals = Hrows, Hcols, Hvals
  qps.arows, qps.acols, qps.avals = Arows, Acols, Avals
  qps.lcon, qps.ucon = lcon, ucon
  qps.lvar, qps.uvar = lvar, uvar
  qps.nvar = length(x0)
  return QuadraticModel(qps, x0)
end

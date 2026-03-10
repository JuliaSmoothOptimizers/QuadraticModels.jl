
function uncqp_autodiff(; σ = 0.0)
  x0 = [-1.2; 1.0]
  f(x) =
    4 * x[1]^2 + 5 * x[2]^2 - x[1] * x[2] + 3.0 * x[1] - 2 * x[2] +
    1.0 +
    0.5 * σ * (x[1]^2 + x[2]^2)

  return ADNLPModel(f, x0, name = "uncqp_autodiff")
end

function uncqp_QP(; σ = 0.0)
  c = [3.0; -2.0]
  Hrows = [1, 2, 2]
  Hcols = [1, 1, 2]
  Hvals = [8.0; -1.0; 10.0]
  c0 = 1.0
  x0 = [-1.2; 1.0]

  σ == 0.0 && return QuadraticModel(c, Hrows, Hcols, Hvals, c0 = c0, x0 = x0, name = "uncqp_QP")
  return QuadraticModel(
    c,
    Hrows,
    Hcols,
    Hvals,
    c0 = c0,
    x0 = x0,
    name = "uncqp_QP",
    regularize = true,
    σ = σ,
  )
end

function uncqp_QPSData(; σ = 0.0)
  c = [3.0; -2.0]
  Hrows = [1, 2, 2]
  Hcols = [1, 1, 2]
  Hvals = [8.0; -1.0; 10.0]
  lvar = [-Inf; -Inf]
  uvar = [Inf; Inf]
  c0 = 1.0
  x0 = [-1.2; 1.0]
  qps = QPSData()
  qps.c = c
  qps.qrows, qps.qcols, qps.qvals = Hrows, Hcols, Hvals
  qps.c0 = c0
  qps.lvar, qps.uvar = lvar, uvar
  qps.nvar = length(x0)
  σ == 0.0 && return QuadraticModel(qps, x0)
  return QuadraticModel(qps, x0, regularize = true, σ = σ)
end

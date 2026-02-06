
function bndqp_autodiff(; σ = 0.0)
  x0 = [0.5; 0.5]
  f(x) = -x[1]^2 + 2x[2]^2 + 3x[1] * x[2] + x[1] + x[2] + 0.5 * σ *(x[1]^2 + x[2]^2)
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]

  return ADNLPModel(f, x0, lvar, uvar, name = "bndqp_autodiff")
end

function bndqp_QP_dense(; σ = 0.0)
  c = [1.0; 1.0]
  H = [-2.0 3.0; 3.0 4.0]
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0 = [0.5; 0.5]

  σ == 0.0 && 
    return QuadraticModel(c, H, lvar = lvar, uvar = uvar, x0 = x0, name = "bndqp_QP")
  return QuadraticModel(c, H, lvar = lvar, uvar = uvar, x0 = x0, name = "bndqp_QP", regularize = true, σ = σ)
end

function bndqp_QP_sparse(; σ = 0.0)
  c = [1.0; 1.0]
  H = sparse([-2.0 0.0; 3.0 4.0])
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0 = [0.5; 0.5]

  σ == 0.0 &&
    return QuadraticModel(c, H, lvar = lvar, uvar = uvar, x0 = x0, name = "bndqp_QP")
  return QuadraticModel(c, H, lvar = lvar, uvar = uvar, x0 = x0, name = "bndqp_QP", regularize = true, σ = σ)
end

function bndqp_QP_symmetric(; σ = 0.0)
  c = [1.0; 1.0]
  H = Symmetric([-2.0 0.0; 3.0 4.0], :L)
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0 = [0.5; 0.5]

  σ == 0.0 &&
    return QuadraticModel(c, H, lvar = lvar, uvar = uvar, x0 = x0, name = "bndqp_QP")
  return QuadraticModel(c, H, lvar = lvar, uvar = uvar, x0 = x0, name = "bndqp_QP", regularize = true, σ = σ)
end

function bndqp_QPSData(; σ = 0.0)
  c = [1.0; 1.0]
  H = [-2.0 0.0; 3.0 4.0]
  uvar = [1.0; 1.0]
  lvar = [0.0; 0.0]
  x0 = [0.5; 0.5]
  qps = QPSData()
  qps.c = c
  qps.qrows, qps.qcols, qps.qvals = findnz(sparse(H))
  qps.lvar, qps.uvar = lvar, uvar
  qps.nvar = length(x0)
  σ == 0.0 &&
    return QuadraticModel(qps, x0)
  return QuadraticModel(qps, x0, regularize = true, σ = σ)
end

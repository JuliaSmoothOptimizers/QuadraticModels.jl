function check_quadratic_model(model, quadraticmodel)
  @test typeof(quadraticmodel) <: QuadraticModels.AbstractQuadraticModel
  rtol  = 1e-8
  @test quadraticmodel.meta.nvar == model.meta.nvar
  @test quadraticmodel.meta.ncon == model.meta.ncon

  x = [-(-1.0)^i for i = 1:quadraticmodel.meta.nvar]

  @test isapprox(obj(model, x), obj(quadraticmodel, x), rtol=rtol)

  f, g = objgrad(model, x)
  f_quad, g_quad = objgrad(quadraticmodel, x)

  @test isapprox(f, f_quad, rtol=rtol)
  @test isapprox(g, g_quad, rtol=rtol)
  @test isapprox(cons(model, x), cons(quadraticmodel, x), rtol=rtol)
  @test isapprox(jac(model, x), jac(quadraticmodel, x), rtol=rtol)

  v = [-(-1.0)^i for i = 1:quadraticmodel.meta.nvar]
  u = [-(-1.0)^i for i = 1:quadraticmodel.meta.ncon]

  @test isapprox(jprod(model, x, v), jprod(quadraticmodel, x, v), rtol=rtol)
  @test isapprox(jtprod(model, x, u), jtprod(quadraticmodel, x, u), rtol=rtol)

  H = hess_op(quadraticmodel, x)
  @test typeof(H) <: LinearOperators.AbstractLinearOperator
  @test size(H) == (model.meta.nvar, model.meta.nvar)
  @test isapprox(H * v, hprod(model, x, v), rtol=rtol)

  reset!(quadraticmodel)
end

for problem in qp_problems_Matrix
  @info "Checking consistency of problem $problem"
  nlp_ad = eval(Symbol(problem * "_autodiff"))()
  nlp_qps = eval(Symbol(problem * "_QPSData"))()
  nlp_qm_dense = eval(Symbol(problem * "_QP_dense"))()
  nlp_qm_sparse = eval(Symbol(problem * "_QP_sparse"))()
  nlp_qm_symmetric = eval(Symbol(problem * "_QP_symmetric"))()
  nlps = [nlp_ad, nlp_qm_dense, nlp_qm_sparse, nlp_qm_symmetric, nlp_qps]
  consistent_nlps(nlps)
  @info "  Consistency checks ✓"
end

for problem in qp_problems_COO
  @info "Checking consistency of problem $problem"
  nlp_ad = eval(Symbol(problem * "_autodiff"))()
  nlp_qm = eval(Symbol(problem * "_QP"))()
  nlp_qps = eval(Symbol(problem * "_QPSData"))()
  nlps = [nlp_ad, nlp_qm, nlp_qps]
  consistent_nlps(nlps)
  @info "  Consistency checks ✓"
end

for problem in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14, :lincon]
  @info "Testing consistency of quadratic approximation of problem $problem"
  include(joinpath(nlpmodels_problems_path, "$problem.jl"))
  problem_s = string(problem)
  nlp = eval(Meta.parse("$(problem)_autodiff"))()
  x = nlp.meta.x0

  fx, gx, Hx = obj(nlp, x), grad(nlp, x), Symmetric(hess(nlp, x), :L)
  nlp_ad = if nlp.meta.ncon > 0
    cx, Ax = cons(nlp, x), jac(nlp, x)
    ADNLPModel(s -> fx + dot(gx, s) + dot(s, Hx * s) / 2, zeros(nlp.meta.nvar),
               nlp.meta.lvar - x, nlp.meta.uvar - x,
               s -> Ax * s, nlp.meta.lcon - cx, nlp.meta.ucon - cx)
  else
    ADNLPModel(s -> fx + dot(gx, s) + dot(s, Hx * s) / 2, zeros(nlp.meta.nvar),
               nlp.meta.lvar - x, nlp.meta.uvar - x)
  end
  nlp_qm = QuadraticModel(nlp, x)
  nlps = [nlp_ad, nlp_qm]
  consistent_nlps(nlps)
end

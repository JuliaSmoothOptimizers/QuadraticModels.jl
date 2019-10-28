function check_quadratic_model(model, quadraticmodel)
    rtol  = 1e-8
    # @assert typeof(quadraticmodel) <: NLPModels.QuadraticModel
    @assert quadraticmodel.meta.nvar == model.meta.nvar
    @assert quadraticmodel.meta.ncon == model.meta.ncon
  
    x = [-(-1.0)^i for i = 1:quadraticmodel.meta.nvar]
  
    @assert isapprox(obj(model, x), obj(quadraticmodel, x), rtol=rtol)

    f, g = grad(model, x)
    f_quad, g_quad = grad(quadraticmodel, x)
  
    @assert isapprox(f, f_quad, rtol=rtol)

    @assert isapprox(g, g_quad, rtol=rtol)
  
    @assert isapprox(cons(model, x), cons(quadraticmodel, x), rtol=rtol)
  
    @assert isapprox(jac(model, x), jac(quadraticmodel, x), rtol=rtol)
  
    v = [-(-1.0)^i for i = 1:quadraticmodel.meta.nvar]
    u = [-(-1.0)^i for i = 1:quadraticmodel.meta.ncon]
  
    @assert isapprox(jprod(model, x, v), jprod(quadraticmodel, x, v), rtol=rtol)
  
    @assert isapprox(jtprod(model, x, u), jtprod(quadraticmodel, x, u), rtol=rtol)
  
    H = hess_op(quadraticmodel, x)
    @assert typeof(H) <: LinearOperators.AbstractLinearOperator
    @assert size(H) == (model.meta.nvar, model.meta.nvar)
    @assert isapprox(H * v, hprod(model, x, v), rtol=rtol)
  
    reset!(quadraticmodel)
  end

  for problem in ["simpleqp"]
    problem_f = eval(Symbol(problem * "_autodiff"))
    nlp = problem_f()
    @printf("Checking QuadraticModel formulation of %-8s\t", problem)
    quadratic_model = QuadraticModel(nlp)
    check_quadratic_model(nlp, quadratic_model)
    @printf("✓\n")
  end

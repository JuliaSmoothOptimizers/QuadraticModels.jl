

function check_quadratic_model(model, quadraticmodel)
    rtol  = 1e-8
    # @assert typeof(quadraticmodel) <: NLPModels.QuadraticModel
    @assert quadraticmodel.meta.nvar == model.meta.nvar
    @assert quadraticmodel.meta.ncon == model.meta.ncon
  
    x = [-(-1.0)^i for i = 1:quadraticmodel.meta.nvar]
  
    @assert isapprox(obj(model, x), obj(quadraticmodel, x), rtol=rtol)
  
    @assert isapprox(grad(model, x), grad(quadraticmodel, x), rtol=rtol)
  
    @assert isapprox(cons(model, x), cons(quadraticmodel, x), rtol=rtol)
  
    @assert isapprox(jac(model, x), jac(quadraticmodel, x), rtol=rtol)
  
    v = [-(-1.0)^i for i = 1:quadraticmodel.meta.nvar]
    u = [-(-1.0)^i for i = 1:quadraticmodel.meta.ncon]
  
    @assert isapprox(jprod(model, x, v), jprod(quadraticmodel, x, v), rtol=rtol)
  
    @assert isapprox(jtprod(model, x, u), jtprod(quadraticmodel, x, u), rtol=rtol)
  
    H = hess_op(quadraticmodel, x)
    @assert typeof(H) <: LinearOperators.AbstractLinearOperator
    @assert size(H) == (model.meta.nvar, model.meta.nvar)
    @assert isapprox(H * v, hprod(quadraticmodel, x, v), rtol=rtol)
  
    g = grad(quadraticmodel, x)
    gp = grad(quadraticmodel, x - g)
    # the quasi-Newton operator itself is tested in LinearOperators
  
    reset!(quadraticmodel)
  end

  for problem in ["simpleqp"]
    try
      eval(Symbol(problem))
    catch
      include("$problem.jl")
    end
    problem_f = eval(Symbol(problem * "_autodiff"))
    nlp = problem_f()
    @printf("Checking QuadraticModel formulation of %-8s\t", problem)
    quadratic_model = QuadraticModel(nlp)
    check_quadratic_model(nlp, quadratic_model)
    @printf("✓\n")
  end

# stdlib
using LinearAlgebra, Printf, SparseArrays, Test

# our packages
using ADNLPModels,
  LinearOperators, NLPModels, NLPModelsModifiers, NLPModelsTest, QPSReader, QuadraticModels

# Definition of quadratic problems
qp_problems_Matrix = ["bndqp", "eqconqp"]
qp_problems_COO = ["uncqp", "ineqconqp"]
for qp in [qp_problems_Matrix; qp_problems_COO]
  include(joinpath("problems", "$qp.jl"))
end

include("test_consistency.jl")

function testSM(sm) # test function for a specific problem
  @test (sm.meta.ncon == 2 && sm.meta.nvar == 5)
  @test sm.meta.lcon == sm.meta.ucon
  lvar_sm_true = [0.0; 0.0; 0.0; -4.0; -3.0]
  uvar_sm_true = [Inf; Inf; Inf; Inf; -2.0]
  H_sm_true = sparse(
    [
      6.0 0.0 0.0 0.0 0.0
      2.0 5.0 0.0 0.0 0.0
      1.0 2.0 4.0 0.0 0.0
      0.0 0.0 0.0 0.0 0.0
      0.0 0.0 0.0 0.0 0.0
    ],
  )
  A_sm_true = sparse([
    1.0 0.0 1.0 0.0 -1.0
    0.0 2.0 1.0 -1.0 0.0
  ])

  @test all(lvar_sm_true .== sm.meta.lvar)
  @test all(uvar_sm_true .== sm.meta.uvar)
  @test all(sparse(sm.data.Arows, sm.data.Acols, sm.data.Avals, 2, 5) .== A_sm_true)
  @test all(sparse(sm.data.Hrows, sm.data.Hcols, sm.data.Hvals, 5, 5) .== H_sm_true)
end

@testset "SlackModel" begin
  H = [
    6.0 2.0 1.0
    2.0 5.0 2.0
    1.0 2.0 4.0
  ]
  c = [-8.0; -3; -3]
  A = [
    1.0 0.0 1.0
    0.0 2.0 1.0
  ]
  b = [0.0; 3]
  l = [0.0; 0; 0]
  u = [Inf; Inf; Inf]
  T = eltype(c)
  qp = QuadraticModel(
    c,
    sparse(H),
    A = A,
    lcon = [-3.0; -4.0],
    ucon = [-2.0; Inf],
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )
  sm = SlackModel(qp)
  testSM(sm)

  SlackModel!(qp)
  testSM(qp)
end

@testset "sort cols COO" begin
  Hrows = [1; 2; 1; 3]
  Hcols = [2; 1; 4; 4]
  Hvals = [1.0; 2.0; 3.0; 4.0]
  Arows = [2; 2; 1; 4]
  Acols = [4; 3; 1; 2]
  Avals = [-1.0; -2.0; -3.0; -4.0]
  c = [-8.0; -3; -3; 2.0]
  l = [0.0; 0; 0; 0]
  u = [Inf; Inf; Inf; Inf]
  qp = QuadraticModel(
    c,
    Hrows,
    Hcols,
    Hvals,
    Arows = Arows,
    Acols = Acols,
    Avals = Avals,
    lcon = [-3.0; -4.0; 2.0; 1.0],
    ucon = [-2.0; Inf; Inf; Inf],
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
    sortcols = true,
  )
  @test issorted(Hcols)
  @test issorted(Acols)
end

include("test_presolve.jl")

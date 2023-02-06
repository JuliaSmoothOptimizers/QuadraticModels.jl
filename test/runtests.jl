# stdlib
using LinearAlgebra, Printf, SparseArrays, Test

# our packages
using ADNLPModels,
  LinearOperators,
  NLPModels,
  NLPModelsModifiers,
  NLPModelsTest,
  QPSReader,
  QuadraticModels,
  SparseMatricesCOO

@testset "test utils" begin
  A = rand(10, 10)
  @test nnz(A) == 100
  @test nnz(Diagonal(A)) == 10
  @test nnz(Symmetric(A)) == 55
  v1, v2 = rand(10), rand(9)
  @test nnz(SymTridiagonal(v1, v2)) == 19
  Asp = sparse([1.0 0.0 0.0; 1.0 0.0 0.0; 0.0 3.0 2.0])
  @test nnz(Symmetric(Asp, :L)) == 4
end

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
  @test all(sparse(sm.data.A.rows, sm.data.A.cols, sm.data.A.vals, 2, 5) .≈ A_sm_true)
  @test all(sparse(sm.data.H.rows, sm.data.H.cols, sm.data.H.vals, 5, 5) .≈ H_sm_true)
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
    SparseMatrixCOO(tril(H)),
    A = SparseMatrixCOO(A),
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

@testset "dense QP, convert" begin
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

  qpdense = QuadraticModel(
    c,
    tril(H),
    A = A,
    lcon = [-3.0; -4.0],
    ucon = [-2.0; Inf],
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )

  smdense = SlackModel(qpdense)
  testSM(smdense)

  qpcoo =
    convert(QuadraticModel{T, Vector{T}, SparseMatrixCOO{T, Int}, SparseMatrixCOO{T, Int}}, qpdense)
  @test typeof(qpcoo.data.H) <: SparseMatrixCOO
  @test typeof(qpcoo.data.A) <: SparseMatrixCOO
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

@testset "LinearOperators" begin
  nvar, ncon = 10, 7
  T = Float64
  H = Symmetric(tril(sprand(T, nvar, nvar, 0.3)), :L)
  A = sprand(T, ncon, nvar, 0.4)
  c = rand(nvar)
  lvar = fill(-Inf, nvar)
  uvar = fill(0.0, nvar)
  lcon = rand(ncon)
  ucon = lcon .+ 100.0
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(H.data),
    A = SparseMatrixCOO(A),
    lcon = lcon,
    ucon = ucon,
    lvar = lvar,
    uvar = uvar,
    c0 = 0.0,
    name = "QM",
  )
  qpLO = QuadraticModel(
    c,
    LinearOperator(H),
    A = LinearOperator(A),
    lcon = lcon,
    ucon = ucon,
    lvar = lvar,
    uvar = uvar,
    c0 = 0.0,
    name = "QMLO",
  )
  x = ones(10)
  @test obj(qp, x) ≈ obj(qpLO, x)
  @test grad(qp, x) ≈ grad(qpLO, x)
  @test cons(qp, x) ≈ cons(qpLO, x)

  SM = SlackModel(qp)
  SMLO = SlackModel(qpLO)
  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix
  x = rand(SM.meta.nvar)
  y = rand(SM.meta.ncon)
  @test SM.meta.nvar == qp.meta.nvar + ns
  @test obj(SM, x) ≈ obj(SMLO, x)
  @test grad(SM, x) ≈ grad(SMLO, x)
  @test cons(SM, x) ≈ cons(SMLO, x)
  @test hprod(SMLO, x, x) ≈ hprod(SMLO, x, x)
  @test jtprod(SMLO, x, y) ≈ jtprod(SM, x, y)
  @test objgrad(SMLO, x)[1] ≈ objgrad(SM, x)[1]
  @test objgrad(SMLO, x)[2] ≈ objgrad(SM, x)[2]

  # tests default A value
  qp = QuadraticModel(c, SparseMatrixCOO(H.data), lvar = lvar, uvar = uvar, c0 = 0.0, name = "QM")
  qpLO = QuadraticModel(c, LinearOperator(H), lvar = lvar, uvar = uvar, c0 = 0.0, name = "QMLO")
  @test qp.data.A isa SparseMatrixCOO
  @test qpLO.data.A isa AbstractLinearOperator
end

@testset "struct and coord CSC" begin
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
    tril(sparse(H)),
    A = sparse(A),
    lcon = b,
    ucon = b,
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM",
  )

  x = zeros(3)
  rowsH, colsH = hess_structure(qp)
  valsH = hess_coord(qp, x)
  rowsHtrue, colsHtrue, valsHtrue = findnz(tril(sparse(H)))
  @test rowsH == rowsHtrue
  @test colsH == colsHtrue
  @test valsH == valsHtrue
  rowsA, colsA = jac_structure(qp)
  valsA = jac_coord(qp, x)
  rowsAtrue, colsAtrue, valsAtrue = findnz(sparse(A))
  @test rowsA == rowsAtrue
  @test colsA == colsAtrue
  @test valsA == valsAtrue
end

include("test_presolve.jl")

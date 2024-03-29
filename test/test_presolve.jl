@testset "presolve ifix" begin
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
  l = [0.0; 2.0; 0]
  u = [Inf; 2.0; Inf]
  T = eltype(c)
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(tril(H)),
    A = SparseMatrixCOO(A),
    lcon = [-3.0; -4.0],
    ucon = [20.0; Inf],
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )

  statsps = presolve(qp)
  psqp = statsps.solver_specific[:presolvedQM]

  c_true = [-4.0; 1.0]
  c0_true = 4.0
  Hps_true = [
    6.0 1.0
    1.0 4.0
  ]
  Aps_true = [1.0 1.0]
  lvarps_true, uvarps_true = [0.0; 0.0], [Inf; Inf]

  @test psqp.data.c == [-4.0; 1.0]
  @test psqp.data.c0 == 4.0
  Hps = sparse(psqp.data.H.rows, psqp.data.H.cols, psqp.data.H.vals, psqp.meta.nvar, psqp.meta.nvar)
  @test norm(Symmetric(Hps, :L) - sparse(Hps_true)) ≤ sqrt(eps(T))
  Aps = sparse(psqp.data.A.rows, psqp.data.A.cols, psqp.data.A.vals, psqp.meta.ncon, psqp.meta.nvar)
  @test norm(Aps - sparse(Aps_true)) ≤ sqrt(eps(T))
  @test psqp.meta.lvar == lvarps_true
  @test psqp.meta.uvar == uvarps_true
  @test psqp.meta.ifix == Int[]
  @test psqp.meta.nvar == 2

  sol_in = QMSolution([4.0; 7.0], [2.0; 2.0], [3.0; 2.0], [0.0; 0.0])
  sol = postsolve(qp, psqp, sol_in)
  @test sol.x == [4.0; 2.0; 7.0]

  # test that solves the problem
  qp2 = QuadraticModel(
    zeros(2),
    SparseMatrixCOO(zeros(2, 2)),
    A = SparseMatrixCOO(zeros(0, 2)),
    lvar = zeros(2),
    uvar = zeros(2),
  )
  stats_ps2 = presolve(qp2)
  @test stats_ps2.status == :first_order
end

@testset "presolve empty rows" begin
  H = [
    6.0 2.0 1.0
    2.0 5.0 2.0
    1.0 2.0 4.0
  ]
  c = [-8.0; -3; -3]
  A = [
    1.0 0.0 1.0
    0.0 0.0 0.0
    0.0 0.0 0.0
    3.2 0.0 2.0
    0.0 0.0 0.0
    0.0 2.0 1.0
  ]
  b = [0.0; 0.0; 0.0; 4.0; 0.0; 3]
  l = [-2.0; -4.0; 0]
  u = [Inf; 2.0; Inf]
  T = eltype(c)
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(tril(H)),
    A = SparseMatrixCOO(A),
    lcon = b,
    ucon = b,
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )

  statsps = presolve(qp)
  psqp = statsps.solver_specific[:presolvedQM]

  Aps_true = [
    1.0 0.0 1.0
    3.2 0.0 2.0
    0.0 2.0 1.0
  ]
  bps_true = [0.0; 4.0; 3.0]

  Aps = sparse(psqp.data.A.rows, psqp.data.A.cols, psqp.data.A.vals, psqp.meta.ncon, psqp.meta.nvar)
  @test Aps == sparse(Aps_true)
  @test psqp.meta.lcon == psqp.meta.ucon == bps_true
  @test psqp.meta.ncon == 3

  sol_in = QMSolution([4.0; 7.0; 4.0], [2.0; 2.0; 4.0], [3.0; 4.0; 2.0], [0.0; 3.0; 0.0])
  sol = postsolve(qp, psqp, sol_in)
  @test sol.y == [2.0; 0.0; 0.0; 2.0; 0.0; 4.0]
end

@testset "presolve singleton rows" begin
  H = [
    6.0 2.0 1.0
    2.0 5.0 2.0
    1.0 2.0 4.0
  ]
  c = [-8.0; -3; -3]
  A = [
    1.0 0.0 1.0
    0.0 0.0 0.0
    0.0 0.0 0.0
    3.0 0.0 0.0
    0.0 0.0 0.0
    0.0 2.0 1.0
  ]
  lcon = [0.0; 0.0; 0.0; 4.0; 0.0; 3]
  ucon = [3.0; 0.0; 0.0; 7.0; 0.0; 8.0]
  l = [0.0; 0.0; 0]
  u = [Inf; 2.0; Inf]
  T = eltype(c)
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(tril(H)),
    A = SparseMatrixCOO(A),
    lcon = lcon,
    ucon = ucon,
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )

  statsps = presolve(qp)
  psqp = statsps.solver_specific[:presolvedQM]

  Aps_true = [
    1.0 0.0 1.0
    0.0 2.0 1.0
  ]

  Aps = sparse(psqp.data.A.rows, psqp.data.A.cols, psqp.data.A.vals, psqp.meta.ncon, psqp.meta.nvar)
  @test Aps == sparse(Aps_true)
  @test psqp.meta.ncon == 2

  # test sparse s_l and s_u multipliers:
  sol_in = QMSolution([4.0; 7.0; 4.0], [2.0; 4.0], sparse([3.0; 4.0; 2.0]), sparse([0.0; 3.0; 0.0]))
  sol = postsolve(qp, psqp, sol_in)
  @test sol.y == [2.0; 0.0; 0.0; 1.0; 0.0; 4.0]
end

@testset "presolve solves problem" begin
  Q = [
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
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(tril(Q)),
    A = SparseMatrixCOO(A),
    lcon = b,
    ucon = b,
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM",
  )
  statsps = presolve(qp)
  psqp = statsps.solver_specific[:presolvedQM]
  atol = eps()
  @test isapprox(statsps.solution, [0.0; 1.5; 0.0], atol = atol)
  @test isapprox(statsps.objective, 1.125, atol = atol)
  @test isapprox(statsps.multipliers_L[2], 0.0, atol = atol)
  @test isapprox(statsps.multipliers_U, [0.0; 0.0; 0.0], atol = atol)
end

@testset "presolve singleton rows and ifix" begin
  H = [
    6.0 2.0 1.0
    2.0 5.0 2.0
    1.0 2.0 4.0
  ]
  c = [-8.0; -3; -3]
  A = [
    1.0 0.0 1.0
    0.0 0.0 0.0
    0.0 0.0 0.0
    3.0 0.0 0.0
    0.0 0.0 0.0
    0.0 2.0 1.0
  ]
  b = [2.0; 0.0; 0.0; 4.0; 0.0; 3]
  l = [0.0; 0.0; 0]
  u = [Inf; 2.0; Inf]
  T = eltype(c)
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(tril(H)),
    A = SparseMatrixCOO(A),
    lcon = b,
    ucon = b,
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )

  statsps = presolve(qp)
  psqp = statsps.solver_specific[:presolvedQM]

  @test statsps.status == :first_order
  @test statsps.solution ≈ [4.0 / 3.0; 7.0 / 6.0; 2.0 / 3.0]
end

@testset "presolve free col singleton" begin
  H = zeros(Float64, 3, 3)
  c = [-8.0; 3; -3]
  A = [
    1.0 0.0 1.0
    3.0 0.0 2.0
    1.0 2.0 1.0
  ]
  b = [2.0; 4.0; 3]
  l = [0.0; -Inf; 0]
  u = [Inf; Inf; Inf]
  T = eltype(c)
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(tril(H)),
    A = SparseMatrixCOO(A),
    lcon = b,
    ucon = copy(b),
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )

  statsps = presolve(qp)
  psqp = statsps.solver_specific[:presolvedQM]

  sol_in = QMSolution([4.0; 4.0], [2.0; 4.0], [3.0; 2.0], [0.0; 0.0])
  sol = postsolve(qp, psqp, sol_in)
  @test sol.x ≈ [4.0, -2.5, 4.0]
end

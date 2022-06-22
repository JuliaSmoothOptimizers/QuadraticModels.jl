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
    ucon = [-2.0; Inf],
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
  Aps_true = [
    1.0 1.0
    0.0 1.0
  ]
  lvarps_true, uvarps_true = [0.0; 0.0], [Inf; Inf]

  @test psqp.data.c == [-4.0; 1.0]
  @test psqp.data.c0 == 4.0
  Hps = sparse(psqp.data.H.rows, psqp.data.H.cols, psqp.data.H.vals, psqp.meta.ncon, psqp.meta.nvar)
  @test norm(Symmetric(Hps, :L) - sparse(Hps_true)) ≤ sqrt(eps(T))
  Aps = sparse(psqp.data.A.rows, psqp.data.A.cols, psqp.data.A.vals, psqp.meta.ncon, psqp.meta.nvar)
  @test norm(Aps - sparse(Aps_true)) ≤ sqrt(eps(T))
  @test psqp.meta.lvar == lvarps_true
  @test psqp.meta.uvar == uvarps_true
  @test psqp.psd.xrm == [2.0]
  @test psqp.meta.ifix == Int[]
  @test psqp.meta.nvar == 2

  x_in = [4.0; 7.0]
  y_in = [2.0; 2.0]
  s_l = sparse([3.0; 2.0])
  s_u = sparse([0.0; 0.0])
  x_out, y_out, s_l, s_u = postsolve(qp, psqp, x_in, y_in, s_l, s_u)
  @test x_out == [4.0; 2.0; 7.0]

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

  x_in = [4.0; 7.0; 4.0]
  y_in = [2.0; 2.0; 4.0]
  s_l = sparse([3.0; 4.0; 2.0])
  s_u = sparse([0.0; 3.0; 0.0])
  x_out, y_out, s_l, s_u = postsolve(qp, psqp, x_in, y_in, s_l, s_u)
  @test y_out == [2.0; 0.0; 0.0; 2.0; 0.0; 4.0]
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

  x_in = [4.0; 7.0; 4.0]
  y_in = [2.0; 4.0]
  s_l = sparse([3.0; 4.0; 2.0])
  s_u = sparse([0.0; 3.0; 0.0])
  x_out, y_out, s_l, s_u = postsolve(qp, psqp, x_in, y_in, s_l, s_u)
  @test y_out == [2.0; 0.0; 0.0; 0.0; 0.0; 4.0]
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

  Aps_true = [
    0.0 1.0
    2.0 1.0
  ]

  Aps = sparse(psqp.data.A.rows, psqp.data.A.cols, psqp.data.A.vals, psqp.meta.ncon, psqp.meta.nvar)
  @test Aps == sparse(Aps_true)
  @test psqp.meta.ncon == 2

  x_in = [7.0; 4.0]
  y_in = [2.0; 4.0]
  s_l = sparse([4.0; 2.0])
  s_u = sparse([3.0; 0.0])
  x_out, y_out, s_l, s_u = postsolve(qp, psqp, x_in, y_in, s_l, s_u)
  @test y_out == [2.0; 0.0; 0.0; 0.0; 0.0; 4.0]
  @test x_out == [4.0 / 3.0; 7.0; 4.0]
end

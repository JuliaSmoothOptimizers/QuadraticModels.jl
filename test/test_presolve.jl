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

  psqp = presolve(qp)

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
  @test psqp.xrm == [2.0]
  @test psqp.meta.ifix == Int[]
  @test psqp.meta.nvar == 2

  x_in = [4.0; 7.0]
  x_out = zeros(3)
  postsolve!(qp, psqp, x_in, x_out)
  @test x_out == [4.0; 2.0; 7.0]
end

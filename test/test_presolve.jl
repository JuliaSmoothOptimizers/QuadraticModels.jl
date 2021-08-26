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
    H,
    A = A,
    lcon = [-3.0; -4.0],
    ucon = [-2.0; Inf],
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )
    
  psqp = presolve(qp)

  Hps_true =   H = [
    6.0 1.0
    1.0 4.0
  ]
  Aps_true = [
    1.0 1.0
    0.0 1.0
  ]

  lvarps_true, uvarps_true = [0. ; 0.], [Inf; Inf]
  @test psqp.meta.lvar == lvarps_true
  @test psqp.meta.uvar == uvarps_true
  println(psqp.data.c)
  println(psqp.data.c0)
  Aps = sparse(psqp.data.Arows, psqp.data.Acols, psqp.data.Avals, psqp.meta.ncon, psqp.meta.nvar)
  @test norm(Aps - sparse(Aps_true)) ≤ sqrt(eps(T))
  Hps = sparse(psqp.data.Hrows, psqp.data.Hcols, psqp.data.Hvals, psqp.meta.ncon, psqp.meta.nvar)
  @test norm(Symmetric(Hps, :L) - sparse(Hps_true)) ≤ sqrt(eps(T))
  @test psqp.xrm == [2.0]
  @test psqp.meta.ifix == Int[]
  @test psqp.meta.nvar == 2

end
function test_only_zeros(table)
  for (key, vals) in table
    if !isnan(vals)
      @test vals == 0
      vals != 0 && println(key)
    end
  end
end

@testset "allocations" begin
  @testset "allocs QPSData" begin
    for problem in qp_problems_Matrix
      nlp_qps = eval(Symbol(problem * "_QPSData"))()
      test_only_zeros(test_allocs_nlpmodels(nlp_qps))
    end
  end

  @testset "allocs QP_dense" begin
    for problem in qp_problems_Matrix
      nlp_qm_dense = eval(Symbol(problem * "_QP_dense"))()
      test_only_zeros(test_allocs_nlpmodels(nlp_qm_dense))
    end
  end

  @testset "allocs COO QPSData" begin
    for problem in qp_problems_COO
      nlp_qps = eval(Symbol(problem * "_QPSData"))()
      test_only_zeros(test_allocs_nlpmodels(nlp_qps))
    end
  end

  @testset "allocs COO QP" begin
    for problem in qp_problems_COO
      nlp_qm_dense = eval(Symbol(problem * "_QP"))()
      test_only_zeros(test_allocs_nlpmodels(nlp_qm_dense))
    end
  end

  @testset "allocs quadratic approximation" begin
    for problem in NLPModelsTest.nlp_problems
      nlp = eval(Symbol(problem))()
      x = nlp.meta.x0
      nlp_qm = QuadraticModel(nlp, x)
      test_only_zeros(test_allocs_nlpmodels(nlp_qm))
    end
  end
end

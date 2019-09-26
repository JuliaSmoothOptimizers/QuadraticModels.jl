# stdlib
using SparseArrays, Test

# our packages
using QuadraticModels, NLPModels, NLPModelsIpopt, LinearOperators

@testset "QuadraticModelsTests" begin
	c0        = 0.0
	c         = [1.5; -2.0]
	Q         = sparse([1; 2; 2], [1; 1; 2], [8.0; 2.0; 10.0])
	A         = sparse([1; 2; 1; 2], [1; 1; 2; 2], [2.0; -1.0; 1.0; 2.0])
	lcon      = [2.0; -Inf]
	ucon      = [Inf; 6.0]
	uvar      = [20; Inf]
	lvar      = [0.0; 0.0]
	objective = 4.3718750e+00

	qp     = QuadraticModel(c, Q, opHermitian(Q), A, lcon, ucon, lvar, uvar, c0 = c0)
	output = ipopt(qp, print_level = 0)

	@test output.dual_feas                      < 1e-6
	@test output.primal_feas                    < 1e-6
	@test abs(output.objective - objective)     < 1e-6

end # testset

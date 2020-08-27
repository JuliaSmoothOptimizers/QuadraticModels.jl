# stdlib
using Printf, SparseArrays, Test

# our packages
using QPSReader
using LinearAlgebra, LinearOperators, NLPModels, QuadraticModels

nlpmodels_path = joinpath(dirname(pathof(NLPModels)), "..", "test")
nlpmodels_problems_path = joinpath(nlpmodels_path, "problems")

# Definition of quadratic problems
qp_problems = ["uncqp", "bndqp", "eqconqp", "ineqconqp"]
for qp in qp_problems
  include(joinpath("problems", "$qp.jl"))
end

include(joinpath(nlpmodels_path, "consistency.jl"))
include("test_consistency.jl")

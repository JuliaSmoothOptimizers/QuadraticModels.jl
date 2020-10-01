# stdlib
using Printf, SparseArrays, Test

# our packages
using LinearAlgebra, LinearOperators, NLPModels, QPSReader, QuadraticModels

nlpmodels_path = joinpath(dirname(pathof(NLPModels)), "..", "test")
nlpmodels_problems_path = joinpath(nlpmodels_path, "problems")

# Definition of quadratic problems
qp_problems_Matrix = ["bndqp", "eqconqp"]
qp_problems_COO = ["uncqp", "ineqconqp"]
for qp in vcat(qp_problems_Matrix, qp_problems_COO)
  include(joinpath("problems", "$qp.jl"))
end

include(joinpath(nlpmodels_path, "consistency.jl"))
include("test_consistency.jl")

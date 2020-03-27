# stdlib
using Printf, SparseArrays, Test

# our packages
using LinearAlgebra, LinearOperators, NLPModels, QuadraticModels

nlpmodels_path = joinpath(dirname(pathof(NLPModels)), "..", "test")
nlpmodels_problems_path = joinpath(nlpmodels_path, "problems")

# Definition of the quadratic problem in ADNLPModel
include("simpleqp.jl")

include(joinpath(nlpmodels_path, "consistency.jl"))
include("test_consistency.jl")

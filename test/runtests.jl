# stdlib
using Printf, SparseArrays, Test

# our packages
using LinearOperators, NLPModels, NLPModelsIpopt, QuadraticModels

include("test_solving_with_ipopt.jl")
include("simpleqp.jl")
include("test_consistency.jl")

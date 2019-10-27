# stdlib
using SparseArrays, Test, Printf

# our packages
using LinearOperators, NLPModels, NLPModelsIpopt, QuadraticModels

include("test_solving_with_ipopt.jl")
include("test_consistency.jl")

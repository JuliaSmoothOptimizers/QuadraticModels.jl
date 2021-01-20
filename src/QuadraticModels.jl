module QuadraticModels

# stdlib
using LinearAlgebra, SparseArrays

# our packages
using LinearOperators, NLPModels

# auxiliary packages
using FastClosures, Requires

import NLPModels:
    objgrad, objgrad!, obj, grad, grad!,
    hess_coord, hess, hess_op, hprod,
    cons, cons!,
    jac_coord, jac, jac_op, jprod, jtprod,
    SlackModel, slack_meta

export QuadraticModel

include("qpmodel.jl")

function __init__()
    @require QPSReader = "10f199a5-22af-520b-b891-7ce84a7b1bd0" include("qps.jl")
end

end # module

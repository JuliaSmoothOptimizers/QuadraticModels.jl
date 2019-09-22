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
    jac_coord, jac, jac_op, jprod, jtprod

export QuadraticModel

include("qpmodel.jl")

function __init__()
    @require QPSReader = "758ba83c-e923-11e8-036a-3f93d0cb3d0c" include("qps.jl")
end

end # module

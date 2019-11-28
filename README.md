# QuadraticModels

Linux and macOS: [![Build Status](https://travis-ci.org/JuliaSmoothOptimizers/QuadraticModels.jl.svg?branch=master)](https://travis-ci.org/JuliaSmoothOptimizers/QuadraticModels.jl)
Windows: [![Build Status](https://ci.appveyor.com/api/projects/status/github/JuliaSmoothOptimizers/QuadraticModels.jl?branch=master&svg=true)](https://ci.appveyor.com/project/tkelman/example-jl/branch/master)
FreeBSD: [![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/QuadraticModels.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/QuadraticModels.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaSmoothOptimizers/QuadraticModels.jl/badge.svg?branch=master)](https://coveralls.io/r/JuliaSmoothOptimizers/QuadraticModels.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaSmoothOptimizers/QuadraticModels.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaSmoothOptimizers/QuadraticModels.jl?branch=master)

A package to model linear and quadratic optimization problems using the [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) data structures.

The problems represented have the form

<p align="center">
optimize &nbsp; c₀ + cᵀ x + ½ xᵀ Q x
&nbsp;&nbsp;
subject to &nbsp; L ≤ Ax ≤ U and ℓ ≤ x ≤ u,
</p>

where the square symmetric matrix Q is zero for linear optimization problems.

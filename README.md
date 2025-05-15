# QuadraticModels

Linux/macOS/Windows: ![CI](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl/workflows/CI/badge.svg?branch=main)
FreeBSD: [![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/QuadraticModels.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/QuadraticModels.jl)
[![codecov.io](http://codecov.io/github/JuliaSmoothOptimizers/QuadraticModels.jl/coverage.svg?branch=main)](http://codecov.io/github/JuliaSmoothOptimizers/QuadraticModels.jl?branch=main)
[![Documentation/stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSmoothOptimizers.github.io/QuadraticModels.jl/stable)
[![Documentation/dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/QuadraticModels.jl/dev)

A package to model linear and quadratic optimization problems using the [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) data structures.

The problems represented have the form

<p align="center">
optimize &nbsp; c₀ + cᵀ x + ½ xᵀ Q x
&nbsp;&nbsp;
subject to &nbsp; L ≤ Ax ≤ U and ℓ ≤ x ≤ u,
</p>

where the square symmetric matrix Q is zero for linear optimization problems.

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.

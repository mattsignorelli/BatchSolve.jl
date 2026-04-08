# BatchSolve

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mattsignorelli.github.io/BatchSolve.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattsignorelli.github.io/BatchSolve.jl/dev/)
[![Build Status](https://github.com/mattsignorelli/BatchSolve.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mattsignorelli/BatchSolve.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mattsignorelli/BatchSolve.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mattsignorelli/BatchSolve.jl)

## Should you use this package?

Do you have a (GPU) vectorized residual/merit function that you want to solve? Do you want to solve it using automatic differentiation for 100,000 different parameter configurations in parallel? Or is your problem not autodifferentiable, and you're looking for a SIMD-friendly finite differences method? Something lightweight, easily precompileable, and supporting a mutable interface? Look no further!

`BatchSolve.jl` aims to provide various functionalities for solving batch-able, vectorized residual functions (for root finding) and objective functions (for minimizing). As long as your function is vectorizable, then `BatchSolve.jl` will take care of the rest!

What do we mean by "vectorizable"? Say you want to find the root `x` of this function given some parameters `p`:

```julia
f(x, p) = (x-1)^2 - p^2
```

You can do this easily with a Newton or Brent method, of course, probably available in some other root-finding package. But what if you want to solve this for 1 million different `p`? Well, if we let `p` be a vector of those million different parameters, then we can evaluate this function for a million different `x` using the broadcast operator "`.`" in Julia:

```julia
f_vectorized(x, p) = f.(x, p)
```

The result of `f_vectorized` will be a vector where each element is the corresponding `f(x,p)`, and the evaluation will take advantage of SIMD if your CPU supports it. Better yet, if `x` and `p` are GPU arrays (e.g. `CuArray` for NVIDIA CUDA or `MtlArray` for Apple Metal), then the evaluation will be GPU-parallelized!

Now we can easily do a Newton method or Brent method with this:

## Should you NOT use this package?

This package is geared towards lightweight but high performance batched solvers, and requires the user to have some knowledge of the appropriate solver to choose in order to get reasonable results for their problem. This is quite different from the polyalgorithm approaches in other packages, e.g. `Optimization.jl` or `BlackBoxOptim.jl`, which is generally better for cases where you just want to "press go" and get a solution, even for hard problems. Furthermore, in cases where a solution doesn't converge or blows up, all this package will tell you is that it failed (and give you the last Jacobian if derivatives are used). 



Of course, `BatchSolve.jl` is not limited to only 1D functions. Consider this example, where we want to find the roots of this 2x2 system, for 100,000 different paramter configurations in parallel:





`BatchSolve.jl` makes heavy use of the powerful `DifferentiationInterface.jl` library, in particular its sparse autodifferentiation features . It goes without saying that the features here wiol

Julia's optimization ecosystem was missing. 
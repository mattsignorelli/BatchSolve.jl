# BatchSolve

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mattsignorelli.github.io/BatchSolve.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattsignorelli.github.io/BatchSolve.jl/dev/)
[![Build Status](https://github.com/mattsignorelli/BatchSolve.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mattsignorelli/BatchSolve.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mattsignorelli/BatchSolve.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mattsignorelli/BatchSolve.jl)

`BatchSolve.jl` aims to provide various functionalities for solving batch-able, vectorized residual functions (for root finding) and objective functions (for minimizing). As long as your function is vectorized, then `BatchSolve.jl` will take care of the rest! 

In summary, this package features:

- Solvers for (GPU) vectorized residual/merit functions
- The `AutoBatch` automatic-differentiation (AD) type for computing batches of derivatives of CPU/GPU vectorized/parallel functions, and is also fully compatible with [`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl) and any of its supported AD backends
- Specialized bindings for [`FiniteDiff.jl`](https://github.com/JuliaDiff/FiniteDiff.jl) with `AutoBatch` to compute CPU/GPU vectorized finite differences
- Solvers that directly plug into CUDA's cuBLAS library for batched linear solving 

What do we mean by "vectorized"? Say you want to find the root `x` of this function given some parameters `p`:

```julia
f(x, p) = (x-1)^2 - p^2
```

You can do this easily with a Newton or Brent method, of course, probably available in some other root-finding package. But what if you want to solve this for 1 million different `p`? Well, if we let `p` be a vector of those million different parameters, then we can evaluate this function for a million different `x` using the broadcast operator "`.`" in Julia:

```julia
f_vectorized(x, p) = f.(x, p)
```

The result of `f_vectorized` will be a vector where each element is the corresponding `f(x,p)`, and the evaluation will (hopefully) take advantage of SIMD if your CPU supports it. Better yet, if `x` and `p` are GPU arrays (e.g. `CuArray` for NVIDIA CUDA), then the evaluation will be GPU-parallelized!

With `BatchSolve.jl`, we can do root finding on this function in a vectorized way too. All we need to do is specify the `batchdim` - the dimension of the input array along which each *independent* element in a batch lies, where 1 corresponds to the rows and 2 to the columns. This allows for batched solving of many n-dimensional dimensional systems. For our 1D example here we would just choose 1. E.g. for a batch with `p=1:10000`,

```julia
sol = newton(f_vectorized, zeros(10000), Constant(1:10000), batchdim=1)
```

where we specified that `p` is a `Constant` - not mutated throughout function evaluation. Arguments could also be specified as `Cache` if they are mutated. These constructs are re-exported from [`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl), and more details can be found [here](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/tutorials/advanced/#Contexts). For all AD purposes, `BatchSolve` uses `DifferentiationInterface`. Therefore it is easy to use a different AD backend, e.g. `Enzyme`:

```julia
using Enzyme
sol = newton(f_vectorized, zeros(10000), Constant(1:10000), batchdim=1, autodiff=AutoEnzyme())
```

The returned object is a `NamedTuple` with the fields:
- `u`: an array with size equal to the input array, containing the inputs at the roots
- `jac`: a (sparse) matrix storing the Jacobian for the entire batched-system at the last iteration
- `retcode`: an array of return codes for each element in the batch, where `0x0` is success, `0x1` is failure, and `0x2` means the maximum number of iterations was reached
- `iters`: an array storing the number of iterations until convergence for each element in the batch

Of course, `BatchSolve.jl` is not limited to only 1D functions. Consider this example:

```julia
g(x, p) = x .+ p
```

This function takes in a vector `x` and adds a vector `p` to it. Suppose we want to do a Newton search on this system now for vectors of length 2, for many different `p`, say 10000. In this simple case we can evaluate `g` in a vectorized way for all systems simply by initializing `x` and `p` as matrices, either 2 x 10000 or 10000 x 2. For `batchdim=1`, this would be 10000 x 2:

```julia
x0 = zeros(10000, 2)
p = rand(10000, 2)
sol = newton(g, x0, Constant(p), batchdim=1)
```

`sol.u` will be a 10000 x 2 matrix with the roots. For `batchdim=2`:

```julia
x0 = zeros(2, 10000)
p = rand(2, 10000)
sol = newton(g, x0, Constant(p), batchdim=2)
```

Of course, you could just treat this batch all together as one vector of size 20000 (number of elements in the "batch" equal to one). This can be done by not setting `batchdim`, which basically gives a regular non-batched Newton solver:

```julia
x0 = zeros(20000)
p = rand(20000)
sol = newton(g, x0, Constant(p))
```

The problem with this approach is that now instead of solving many small 2x2 systems, the linear algebra solver needs to solve a giant 20000 x 20000 matrix equation. This will generally be much slower than just solving many small systems. In fact, we tested this approach with CUDA's cuSPARSE solver, and unfortunately we found performance to scale poorly beyond 60000 x 60000 systems. 

For CUDA arrays, `BatchSolve.jl` plugs directly into CUDA's cuBLAS library which provides GPU-accelerated batched linear system solving.

Currently implemented batched solvers include:

### Root Finders
- Newton-Raphson `newton` (uses derivatives)

### Minimizers
- Differential evolution `de`
- Genetic algorithm `ga`
- 1D Brent's method `brent`
 
## Data Structures, SIMD, and which `batchdim`?

While the example shown above uses the broadcast operators with GPU arrays, you can of course use this package with any vectorized function. We generally recommend writing kernels explicitly with [`KernelAbstractions.jl`](https://juliagpu.github.io/KernelAbstractions.jl/stable/), and laying ouy memory in a SIMD-able, structure-of-arrays format. This would correspond to `batchdim=1` in Julia as a column-major language. The only thing to note is that for CUDA arrays with `batchdim=1`, the sparse Jacobian matrix must be `permutedims`-ed in order to be inputted into CUBLAS's batched linear system solver, and then the solution `permutedims`-ed back to the solution array. Depending on the cost of your function, it is possible that using `batchdim=2` (for array-of-structures) is faster, because this `permutedims` is then not necessary. Therefore we recommended testing both for your particular use case.

## Vectorized Finite Differences

This package also features special bindings for [`FiniteDiff.jl`](https://github.com/JuliaDiff/FiniteDiff.jl) with `AutoBatch` to accelerate finite differences calculations for vectorized functions: if a function is vectorized, then instead of evaluating the same function many times for each tangent, we can just evaluate it a single time for all tangents. This can offer massive speedups in the case where your function is extremely expensive to evaluate.

Here is a simple example that shows approximately a **250,000x speedup** of vectorized vs non-vectorized finite differences on the GPU:

## Should you NOT use this package?

This package is geared towards lightweight but high performance batched solvers, and requires the user to have some knowledge of the appropriate solver to choose for their problem in order to get reasonable results. This is quite different from the polyalgorithm approaches in other packages, e.g. [`BlackBoxOptim.jl`](https://github.com/SciML/BlackBoxOptim.jl), which is generally better for cases where you just want to "press go" and get a solution, even for hard problems. Furthermore, in cases where a solution doesn't converge or blows up, all `BatchSolve.jl` will tell you is that it failed (and give you the last Jacobian if derivatives are used). This package also lacks many features in comparison to e.g. [`NonlinearSolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/). It is very barebones in comparison, but will get the job done fast and correctly if your problem is well-paired with the selected solver.

## Acknowledgements

`BatchSolve.jl` makes very heavy use of the powerful [`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl) package and its sparse automatic-differentiation features via [`SparseMatrixColorings.jl`](https://github.com/JuliaDiff/SparseMatrixColorings.jl). It goes without saying that `BatchSolve.jl` would not be possible without the massive amount of careful work put into both of these packages. We are extremely thankful for this work.
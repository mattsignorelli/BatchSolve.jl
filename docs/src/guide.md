# How BatchSolve works

BatchSolve solves many independent small problems at once: root finding with Newton's method,
least squares (and root finding) with the Levenberg-Marquardt method, and one-dimensional
minimization with Brent's method. You write one function that evaluates the residuals of *all*
problems together (with broadcasts or a GPU kernel), and every step of the solver is done for all
problems together with array operations. The same code then runs on the CPU and on a GPU.

## Batches

Each problem has `n` unknowns and `m` residuals. The unknowns of all `B` problems are stored in
one array, and `batchdim` says along which dimension the problems lie:

- `batchdim=1`: one problem per row, `x` is `B × n` and the residual `B × m` (each column holds
  one unknown of all problems: the layout that vectorizes best on the CPU and on GPUs)
- `batchdim=2`: one problem per column, `x` is `n × B` and the residual `m × B`
- `batchdim=nothing` (the default): `x` is a single problem

For example, 10,000 equations `(x - 1)² = p²`, one for each `p`:

```julia
using BatchSolve
f(x, p) = (x - 1)^2 - p^2
sol = newton((x, p) -> f.(x, p), fill(3.0, 10000), Constant(collect(1.0:10000)); batchdim=1)
sol.u[1:3]       # [2.000000000000001, 3.0, 4.000000000039321]
sol.retcode[1:3] # [0x00, 0x00, 0x00]: all converged
sol.iters[1:3]   # [5, 0, 4]
```

Further arguments of the function are passed as `Constant` (not changed by the function) or
`Cache` (changed by it), as in
[DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/tutorials/advanced/#Contexts).

Because the problems are independent, the Jacobian of the whole batch is block diagonal: each
residual depends only on the unknowns of its own problem. BatchSolve stores it as a sparse matrix
with this pattern and computes it with the `AutoBatch` differentiation type, which gets the
derivatives with respect to the k-th unknown of all problems together, so a batch costs as many
derivative evaluations as a single problem (`n`). The linear systems of a step are solved for all
problems at once (with CUDA arrays by cuBLAS's batched solvers).

Every problem has its own convergence state: a problem that has converged (or failed) is not
changed any more, while the others continue. The result is a `NamedTuple` with the solution `u`,
the residual `f`, the Jacobian `jac`, and per problem the return code `retcode`
(`0x0` converged, `0x1` failed, `0x2` maximum number of iterations reached) and the number of
iterations `iters`.

## Levenberg-Marquardt

`levenberg_marquardt` minimizes the sum of squares of the residuals, ‖f(x)‖², so the number of
residuals `m` may differ from the number of unknowns `n` (it finds a root where there is one).
Each step solves the damped normal equations

```math
(J^T J + \lambda D)\, \Delta x = -J^T f, \qquad D = \mathrm{clamp}(\mathrm{diag}(J^T J),\ \texttt{diagmin},\ \texttt{diagmax})
```

with the Jacobian `J` at the current point. A small damping λ gives the Gauss-Newton step, a large
one a short step along the steepest descent. The step is tried: it is accepted if the sum of
squares decreases by at least `accept_ratio` times the decrease predicted by the linear model,
and then λ decreases; otherwise the point stays and λ increases (Nielsen's update). Each problem
has its own λ. A problem has converged when the residual is small (`abstol`), the step is small
(`reltol`), or the gradient of the sum of squares is small (`gradtol`, which ends least-squares
problems whose minimum is not zero).

Rosenbrock's function as two residuals, `10(x₂ - x₁²)` and `a - x₁`, for three values of `a`
(minimum at `x₁ = a`, `x₂ = a²`):

```julia
ros(x, a) = hcat(10 .* (x[:, 2] .- x[:, 1] .^ 2), a .- x[:, 1])
a = [1.0, 2.0, 0.5]
x0 = [-1.2 1.0; -1.2 1.0; -1.2 1.0]
sol = levenberg_marquardt(ros, x0, Constant(a); batchdim=1)
sol.u     # [1.0 1.0; 2.0 4.0; 0.5 0.25] (to 1e-8)
sol.iters # [32; 34; 12]
```

### Bounds

`lower` and `upper` bound the unknowns: a number for all of them, or an array of the shape of `x`
(so every problem can have its own bounds; `-Inf`/`Inf` for none). The starting point must be
within the bounds. Every trial point is put back into the bounds, and an unknown that is on a
bound while the sum of squares would decrease by leaving it is held there for that step. The
result is the minimum within the bounds. With `x₁ ≤ 0.8`:

```julia
sol = levenberg_marquardt(ros, x0, Constant(a); batchdim=1, upper=[0.8 Inf])
sol.u       # [0.8 0.64; 0.8 0.64; 0.5 0.25]
sol.retcode # all 0x00: the first two problems at the bound, the third inside
```

### Step limit

`maxstep` limits the change of each unknown in one step (a number, or an array of the shape of
`x`). A longer step is shortened as a whole, keeping its direction. This keeps the iteration in a
region where the linear model of the residuals is good, e.g. when a large step leads to points
where the residuals can't be computed:

```julia
sol = levenberg_marquardt(ros, x0, Constant(a); batchdim=1, maxstep=0.1)
sol.iters # [32; 63; 23]
```

### Residuals and Jacobian from one calculation

The function form `levenberg_marquardt!(val!, val_and_jac!, y, jac, x)` takes the residuals and
the Jacobian from your own functions: `val!(y, x)` fills the residuals `y`, `val_and_jac!(y, jac,
x)` the residuals and the Jacobian (for example when the derivatives come from a Taylor series
calculation, as in SciBmad's `match!`). By default a trial point is evaluated with `val!`, and
`val_and_jac!` is called again after the step is accepted. When the residuals cost about as much
as the residuals with the Jacobian, `trial_jac=true` evaluates every trial point with
`val_and_jac!` and keeps its Jacobian when the step is accepted, which saves that second
evaluation (and wastes a Jacobian for each rejected step):

```julia
val!(y, x) = (y .= [10 * (x[2] - x[1]^2), 1 - x[1]]; y)
function val_and_jac!(y, J, x)
  y .= [10 * (x[2] - x[1]^2), 1 - x[1]]
  J .= [-20x[1] 10; -1 0]
  return y
end
sol = levenberg_marquardt!(val!, val_and_jac!, zeros(2), zeros(2, 2), [-1.2, 1.0])
# 32 evaluations of val! and 23 of val_and_jac!
sol = levenberg_marquardt!(val!, val_and_jac!, zeros(2), zeros(2, 2), [-1.2, 1.0]; trial_jac=true)
# no evaluation of val! and 33 of val_and_jac!, the same solution
```

`lower`, `upper`, `maxstep` and `trial_jac` work with and without `batchdim`.

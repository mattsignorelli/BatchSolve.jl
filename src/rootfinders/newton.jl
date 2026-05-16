"""
    newton(f, x, contexts...; reltol, abstol, maxiter, autodiff, prep, batchdim, solver, verbose)

Finds roots of `f(x, contexts...) = 0` using Newton's method. 

# Arguments
- `f`: Function returning the residual vector; called as `f(x, unwrapped_contexts...)`
- `x`: Initial guess as `AbstractArray`
- `contexts`: Optional `DifferentiationInterface.Context` objects (`Constant`s or
  `Cache`s) forwarded to `f` after unwrapping.

# Keyword Arguments
- `reltol`: Relative convergence tolerance on the Newton step; default `sqrt(eps(eltype(x)))`.
- `abstol`: Absolute convergence tolerance on the residual norm; default `sqrt(eps(eltype(y)))`
  (inferred after the first evaluation of `f`).
- `maxiter`: Maximum number of Newton iterations; default `100`.
- `autodiff`: AD backend used to compute Jacobians. Defaults to `AutoForwardDiff()` on
  CPU and `AutoForwardFromPrimitive(AutoForwardDiff())` for GPU arrays. Wraps in `AutoBatch` 
  automatically when `batchdim` is set.
- `prep`: Pre-allocated `DifferentiationInterface` Jacobian preparation object. When
  `nothing` (default) it is created automatically on the first call.
- `batchdim`: Batch dimension index (`1`, `2`, or `nothing`). When set, independent
  Newton systems are solved in parallel along that dimension; default `nothing`.
- `solver`: Callable `(dx, jac, y) -> ()` that solves the linear system in-place.
  Defaults to `newton_solver(device, y, x, batchdim)`.
- `verbose`: Print iteration table (iteration, ‖y‖, ‖dx‖) when `true`; default `false`.

# Returns
A `NamedTuple` with fields:
- `u`: Solution array (same object as `x`, mutated in-place).
- `f`: Final residual vector (same object as `y`, mutated in-place).
- `jac`: Final Jacobian.
- `retcode`: `RETCODE_SUCCESS`, `RETCODE_FAILURE`, or `RETCODE_MAXITER`.
- `iters`: Number of iterations taken (scalar, or array when `batchdim` is set).
"""
function newton(
  f::Function, 
  x::AbstractArray, 
  contexts::Vararg{DI.Context}; 
  reltol=sqrt(eps(eltype(x))),
  abstol=nothing, 
  maxiter=100, 
  # On GPU need to use ForwardDiff from primitive (pushforward) for no scalar indexing
  autodiff=KA.get_backend(x) isa KA.GPU ? AutoForwardFromPrimitive(AutoForwardDiff()) : AutoForwardDiff(),
  prep=nothing, 
  batchdim::Union{Nothing,Integer}=nothing,
  solver=nothing,
  verbose=false,
)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    y = fc(x)
    f!(_y, _x, _contexts...) = (_y .= f(_x, _contexts...); _y)
    if isnothing(solver)
      solver = newton_solver(KA.get_backend(x), y, x, batchdim)
    end
    if isnothing(abstol)
      abstol = sqrt(eps(eltype(y)))
    end
    return newton!(f!, y, copy(x), contexts...; reltol, abstol, maxiter, autodiff, prep, batchdim, solver, verbose)
end

"""
    newton!(f!, y, x, contexts...; reltol, abstol, maxiter, autodiff, prep, batchdim, solver, verbose, dx)

In-place Newton root-finder for `f!(y, x, contexts...) = 0`. Prepares the AD Jacobian
backend, allocates the Jacobian matrix, then delegates to the three-argument
`newton!(val_and_jac!, y, jac, x, ...)` core.

# Arguments
- `f!`: In-place residual function; must satisfy `f!(y, x, contexts...)` and mutate `y`.
- `y`: Residual vector (mutated in-place).
- `x`: Initial guess (mutated in-place; contains the solution on return).
- `contexts`: Optional `DifferentiationInterface.Context` objects forwarded to `f!`.

# Keyword Arguments
- `reltol`: Relative convergence tolerance; default `sqrt(eps(eltype(x)))`.
- `abstol`: Absolute convergence tolerance; default `sqrt(eps(eltype(y)))`.
- `maxiter`: Maximum iterations; default `100`.
- `autodiff`: AD backend for Jacobian computation. GPU arrays default to
  `AutoForwardFromPrimitive(AutoForwardDiff())`; CPU arrays default to
  `AutoForwardDiff()`. Wraps in `AutoBatch` automatically when `batchdim` is set.
- `prep`: Pre-allocated Jacobian preparation object; created automatically when `nothing`.
- `batchdim`: Batch dimension (`1`, `2`, or `nothing`); default `nothing`.
- `solver`: Linear solver callable; defaults to `newton_solver(device, y, x, batchdim)`.
- `verbose`: Print iteration table (iteration, ‖y‖, ‖dx‖) when `true`; default `false`.
- `dx`: Pre-allocated Newton step buffer; default `zero.(x)`.

# Returns
A `NamedTuple` with fields:
- `u`: Solution array (same object as `x`, mutated in-place).
- `f`: Final residual vector (same object as `y`, mutated in-place).
- `jac`: Final Jacobian.
- `retcode`: `RETCODE_SUCCESS`, `RETCODE_FAILURE`, or `RETCODE_MAXITER`.
- `iters`: Number of iterations taken (scalar, or array when `batchdim` is set).
"""
function newton!(
  f!::Function,  # DO NOT SPECIALIZE ON FUNCTION, no need
  y::Y, 
  x::X,
  contexts::Vararg{DI.Context};
  reltol=sqrt(eps(eltype(x))), 
  abstol=sqrt(eps(eltype(y))), 
  maxiter=100, 
  # On GPU need to use ForwardDiff from primitive (pushforward) for no scalar indexing
  autodiff=KA.get_backend(x) isa KA.GPU ? AutoForwardFromPrimitive(AutoForwardDiff()) : AutoForwardDiff(),
  prep=nothing, 
  batchdim::Union{Nothing,Integer}=nothing,
  solver::T=newton_solver(KA.get_backend(x), y, x, batchdim), # We do specialize on the solver tho
  verbose=false,
  dx=zero.(x), # Temporary
) where {Y,X,T}
  if !isnothing(batchdim) && !(autodiff isa AutoBatch)
    autodiff = AutoBatch(autodiff; batchdim=batchdim)
  end

  if isnothing(prep)
    prep = DI.prepare_jacobian(f!, y, autodiff, x, contexts...)
  end
  if autodiff isa AutoBatch || autodiff isa AutoSparse
    jac = similar(sparsity_pattern(prep), eltype(y))
  else
    if Y <: StaticArray && X <: StaticArray
      jac = similar(y, Size(length(Y), length(X)))
    else
      jac = similar(y, length(y), length(x))
    end
  end
  let _f! = f!, _prep = prep, _backend = autodiff
    val_and_jac!(_y, _jac, _x, _contexts...) = DI.value_and_jacobian!(_f!, _y, _jac, _prep, _backend, _x, _contexts...)
    return newton!(val_and_jac!, y, jac, x, contexts...; reltol, abstol, maxiter, batchdim, solver, dx, verbose)
  end
end

"""
    newton!(val_and_jac!, y, jac, x, contexts...; reltol, abstol, maxiter, batchdim, iters, retcode, solver, verbose, dx)

Core Newton iteration loop. Accepts a combined value-and-Jacobian callable and runs the
Newton update `x = x − inv(J)*f(x)` until convergence or `maxiter` is reached.
Supports both scalar (unbatched) and batched operation.

# Arguments
- `val_and_jac!`: Callable `(y, jac, x, contexts...) -> nothing` that simultaneously
  fills the residual `y` and the Jacobian `jac` in-place.
- `y`: Residual vector / matrix (mutated).
- `jac`: Jacobian array (mutated); may be dense or sparse depending on the AD backend.
- `x`: Current iterate (mutated; holds the solution on return).
- `contexts`: Optional `DifferentiationInterface.Context` objects.

# Keyword Arguments
- `reltol`: Relative convergence tolerance on ‖dx‖ / ‖x‖; default `√eps(eltype(x))`.
- `abstol`: Absolute convergence tolerance on ‖y‖; default `√eps(eltype(y))`.
- `maxiter`: Maximum number of iterations; default `100`.
- `batchdim`: Batch dimension (`1`, `2`, or `nothing`); default `nothing`.
- `iters`: Integer array (shape broadcastable over the batch dimension) in which the
  iteration count at convergence is recorded for each problem in the batch. Allocated
  automatically when `batchdim` is set; ignored (with a warning) when `batchdim=nothing`.
- `retcode`: `UInt8` array (same shape as `iters`) recording the per-problem return code.
  Allocated automatically when `batchdim` is set.
- `solver`: Linear solver `(dx, jac, y) -> nothing`; defaults to
  `newton_solver(device, y, x, batchdim)`.
- `verbose`: Print per-iteration `(iter, ‖y‖, ‖dx‖)` table; default `false`.
- `dx`: Pre-allocated Newton step buffer; default `zero.(x)`.

## Convergence criteria (non-batched)
An iteration is considered converged when either:
1. `norm(y) < abstol` (residual is small enough), or
2. `norm(dx) < reltol * norm(x)` (step is small relative to the current iterate).

In the batched case the same criteria are applied element-wise along `batchdim`, and
only the unconverged sub-problems continue to be updated.

# Returns
A `NamedTuple` with fields:
- `u`: Solution (`x`, mutated in-place).
- `f`: Final residual (`y`, mutated in-place).
- `jac`: Final Jacobian.
- `retcode`: `RETCODE_SUCCESS` / `RETCODE_FAILURE` / `RETCODE_MAXITER` — scalar for
  non-batched runs, array for batched runs.
- `iters`: Iteration count at convergence — scalar or array.
"""
function newton!(
  val_and_jac!::Function,
  y,
  jac,
  x,
  contexts::Vararg{DI.Context};
  reltol=sqrt(eps(eltype(x))),
  abstol=sqrt(eps(eltype(y))), 
  maxiter=100, 
  batchdim::Union{Nothing,Integer}=nothing, 
  iters=isnothing(batchdim) ? nothing : similar(x, Int, ntuple(i-> i == batchdim ? size(x, batchdim) : 1, Val{2}())), # If batch, then array that should be modified in-place with the iteration when convergence reached
  retcode=isnothing(batchdim) ? nothing : similar(x, UInt8, ntuple(i-> i == batchdim ? size(x, batchdim) : 1, Val{2}())),
  solver::T=newton_solver(KA.get_backend(x), y, x, batchdim), 
  verbose=false,
  dx=zero.(x),
) where {T}
  # Setup:
  out = (; u=x, f=y, jac=jac)
  if isnothing(batchdim)
    if !isnothing(iters)
      @warn "You provided `iters`, but this is only used for batched-Newton. Non-batched Newton 
             always returns a scalar `iters`."
    end
    if !isnothing(retcode)
      @warn "You provided `retcode`, but this is only used for batched-Newton. Non-batched Newton 
             always returns a scalar `retcode`."
    end
    out = merge(out, (; retcode=RETCODE_MAXITER, iters=0))
    # Newton:
    dx .= 0
    if verbose
      println("Iteration   norm(y)          norm(dx)")
      println("-" ^ 45)
    end
    val_and_jac!(y, jac, x, contexts...)
    for iter in 1:maxiter
      solver(dx, jac, y)
      if verbose
        @printf("%-11d %-16.6e %-16.6e\n", iter, norm(y), norm(dx))
      end
      if any(isnan.(dx))
        @reset out.retcode = RETCODE_FAILURE
        @reset out.iters = iter-1
        return out
      elseif norm(y) < abstol
        @reset out.retcode = RETCODE_SUCCESS
        @reset out.iters = iter-1
        return out
      end
      x .= x .+ dx
      val_and_jac!(y, jac, x, contexts...)
      if norm(dx) < reltol*norm(x)
        @reset out.retcode = RETCODE_SUCCESS
        @reset out.iters = iter
        return out
      end
    end
    @reset out.iters=maxiter
    return out
  else
    otherdim = mod(batchdim, 2) + 1
    abstol2 = abstol^2
    reltol2 = reltol^2
    fill!(retcode, RETCODE_MAXITER)
    fill!(iters, -1)
    out = merge(out, (; retcode=retcode, iters=iters))
    # Newton:
    dx .= 0
    if verbose
      println("Batched-newton: printed norms are for entire batch")
      println("Iteration   norm(y)          norm(dx)")
      println("-" ^ 45)
    end
    val_and_jac!(y, jac, x, contexts...)
    for iter in 1:maxiter
      solver(dx, jac, y)
      if verbose
        @printf("%-11d %-16.6e %-16.6e\n", iter, norm(y), norm(dx))
      end
      out.retcode .= ifelse.(any(isnan, dx, dims=otherdim), RETCODE_FAILURE, out.retcode)
      out.iters .= ifelse.(
        (sum(abs2, y, dims=otherdim) .< abstol2 .|| out.retcode .== RETCODE_FAILURE) .&& out.iters .== -1,
        iter-1,
        out.iters
      )
      x .= x .+ (out.iters .== -1) .* dx
      out.iters .= ifelse.(
        sum(abs2, dx, dims=otherdim) .< reltol2.*sum(abs2, x, dims=otherdim) .&& out.iters .== -1,
        iter,
        out.iters
      )
      out.retcode .= ifelse.(out.iters .!= -1 .&& out.retcode .== RETCODE_MAXITER, RETCODE_SUCCESS, out.retcode)
      if all(out.retcode .!= RETCODE_MAXITER)
        break
      end
      val_and_jac!(y, jac, x, contexts...)
    end
    return out
  end
end 

"""
    newton_solver(device, y, x, batchdim) -> Function

Construct and return a linear-system solver callable compatible with the given device,
array shapes, and batch configuration. The returned function has the signature

    solver(dx, jac, y) -> nothing

and solves `jac * dx = -y` in-place, writing the Newton step into `dx`.

# Arguments
- `device`: KernelAbstractions backend (e.g. `CPU()`, `CUDABackend()`). 
- `y`: Prototype residual array (used for size introspection; not mutated).
- `x`: Prototype solution array (used for size introspection; not mutated).
- `batchdim`: Batch dimension (`nothing`, `1`, or `2`).

# Returned solver behaviour
| `batchdim` | Jacobian type | Behaviour |
|------------|---------------|-----------|
| `nothing`  | dense matrix  | Single `jac \\ y` solve; writes `NaN` when `jac` is singular. |
| `2`        | `SparseMatrixCSC` (block-diagonal, blocks contiguous in `nzval`) | Iterates over batch index `i`, extracts each `(n_rows × n_cols)` block from `nzval`, solves independently. |
| `1`        | `SparseMatrixCSC` (interleaved columns) | Iterates over batch index `i`, gathers every `batchsize`-th column, solves independently. |

Singular sub-Jacobians (detected via `ArrayInterface.issingular`) result in `NaN` being
written to the corresponding slice of `dx` so that upstream code can detect and handle
failures gracefully.

# Errors
Throws an `ArgumentError`-style error if `batchdim ∉ {nothing, 1, 2}`.
"""
function newton_solver(device, _y, _x, batchdim)
  _lx = length(_x)
  _ly = length(_y)
  if isnothing(batchdim)
    let lx=_lx, ly=_ly
      return (dx, jac, y)->begin
        if ArrayInterface.issingular(jac) || any(isnan, jac) || any(isinf, jac)
          dx .= NaN32
        else
          reshape(dx, lx) .= -jac \ reshape(y, ly)
        end
      end
    end
  elseif batchdim == 2 # Do each serially
    _batchsize = size(_x, 2)
    _n_rows = size(_y, 1)
    _n_cols = size(_x, 1)
    let n_rows=_n_rows, n_cols=_n_cols, batchsize=_batchsize, jacsize=_n_rows*_n_cols
      return (dx, jac::SparseMatrixCSC, y)->begin
        for i in 1:batchsize
          jac_offset = (i-1)*jacsize 
          curjac = reshape(view(jac.nzval, (jac_offset+1):(jac_offset+jacsize)), (n_rows, n_cols))
          dx_offset = (i-1)*n_cols
          y_offset = (i-1)*n_rows
          if ArrayInterface.issingular(curjac) || any(isnan, curjac) || any(isinf, curjac)
            view(dx, (dx_offset+1):(dx_offset+n_cols)) .= NaN32
          else
            view(dx, (dx_offset+1):(dx_offset+n_cols)) .= -curjac \ view(y, (y_offset+1):(y_offset+n_rows))
          end
        end
      end
    end
  elseif batchdim == 1
    _batchsize = size(_x, 1)
    _n_rows = size(_y, 2)
    let n_rows=_n_rows, batchsize=_batchsize, xlen=length(_x), ylen=length(_y)
      return (dx, jac::SparseMatrixCSC, y)->begin
        for i in 1:batchsize
          curjac = view(reshape(jac.nzval, n_rows, :), :, i:batchsize:xlen)
          if ArrayInterface.issingular(curjac) || any(isnan, curjac) || any(isinf, curjac)
            view(dx, i:batchsize:xlen) .= NaN32
          else
            view(dx, i:batchsize:xlen) .= -curjac \ view(y, i:batchsize:ylen)
          end
        end
      end
    end
  else
    error("Invalid batchdim (must be either 1, 2, or nothing)")
  end
end

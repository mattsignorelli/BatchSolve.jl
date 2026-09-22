"""
    levenberg_marquardt(f, x, contexts...; reltol, abstol, gradtol, maxiter, autodiff, prep,
                        batchdim, solver, verbose, damping, damping_min, damping_max,
                        accept_ratio, diagmin, diagmax)

Minimizes `‖f(x, contexts...)‖²` (finds roots of `f(x, contexts...) = 0` when a root exists)
using the Levenberg–Marquardt method. Unlike [`newton`](@ref), the residual may have a
different length than `x` (over- *or* under-determined systems), and the iteration is
globalized by an adaptive damping parameter, so it is far more tolerant of poor initial guesses
and singular Jacobians.

When `batchdim` is set, every batch element is an independent problem with its own damping
parameter, acceptance decision and convergence status. Everything is expressed as array
operations (no per-element branching), so it runs on any device supported by the linear
solver.

# Arguments
- `f`: Function returning the residual array; called as `f(x, unwrapped_contexts...)`.
- `x`: Initial guess as `AbstractArray`.
- `contexts`: Optional `DifferentiationInterface.Context` objects (`Constant`s or
  `Cache`s) forwarded to `f` after unwrapping.

# Keyword Arguments
- `reltol`: Step tolerance. Converged when `‖dx‖ ≤ reltol*(‖x‖ + reltol)`; default
  `sqrt(eps(eltype(x)))`.
- `abstol`: Residual tolerance. Converged when `‖f(x)‖ < abstol`; default
  `sqrt(eps(eltype(y)))` (inferred after the first evaluation of `f`).
- `gradtol`: Stationarity tolerance. Converged when, for every variable `k`,
  `|(Jᵀf)_k| ≤ gradtol * ‖J[:,k]‖ * ‖f‖` (scale-invariant, as in MINPACK's `gtol`); default
  `sqrt(eps(eltype(y)))`. This is what terminates least-squares problems whose minimum has a
  non-zero residual.
- `maxiter`: Maximum number of iterations; default `100`.
- `autodiff`: AD backend used to compute Jacobians. Defaults as in `newton`. Wraps in
  `AutoBatch` automatically when `batchdim` is set.
- `prep`: Pre-allocated `DifferentiationInterface` Jacobian preparation object.
- `batchdim`: Batch dimension index (`1`, `2`, or `nothing`); default `nothing`.
- `solver`: Callable `(x, A, b) -> x` that solves the **square** (batched) system `Ax=b`
  in-place, where `x` and `b` have the same shape as the unknowns. Note that this differs from
  `newton`: here `A` is the damped normal-equations matrix `JᵀJ + λ diag(D)` (`n × n` per
  batch element, stored with the sparsity pattern `make_pattern(x, x, batchdim)`), not the
  Jacobian. Defaults to `make_linear_solver(device, x, x, batchdim)`.
- `verbose`: Print iteration table (iteration, ‖y‖, ‖dx‖, max λ) when `true`; default `false`.
- `damping`: Initial value of the (relative) damping parameter λ; default `1e-3`.
- `damping_min`, `damping_max`: Bounds on λ; defaults `1e-10`, `1e10`.
- `accept_ratio`: A step is accepted when (actual reduction)/(predicted reduction) exceeds
  this value; default `1e-4`.
- `diagmin`, `diagmax`: `diag(JᵀJ)` is clamped to `[diagmin, diagmax]` to form the Marquardt
  damping scale `D`; defaults `1e-6`, `1e16`.

# Returns
A `NamedTuple` with fields:
- `u`: Solution array (same object as `x`, mutated in-place).
- `f`: Final residual (same object as `y`, mutated in-place).
- `jac`: Final Jacobian.
- `retcode`: `RETCODE_SUCCESS`, `RETCODE_FAILURE` (non-finite residual/Jacobian at the current
  point, or steps still being rejected with the damping already at `damping_max`), or
  `RETCODE_MAXITER`, per batch element (scalar when `batchdim=nothing`). `RETCODE_SUCCESS`
  means converged to a root *or* a stationary point of `‖f‖²`; inspect `f` if you need to
  distinguish them.
- `iters`: Number of iterations taken (scalar, or array when `batchdim` is set; `-1` for
  batch elements that did not converge).
"""
function levenberg_marquardt(
  f::Function,
  x::AbstractArray,
  contexts::Vararg{DI.Context};
  reltol=sqrt(eps(eltype(x))),
  abstol=nothing,
  gradtol=nothing,
  maxiter=100,
  # On GPU need to use ForwardDiff from primitive (pushforward) for no scalar indexing
  autodiff=KA.get_backend(x) isa KA.GPU ? AutoForwardFromPrimitive(AutoForwardDiff()) : AutoForwardDiff(),
  prep=nothing,
  batchdim::Union{Nothing,Integer}=nothing,
  solver=nothing,
  verbose=false,
  damping=1e-3,
  damping_min=1e-10,
  damping_max=1e10,
  accept_ratio=1e-4,
  diagmin=1e-6,
  diagmax=1e16,
)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    y = fc(x)
    f!(_y, _x, _contexts...) = (_y .= f(_x, _contexts...); _y)
    if isnothing(solver)
      # The linear systems are the (square) normal equations, n × n per batch element
      solver = make_linear_solver(KA.get_backend(x), x, x, batchdim)
    end
    if isnothing(abstol)
      abstol = sqrt(eps(eltype(y)))
    end
    if isnothing(gradtol)
      gradtol = sqrt(eps(eltype(y)))
    end
    return levenberg_marquardt!(f!, y, copy(x), contexts...;
      reltol, abstol, gradtol, maxiter, autodiff, prep, batchdim, solver, verbose,
      damping, damping_min, damping_max, accept_ratio, diagmin, diagmax)
end

"""
    levenberg_marquardt!(f!, y, x, contexts...; reltol, abstol, gradtol, maxiter, autodiff, prep,
                         batchdim, solver, verbose, dx, damping, damping_min, damping_max,
                         accept_ratio, diagmin, diagmax)

In-place Levenberg–Marquardt solver for `‖f!(y, x, contexts...)‖²`. Prepares the AD Jacobian
backend, allocates the Jacobian, then delegates to the core method
`levenberg_marquardt!(val!, val_and_jac!, y, jac, x, ...)`.

# Arguments
- `f!`: In-place residual function; must satisfy `f!(y, x, contexts...)` and mutate `y`.
- `y`: Residual array (mutated in-place). May have a different length than `x`.
- `x`: Initial guess (mutated in-place; contains the solution on return).
- `contexts`: Optional `DifferentiationInterface.Context` objects forwarded to `f!`.

# Keyword Arguments
See [`levenberg_marquardt`](@ref). Additionally:
- `dx`: Pre-allocated step buffer; default `zero.(x)`.

# Returns
Same `NamedTuple` as [`levenberg_marquardt`](@ref).
"""
function levenberg_marquardt!(
  f!::Function,  # DO NOT SPECIALIZE ON FUNCTION, no need
  y::Y,
  x::X,
  contexts::Vararg{DI.Context};
  reltol=sqrt(eps(eltype(x))),
  abstol=sqrt(eps(eltype(y))),
  gradtol=sqrt(eps(eltype(y))),
  maxiter=100,
  # On GPU need to use ForwardDiff from primitive (pushforward) for no scalar indexing
  autodiff=KA.get_backend(x) isa KA.GPU ? AutoForwardFromPrimitive(AutoForwardDiff()) : AutoForwardDiff(),
  prep=nothing,
  batchdim::Union{Nothing,Integer}=nothing,
  solver::T=make_linear_solver(KA.get_backend(x), x, x, batchdim), # We do specialize on the solver tho
  verbose=false,
  dx=zero.(x), # Temporary
  damping=1e-3,
  damping_min=1e-10,
  damping_max=1e10,
  accept_ratio=1e-4,
  diagmin=1e-6,
  diagmax=1e16,
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
    # Residual only (used to evaluate trial points; contexts arrive wrapped, as for val_and_jac!)
    val!(_y, _x, _contexts...) = _f!(_y, _x, map(DI.unwrap, _contexts)...)
    val_and_jac!(_y, _jac, _x, _contexts...) = DI.value_and_jacobian!(_f!, _y, _jac, _prep, _backend, _x, _contexts...)
    return levenberg_marquardt!(val!, val_and_jac!, y, jac, x, contexts...;
      reltol, abstol, gradtol, maxiter, batchdim, solver, dx, verbose,
      damping, damping_min, damping_max, accept_ratio, diagmin, diagmax)
  end
end

"""
    levenberg_marquardt!(val!, val_and_jac!, y, jac, x, contexts...; reltol, abstol, gradtol,
                         maxiter, batchdim, iters, retcode, solver, verbose, dx, damping,
                         damping_min, damping_max, accept_ratio, diagmin, diagmax)

Core Levenberg–Marquardt loop. Accepts a residual-only callable and a combined
value-and-Jacobian callable, and iterates

    (JᵀJ + λ D) dx = -Jᵀy,    D = clamp(diag(JᵀJ), diagmin, diagmax)

with a Nielsen-style adaptive λ, until convergence or `maxiter`.

# Arguments
- `val!`: Callable `(y, x, contexts...) -> nothing` filling the residual `y` in-place. Used for
  trial points, so that the (expensive) Jacobian is only evaluated once per accepted step.
- `val_and_jac!`: Callable `(y, jac, x, contexts...) -> nothing` filling `y` and `jac`.
- `y`: Residual array (mutated).
- `jac`: Jacobian (mutated). Dense when `batchdim=nothing`; when `batchdim` is set, a sparse
  block-diagonal matrix with the layout produced by `AutoBatch`/`make_pattern`.
- `x`: Current iterate (mutated; holds the solution on return).
- `contexts`: Optional `DifferentiationInterface.Context` objects.

# Keyword Arguments
See [`levenberg_marquardt`](@ref). Additionally:
- `iters`, `retcode`: Optional pre-allocated arrays of shape `(B, 1)` (`batchdim=1`) or
  `(1, B)` (`batchdim=2`) receiving per-problem results (`Int` and `UInt8` respectively).
- `dx`: Pre-allocated step buffer; default `zero.(x)`.

## Algorithm (per batch element, all in masked array form)
1. Compute `g = Jᵀy`, `cost = ‖y‖²`. Elements whose residual (`abstol`) or scaled gradient
   (`gradtol`) is small are marked converged; elements with non-finite `cost`/`g` are marked
   failed. Marked elements are frozen: they never update `x`, `λ`, or `iters` again.
2. Solve the damped normal equations for `dx`.
3. Evaluate the residual at the trial point `x + dx` and compute the gain ratio
   `ρ = (‖y‖² - ‖y_trial‖²) / (dxᵀ(λ D dx - g))`.
4. Accept the step where `ρ > accept_ratio` and the trial cost is finite; `x` is updated with
   `ifelse`, so rejected/non-finite trial points never contaminate `x`.
5. If any element accepted a step, re-evaluate residual and Jacobian (at unchanged `x` for the
   others, which is harmless). Accepted steps whose new residual or Jacobian is non-finite are
   rolled back to the previous `x` and counted as rejected (one extra evaluation, only when
   this happens). A non-finite residual/Jacobian at the *initial* point is a failure.
6. Update `λ ← λ max(1/3, 1 - (2ρ-1)³)`, `ν ← 2` on acceptance, else `λ ← λν`, `ν ← 2ν`.
7. Elements whose (finite, evaluated) step is small relative to `x` (`reltol`) are marked
   converged; elements rejected while `λ == damping_max` are marked failed.

For `batchdim=nothing` the same code runs with a single "lane", and `JᵀJ` is formed with `mul!`.
Sparse Jacobians are only supported together with `batchdim`.

# Returns
Same `NamedTuple` as [`levenberg_marquardt`](@ref).
"""
function levenberg_marquardt!(
  val!::Function,
  val_and_jac!::Function,
  y,
  jac,
  x,
  contexts::Vararg{DI.Context};
  reltol=sqrt(eps(eltype(x))),
  abstol=sqrt(eps(eltype(y))),
  gradtol=sqrt(eps(eltype(y))),
  maxiter=100,
  batchdim::Union{Nothing,Integer}=nothing,
  iters=nothing,
  retcode=nothing,
  solver::T=make_linear_solver(KA.get_backend(x), x, x, batchdim),
  verbose=false,
  dx=zero.(x),
  damping=1e-3,
  damping_min=1e-10,
  damping_max=1e10,
  accept_ratio=1e-4,
  diagmin=1e-6,
  diagmax=1e16,
) where {T}
  # ── Setup ────────────────────────────────────────────────────────────────────────────────
  if !isnothing(batchdim) && !(batchdim in (1, 2))
    error("Invalid batchdim (must be either 1, 2, or nothing)")
  end
  batched = !isnothing(batchdim)
  if !batched && jac isa SparseArrays.AbstractSparseMatrix
    error("levenberg_marquardt: a sparse Jacobian is only supported together with `batchdim`.")
  end

  # All constants are converted to the element type ahead of time (keeps Float32/Metal kernels
  # free of Float64 literals).
  FT = eltype(x)
  zeroT, oneT, twoT = zero(FT), one(FT), FT(2)
  thirdT = oneT / FT(3)
  νmax = FT(1e8)
  reltol = FT(reltol)
  abstol2 = FT(abstol)^2
  gradtol = FT(gradtol)
  λmin, λmax = FT(damping_min), FT(damping_max)
  ρmin = FT(accept_ratio)
  dmin, dmax = FT(diagmin), FT(diagmax)

  # Layout bookkeeping. Unbatched problems are treated as a batch of one with batchdim = 2.
  #   batchdim = 2: x is (n, B), y is (m, B), J.nzval is (m, n, B) and JᵀJ.nzval is (n, n, B)
  #   batchdim = 1: x is (B, n), y is (B, m), J.nzval is (m, B, n) and JᵀJ.nzval is (n, B, n)
  bd = batched ? Int(batchdim) : 2
  od = mod(bd, 2) + 1                      # the non-batch dimension
  B = batched ? size(x, bd) : 1
  n = batched ? size(x, od) : length(x)    # number of unknowns
  m = batched ? size(y, od) : length(y)    # number of residuals
  lane_shape = bd == 1 ? (B, 1) : (1, B)   # shape of per-problem scalars (reductions over `od`)
  lane3 = bd == 1 ? (1, B, 1) : (1, 1, B)
  J_shape = bd == 1 ? (m, B, n) : (m, n, B)
  H_shape = bd == 1 ? (n, B, n) : (n, n, B)
  I_shape = bd == 1 ? (n, 1, n) : (n, n, 1)
  v_shape = bd == 1 ? (1, B, n) : (1, n, B) # a per-variable quantity along J's column dimension
  y_shape = bd == 1 ? (m, B, 1) : (m, 1, B) # residual aligned with J's row dimension
  cdim = bd == 1 ? 3 : 2                    # J's column (variable) dimension
  jdims = bd == 1 ? (1, 3) : (1, 2)         # all of J's dimensions except the batch one

  as2(a) = batched ? a : reshape(a, :, 1)   # 2D "natural" view; unbatched → (n, 1)

  # Normal-equations matrix. Batched: same block-diagonal sparsity layout that the linear solvers
  # expect for square systems. Unbatched: dense.
  A = batched ? similar(make_pattern(x, x, bd), eltype(y)) : similar(y, n, n)
  Jv = reshape(batched ? nonzeros(jac) : jac, J_shape)   # views: alias jac / A memory
  Hv = reshape(batched ? nonzeros(A) : A, H_shape)

  S = similar(Jv)                # scratch for elementwise products
  g = similar(x)                 # gradient Jᵀy
  Dsq = similar(x)               # diag(JᵀJ)
  D = similar(x)                 # Marquardt damping scale
  rhs = similar(x)
  xt = similar(x)                # trial point
  xs = similar(x)                # saved iterate, to roll back a step whose new point is unusable
  yt = similar(y)                # residual at trial point
  ycan = bd == 1 ? similar(y, m, B) : nothing
  eye = similar(x, FT, n, n)
  copyto!(eye, Matrix{FT}(I, n, n))

  x2, y2, dx2, g2, D2, Dsq2, rhs2, xt2, yt2, xs2 = map(as2, (x, y, dx, g, D, Dsq, rhs, xt, yt, xs))
  gv = reshape(g, v_shape)
  Dsqv = reshape(Dsq, v_shape)
  Dv = reshape(D, v_shape)
  Iv = reshape(eye, I_shape)

  # Per-problem state (all shaped like a reduction over the non-batch dimension)
  iters = isnothing(iters) ? similar(x, Int, lane_shape) : iters
  retcode = isnothing(retcode) ? similar(x, UInt8, lane_shape) : retcode
  fill!(iters, -1)
  fill!(retcode, RETCODE_MAXITER)
  λ = similar(x, FT, lane_shape)
  fill!(λ, FT(damping))
  ν = similar(x, FT, lane_shape)
  fill!(ν, twoT)
  λv = reshape(λ, lane3)
  rollback = similar(x, Bool, lane_shape)

  if verbose
    batched && println("Batched-LM: printed norms are for entire batch")
    println("Iteration   norm(y)          norm(dx)         max(lambda)")
    println("-" ^ 63)
  end

  # ── Levenberg–Marquardt ──────────────────────────────────────────────────────────────────
  dx .= 0
  val_and_jac!(y, jac, x, contexts...)
  # maxiter+1 passes: the last one only performs the convergence checks
  for iter in 1:(maxiter + 1)
    # ── 1. Gradient, damping scale and convergence checks at the current point ─────────────
    cost = sum(abs2, y2; dims=od)                    # ‖y‖² per problem
    yb = if bd == 1
      permutedims!(ycan, y, (2, 1))                  # (B, m) → (m, B)
      reshape(ycan, y_shape)
    else
      reshape(y, y_shape)
    end
    @. S = Jv * yb
    gv .= sum(S; dims=1)                             # g = Jᵀy
    Dsqv .= sum(abs2, Jv; dims=1)                    # diag(JᵀJ)

    gn2 = sum(abs2, g2; dims=od)
    finite = @. isfinite(cost) & isfinite(gn2)
    grad_ok = all(abs.(g2) .<= gradtol .* sqrt.(Dsq2) .* sqrt.(cost); dims=od)
    active = iters .== -1
    conv = @. active & finite & ((cost < abstol2) | grad_ok)
    fail = @. active & !finite
    @. retcode = ifelse(fail, RETCODE_FAILURE, ifelse(conv, RETCODE_SUCCESS, retcode))
    @. iters = ifelse(conv | fail, iter - 1, iters)
    active = iters .== -1
    if !any(active) || iter > maxiter
      break
    end

    # ── 2. Damped normal equations: (JᵀJ + λ D) dx = -g ────────────────────────────────────
    if batched
      # H[c, :] = Σ_r J[r, c] * J[r, :], one variable at a time (memory-light, one broadcast
      # and one reduction per variable, independent of the number of residuals)
      for c in 1:n
        Jc = selectdim(Jv, cdim, c:c)
        @. S = Jv * Jc
        selectdim(Hv, 1, c:c) .= sum(S; dims=1)
      end
    else
      mul!(A, transpose(jac), jac)
    end
    @. D = clamp(Dsq, dmin, dmax)
    # Add λ D to the diagonal, per problem. Finished/failed problems get an identity block and a
    # zero right-hand side instead, so the linear solver never sees their (possibly NaN) data.
    act3 = reshape(active, lane3)
    @. Hv = ifelse(act3, Hv + λv * Iv * Dv, Iv)
    @. rhs = ifelse(active, -g, zeroT)
    solver(dx, A, rhs)

    # ── 3. Trial point and gain ratio ──────────────────────────────────────────────────────
    # Only step where the solve produced a finite dx; elsewhere evaluate at the current x
    # (which yields ρ = 0 → rejection and a larger λ).
    use = active .& all(isfinite, dx2; dims=od)
    @. xt2 = ifelse(use, x2 + dx2, x2)
    val!(yt, xt, contexts...)
    costt = sum(abs2, yt2; dims=od)
    @. rhs2 = dx2 * (λ * D2 * dx2 + rhs2)            # dxᵀ(λ D dx - g), elementwise (rhs = -g)
    pred = sum(rhs2; dims=od)                        # 2 × predicted reduction
    ρ = @. ifelse(pred > zeroT, (cost - costt) / pred, -oneT)
    accept = @. use & isfinite(costt) & (ρ > ρmin)

    # ── 4. Accept, then verify the new point ───────────────────────────────────────────────
    copyto!(xs, x)                                   # saved so unusable steps can be undone
    fill!(rollback, false)
    @. x2 = ifelse(accept, xt2, x2)
    if any(accept)
      val_and_jac!(y, jac, x, contexts...)
      # A step is only kept if the residual AND Jacobian at the new point are finite. Otherwise
      # it is rolled back and counted as a rejection (so damping grows below).
      jok = reshape(all(isfinite, Jv; dims=jdims), lane_shape)
      yok = all(isfinite, y2; dims=od)
      @. rollback = accept & !(jok & yok)
      if any(rollback)
        @. x2 = ifelse(rollback, xs2, x2)
        @. accept = accept & !rollback
        val_and_jac!(y, jac, x, contexts...)         # restore y, jac at the rolled-back x
      end
    end

    # ── 5. Damping update (rejected = not accepted, incl. rolled-back steps) ───────────────
    trial_ok = @. use & isfinite(costt)              # the step and its residual were finite
    atmax = @. active & !accept & (λ >= λmax)        # rejected although damping was already maximal
    @. λ = ifelse(active,
      clamp(ifelse(accept, λ * max(thirdT, oneT - (2ρ - oneT)^3), λ * ν), λmin, λmax),
      λ)
    @. ν = ifelse(active, ifelse(accept, twoT, min(twoT * ν, νmax)), ν)

    # ── 6. Step-size convergence / stalling ────────────────────────────────────────────────
    # A tiny step only counts as convergence if it was actually evaluated: a step that is tiny
    # merely because repeated non-finite trials inflated λ is not convergence.
    dxn = sum(abs2, dx2; dims=od)
    xn = sum(abs2, x2; dims=od)
    small = @. active & trial_ok & !rollback & (dxn <= (reltol * (sqrt(xn) + reltol))^2)
    stalled = @. atmax & !small
    @. retcode = ifelse(small, RETCODE_SUCCESS, ifelse(stalled, RETCODE_FAILURE, retcode))
    @. iters = ifelse(small | stalled, iter, iters)

    if verbose
      @printf("%-11d %-16.6e %-16.6e %-16.6e\n", iter, sqrt(sum(cost)), norm(dx), maximum(λ))
    end
  end

  if batched
    return (; u=x, f=y, jac=jac, retcode=retcode, iters=iters)
  else
    it = only(Array(iters))
    return (; u=x, f=y, jac=jac, retcode=only(Array(retcode)), iters=(it == -1 ? maxiter : it))
  end
end
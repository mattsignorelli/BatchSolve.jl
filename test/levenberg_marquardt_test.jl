using Test
using Random
using LinearAlgebra
using SparseArrays
using BatchSolve
using BatchSolve: RETCODE_SUCCESS, RETCODE_FAILURE, RETCODE_MAXITER, make_pattern

# ══════════════════════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════════════════════

# Layout conventions (see `newton`): batchdim = 1 → arrays are (B, n) ("SoA"), batchdim = 2 → (n, B).

"Reshape a length-B vector of per-problem scalars so it broadcasts against a (B, 1) / (1, B) column."
lane(v, bd) = bd == 1 ? reshape(v, :, 1) : reshape(v, 1, :)
"The k-th batch element of a (B, n) / (n, B) array, as a vector."
lane_of(A, k, bd) = bd == 1 ? A[k, :] : A[:, k]
"A sub-batch (indices `ks`) of a (B, n) / (n, B) array."
sub_batch(A, ks, bd) = bd == 1 ? A[ks, :] : A[:, ks]
"Row and column index ranges of batch element `i`'s block in the block-diagonal sparse Jacobian."
block_idx(bd, i, m, n, B) =
    bd == 2 ? (((i - 1) * m + 1):(i * m), ((i - 1) * n + 1):(i * n)) : (i:B:(B * m), i:B:(B * n))

"Fill the nonzeros of a batched sparse Jacobian from J3[r, c, i], in the layout `make_pattern` produces."
function fill_jac!(jac, J3, bd)
    m, n, B = size(J3)
    nz = nonzeros(jac)
    if bd == 2
        reshape(nz, m, n, B) .= J3
    else
        reshape(nz, m, B, n) .= permutedims(J3, (1, 3, 2))
    end
    return jac
end

"Batched linear residual r_i = A_i x_i - b_i, A[r, c, i], b[r, i]."
function lin_resid!(y, x, A, b, bd)
    m, n, B = size(A)
    xc = bd == 1 ? permutedims(x) : x
    yc = dropdims(sum(A .* reshape(xc, 1, n, B); dims=2); dims=2) .- b
    y .= bd == 1 ? permutedims(yc) : yc
    return nothing
end

"Batched Rosenbrock residual with per-problem scale s: [10 s (x₂ - x₁²), 1 - x₁]. Root: (1, 1)."
function make_ros(bd)
    od = mod(bd, 2) + 1
    return (x, s) -> cat(
        10 .* s .* (selectdim(x, od, 2:2) .- selectdim(x, od, 1:1) .^ 2),
        1 .- selectdim(x, od, 1:1);
        dims=od,
    )
end
ros_vec(x, s) = [10 * s * (x[2] - x[1]^2), 1 - x[1]]           # unbatched version

function ros_start(rng, B, bd)
    X = hcat(-1.2 .+ 0.6 .* (rand(rng, B) .- 0.5), 1.0 .+ 0.6 .* (rand(rng, B) .- 0.5))
    return bd == 1 ? X : Matrix(permutedims(X))
end

solve_ros(bd, X0, s; maxiter=300, kw...) =
    levenberg_marquardt(make_ros(bd), X0, Constant(lane(s, bd)); batchdim=bd, maxiter, kw...)

"Run `f()` and return everything it printed to stdout."
function capture_stdout(f)
    mktemp() do path, io
        redirect_stdout(f, io)
        flush(io)
        return read(path, String)
    end
end

# ── Instrumented problem: f(x) = x - c for every batch element (n = m = 1, J = 1) ─────────────
# Lets us observe every trial point the solver evaluates, and inject NaN/Inf on demand.
# For this problem the damped step is dx = -(x - c) / (1 + λ), so the damping λ that was used at
# each trial point can be recovered exactly from the trial points themselves.
mutable struct TrialLog
    trials::Vector{Vector{Float64}}   # x passed to every residual-only (trial) evaluation
    njac_at_trial::Vector{Int}        # number of val_and_jac! calls made before each trial
    njac::Int                         # total val_and_jac! calls
    nbad::Int                         # number of NaN Jacobians injected so far
    ntrial::Int                       # number of trial evaluations so far
end
TrialLog() = TrialLog(Vector{Float64}[], Int[], 0, 0, 0)

"""
    run_instrumented(bd, x0v, cv; mode, K, bad, kwargs...)

`mode`:
- `:none`        no faults
- `:res`         residual of element `bad` is NaN at the first `K` trial points
- `:resinf`      same, with Inf
- `:res_forever` residual of element `bad` is NaN at every trial point
- `:jac`         Jacobian of element `bad` is NaN at the first `K` *new* points reached by an accepted step
- `:jac0`        Jacobian of element `bad` is always NaN (including at the initial point)
"""
function run_instrumented(bd, x0v, cv; mode=:none, K=3, bad=2, kwargs...)
    B = length(x0v)
    shape = isnothing(bd) ? (B,) : (bd == 1 ? (B, 1) : (1, B))
    x = copy(reshape(x0v, shape))
    cc = reshape(cv, shape)
    y = zero(x)
    jac = isnothing(bd) ? zeros(1, 1) : similar(make_pattern(x, y, bd), Float64)
    lg = TrialLog()
    nzs(j) = isnothing(bd) ? vec(j) : nonzeros(j)

    val! = function (yo, xo)
        lg.ntrial += 1
        push!(lg.trials, vec(copy(xo)))
        push!(lg.njac_at_trial, lg.njac)
        yo .= xo .- cc
        if mode == :res_forever || (mode in (:res, :resinf) && lg.ntrial <= K)
            yo[bad] = mode == :resinf ? Inf : NaN
        end
        return nothing
    end
    val_and_jac! = function (yo, jo, xo)
        lg.njac += 1
        yo .= xo .- cc
        nz = nzs(jo)
        nz .= 1.0
        if mode == :jac && xo[bad] != x0v[bad] && lg.nbad < K
            lg.nbad += 1
            nz[bad] = NaN
        elseif mode == :jac0
            nz[bad] = NaN
        end
        return nothing
    end
    out = levenberg_marquardt!(val!, val_and_jac!, y, jac, x; batchdim=bd, kwargs...)
    return (; out, lg, x, y, jac, c=cv, x0=x0v)
end

"Damping used for a step from `x0` to trial point `xt` on f(x) = x - c."
lam_from_trial(x0, c, xt) = (x0 - c) / (x0 - xt) - 1
"Damping after k consecutive rejections from λ0: λ ← λν, ν ← 2ν with ν = 2 initially ⇒ λ0 · 2^(k(k+1)/2)."
lam_expected(k; λ0=1e-3) = λ0 * 2.0^(k * (k + 1) / 2)

# ══════════════════════════════════════════════════════════════════════════════════════════════
@testset "levenberg_marquardt" begin

# ── Unbatched ────────────────────────────────────────────────────────────────────────────────
@testset "unbatched problems" begin
    @testset "scalar root" begin
        out = levenberg_marquardt(x -> x .^ 2 .- 2, [1.0])
        @test out.retcode == RETCODE_SUCCESS
        @test out.u[1] ≈ sqrt(2) atol = 1e-7
        @test abs(out.f[1]) < 1e-7
        @test 0 < out.iters < 100
    end

    @testset "square linear system" begin
        A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
        b = [1.0, 2.0, 3.0]
        out = levenberg_marquardt(x -> A * x .- b, zeros(3))
        @test out.retcode == RETCODE_SUCCESS
        @test out.u ≈ A \ b atol = 1e-7
    end

    @testset "Rosenbrock from the classic start (needs damping)" begin
        out = levenberg_marquardt(ros_vec, [-1.2, 1.0], Constant(1.0); maxiter=300)
        @test out.retcode == RETCODE_SUCCESS
        @test out.u ≈ [1.0, 1.0] atol = 1e-6
    end

    @testset "contexts are forwarded (Constant)" begin
        out = levenberg_marquardt((x, a) -> x .^ 2 .- a, [1.0], Constant(3.0))
        @test out.u[1] ≈ sqrt(3) atol = 1e-7
    end

    @testset "over-determined linear least squares (non-zero residual) = A \\ b" begin
        rng = MersenneTwister(1)
        A = randn(rng, 6, 3)
        b = randn(rng, 6)
        out = levenberg_marquardt((x, A, b) -> A * x .- b, zeros(3), Constant(A), Constant(b))
        @test out.retcode == RETCODE_SUCCESS          # terminates via gradtol, not abstol
        @test out.u ≈ A \ b atol = 1e-6
        @test norm(out.f) > 1e-3                      # residual really is non-zero
        @test out.iters < 100
    end

    @testset "over-determined nonlinear fit: stationary point of ‖f‖²" begin
        rng = MersenneTwister(2)
        t = collect(range(0, 3; length=8))
        d = 2.0 .* exp.(-1.3 .* t) .+ 0.02 .* randn(rng, 8)
        f(p, t, d) = p[1] .* exp.(-p[2] .* t) .- d
        out = levenberg_marquardt(f, [1.0, 1.0], Constant(t), Constant(d))
        @test out.retcode == RETCODE_SUCCESS
        @test out.u ≈ [2.0, 1.3] atol = 0.2
        # gradient Jᵀf must vanish, relative to ‖J‖‖f‖
        @test norm(out.jac' * out.f) < 1e-6 * norm(out.jac) * norm(out.f)
        # and the cost must have decreased from the start
        @test norm(out.f) < norm(f([1.0, 1.0], t, d))
    end

    @testset "under-determined system (m < n)" begin
        out = levenberg_marquardt(x -> [x[1]^2 + x[2]^2 - 4], [1.0, 0.5])
        @test out.retcode == RETCODE_SUCCESS
        @test abs(out.f[1]) < 1e-6
        @test norm(out.u) ≈ 2 atol = 1e-6
    end

    @testset "Float32 is preserved" begin
        out = levenberg_marquardt(x -> x .^ 2 .- 2f0, [1f0])
        @test eltype(out.u) == Float32
        @test out.retcode == RETCODE_SUCCESS
        @test out.u[1] ≈ sqrt(2f0) atol = 1e-4
    end

    @testset "starting at a root does no work" begin
        out = levenberg_marquardt(x -> x .^ 2 .- 4, [2.0])
        @test out.retcode == RETCODE_SUCCESS
        @test out.iters == 0
        @test out.u == [2.0]
    end
end

# ── API ──────────────────────────────────────────────────────────────────────────────────────
@testset "API and return values" begin
    @testset "in-place vs non-mutating" begin
        f!(y, x) = (y .= x .^ 2 .- 2; y)
        x = [1.0]
        y = zeros(1)
        out = levenberg_marquardt!(f!, y, x)
        @test out.u === x
        @test out.f === y
        @test x[1] ≈ sqrt(2) atol = 1e-7

        x0 = [1.0]
        out2 = levenberg_marquardt(z -> z .^ 2 .- 2, x0)
        @test x0 == [1.0]                              # input untouched
        @test out2.u !== x0
        @test keys(out2) == (:u, :f, :jac, :retcode, :iters)
    end

    @testset "batched return shapes and eltypes" begin
        B = 5
        s = fill(1.0, B)
        for bd in (1, 2)
            X0 = ros_start(MersenneTwister(3), B, bd)
            out = solve_ros(bd, X0, s)
            @test size(out.u) == size(X0)
            @test size(out.retcode) == (bd == 1 ? (B, 1) : (1, B))
            @test size(out.iters) == size(out.retcode)
            @test eltype(out.retcode) == UInt8
            @test eltype(out.iters) == Int
            @test out.jac isa SparseMatrixCSC
        end
    end

    @testset "final f and jac correspond to the returned u" begin
        B = 6
        rng = MersenneTwister(4)
        s = 0.5 .+ 2.5 .* rand(rng, B)
        for bd in (1, 2)
            out = solve_ros(bd, ros_start(rng, B, bd), s)
            @test out.f ≈ make_ros(bd)(out.u, lane(s, bd)) atol = 1e-12
            D = Matrix(out.jac)
            for i in 1:B
                x1 = bd == 1 ? out.u[i, 1] : out.u[1, i]
                rows, cols = block_idx(bd, i, 2, 2, B)
                @test D[rows, cols] ≈ [(-20 * s[i] * x1) (10 * s[i]); -1.0 0.0] atol = 1e-9
            end
        end
    end

    @testset "custom solver receives the square damped normal equations" begin
        rng = MersenneTwister(5)
        # unbatched: 6 residuals, 3 unknowns → solver sees a dense 3×3
        A = randn(rng, 6, 3)
        b = randn(rng, 6)
        x0 = zeros(3)
        base = BatchSolve.make_linear_solver(BatchSolve.KA.get_backend(x0), x0, x0, nothing)
        sizes = Set{Tuple{Int,Int}}()
        spy = (x, M, r) -> (push!(sizes, size(M)); base(x, M, r))
        out = levenberg_marquardt((x, A, b) -> A * x .- b, x0, Constant(A), Constant(b); solver=spy)
        @test sizes == Set([(3, 3)])
        @test out.u ≈ A \ b atol = 1e-6

        # batched: solver sees the (B·n)×(B·n) block-diagonal sparse matrix with B·n² stored entries
        m, n, B = 5, 3, 4
        A3 = randn(rng, m, n, B)
        b2 = randn(rng, m, B)
        for bd in (1, 2)
            x = bd == 1 ? zeros(B, n) : zeros(n, B)
            y = bd == 1 ? zeros(B, m) : zeros(m, B)
            jac = similar(make_pattern(x, y, bd), Float64)
            base = BatchSolve.make_linear_solver(BatchSolve.KA.get_backend(x), x, x, bd)
            seen = Set{Tuple{Int,Int,Int}}()
            spy = (xx, M, r) -> (push!(seen, (size(M)..., nnz(M))); base(xx, M, r))
            val! = (yo, xo) -> lin_resid!(yo, xo, A3, b2, bd)
            vj! = (yo, jo, xo) -> (lin_resid!(yo, xo, A3, b2, bd); fill_jac!(jo, A3, bd); nothing)
            levenberg_marquardt!(val!, vj!, y, jac, x; batchdim=bd, solver=spy)
            @test seen == Set([(B * n, B * n, B * n * n)])
        end
    end

    @testset "verbose output" begin
        txt = capture_stdout(() -> levenberg_marquardt(x -> x .^ 2 .- 2, [1.0]; verbose=true))
        @test occursin("Iteration", txt)
        @test count(==('\n'), txt) > 3
        X0 = ros_start(MersenneTwister(6), 3, 2)
        txt = capture_stdout(() -> solve_ros(2, X0, ones(3); verbose=true))
        @test occursin("Batched-LM", txt)
    end

    @testset "invalid arguments" begin
        @test_throws ErrorException levenberg_marquardt(x -> x .^ 2 .- 1, ones(2, 2); batchdim=3)
        val! = (y, x) -> (y .= x .^ 2 .- 1; nothing)
        vj! = (y, J, x) -> (y .= x .^ 2 .- 1; J .= 2 .* x; nothing)
        @test_throws ErrorException levenberg_marquardt!(val!, vj!, zeros(2), sparse(1.0I, 2, 2), ones(2))
    end
end

# ── Termination controls ─────────────────────────────────────────────────────────────────────
@testset "termination controls" begin
    @testset "maxiter" begin
        out = levenberg_marquardt(ros_vec, [-1.2, 1.0], Constant(1.0); maxiter=3)
        @test out.retcode == RETCODE_MAXITER
        @test out.iters == 3                           # unbatched: scalar, = maxiter when not converged

        B = 4
        for bd in (1, 2)
            out = solve_ros(bd, ros_start(MersenneTwister(7), B, bd), ones(B); maxiter=3)
            @test all(==(RETCODE_MAXITER), out.retcode)
            @test all(==(-1), out.iters)               # batched: -1 for non-converged elements
        end
    end

    @testset "abstol is honoured" begin
        tight = levenberg_marquardt(ros_vec, [-1.2, 1.0], Constant(1.0); maxiter=300)
        loose = levenberg_marquardt(ros_vec, [-1.2, 1.0], Constant(1.0); maxiter=300, abstol=1e-2)
        @test loose.retcode == RETCODE_SUCCESS
        @test norm(loose.f) < 1e-2
        @test loose.iters <= tight.iters
    end

    @testset "answer does not depend on the damping settings" begin
        for damping in (1e-4, 1e-2, 1.0), accept_ratio in (1e-4, 1e-2)
            out = levenberg_marquardt(ros_vec, [-1.2, 1.0], Constant(1.0); maxiter=300, damping, accept_ratio)
            @test out.retcode == RETCODE_SUCCESS
            @test out.u ≈ [1.0, 1.0] atol = 1e-6
        end
    end

    @testset "start at a stationary point of a least-squares problem" begin
        rng = MersenneTwister(8)
        A = randn(rng, 6, 3)
        b = randn(rng, 6)
        out = levenberg_marquardt((x, A, b) -> A * x .- b, A \ b, Constant(A), Constant(b))
        @test out.retcode == RETCODE_SUCCESS
        @test out.iters <= 1
    end
end

# ── Batched correctness ──────────────────────────────────────────────────────────────────────
@testset "batched problems" begin
    rng = MersenneTwister(2024)
    B = 9
    s = 0.5 .+ 2.5 .* rand(rng, B)

    for bd in (1, 2)
        X0 = ros_start(rng, B, bd)

        @testset "heterogeneous Rosenbrock batch, batchdim=$bd" begin
            out = solve_ros(bd, X0, s)
            @test all(==(RETCODE_SUCCESS), out.retcode)
            @test all(>=(0), out.iters)
            @test all(isapprox.(out.u, 1.0; atol=1e-6))
        end

        @testset "batched == unbatched, batchdim=$bd" begin
            out = solve_ros(bd, X0, s)
            for k in 1:B
                ref = levenberg_marquardt(ros_vec, lane_of(X0, k, bd), Constant(s[k]); maxiter=300)
                @test lane_of(out.u, k, bd) ≈ ref.u atol = 1e-6
                @test out.retcode[k] == ref.retcode
            end
        end

        @testset "agrees with newton where newton converges, batchdim=$bd" begin
            nt = newton(make_ros(bd), X0, Constant(lane(s, bd)); batchdim=bd)
            lm = solve_ros(bd, X0, s)
            @test all(isapprox.(nt.u, 1.0; atol=1e-6))      # precondition: newton solved it
            @test lm.u ≈ nt.u atol = 1e-6
        end

        @testset "Float32 batch, batchdim=$bd" begin
            out = levenberg_marquardt(
                make_ros(bd), Float32.(X0), Constant(Float32.(lane(s, bd)));
                batchdim=bd, maxiter=300, reltol=1f-4, abstol=1f-5, gradtol=1f-5,
            )
            @test eltype(out.u) == Float32
            @test all(isapprox.(out.u, 1f0; atol=5f-3))
        end
    end

    @testset "batchdim=1 and batchdim=2 solve the same problems the same way" begin
        X1 = ros_start(MersenneTwister(99), B, 1)
        o1 = solve_ros(1, X1, s)
        o2 = solve_ros(2, Matrix(permutedims(X1)), s)
        @test o1.u ≈ permutedims(o2.u) rtol = 1e-12
        @test vec(o1.iters) == vec(o2.iters)
        @test vec(o1.retcode) == vec(o2.retcode)
    end

    @testset "linear least squares with analytic Jacobians, m ≠ n" begin
        m, n, Bl = 5, 3, 6
        rl = MersenneTwister(11)
        A = randn(rl, m, n, Bl)
        b = randn(rl, m, Bl)
        for bd in (1, 2)
            x = bd == 1 ? zeros(Bl, n) : zeros(n, Bl)
            y = bd == 1 ? zeros(Bl, m) : zeros(m, Bl)
            jac = similar(make_pattern(x, y, bd), Float64)
            val! = (yo, xo) -> lin_resid!(yo, xo, A, b, bd)
            vj! = (yo, jo, xo) -> (lin_resid!(yo, xo, A, b, bd); fill_jac!(jo, A, bd); nothing)
            out = levenberg_marquardt!(val!, vj!, y, jac, x; batchdim=bd)
            @test all(==(RETCODE_SUCCESS), out.retcode)
            xc = bd == 1 ? permutedims(out.u) : out.u
            for i in 1:Bl
                @test xc[:, i] ≈ A[:, :, i] \ b[:, i] atol = 1e-6
            end
        end
    end

    @testset "block-diagonal sparse layout matches what the solver assumes" begin
        m, n, Bl = 3, 2, 4
        J3 = randn(MersenneTwister(12), m, n, Bl)
        for bd in (1, 2)
            x = bd == 1 ? zeros(Bl, n) : zeros(n, Bl)
            y = bd == 1 ? zeros(Bl, m) : zeros(m, Bl)
            jac = fill_jac!(similar(make_pattern(x, y, bd), Float64), J3, bd)
            D = Matrix(jac)
            for i in 1:Bl
                rows, cols = block_idx(bd, i, m, n, Bl)
                @test D[rows, cols] == J3[:, :, i]
            end
            @test nnz(jac) == m * n * Bl
        end
    end
end

# ── Batch independence: a problem behaves identically whether alone or in a batch ────────────
# Same layout and same code path ⇒ results are compared with `==` (bit-for-bit). These problems
# have m = n = 2, so every reduction is a two-term sum and is independent of summation order.
@testset "identical convergence alone vs. in a batch" begin
    rng = MersenneTwister(31337)
    B = 9
    s = 0.5 .+ 2.5 .* rand(rng, B)

    for bd in (1, 2)
        X0 = ros_start(rng, B, bd)
        full = solve_ros(bd, X0, s)
        @test all(==(RETCODE_SUCCESS), full.retcode)
        @test length(unique(vec(full.iters))) > 1        # elements really do need different numbers of iterations

        @testset "every element alone == the same element in the batch, batchdim=$bd" begin
            for k in 1:B
                alone = solve_ros(bd, sub_batch(X0, k:k, bd), s[k:k])
                @test lane_of(alone.u, 1, bd) == lane_of(full.u, k, bd)
                @test lane_of(alone.f, 1, bd) == lane_of(full.f, k, bd)
                @test alone.iters[1] == full.iters[k]
                @test alone.retcode[1] == full.retcode[k]
            end
        end

        @testset "position in the batch does not matter, batchdim=$bd" begin
            perm = reverse(1:B)
            rev = solve_ros(bd, sub_batch(X0, perm, bd), s[perm])
            @test rev.u == sub_batch(full.u, perm, bd)
            @test vec(rev.iters) == vec(full.iters)[perm]
            @test vec(rev.retcode) == vec(full.retcode)[perm]
        end

        @testset "a failing neighbour (NaN initial guess) changes nothing, batchdim=$bd" begin
            bad = 4
            Xn = copy(X0)
            bd == 1 ? (Xn[bad, :] .= NaN) : (Xn[:, bad] .= NaN)
            nanrun = solve_ros(bd, Xn, s)
            @test nanrun.retcode[bad] == RETCODE_FAILURE
            @test nanrun.iters[bad] == 0
            others = setdiff(1:B, bad)
            @test sub_batch(nanrun.u, others, bd) == sub_batch(full.u, others, bd)
            @test vec(nanrun.iters)[others] == vec(full.iters)[others]
            @test vec(nanrun.retcode)[others] == vec(full.retcode)[others]
        end

        @testset "an already-converged element stays put while others iterate, batchdim=$bd" begin
            Xe = copy(X0)
            bd == 1 ? (Xe[1, :] .= 1.0) : (Xe[:, 1] .= 1.0)   # element 1 starts exactly at the root
            out = solve_ros(bd, Xe, s)
            @test out.iters[1] == 0
            @test lane_of(out.u, 1, bd) == [1.0, 1.0]
            @test maximum(out.iters) > 5                        # others kept going long after
        end
    end
end

# ── Non-finite residuals / Jacobians ─────────────────────────────────────────────────────────
# Instrumented f(x) = x - c, n = m = 1. With Marquardt scaling D = 1 and J = 1, so after k
# consecutive rejections the damping used is λ0 · 2^(k(k+1)/2) and, on acceptance, λ ← λ/3 (ρ = 1
# for a linear residual). Both are read back from the trial points.
@testset "non-finite residual or Jacobian ⇒ step rejected, damping increased" begin
    c = [2.0, -1.0, 0.5]
    x0 = [5.0, 3.0, -4.0]
    K = 3           # number of poisoned evaluations
    bad = 2

    @testset "batched, batchdim=$bd, $mode" for bd in (1, 2), mode in (:res, :resinf, :jac)
        base = run_instrumented(bd, x0, c; mode=:none)
        r = run_instrumented(bd, x0, c; mode, K, bad)
        tr = [t[bad] for t in r.lg.trials]                 # trial points of the poisoned element

        # K rejections, then one accepted step: damping used was λ0, 2λ0, 8λ0, 64λ0, ...
        λs = lam_from_trial.(x0[bad], c[bad], tr[1:(K + 1)])
        @test λs ≈ lam_expected.(0:K) rtol = 1e-8

        # after acceptance the damping is divided by 3 (Nielsen update with ρ = 1)
        x1 = tr[K + 1]
        @test lam_from_trial(x1, c[bad], tr[K + 2]) ≈ lam_expected(K) / 3 rtol = 1e-6

        # x was not moved by the rejected steps (every rejected trial starts from x0, checked above)
        # and the problem still converges, just more slowly
        @test all(==(RETCODE_SUCCESS), r.out.retcode)
        @test r.x ≈ reshape(c, size(r.x)) atol = 1e-6
        @test r.out.iters[bad] >= base.out.iters[bad] + K

        # the other elements are bit-for-bit unaffected
        others = [1, 3]
        @test r.x[others] == base.x[others]
        @test r.out.iters[others] == base.out.iters[others]

        # final residual/Jacobian are those of the final x (i.e. restored after any rollback)
        @test r.y ≈ r.x .- reshape(c, size(r.x)) atol = 1e-12
        @test all(isfinite, r.y)
        @test all(isfinite, nonzeros(r.jac))
        mode == :jac && @test r.lg.nbad == K
    end

    @testset "unbatched, $mode" for mode in (:res, :resinf, :jac)
        r = run_instrumented(nothing, [5.0], [2.0]; mode, K, bad=1)
        tr = [t[1] for t in r.lg.trials]
        @test lam_from_trial.(5.0, 2.0, tr[1:(K + 1)]) ≈ lam_expected.(0:K) rtol = 1e-8
        @test lam_from_trial(tr[K + 1], 2.0, tr[K + 2]) ≈ lam_expected(K) / 3 rtol = 1e-6
        @test r.out.retcode == RETCODE_SUCCESS
        @test r.x ≈ [2.0] atol = 1e-6
        @test all(isfinite, r.jac)
        if mode != :jac
            # rejected trials must not trigger Jacobian re-evaluation (x didn't move)
            @test all(==(1), r.lg.njac_at_trial[1:(K + 1)])
        else
            @test r.lg.nbad == K
        end
    end

    @testset "damping keywords" begin
        # initial damping is what the first step uses
        r = run_instrumented(nothing, [5.0], [2.0]; mode=:none, bad=1, damping=0.5)
        @test lam_from_trial(5.0, 2.0, r.lg.trials[1][1]) ≈ 0.5 rtol = 1e-9

        # damping_max caps the growth and makes a hopeless problem fail sooner
        dflt = run_instrumented(nothing, [5.0], [2.0]; mode=:res_forever, bad=1)
        capped = run_instrumented(nothing, [5.0], [2.0]; mode=:res_forever, bad=1, damping_max=10.0)
        @test dflt.out.retcode == RETCODE_FAILURE
        @test capped.out.retcode == RETCODE_FAILURE
        @test capped.out.iters < dflt.out.iters
    end

    @testset "non-finite at the initial point is a failure, batchdim=$bd" for bd in (nothing, 1, 2)
        if isnothing(bd)
            r = run_instrumented(nothing, [5.0], [2.0]; mode=:jac0, bad=1)
            @test r.out.retcode == RETCODE_FAILURE
            @test r.out.iters == 0
            @test r.x == [5.0]
        else
            base = run_instrumented(bd, x0, c; mode=:none)
            r = run_instrumented(bd, x0, c; mode=:jac0, bad)
            @test r.out.retcode[bad] == RETCODE_FAILURE
            @test r.out.iters[bad] == 0
            @test r.x[bad] == x0[bad]                              # never moved
            @test r.out.retcode[[1, 3]] == [RETCODE_SUCCESS, RETCODE_SUCCESS]
            @test r.x[[1, 3]] == base.x[[1, 3]]
        end
    end

    @testset "persistently non-finite residual ⇒ FAILURE (damping saturates), never SUCCESS, batchdim=$bd" for bd in (nothing, 1, 2)
        if isnothing(bd)
            r = run_instrumented(nothing, [5.0], [2.0]; mode=:res_forever, bad=1)
            @test r.out.retcode == RETCODE_FAILURE
            @test r.out.iters < 100
            @test r.x == [5.0]
        else
            base = run_instrumented(bd, x0, c; mode=:none)
            r = run_instrumented(bd, x0, c; mode=:res_forever, bad)
            @test r.out.retcode[bad] == RETCODE_FAILURE
            @test 0 < r.out.iters[bad] < 100
            @test r.x[bad] == x0[bad]                              # rejected every time
            @test r.out.retcode[[1, 3]] == [RETCODE_SUCCESS, RETCODE_SUCCESS]
            @test r.x[[1, 3]] == base.x[[1, 3]]
            @test r.out.iters[[1, 3]] == base.out.iters[[1, 3]]
        end
    end

    # Through the public (AD-based) API: f(x) = x - c, but NaN once x > 10, and the root of the
    # second problem lies outside that domain. A non-finite trial point must never be accepted.
    @testset "leaving the domain of f (public API), batchdim=$bd" for bd in (1, 2)
        cvec = [5.0, 100.0, 3.0]
        xs0 = [1.0, 1.0, 2.0]
        f(x, cc) = x .- cc .+ (x .> 10) .* NaN
        xs = bd == 1 ? reshape(xs0, :, 1) : reshape(xs0, 1, :)
        out = levenberg_marquardt(f, xs, Constant(lane(cvec, bd)); batchdim=bd, maxiter=300)
        u = vec(out.u)
        @test all(isfinite, u)
        @test all(isfinite, out.f)
        @test u[1] ≈ 5.0 atol = 1e-6
        @test u[3] ≈ 3.0 atol = 1e-6
        @test 9.99 < u[2] <= 10.0                               # crept up to the boundary, never past it
        # ... and the element with the infeasible root behaves exactly as if it were alone
        alone = levenberg_marquardt(
            f, fill(1.0, 1, 1), Constant(lane([100.0], bd)); batchdim=bd, maxiter=300,
        )
        @test alone.u[1] == u[2]
        @test alone.iters[1] == out.iters[2]
    end
end

end # @testset "levenberg_marquardt"
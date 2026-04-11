"""
Test suite for BatchSolve.newton!

Interface: newton!(f!, y, x; kwargs...)
  f! is mutating: f!(y, x)
  y must be pre-allocated and pre-filled before calling.
  retcode/iters shape matches x shape along batchdim:
    x shape (n,)   batchdim=1 → retcode/iters shape (n, 1)
    x shape (1,n)  batchdim=2 → retcode/iters shape (1, n)
  Default convergence: reltol=sqrt(eps(eltype(x))), abstol=sqrt(eps(eltype(y)))
  ~1e-8 for Float64 in practice, not 1e-12.
"""

using Test
using Random
using LinearAlgebra
using SparseArrays
using BatchSolve

const SUCCESS  = BatchSolve.RETCODE_SUCCESS
const FAILURE  = BatchSolve.RETCODE_FAILURE
const MAXITERS = BatchSolve.RETCODE_MAXITERS

# ===========================================================================
# 1-18: Scalar (non-batched)
# ===========================================================================

@testset "1. Linear f(x)=x-c" begin
    y = [0.0]; x = [0.0]; y[1] = x[1] - 3.7
    sol = newton!((y,x)->(y[1]=x[1]-3.7), y, x)
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ 3.7
end

@testset "2. Quadratic x^2-2, root at sqrt(2)" begin
    f!(y,x) = (y[1] = x[1]^2 - 2.0)
    x = [1.0]; y = [x[1]^2 - 2.0]
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ sqrt(2.0)
    @test sol.u[1] > 0
end

@testset "3. Trig sin(x), root near pi" begin
    f!(y,x) = (y[1] = sin(x[1]))
    x = [3.0]; y = [sin(3.0)]
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ π
end

@testset "4. Exponential exp(x)-3" begin
    f!(y,x) = (y[1] = exp(x[1]) - 3.0)
    x = [1.0]; y = [exp(1.0) - 3.0]
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ log(3.0)
end

@testset "5. Multiple roots: initial guess selects branch" begin
    f!(y,x) = (y[1] = x[1]^2 - 4.0)
    xp = [1.5]; yp = [xp[1]^2 - 4.0]
    xn = [-1.5]; yn = [xn[1]^2 - 4.0]
    @test newton!(f!, yp, xp).u[1] ≈  2.0
    @test newton!(f!, yn, xn).u[1] ≈ -2.0
end

@testset "6. High-multiplicity root (x-1)^3: converges linearly" begin
    f!(y,x) = (y[1] = (x[1] - 1.0)^3)
    x = [0.5]; y = [(0.5 - 1.0)^3]
    sol = newton!(f!, y, x; abstol=1e-30, reltol=1e-30)
    @test sol.u[1] ≈ 1.0
end

@testset "7. 2D linear system" begin
    A = [2.0 1.0; 1.0 3.0]; b = [5.0; 10.0]
    f!(y,x) = (y .= A * x .- b)
    x = [0.0; 0.0]; y = A*x .- b
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.u ≈ A \ b
end

@testset "8. 3D linear system" begin
    A = [4.0 1 0; 2 5 1; 0 1 3]; b = [7.0; 14; 9]
    f!(y,x) = (y .= A * x .- b)
    x = zeros(3); y = A*x .- b
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.u ≈ A \ b
end

@testset "9. Nonlinear 2D: unit circle + diagonal" begin
    f!(y,x) = (y[1] = x[1]^2 + x[2]^2 - 1; y[2] = x[1] - x[2])
    x = [0.6; 0.6]; y = zeros(2); f!(y, x)
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.u ≈ fill(1/sqrt(2), 2)
end

@testset "10. retcode == SUCCESS on convergence" begin
    f!(y,x) = (y[1] = x[1] - 1.0)
    x = [0.0]; y = [-1.0]
    @test newton!(f!, y, x).retcode == SUCCESS
end

@testset "11. retcode valid UInt8 with maxiter=1" begin
    f!(y,x) = (y[1] = x[1]^2 - 2.0)
    x = [100.0]; y = [x[1]^2 - 2.0]
    sol = newton!(f!, y, x; maxiter=1)
    @test sol.retcode isa UInt8
    @test sol.retcode in (SUCCESS, MAXITERS)
end

@testset "12. retcode == FAILURE on singular Jacobian" begin
    f!(y,x) = (y[1] = x[1]^2)
    x = [0.0]; y = [0.0]
    @test newton!(f!, y, x).retcode == FAILURE
end

@testset "13. iters is non-negative integer" begin
    f!(y,x) = (y[1] = x[1] - 2.5)
    x = [0.0]; y = [-2.5]
    sol = newton!(f!, y, x)
    @test sol.iters >= 0
    @test sol.iters isa Integer
end

@testset "14. u contains the root: residual near zero" begin
    f!(y,x) = (y[1] = cos(x[1]) - 0.5)
    x = [1.0]; y = [cos(1.0) - 0.5]
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test cos(sol.u[1]) ≈ 0.5
end

@testset "15. jac size scalar (1,1)" begin
    f!(y,x) = (y[1] = x[1] - 1.0)
    x = [0.0]; y = [-1.0]
    @test size(newton!(f!, y, x).jac) == (1, 1)
end

@testset "15b. jac size 2D is (2,2)" begin
    f!(y,x) = (y .= x .- 1.0)
    x = zeros(2); y = x .- 1.0
    sol = newton!(f!, y, x)
    @test size(sol.jac, 1) == 2
    @test size(sol.jac, 2) == 2
end

@testset "16. Loose abstol needs fewer iters than tight" begin
    f!(y,x) = (y[1] = x[1]^3 - 2.0)
    xt = [1.0]; yt = [xt[1]^3 - 2.0]
    xl = [1.0]; yl = [xl[1]^3 - 2.0]
    tight = newton!(f!, yt, xt; abstol=1e-12)
    loose = newton!(f!, yl, xl; abstol=1e-2)
    @test tight.retcode == SUCCESS
    @test loose.retcode == SUCCESS
    @test loose.iters <= tight.iters
    @test tight.u[1] ≈ 2^(1/3)
end

@testset "17. Float32 type preserved" begin
    f!(y,x) = (y[1] = x[1] - 1.5f0)
    x = Float32[0.0]; y = Float32[-1.5]
    sol = newton!(f!, y, x)
    @test eltype(sol.u) == Float32
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ 1.5f0
end

@testset "18. reltol parameter" begin
    f!(y,x) = (y[1] = x[1] - Float64(π))
    x = [0.0]; y = [-Float64(π)]
    sol = newton!(f!, y, x; reltol=1e-14)
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ π
end

# ===========================================================================
# 19-30: Batched batchdim=1
# Note: for x shape (n,), retcode/iters shape is (n, 1)
# ===========================================================================

@testset "19. Batch 1D linear, batchdim=1, n=10" begin
    cs = collect(LinRange(1.0, 10.0, 10))
    f!(y,x) = (y .= x .- cs)
    x = zeros(10); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test sol.u ≈ cs
end

@testset "20. Batch 1D quadratic x^2=p, batchdim=1" begin
    ps = collect(LinRange(1.0, 4.0, 20))
    f!(y,x) = (y .= x.^2 .- ps)
    x = ones(20); y = x.^2 .- ps
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test sol.u ≈ sqrt.(ps)
end

@testset "21. Batch 1D trig sin(x)=c, batchdim=1" begin
    cs = collect(LinRange(0.1, 0.9, 9))
    f!(y,x) = (y .= sin.(x) .- cs)
    x = asin.(cs) .+ 0.1; y = sin.(x) .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test sin.(sol.u) ≈ cs
end

@testset "22. Large batch n=1000, batchdim=1" begin
    cs = rand(MersenneTwister(42), 1000) .* 10
    f!(y,x) = (y .= x .- cs)
    x = zeros(1000); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test sol.u ≈ cs
end

@testset "23. 2D systems batched, batchdim=1" begin
    P = rand(MersenneTwister(7), 50, 2)
    f!(y,x) = (y[:,1] .= 2 .* x[:,1] .- P[:,1]; y[:,2] .= 3 .* x[:,2] .- P[:,2])
    x = zeros(50, 2); y = similar(x); f!(y, x)
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test sol.u[:,1] ≈ P[:,1] ./ 2
    @test sol.u[:,2] ≈ P[:,2] ./ 3
end

@testset "24. Heterogeneous difficulty: cubic roots, batchdim=1" begin
    cs = collect(LinRange(0.5, 2.0, 20))
    f!(y,x) = (y .= x.^3 .- cs)
    x = ones(20); y = x.^3 .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test sol.u ≈ cs.^(1/3)
end

@testset "25. retcode/iters shape for 1D batch, batchdim=1" begin
    # x shape (15,) → retcode/iters shape (15, 1)
    cs = ones(15)
    f!(y,x) = (y .= x .- cs)
    x = zeros(15); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test size(sol.retcode) == (15, 1)
    @test size(sol.iters)   == (15, 1)
    @test eltype(sol.retcode) == UInt8
    @test all(sol.retcode .== SUCCESS)
    @test all(sol.iters .>= 0)
end

@testset "26. retcode/iters shape for 2D batch, batchdim=1" begin
    # x shape (10, 2) → retcode/iters shape (10, 1)
    P = rand(MersenneTwister(3), 10, 2)
    f!(y,x) = (y[:,1] .= x[:,1] .- P[:,1]; y[:,2] .= x[:,2] .- P[:,2])
    x = zeros(10, 2); y = similar(x); f!(y, x)
    sol = newton!(f!, y, x; batchdim=1)
    @test size(sol.retcode) == (10, 1)
    @test size(sol.iters)   == (10, 1)
    @test all(sol.retcode .== SUCCESS)
end

@testset "27. Each lane matches its root, batchdim=1" begin
    cs = [1.0, 2.0, 3.0, 4.0, 5.0]
    f!(y,x) = (y .= x .- cs)
    x = zeros(5); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test sol.u ≈ cs
end

@testset "28. Singular J lanes → FAILURE, good lanes → SUCCESS, batchdim=1" begin
    cs = [0.0, 1.0, 0.0, 1.0]
    x0 = [0.0, 1.0, 0.0, 1.0]
    f!(y,x) = (y .= x.^2 .- cs)
    y = x0.^2 .- cs
    sol = newton!(f!, y, copy(x0); batchdim=1)
    @test sol.retcode[1] == FAILURE
    @test sol.retcode[3] == FAILURE
    @test sol.retcode[2] == SUCCESS
    @test sol.retcode[4] == SUCCESS
end

@testset "29. All u values in expected range, batchdim=1" begin
    cs = rand(MersenneTwister(11), 100) .+ 0.1
    f!(y,x) = (y .= x .- cs)
    x = zeros(100); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test all(sol.u .>= 0.0)
    @test all(sol.u .<= 2.0)
end

@testset "30. Permutation invariance, batchdim=1" begin
    n    = 20
    cs   = collect(LinRange(0.1, 2.0, n))
    perm = randperm(MersenneTwister(3), n)
    f1!(y,x) = (y .= x .- cs)
    f2!(y,x) = (y .= x .- cs[perm])
    x1 = zeros(n); y1 = x1 .- cs
    x2 = zeros(n); y2 = x2 .- cs[perm]
    sol1 = newton!(f1!, y1, x1; batchdim=1)
    sol2 = newton!(f2!, y2, x2; batchdim=1)
    @test sol1.u ≈ sol2.u[invperm(perm)]
end

# ===========================================================================
# 31-35: Batched batchdim=2
# Note: for x shape (1,n), retcode/iters shape is (1, n)
# ===========================================================================

@testset "31. Batch 1D linear, batchdim=2, n=10" begin
    cs = reshape(collect(LinRange(1.0, 10.0, 10)), 1, 10)
    f!(y,x) = (y .= x .- cs)
    x = zeros(1, 10); y = x .- cs
    sol = newton!(f!, y, x; batchdim=2)
    @test all(sol.retcode .== SUCCESS)
    @test vec(sol.u) ≈ vec(cs)
end

@testset "32. Large batch n=1000, batchdim=2" begin
    cs = rand(MersenneTwister(42), 1, 1000)
    f!(y,x) = (y .= x .- cs)
    x = zeros(1, 1000); y = x .- cs
    sol = newton!(f!, y, x; batchdim=2)
    @test all(sol.retcode .== SUCCESS)
    @test sol.u ≈ cs
end

@testset "33. 2D systems batched, batchdim=2" begin
    P = rand(MersenneTwister(7), 2, 50)
    f!(y,x) = (y[1,:] .= 2 .* x[1,:] .- P[1,:]; y[2,:] .= 3 .* x[2,:] .- P[2,:])
    x = zeros(2, 50); y = similar(x); f!(y, x)
    sol = newton!(f!, y, x; batchdim=2)
    @test all(sol.retcode .== SUCCESS)
    @test sol.u[1,:] ≈ P[1,:] ./ 2
    @test sol.u[2,:] ≈ P[2,:] ./ 3
end

@testset "34. retcode/iters shape for batchdim=2" begin
    # x shape (1, 30) → retcode/iters shape (1, 30)
    cs = rand(MersenneTwister(1), 1, 30)
    f!(y,x) = (y .= x .- cs)
    x = zeros(1, 30); y = x .- cs
    sol = newton!(f!, y, x; batchdim=2)
    @test size(sol.retcode) == (1, 30)
    @test size(sol.iters)   == (1, 30)
    @test all(sol.retcode .== SUCCESS)
    @test all(sol.iters .>= 0)
end

@testset "35. batchdim=1 and batchdim=2 give same answers" begin
    cs  = rand(MersenneTwister(99), 20)
    f1!(y,x) = (y .= x .- cs)
    cs2 = reshape(cs, 1, 20)
    f2!(y,x) = (y .= x .- cs2)
    x1 = zeros(20);    y1 = x1 .- cs
    x2 = zeros(1, 20); y2 = x2 .- cs2
    sol1 = newton!(f1!, y1, x1; batchdim=1)
    sol2 = newton!(f2!, y2, x2; batchdim=2)
    @test vec(sol1.u) ≈ vec(sol2.u)
    @test all(vec(sol1.retcode) .== vec(sol2.retcode))
end

# ===========================================================================
# 36-45: Edge cases
# ===========================================================================

@testset "36. Already at root: converges in <=1 iters" begin
    f!(y,x) = (y[1] = x[1] - 1.0)
    x = [1.0]; y = [0.0]
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ 1.0
    @test sol.iters <= 1
end

@testset "37. maxiter=1: no error, result is finite" begin
    f!(y,x) = (y[1] = x[1]^2 - 2.0)
    x = [100.0]; y = [x[1]^2 - 2.0]
    sol = newton!(f!, y, x; maxiter=1)
    @test !isnan(sol.u[1])
    @test sol.retcode isa UInt8
end

@testset "38. Very tight abstol=1e-14" begin
    f!(y,x) = (y[1] = x[1] - sqrt(2.0))
    x = [1.0]; y = [1.0 - sqrt(2.0)]
    sol = newton!(f!, y, x; abstol=1e-14)
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ sqrt(2.0)
end

@testset "39. Loose abstol fewer iters than tight" begin
    f!(y,x) = (y[1] = sin(x[1]))
    xt = [3.0]; yt = [sin(3.0)]
    xl = [3.0]; yl = [sin(3.0)]
    tight = newton!(f!, yt, xt; abstol=1e-14)
    loose = newton!(f!, yl, xl; abstol=1e-2)
    @test tight.retcode == SUCCESS
    @test loose.retcode == SUCCESS
    @test loose.iters <= tight.iters
end

@testset "40. Invalid batchdim raises error" begin
    f!(y,x) = (y .= x)
    y = zeros(4); x = zeros(4)
    @test_throws Exception newton!(f!, copy(y), copy(x); batchdim=3)
    @test_throws Exception newton!(f!, copy(y), copy(x); batchdim=0)
end

@testset "41. Reproducibility: identical calls give identical results" begin
    f!(y,x) = (y[1] = x[1]^2 - 3.0)
    x1 = [1.0]; y1 = [x1[1]^2 - 3.0]
    x2 = [1.0]; y2 = [x2[1]^2 - 3.0]
    sol1 = newton!(f!, y1, x1)
    sol2 = newton!(f!, y2, x2)
    @test sol1.u      == sol2.u
    @test sol1.retcode == sol2.retcode
    @test sol1.iters   == sol2.iters
end

@testset "42. Residual near zero at returned u" begin
    f!(y,x) = (y[1] = x[1]^3 - 5.0)
    x = [1.5]; y = [x[1]^3 - 5.0]
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.u[1]^3 ≈ 5.0
end

@testset "43. Batch vs serial: same cubic roots, batchdim=1" begin
    cs = collect(1.0:10.0)
    f!(y,x) = (y .= x.^3 .- cs)
    x = ones(10); y = x.^3 .- cs
    sol_batch = newton!(f!, y, x; batchdim=1)
    roots_serial = map(1:10) do i
        fi!(y,x) = (y[1] = x[1]^3 - cs[i])
        xi = [1.0]; yi = [xi[1]^3 - cs[i]]
        newton!(fi!, yi, xi).u[1]
    end
    @test all(sol_batch.retcode .== SUCCESS)
    @test sol_batch.u ≈ roots_serial
end

@testset "44. x0 already at root: immediately succeeds, batchdim=1" begin
    cs = collect(1.0:5.0)
    f!(y,x) = (y .= x .- cs)
    x = copy(cs); y = x .- cs   # y = 0 everywhere
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    @test all(sol.iters .<= 1)
end

@testset "45. maxiter=1 from far start → not SUCCESS, batchdim=1" begin
    cs = zeros(5)
    f!(y,x) = (y .= x.^3 .- cs)
    x = fill(1e6, 5); y = x.^3 .- cs
    sol = newton!(f!, y, x; batchdim=1, maxiter=1)
    @test all(sol.retcode .!= SUCCESS)
end

# ===========================================================================
# 46-50: newton_solver CPU closures
# ===========================================================================

@testset "46. newton_solver(nothing) solves J*dx=-y" begin
    y  = [3.0; 4.0]
    J  = [2.0 0.0; 0.0 4.0]
    dx = zeros(2)
    solve! = BatchSolve.newton_solver(nothing, y, zeros(2), nothing)
    solve!(dx, J, y)
    @test dx ≈ [-1.5; -1.0]
end

@testset "47. newton_solver batchdim=1: two 1x1 systems" begin
    y  = [4.0; 6.0]
    J  = sparse([1, 2], [1, 2], [2.0, 3.0], 2, 2)
    dx = zeros(2)
    solve! = BatchSolve.newton_solver(nothing, reshape(y, 2, 1), reshape(zeros(2), 2, 1), 1)
    solve!(dx, J, y)
    @test dx[1] ≈ -2.0
    @test dx[2] ≈ -2.0
end

@testset "48. newton_solver batchdim=2: two 1x1 systems" begin
    y  = [4.0; 6.0]
    J  = sparse([1, 2], [1, 2], [2.0, 3.0], 2, 2)
    dx = zeros(2)
    solve! = BatchSolve.newton_solver(nothing, reshape(y, 1, 2), reshape(zeros(2), 1, 2), 2)
    solve!(dx, J, y)
    @test dx[1] ≈ -2.0
    @test dx[2] ≈ -2.0
end

@testset "49. Singular J in batchdim=1 → NaN for that lane only" begin
    y  = [1.0; 2.0]
    J  = sparse([1, 2], [1, 2], [0.0, 2.0], 2, 2)
    dx = zeros(2)
    solve! = BatchSolve.newton_solver(nothing, reshape(y, 2, 1), reshape(zeros(2), 2, 1), 1)
    solve!(dx, J, y)
    @test isnan(dx[1])
    @test dx[2] ≈ -1.0
end

@testset "50. Singular J in batchdim=2 → NaN for that lane only" begin
    y  = [1.0; 2.0]
    J  = sparse([1, 2], [1, 2], [0.0, 2.0], 2, 2)
    dx = zeros(2)
    solve! = BatchSolve.newton_solver(nothing, reshape(y, 1, 2), reshape(zeros(2), 1, 2), 2)
    solve!(dx, J, y)
    @test isnan(dx[1])
    @test dx[2] ≈ -1.0
end

# ===========================================================================
# Bonus
# ===========================================================================

@testset "B1. Direct val_and_jac! interface" begin
    vj!(y, J, x) = (y[1] = x[1] - 3.0; J[1,1] = 1.0)
    y = [0.0]; J = zeros(1,1); x = [0.0]
    sol = newton!(vj!, y, J, x)
    @test sol.retcode == SUCCESS
    @test sol.u[1] ≈ 3.0
end

@testset "B2. iters <= maxiter for all lanes, batchdim=1" begin
    cs = rand(MersenneTwister(5), 50) .+ 1.0
    f!(y,x) = (y .= x .- cs)
    x = zeros(50); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1, maxiter=100)
    @test all(sol.iters .<= 100)
end

@testset "B3. Float32 batch preserves type, batchdim=1" begin
    cs = Float32.(collect(1.0:10.0))
    f!(y,x) = (y .= x .- cs)
    x = zeros(Float32, 10); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test eltype(sol.u) == Float32
    @test all(sol.retcode .== SUCCESS)
    @test sol.u ≈ cs
end

@testset "B4. Residual near zero at solution, 2D batch batchdim=1" begin
    P = rand(MersenneTwister(42), 100, 2)
    f!(y,x) = (y[:,1] .= 2.0 .* x[:,1] .- P[:,1]; y[:,2] .= 3.0 .* x[:,2] .- P[:,2])
    x = zeros(100, 2); y = similar(x); f!(y, x)
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== SUCCESS)
    res = similar(sol.u); f!(res, sol.u)
    @test maximum(abs.(res)) < 1e-8
end

@testset "B5. Newton converges in < 15 iters for smooth scalar problem" begin
    f!(y,x) = (y[1] = x[1]^5 - 3.0)
    x = [1.5]; y = [x[1]^5 - 3.0]
    sol = newton!(f!, y, x)
    @test sol.retcode == SUCCESS
    @test sol.iters < 15
    @test sol.u[1] ≈ 3.0^(1/5)
end

@testset "B6. Batch: solution independent of lane ordering, batchdim=1" begin
    n    = 30
    cs   = rand(MersenneTwister(77), n) .* 5
    perm = randperm(MersenneTwister(77), n)
    f1!(y,x) = (y .= x .- cs)
    f2!(y,x) = (y .= x .- cs[perm])
    x1 = zeros(n); y1 = x1 .- cs
    x2 = zeros(n); y2 = x2 .- cs[perm]
    sol1 = newton!(f1!, y1, x1; batchdim=1)
    sol2 = newton!(f2!, y2, x2; batchdim=1)
    @test sol1.u ≈ sol2.u[invperm(perm)]
end
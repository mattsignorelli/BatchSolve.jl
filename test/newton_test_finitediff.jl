using Random
using SparseArrays
using BatchSolve
import BatchSolve: RETCODE_SUCCESS, RETCODE_FAILURE, RETCODE_MAXITER

# ===========================================================================
# 19-30: Batched batchdim=1
# Note: for x shape (n,), retcode/iters shape is (n, 1)
# ===========================================================================

@testset "19. Batch 1D linear, batchdim=1, n=10" begin
    cs = collect(LinRange(1.0, 10.0, 10))
    f!(y,x) = (y .= x .- cs)
    x = zeros(10); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs
end

@testset "20. Batch 1D quadratic x^2=p, batchdim=1" begin
    ps = collect(LinRange(1.0, 4.0, 20))
    f!(y,x) = (y .= x.^2 .- ps)
    x = ones(20); y = x.^2 .- ps
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ sqrt.(ps)
end

@testset "21. Batch 1D trig sin(x)=c, batchdim=1" begin
    cs = collect(LinRange(0.1, 0.9, 9))
    f!(y,x) = (y .= sin.(x) .- cs)
    x = asin.(cs) .+ 0.1; y = sin.(x) .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sin.(sol.u) ≈ cs
end

@testset "22. Large batch n=1000, batchdim=1" begin
    cs = rand(MersenneTwister(42), 1000) .* 10
    f!(y,x) = (y .= x .- cs)
    x = zeros(1000); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs
end

@testset "23. 2D systems batched, batchdim=1" begin
    P = rand(MersenneTwister(7), 50, 2)
    f!(y,x) = (y[:,1] .= 2 .* x[:,1] .- P[:,1]; y[:,2] .= 3 .* x[:,2] .- P[:,2])
    x = zeros(50, 2); y = similar(x); f!(y, x)
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u[:,1] ≈ P[:,1] ./ 2
    @test sol.u[:,2] ≈ P[:,2] ./ 3
end

@testset "24. Heterogeneous difficulty: cubic roots, batchdim=1" begin
    cs = collect(LinRange(0.5, 2.0, 20))
    f!(y,x) = (y .= x.^3 .- cs)
    x = ones(20); y = x.^3 .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
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
    @test all(sol.retcode .== RETCODE_SUCCESS)
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
    @test all(sol.retcode .== RETCODE_SUCCESS)
end

@testset "27. Each lane matches its root, batchdim=1" begin
    cs = [1.0, 2.0, 3.0, 4.0, 5.0]
    f!(y,x) = (y .= x .- cs)
    x = zeros(5); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs
end

@testset "28. Singular J lanes → RETCODE_FAILURE, good lanes → RETCODE_SUCCESS, batchdim=1" begin
    cs = [0.0, 1.0, 0.0, 1.0]
    x0 = [0.0, 1.0, 0.0, 1.0]
    f!(y,x) = (y .= x.^2 .- cs)
    y = x0.^2 .- cs
    sol = newton!(f!, y, copy(x0); batchdim=1)
    @test sol.retcode[1] == RETCODE_FAILURE
    @test sol.retcode[3] == RETCODE_FAILURE
    @test sol.retcode[2] == RETCODE_SUCCESS
    @test sol.retcode[4] == RETCODE_SUCCESS
end

@testset "29. All u values in expected range, batchdim=1" begin
    cs = rand(MersenneTwister(11), 100) .+ 0.1
    f!(y,x) = (y .= x .- cs)
    x = zeros(100); y = x .- cs
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
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
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test vec(sol.u) ≈ vec(cs)
end

@testset "32. Large batch n=1000, batchdim=2" begin
    cs = rand(MersenneTwister(42), 1, 1000)
    f!(y,x) = (y .= x .- cs)
    x = zeros(1, 1000); y = x .- cs
    sol = newton!(f!, y, x; batchdim=2)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs
end

@testset "33. 2D systems batched, batchdim=2" begin
    P = rand(MersenneTwister(7), 2, 50)
    f!(y,x) = (y[1,:] .= 2 .* x[1,:] .- P[1,:]; y[2,:] .= 3 .* x[2,:] .- P[2,:])
    x = zeros(2, 50); y = similar(x); f!(y, x)
    sol = newton!(f!, y, x; batchdim=2)
    @test all(sol.retcode .== RETCODE_SUCCESS)
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
    @test all(sol.retcode .== RETCODE_SUCCESS)
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
    @test all(sol_batch.retcode .== RETCODE_SUCCESS)
    @test sol_batch.u ≈ roots_serial
end

@testset "44. x0 already at root: immediately succeeds, batchdim=1" begin
    cs = collect(1.0:5.0)
    f!(y,x) = (y .= x .- cs)
    x = copy(cs); y = x .- cs   # y = 0 everywhere
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test all(sol.iters .<= 1)
end

@testset "45. maxiter=1 from far start → not RETCODE_SUCCESS, batchdim=1" begin
    cs = zeros(5)
    f!(y,x) = (y .= x.^3 .- cs)
    x = fill(1e6, 5); y = x.^3 .- cs
    sol = newton!(f!, y, x; batchdim=1, maxiter=1)
    @test all(sol.retcode .!= RETCODE_SUCCESS)
end

# ===========================================================================
# Bonus
# ===========================================================================

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
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs
end

@testset "B4. Residual near zero at solution, 2D batch batchdim=1" begin
    P = rand(MersenneTwister(42), 100, 2)
    f!(y,x) = (y[:,1] .= 2.0 .* x[:,1] .- P[:,1]; y[:,2] .= 3.0 .* x[:,2] .- P[:,2])
    x = zeros(100, 2); y = similar(x); f!(y, x)
    sol = newton!(f!, y, x; batchdim=1)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    res = similar(sol.u); f!(res, sol.u)
    @test maximum(abs.(res)) < 1e-8
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
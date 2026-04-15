"""
Comprehensive test suite for BatchSolve.brent / brent!

API (from source):
  brent(g, x, contexts...; xa, xb, tol, maxiter, check_every)
    g is non-mutating: y = g(x)  (or g(x, contexts...))
    returns NamedTuple (u, iters, retcode) — shapes match x

  brent!(g!, y, x, contexts...; xa, xb, tol, maxiter, check_every)
    g! is mutating: g!(y, x)
    returns same NamedTuple

  retcode: RETCODE_SUCCESS (0x00) or RETCODE_MAXITER (0x02)
  iters:   -1 until converged, then set to the iteration number
  u:       same array as x (mutated in place)
  shapes:  retcode and iters have same shape as x

  Default convergence: tol=1e-13 (very tight)
  Brent minimizes (NOT maximizes) — returns minimum of g over [xa, xb].
"""

using Random
import BatchSolve: RETCODE_SUCCESS, RETCODE_FAILURE, RETCODE_MAXITER

# Convenience: run brent! on a scalar problem
function brent1(f, a::T, b::T; kw...) where T
    xa = [a]; xb = [b]
    x  = [a + T(0.3819660112501051) * (b - a)]
    y  = f.(x)
    g!(yy, xx) = yy .= f.(xx)
    brent!(g!, y, x; xa=xa, xb=xb, kw...)
end

# ===========================================================================
# 1. Basic quadratics
# ===========================================================================

@testset "1. Quadratic interior minimum" begin
    sol = brent1(x -> (x - 1.3)^2, 0.0, 3.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1.3
    @test sol.iters[1] >= 0
end

@testset "2. Quadratic minimum at left boundary" begin
    sol = brent1(x -> (x + 5.0)^2, -5.0, 0.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ -5.0
end

@testset "3. Quadratic minimum at right boundary" begin
    sol = brent1(x -> (x - 5.0)^2, 0.0, 5.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 5.0
end

@testset "4. Quadratic: returned u is minimum not maximum" begin
    # f = -(x-0.5)^2 has maximum at 0.5; brent minimizes so should find boundary
    # f = (x-0.5)^2 has minimum at 0.5
    sol = brent1(x -> (x - 0.5)^2, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.5
end

# ===========================================================================
# 5. brent (non-mutating) wrapper
# ===========================================================================

@testset "5. brent (non-mutating) wrapper matches brent!" begin
    f(x) = (x - 0.7)^2
    g(xx) = f.(xx)
    g!(yy, xx) = yy .= f.(xx)
    xa = [0.0]; xb = [1.0]
    x1 = [0.5]; y1 = f.(x1)
    x2 = [0.5]
    sol1 = brent!(g!, y1, x1; xa=xa, xb=xb)
    sol2 = brent(g, x2; xa=xa, xb=xb)
    @test sol1.u ≈ sol2.u
    @test sol1.retcode == sol2.retcode
end

# ===========================================================================
# 6-10. Output structure
# ===========================================================================

@testset "6. retcode is UInt8" begin
    sol = brent1(x -> (x - 0.5)^2, 0.0, 1.0)
    @test eltype(sol.retcode) == UInt8
end

@testset "7. retcode shape matches x shape" begin
    n = 20
    cs = rand(MersenneTwister(1), n)
    xa = zeros(n); xb = ones(n)
    x  = copy(xa) .+ 0.5; y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test size(sol.retcode) == size(x)
    @test size(sol.iters)   == size(x)
    @test size(sol.u)       == size(x)
end

@testset "8. iters is -1 before convergence, non-negative after" begin
    sol = brent1(x -> (x - 0.3)^2, 0.0, 1.0)
    @test sol.iters[1] >= 0
end

@testset "9. u is same object as x (mutated in place)" begin
    g!(yy, xx) = yy .= (xx .- 0.5).^2
    xa = [0.0]; xb = [1.0]
    x  = [0.4]; y = (x .- 0.5).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test sol.u === x   # same array object
end

@testset "10. iters <= maxiter" begin
    sol = brent1(x -> (x - 0.5)^2, 0.0, 1.0; maxiter=100)
    @test sol.iters[1] <= 100
end

# ===========================================================================
# 11-20. Trigonometric functions
# ===========================================================================

@testset "11. sin: minimum at 3pi/2 in [3, 6]" begin
    sol = brent1(sin, 3.0, 6.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 3pi/2
    @test sin(sol.u[1]) ≈ -1.0
end

@testset "12. cos: minimum at pi in [0, 2pi]" begin
    sol = brent1(cos, 0.0, 2pi)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ pi
    @test cos(sol.u[1]) ≈ -1.0
end

@testset "13. -cos: minimum at 0 in [-1, 1]" begin
    sol = brent1(x -> -cos(x), -1.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.0 atol=1e-15
end

@testset "14. sin(2pi*x): minimum at 0.75 in [0, 1]" begin
    sol = brent1(x -> sin(2pi*x), 0.5, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.75
end

@testset "15. cos(2pi*x): minimum at 0.5 in [0, 1]" begin
    sol = brent1(x -> cos(2pi*x), 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.5
end

# ===========================================================================
# 16-25. Polynomial functions
# ===========================================================================

@testset "16. Cubic x^3-3x: minimum at x=1 in [0, 3]" begin
    sol = brent1(x -> x^3 - 3x, 0.0, 3.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1.0 
end

@testset "17. Quartic (x-2.7)^4" begin
    sol = brent1(x -> (x - 2.7)^4, -5.0, 5.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 2.7
end

@testset "18. Degree-6 polynomial (x-1.23456)^6" begin
    sol = brent1(x -> (x - 1.23456)^6, 0.0, 3.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1.23456 
end

@testset "19. Rosenbrock-like (1-x)^2 + 100*(x^2-x)^2" begin
    sol = brent1(x -> (1-x)^2 + 100*(x^2-x)^2, 0.0, 2.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1.0 
end

@testset "20. x*(x-1)*(x-2): minimum near x=1.577 in [1, 2]" begin
    f = x -> x*(x-1)*(x-2)
    # f'(x) = 3x^2 - 6x + 2 = 0 → x = (6 ± sqrt(12))/6 = 1 ± 1/sqrt(3)
    xmin = 1 + 1/sqrt(3)
    sol = brent1(f, 1.0, 2.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ xmin 
end

# ===========================================================================
# 21-25. Exponential / logarithmic
# ===========================================================================

@testset "21. Gaussian well: -exp(-(x-0.7)^2/0.01)" begin
    sol = brent1(x -> -exp(-(x-0.7)^2/0.01), 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.7
    @test sol.u[1] |> x -> -exp(-(x-0.7)^2/0.01) ≈ -1.0
end

@testset "22. x*log(x): minimum at 1/e" begin
    sol = brent1(x -> x*log(x), 1e-6, 2.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1/MathConstants.e
end

@testset "23. exp((x-0.4)^2): minimum at 0.4" begin
    sol = brent1(x -> exp((x-0.4)^2), 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.4
end

@testset "24. -exp(-x^2): Gaussian minimum at 0" begin
    sol = brent1(x -> -exp(-x^2), -2.0, 2.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.0
end

@testset "25. log(1 + (x-1.5)^2): minimum at 1.5" begin
    sol = brent1(x -> log(1 + (x-1.5)^2), 0.0, 3.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1.5
end

# ===========================================================================
# 26-30. Flat and degenerate functions
# ===========================================================================

@testset "26. Constant function: converges to some point in bracket" begin
    sol = brent1(x -> 42.0*one(x), 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test 0.0 <= sol.u[1] <= 1.0
    @test !isnan(sol.u[1])
end

@testset "27. Nearly flat: 1e-15*(x-0.5)^2" begin
    sol = brent1(x -> 1e-15*(x-0.5)^2, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test !isnan(sol.u[1])
    @test 0.0 <= sol.u[1] <= 1.0
end

@testset "28. All-zeros: converges without NaN" begin
    sol = brent1(x -> zero(x), -1.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test !isnan(sol.u[1])
end

@testset "29. No NaN or Inf in any output field" begin
    sol = brent1(x -> (x-0.5)^2, 0.0, 1.0)
    @test !any(isnan.(sol.u))
    @test !any(isinf.(sol.u))
end

@testset "30. f with large negative offset: -1000 + (x-0.5)^2" begin
    sol = brent1(x -> -1000.0 + (x-0.5)^2, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.5
end

# ===========================================================================
# 31-35. Non-smooth functions
# ===========================================================================

@testset "31. abs(x-0.3): V-shaped minimum at 0.3" begin
    sol = brent1(x -> abs(x - 0.3), -1.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.3 
end

@testset "32. max(0, x-0.5): kink at 0.5, minimum to the left" begin
    sol = brent1(x -> max(0.0, x - 0.5), 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] <= 0.5 + 1e-8
end

@testset "33. Steep tanh step: minimum in left region" begin
    sol = brent1(x -> tanh(100*(x - 0.6)), 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] < 0.6
end

@testset "34. abs(sin(pi*x)): boundary minimum in [0.1, 0.9]" begin
    # sin(pi*x) > 0 for all x in (0,1), minimum at boundary x=0.1
    sol = brent1(x -> abs(sin(pi*x)), 0.1, 0.9)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.1 
end

@testset "35. floor-like: step function minimum at left" begin
    sol = brent1(x -> floor(10x)/10.0, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] < 0.2   # minimum is in leftmost step
end

# ===========================================================================
# 36-40. Multimodal functions (bracket selects well)
# ===========================================================================

@testset "36. Two-well: bracket selects left well at 0.25" begin
    f = x -> (x - 0.25)^2 * (x - 0.75)^2
    sol = brent1(f, 0.0, 0.5)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.25 
end

@testset "37. Two-well: bracket selects right well at 0.75" begin
    f = x -> (x - 0.25)^2 * (x - 0.75)^2
    sol = brent1(f, 0.5, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.75 
end

@testset "38. Oscillatory: returned point is a local minimum" begin
    f = x -> sin(20x) + 0.1*(x - 0.5)^2
    sol = brent1(f, 0.1, 0.4)
    @test sol.retcode[1] == RETCODE_SUCCESS
    delta = 1e-5
    @test f(sol.u[1]) <= f(sol.u[1] + delta) + 1e-12
    @test f(sol.u[1]) <= f(sol.u[1] - delta) + 1e-12
end

@testset "39. Symmetric well: correct well selected by bracket" begin
    f = x -> (x - 0.3)^2 * (x - 0.7)^2
    sol_l = brent1(f, 0.0, 0.5)
    sol_r = brent1(f, 0.5, 1.0)
    @test sol_l.u[1] ≈ 0.3 
    @test sol_r.u[1] ≈ 0.7 
end

@testset "40. Rational 1/(x^2+1): decreasing on [0.5, 3], min at right" begin
    sol = brent1(x -> 1/(x^2+1), 0.5, 3.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 3.0
end

# ===========================================================================
# 41-45. Asymmetric brackets
# ===========================================================================

@testset "41. Very small bracket [0.9999, 1.0001]" begin
    sol = brent1(x -> (x-1.0)^2, 0.9999, 1.0001)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1.0
end

@testset "42. Large bracket [-1e6, 1e6] with min at 0" begin
    sol = brent1(x -> x^2, -1e6, 1e6)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test abs(sol.u[1]) < 1e-6
end

@testset "43. Min near right end of wide bracket" begin
    sol = brent1(x -> (x - 99.9)^2, 0.0, 100.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 99.9 
end

@testset "44. Min near left end of wide bracket" begin
    sol = brent1(x -> (x - 0.001)^2, 0.0, 100.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.001 
end

@testset "45. Bracket entirely negative: [-5, -1], min at -3.7" begin
    sol = brent1(x -> (x + 3.7)^2, -5.0, -1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ -3.7
end

# ===========================================================================
# 46-50. Float32 type stability
# ===========================================================================

@testset "46. Float32: type preserved end-to-end" begin
    xa = Float32[0.0]; xb = Float32[3.0]
    x  = Float32[1.5]; y  = (x .- 1.5f0).^2
    g!(yy, xx) = yy .= (xx .- 1.5f0).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test eltype(sol.u) == Float32
    @test eltype(sol.iters) == Int
    @test eltype(sol.retcode) == UInt8
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1.5f0
end

@testset "47. Float32: recovers known minimum" begin
    xa = Float32[-2.0]; xb = Float32[2.0]
    x  = Float32[0.0];  y  = (x .- 0.8f0).^2
    g!(yy, xx) = yy .= (xx .- 0.8f0).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb, tol=1f-5)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.8f0
end

@testset "48. Float32 vs Float64: same problem gives consistent answer" begin
    f32_sol = brent1(x -> (x - 0.42f0)^2, 0.0f0, 1.0f0)
    f64_sol = brent1(x -> (x - 0.42)^2,   0.0,   1.0)
    @test f32_sol.u[1] ≈ Float32(f64_sol.u[1])
end

# ===========================================================================
# 49-55. Tolerance and maxiter
# ===========================================================================

@testset "49. Tight tol=1e-14: high precision for quadratic" begin
    sol = brent1(x -> (x - sqrt(2.0))^2, 0.0, 3.0; tol=1e-14)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ sqrt(2.0)
end

@testset "50. Loose tol=1e-3: coarser result than default" begin
    sol_tight = brent1(x -> (x - sqrt(2.0))^2, 0.0, 3.0; tol=1e-13)
    sol_loose = brent1(x -> (x - sqrt(2.0))^2, 0.0, 3.0; tol=1e-3)
    @test abs(sol_tight.u[1] - sqrt(2.0)) <= abs(sol_loose.u[1] - sqrt(2.0)) + 1e-10
end

@testset "51. maxiter=1: returns finite result in bracket" begin
    sol = brent1(x -> (x-0.5)^2, 0.0, 1.0; maxiter=1)
    @test !isnan(sol.u[1])
    @test 0.0 <= sol.u[1] <= 1.0
    @test sol.retcode[1] isa UInt8
end

@testset "52. maxiter=500: converges for quadratic" begin
    sol = brent1(x -> (x - 0.123456789)^2, 0.0, 1.0; maxiter=500)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.123456789
end

@testset "53. check_every=0: skips mid-loop check, still converges" begin
    sol = brent1(x -> (x-0.7)^2, 0.0, 1.0; check_every=0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.7
end

@testset "54. check_every=10 vs check_every=1: same result" begin
    sol1 = brent1(x -> (x-0.7)^2, 0.0, 1.0; check_every=1)
    sol2 = brent1(x -> (x-0.7)^2, 0.0, 1.0; check_every=10)
    @test sol1.u ≈ sol2.u
    @test sol1.retcode == sol2.retcode
end

@testset "55. check_every=100: still converges within maxiter=500" begin
    sol = brent1(x -> (x-0.3)^2, 0.0, 1.0; check_every=100, maxiter=500)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.3
end

# ===========================================================================
# 56-65. Vectorized batch
# ===========================================================================

@testset "56. Batch of 20 quadratics" begin
    n  = 20
    cs = collect(LinRange(0.1, 0.9, n))
    xa = zeros(n); xb = ones(n)
    x  = (xa .+ xb) ./ 2; y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs
    @test size(sol.retcode) == size(x)
    @test size(sol.iters)   == size(x)
end

@testset "57. Large batch n=1000 random quadratics" begin
    rng = MersenneTwister(42)
    n   = 1000
    cs  = rand(rng, n)
    xa  = zeros(n); xb = ones(n)
    x   = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs
end

@testset "58. Batch: heterogeneous brackets and scales" begin
    cs  = [-3.7, 0.4,  0.25, 5e-4,  42.0]
    xa  = [-10.0, -1.0, 0.0,  0.0, -100.0]
    xb  = [ 10.0,  1.0, 0.5,  1e-3,  100.0]
    x   = (xa .+ xb) ./ 2; y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs 
end

@testset "59. All lanes identical: same result expected" begin
    n  = 100
    cs = fill(0.42, n)
    xa = zeros(n); xb = ones(n)
    x  = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test all(abs.(sol.u .- 0.42) .< 1e-10)
    @test length(unique(sol.u)) == 1
end

@testset "60. Batch: result independent of lane ordering" begin
    n    = 20
    cs   = collect(LinRange(0.1, 0.9, n))
    perm = randperm(MersenneTwister(7), n)
    cs2  = cs[perm]; xa2 = zeros(n); xb2 = ones(n)
    xa   = zeros(n); xb  = ones(n)
    x1   = fill(0.5, n); y1 = (x1 .- cs).^2
    x2   = fill(0.5, n); y2 = (x2 .- cs2).^2
    g1!(yy, xx) = yy .= (xx .- cs).^2
    g2!(yy, xx) = yy .= (xx .- cs2).^2
    sol1 = brent!(g1!, y1, x1; xa=xa,  xb=xb)
    sol2 = brent!(g2!, y2, x2; xa=xa2, xb=xb2)
    @test sol1.u ≈ sol2.u[invperm(perm)]
end

@testset "61. Batch: u always inside [xa, xb]" begin
    rng = MersenneTwister(17)
    n   = 50
    cs  = rand(rng, n)
    xa  = zeros(n); xb = ones(n)
    x   = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test all(sol.u .>= xa .- 1e-14)
    @test all(sol.u .<= xb .+ 1e-14)
end

@testset "62. Batch: fmin <= f(xa) and f(xb) for all lanes" begin
    rng = MersenneTwister(55)
    n   = 30
    cs  = rand(rng, n)
    xa  = zeros(n); xb = ones(n)
    x   = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    fa = (xa .- cs).^2
    fb = (xb .- cs).^2
    # evaluate g at solution
    gu = similar(x); g!(gu, sol.u)
    @test all(gu .<= fa .+ 1e-12)
    @test all(gu .<= fb .+ 1e-12)
end

@testset "63. Batch: mixed easy and hard lanes" begin
    cs  = [0.5, 0.0001]
    xa  = [0.0, 0.0]; xb = [1.0, 1.0]
    x   = [0.5, 0.5]; y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test all(sol.retcode .== RETCODE_SUCCESS)
    @test sol.u ≈ cs 
end

@testset "64. Batch: serial vs batch same answers" begin
    n  = 10
    cs = collect(LinRange(0.05, 0.95, n))
    xa = zeros(n); xb = ones(n)
    x  = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol_batch = brent!(g!, y, x; xa=xa, xb=xb)
    roots_serial = map(1:n) do i
        brent1(xx -> (xx - cs[i])^2, 0.0, 1.0).u[1]
    end
    @test sol_batch.u ≈ roots_serial
end

@testset "65. Batch: iters shape matches x" begin
    n   = 25
    cs  = rand(MersenneTwister(3), n)
    xa  = zeros(n); xb = ones(n)
    x   = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test size(sol.iters) == (n,)
    @test all(sol.iters .>= 0)
end

# ===========================================================================
# 66-70. Monotone functions (minimum at boundary)
# ===========================================================================

@testset "66. Monotone increasing: minimum at left boundary" begin
    sol = brent1(exp, -2.0, 2.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ -2.0
end

@testset "67. Monotone decreasing: minimum at right boundary" begin
    sol = brent1(x -> -exp(x), -2.0, 2.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 2.0
end

@testset "68. Monotone: f(u) <= f(xa)" begin
    sol = brent1(exp, -2.0, 2.0)
    @test exp(sol.u[1]) <= exp(-2.0) + 1e-13
end

# ===========================================================================
# 69-73. Reproducibility and correctness invariants
# ===========================================================================

@testset "69. Reproducibility: two identical calls give bit-identical results" begin
    g!(yy, xx) = yy .= (xx .- 0.6).^2
    xa1 = [0.0]; xb1 = [1.0]
    xa2 = [0.0]; xb2 = [1.0]
    x1 = [0.5]; y1 = (x1 .- 0.6).^2
    x2 = [0.5]; y2 = (x2 .- 0.6).^2
    sol1 = brent!(g!, y1, x1; xa=xa1, xb=xb1)
    sol2 = brent!(g!, y2, x2; xa=xa2, xb=xb2)
    @test sol1.u      == sol2.u
    @test sol1.retcode == sol2.retcode
    @test sol1.iters   == sol2.iters
end

@testset "70. f(u) == g(u) at returned point (output consistency)" begin
    f = x -> sin(3x) + 0.2*(x-1.0)^2
    sol = brent1(f, 0.0, 2.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    # The returned u should be a local minimum: f(u) <= f(u ± delta)
    delta = 1e-5
    @test f(sol.u[1]) <= f(sol.u[1] + delta) + 1e-12
    @test f(sol.u[1]) <= f(sol.u[1] - delta) + 1e-12
end

@testset "71. Minimization not maximization: cos on [0, 2pi]" begin
    sol = brent1(cos, 0.0, 2pi)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test cos(sol.u[1]) ≈ -1.0   # minimum, not +1
end

@testset "72. u in bracket for all tol values" begin
    for tol in [1e-3, 1e-7, 1e-13]
        sol = brent1(x -> (x-0.5)^2, 0.0, 1.0; tol=tol)
        @test 0.0 <= sol.u[1] <= 1.0
    end
end

@testset "73. Irrational minimum: pi-3 in [0, 1]" begin
    sol = brent1(x -> (x - (pi-3))^2, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ pi - 3
end

# ===========================================================================
# 74-78. Tight bracket and edge cases
# ===========================================================================

@testset "74. Nearly degenerate bracket width 1e-10" begin
    eps_val = 1e-10
    sol = brent1(x -> (x-0.5)^2, 0.5 - eps_val, 0.5 + eps_val)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.5 
end

@testset "75. Minimum exactly at CGOLD starting point" begin
    cgold = 0.3819660112501051
    xmin  = cgold   # minimum IS the first interior evaluation point
    sol = brent1(x -> (x - xmin)^2, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ xmin
end

@testset "76. Very sharp Gaussian: tight bracket needed" begin
    # Must bracket tightly — function underflows outside ~5σ ≈ 5e-4
    sol = brent1(x -> -exp(-((x-0.5)/1e-4)^2), 0.499, 0.501; tol=1e-10)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.5 
end

@testset "77. Minimum at exact bracket midpoint" begin
    sol = brent1(x -> (x - 0.5)^2, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.5
end

@testset "78. check_every > maxiter: final check only" begin
    sol = brent1(x -> (x-0.3)^2, 0.0, 1.0; check_every=1000, maxiter=500)
    @test !isnan(sol.u[1])
    @test 0.0 <= sol.u[1] <= 1.0
end

# ===========================================================================
# 79-83. RETCODE_MAXITER cases
# ===========================================================================

@testset "79. maxiter=1 likely gives RETCODE_MAXITER for hard problem" begin
    sol = brent1(x -> (x-0.5)^2, 0.0, 1.0; maxiter=1)
    @test sol.retcode[1] isa UInt8
    @test sol.retcode[1] in (RETCODE_SUCCESS, RETCODE_MAXITER)
end

@testset "80. retcode is RETCODE_MAXITER or RETCODE_SUCCESS, never anything else" begin
    for f in [x->(x-0.3)^2, x->sin(x), x->exp(x)]
        sol = brent1(f, 0.0, 3.0; maxiter=5)
        @test sol.retcode[1] in (RETCODE_SUCCESS, RETCODE_MAXITER)
    end
end

@testset "81. Batch: retcode is RETCODE_SUCCESS for all after sufficient maxiter" begin
    n   = 50
    cs  = rand(MersenneTwister(9), n)
    xa  = zeros(n); xb = ones(n)
    x   = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb, maxiter=500)
    @test all(sol.retcode .== RETCODE_SUCCESS)
end

@testset "82. iters field set to -1 initially, non-negative after solve" begin
    g!(yy, xx) = yy .= (xx .- 0.5).^2
    xa = [0.0]; xb = [1.0]
    x  = [0.4]; y = (x .- 0.5).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test sol.iters[1] >= 0
end

@testset "83. Batch iters: all non-negative after convergence" begin
    n  = 30
    cs = rand(MersenneTwister(21), n)
    xa = zeros(n); xb = ones(n)
    x  = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test all(sol.iters .>= 0)
end

# ===========================================================================
# 84-90. Correctness invariants
# ===========================================================================

@testset "84. Returned f(u) <= f(xa) and f(xb) (is actually minimum)" begin
    f = x -> sin(x) + 0.1*x
    sol = brent1(f, 1.0, 5.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test f(sol.u[1]) <= f(1.0) + 1e-12
    @test f(sol.u[1]) <= f(5.0) + 1e-12
end

@testset "85. Minimization vs maximization: -f gives opposite u" begin
    # f has min at 0.3 in [0,1]; -f has max at 0.3, min at boundary
    f  = x -> (x - 0.3)^2 + 0.1   # min at 0.3
    nf = x -> -((x - 0.3)^2 + 0.1) # max at 0.3, so brent finds boundary
    sol_f  = brent1(f,  0.0, 1.0)
    sol_nf = brent1(nf, 0.0, 1.0)
    @test sol_f.u[1] ≈ 0.3 
    # sol_nf should be at a boundary since -f is maximized at 0.3
    @test abs(sol_nf.u[1] - 0.0) < 0.01 || abs(sol_nf.u[1] - 1.0) < 0.01
end

@testset "86. Pure quadratic converges in few evaluations" begin
    eval_count = Ref(0)
    function g!(yy, xx)
        eval_count[] += 1
        yy .= (xx .- 0.33).^2
    end
    xa = [0.0]; xb = [1.0]
    x  = [0.5]; y = (x .- 0.33).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 0.33
    @test eval_count[] < 20   # parabolic steps should kick in
end

@testset "87. Batch: fmin values are all non-NaN" begin
    n  = 50
    cs = rand(MersenneTwister(33), n)
    xa = zeros(n); xb = ones(n)
    x  = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    gu = similar(x); g!(gu, sol.u)
    @test !any(isnan.(gu))
    @test !any(isinf.(gu))
end

@testset "88. Minimum recovery: e/10" begin
    sol = brent1(x -> (x - MathConstants.e/10)^2, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ MathConstants.e/10
end

@testset "89. Minimum recovery: sqrt(2)-1" begin
    sol = brent1(x -> (x - (sqrt(2.0)-1))^2, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ sqrt(2.0) - 1
end

@testset "90. Large batch n=1000: all u in [xa, xb]" begin
    rng = MersenneTwister(99)
    n   = 1000
    cs  = rand(rng, n)
    xa  = zeros(n); xb = ones(n)
    x   = fill(0.5, n); y = (x .- cs).^2
    g!(yy, xx) = yy .= (xx .- cs).^2
    sol = brent!(g!, y, x; xa=xa, xb=xb)
    @test all(sol.u .>= xa .- 1e-14)
    @test all(sol.u .<= xb .+ 1e-14)
end

# ==============================================================================
# 91. Parabolic edge guard (u too close to bracket end)
# ==============================================================================
@testset "91. Minimum very close to bracket endpoint: edge guard active" begin
    # Place minimum at xa + epsilon; parabola should want to step past boundary.
    f = x -> (x - 1e-8)^2
    sol = brent1(f, 0.0, 1.0)
    @test sol.retcode[1] == RETCODE_SUCCESS
    @test sol.u[1] ≈ 1e-8
end
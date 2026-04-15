"""
Test suite for AutoBatch{<:AutoFiniteDiff} Jacobian bindings (batch-ad/finitediff.jl).

Covers:
  - batchdim ∈ {1, 2}
  - fdmode  ∈ {Val(:forward), Val(:central)}
  - square and non-square (tall ny>nx, wide ny<nx) Jacobians
  - all 4 DI interface variants: jacobian, jacobian!, value_and_jacobian, value_and_jacobian!
  - one-arg (f) and two-arg (f!) signatures
  - scalar Constant contexts
  - Float32 type preservation
  - edge cases: batch size 1, large batch, single element per lane
  - permutation invariance

Correctness is verified by comparing with AutoBatch(AutoForwardDiff()) and, where simple,
against analytical Jacobians.
"""

using Test
using JET
using Random
using SparseArrays
using LinearAlgebra
using BatchSolve
import DifferentiationInterface as DI
using ADTypes: AutoForwardDiff, AutoFiniteDiff

# ============================================================================
# Helpers
# ============================================================================

# Extract per-lane (ny×nx) Jacobian from the full batch sparse Jacobian.
#
# batchdim=2 (block-diagonal): lane i occupies rows (i-1)*ny+1:i*ny, cols (i-1)*nx+1:i*nx
# batchdim=1 (banded):         lane i occupies rows {(r-1)*nlanes+i : r=1..ny},
#                                               cols {(c-1)*nlanes+i : c=1..nx}
function lane_jac(jac_full, lane, nlanes, ny, nx, batchdim)
    dense = Matrix(jac_full)
    if batchdim == 2
        return dense[(lane-1)*ny+1 : lane*ny, (lane-1)*nx+1 : lane*nx]
    else
        rows = [(r-1)*nlanes + lane for r in 1:ny]
        cols = [(c-1)*nlanes + lane for c in 1:nx]
        return dense[rows, cols]
    end
end

make_fd_fwd(bd) = AutoBatch(AutoFiniteDiff();                    batchdim=bd)
make_fd_cen(bd) = AutoBatch(AutoFiniteDiff(fdjtype=Val(:central)); batchdim=bd)
make_fad(bd)    = AutoBatch(AutoForwardDiff();                   batchdim=bd)

const ATOL_FWD = 1e-5   # forward-difference truncation error ~O(h),  h≈sqrt(eps)≈1.5e-8
const ATOL_CEN = 1e-8   # central-difference truncation error ~O(h²), h≈cbrt(eps)≈6e-6

# Quick check: do two sparse Jacobians (one from FD, one reference) agree?
jac_ok(jfd, jref; atol) = isapprox(Matrix(jfd), Matrix(jref); atol=atol)

# ============================================================================
# Section 1: Square Jacobians, batchdim=1
# ============================================================================

@testset "Square Jac batchdim=1 forward" begin

    @testset "Identity f(x)=x, nx=3, nlanes=5" begin
        nlanes, nx = 5, 3
        x  = randn(MersenneTwister(1), nlanes, nx)
        f(x) = copy(x)
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            @test lane_jac(Jfd, lane, nlanes, nx, nx, 1) ≈ I(nx)  atol=ATOL_FWD
        end
    end

    @testset "Elementwise sin, nx=4, nlanes=8" begin
        nlanes, nx = 8, 4
        x  = randn(MersenneTwister(2), nlanes, nx)
        f(x) = sin.(x)
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 1)
            @test diag(J) ≈ cos.(x[lane, :])  atol=ATOL_FWD
            @test norm(J - Diagonal(cos.(x[lane,:]))) < ATOL_FWD * nx
        end
    end

    @testset "Linear f(x)=x*A', nx=3, nlanes=4" begin
        nlanes, nx = 4, 3
        A = rand(MersenneTwister(3), nx, nx)
        x = randn(MersenneTwister(3), nlanes, nx)
        f(x) = x * A'
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            @test lane_jac(Jfd, lane, nlanes, nx, nx, 1) ≈ A  atol=ATOL_FWD
        end
    end

    @testset "Quadratic f(x)=x.^2, nx=5, nlanes=6" begin
        nlanes, nx = 6, 5
        x  = randn(MersenneTwister(4), nlanes, nx) .+ 1.0
        f(x) = x .^ 2
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 1)
            @test diag(J) ≈ 2 .* x[lane, :]  atol=ATOL_FWD
        end
    end

    @testset "Mixed nonlinear [sin, exp, x^3], nx=3, nlanes=7" begin
        nlanes, nx = 7, 3
        x  = randn(MersenneTwister(5), nlanes, nx) .+ 0.5
        f(x) = hcat(sin.(x[:,1:1]), exp.(x[:,2:2]), x[:,3:3].^3)
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end
end

@testset "Square Jac batchdim=1 central" begin

    @testset "Elementwise sin, nx=4, nlanes=8" begin
        nlanes, nx = 8, 4
        x  = randn(MersenneTwister(10), nlanes, nx)
        f(x) = sin.(x)
        pfd = DI.prepare_jacobian(f, make_fd_cen(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_cen(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 1)
            @test diag(J) ≈ cos.(x[lane, :])  atol=ATOL_CEN
        end
    end

    @testset "Cubic f(x)=x.^3, nx=3, nlanes=10" begin
        nlanes, nx = 10, 3
        x  = randn(MersenneTwister(11), nlanes, nx) .+ 0.5
        f(x) = x .^ 3
        pfd = DI.prepare_jacobian(f, make_fd_cen(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_cen(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 1)
            @test diag(J) ≈ 3 .* x[lane, :].^2  atol=ATOL_CEN
        end
    end

    @testset "Central more accurate than forward on x^3, batchdim=1" begin
        nlanes, nx = 4, 3
        x = randn(MersenneTwister(12), nlanes, nx) .+ 1.0
        f(x) = x .^ 3
        pfwd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pcen = DI.prepare_jacobian(f, make_fd_cen(1), x)
        pfa  = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfwd = DI.jacobian(f, pfwd, make_fd_fwd(1), x)
        Jcen = DI.jacobian(f, pcen, make_fd_cen(1), x)
        Jfa  = DI.jacobian(f, pfa,  make_fad(1),    x)
        err_fwd = norm(Matrix(Jfwd) .- Matrix(Jfa))
        err_cen = norm(Matrix(Jcen) .- Matrix(Jfa))
        @test err_cen < err_fwd
    end
end

# ============================================================================
# Section 2: Square Jacobians, batchdim=2
# ============================================================================

@testset "Square Jac batchdim=2 forward" begin

    @testset "Elementwise sin, nx=4, nlanes=8" begin
        nx, nlanes = 4, 8
        x  = randn(MersenneTwister(20), nx, nlanes)
        f(x) = sin.(x)
        pfd = DI.prepare_jacobian(f, make_fd_fwd(2), x)
        pfa = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(2), x)
        Jfa = DI.jacobian(f, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 2)
            @test diag(J) ≈ cos.(x[:, lane])  atol=ATOL_FWD
        end
    end

    @testset "Linear f(x)=A*x, nx=3, nlanes=5" begin
        nx, nlanes = 3, 5
        A = rand(MersenneTwister(21), nx, nx)
        x = randn(MersenneTwister(21), nx, nlanes)
        f(x) = A * x
        pfd = DI.prepare_jacobian(f, make_fd_fwd(2), x)
        pfa = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(2), x)
        Jfa = DI.jacobian(f, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            @test lane_jac(Jfd, lane, nlanes, nx, nx, 2) ≈ A  atol=ATOL_FWD
        end
    end

    @testset "Quadratic f(x)=x.^2, nx=5, nlanes=6" begin
        nx, nlanes = 5, 6
        x  = randn(MersenneTwister(22), nx, nlanes) .+ 1.0
        f(x) = x .^ 2
        pfd = DI.prepare_jacobian(f, make_fd_fwd(2), x)
        pfa = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(2), x)
        Jfa = DI.jacobian(f, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 2)
            @test diag(J) ≈ 2 .* x[:, lane]  atol=ATOL_FWD
        end
    end
end

@testset "Square Jac batchdim=2 central" begin

    @testset "Cubic f(x)=x.^3, nx=3, nlanes=10" begin
        nx, nlanes = 3, 10
        x  = randn(MersenneTwister(30), nx, nlanes) .+ 0.5
        f(x) = x .^ 3
        pfd = DI.prepare_jacobian(f, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(f, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 2)
            @test diag(J) ≈ 3 .* x[:, lane].^2  atol=ATOL_CEN
        end
    end

    @testset "Elementwise exp, nx=4, nlanes=6" begin
        nx, nlanes = 4, 6
        x  = randn(MersenneTwister(31), nx, nlanes) .* 0.5
        f(x) = exp.(x)
        pfd = DI.prepare_jacobian(f, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(f, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
    end

    @testset "batchdim=1 and batchdim=2 give same per-lane Jacobians (forward)" begin
        nlanes, nx = 6, 4
        x1 = randn(MersenneTwister(32), nlanes, nx)
        x2 = copy(x1')
        f(x) = sin.(x)
        pfd1 = DI.prepare_jacobian(f, make_fd_fwd(1), x1)
        pfd2 = DI.prepare_jacobian(f, make_fd_fwd(2), x2)
        Jfd1 = DI.jacobian(f, pfd1, make_fd_fwd(1), x1)
        Jfd2 = DI.jacobian(f, pfd2, make_fd_fwd(2), x2)
        for lane in 1:nlanes
            J1 = lane_jac(Jfd1, lane, nlanes, nx, nx, 1)
            J2 = lane_jac(Jfd2, lane, nlanes, nx, nx, 2)
            @test J1 ≈ J2  atol=ATOL_FWD
        end
    end

    @testset "Central more accurate than forward on x^3, batchdim=2" begin
        nx, nlanes = 3, 4
        x  = randn(MersenneTwister(33), nx, nlanes) .+ 1.0
        f(x) = x .^ 3
        pfwd = DI.prepare_jacobian(f, make_fd_fwd(2), x)
        pcen = DI.prepare_jacobian(f, make_fd_cen(2), x)
        pfa  = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfwd = DI.jacobian(f, pfwd, make_fd_fwd(2), x)
        Jcen = DI.jacobian(f, pcen, make_fd_cen(2), x)
        Jfa  = DI.jacobian(f, pfa,  make_fad(2),    x)
        @test norm(Matrix(Jcen) .- Matrix(Jfa)) < norm(Matrix(Jfwd) .- Matrix(Jfa))
    end
end

# ============================================================================
# Section 3: Non-square tall Jacobians (ny > nx)
# ============================================================================
#
# Per-lane test function: f: R^2 → R^3
#   f([x1,x2]) = [x1+x2,  x1*x2,  x1^2]
#   J = [1    1  ]
#       [x2   x1 ]
#       [2x1  0  ]
#

@testset "Tall Jac ny>nx" begin

    @testset "ny=3, nx=2, nlanes=5, batchdim=1, forward" begin
        nlanes, nx, ny = 5, 2, 3
        x  = randn(MersenneTwister(40), nlanes, nx) .+ 1.0
        function ftall1(x)
            out = similar(x, size(x,1), 3)
            out[:,1] .= x[:,1] .+ x[:,2]
            out[:,2] .= x[:,1] .* x[:,2]
            out[:,3] .= x[:,1].^2
            return out
        end
        pfd = DI.prepare_jacobian(ftall1, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(ftall1, make_fad(1),    x)
        Jfd = DI.jacobian(ftall1, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(ftall1, pfa, make_fad(1),    x)
        @test size(Jfd, 1) == nlanes * ny
        @test size(Jfd, 2) == nlanes * nx
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            x1, x2 = x[lane,1], x[lane,2]
            J_expected = [1.0   1.0;
                          x2    x1;
                          2x1   0.0]
            @test lane_jac(Jfd, lane, nlanes, ny, nx, 1) ≈ J_expected  atol=ATOL_FWD
        end
    end

    @testset "ny=3, nx=2, nlanes=5, batchdim=2, forward" begin
        nx, ny, nlanes = 2, 3, 5
        x  = randn(MersenneTwister(41), nx, nlanes) .+ 1.0
        function ftall2(x)
            out = similar(x, 3, size(x,2))
            out[1,:] .= x[1,:] .+ x[2,:]
            out[2,:] .= x[1,:] .* x[2,:]
            out[3,:] .= x[1,:].^2
            return out
        end
        pfd = DI.prepare_jacobian(ftall2, make_fd_fwd(2), x)
        pfa = DI.prepare_jacobian(ftall2, make_fad(2),    x)
        Jfd = DI.jacobian(ftall2, pfd, make_fd_fwd(2), x)
        Jfa = DI.jacobian(ftall2, pfa, make_fad(2),    x)
        @test size(Jfd, 1) == nlanes * ny
        @test size(Jfd, 2) == nlanes * nx
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            x1, x2 = x[1,lane], x[2,lane]
            J_expected = [1.0   1.0;
                          x2    x1;
                          2x1   0.0]
            @test lane_jac(Jfd, lane, nlanes, ny, nx, 2) ≈ J_expected  atol=ATOL_FWD
        end
    end

    @testset "ny=5, nx=2, nlanes=7, batchdim=1, central" begin
        nlanes, nx, ny = 7, 2, 5
        x  = randn(MersenneTwister(42), nlanes, nx) .+ 0.5
        function ftall5_1(x)
            out = similar(x, size(x,1), 5)
            out[:,1] .= x[:,1] .+ x[:,2]
            out[:,2] .= x[:,1] .* x[:,2]
            out[:,3] .= x[:,1].^2
            out[:,4] .= sin.(x[:,1]) .+ cos.(x[:,2])
            out[:,5] .= exp.(x[:,1])
            return out
        end
        pfd = DI.prepare_jacobian(ftall5_1, make_fd_cen(1), x)
        pfa = DI.prepare_jacobian(ftall5_1, make_fad(1),    x)
        Jfd = DI.jacobian(ftall5_1, pfd, make_fd_cen(1), x)
        Jfa = DI.jacobian(ftall5_1, pfa, make_fad(1),    x)
        @test size(Jfd, 1) == nlanes * ny
        @test size(Jfd, 2) == nlanes * nx
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
    end

    @testset "ny=5, nx=2, nlanes=7, batchdim=2, central" begin
        nx, ny, nlanes = 2, 5, 7
        x  = randn(MersenneTwister(43), nx, nlanes) .+ 0.5
        function ftall5_2(x)
            out = similar(x, 5, size(x,2))
            out[1,:] .= x[1,:] .+ x[2,:]
            out[2,:] .= x[1,:] .* x[2,:]
            out[3,:] .= x[1,:].^2
            out[4,:] .= sin.(x[1,:]) .+ cos.(x[2,:])
            out[5,:] .= exp.(x[1,:])
            return out
        end
        pfd = DI.prepare_jacobian(ftall5_2, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(ftall5_2, make_fad(2),    x)
        Jfd = DI.jacobian(ftall5_2, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(ftall5_2, pfa, make_fad(2),    x)
        @test size(Jfd, 1) == nlanes * ny
        @test size(Jfd, 2) == nlanes * nx
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
    end

    @testset "ny=3, nx=2: batchdim=1 and batchdim=2 agree per-lane, forward" begin
        nlanes, nx, ny = 5, 2, 3
        rng = MersenneTwister(44)
        x1 = randn(rng, nlanes, nx) .+ 1.0
        x2 = copy(x1')  # (nx, nlanes)
        function ftall_bd1(x)
            out = similar(x, size(x,1), 3)
            out[:,1] .= x[:,1] .+ x[:,2]
            out[:,2] .= x[:,1] .* x[:,2]
            out[:,3] .= x[:,1].^2
            return out
        end
        function ftall_bd2(x)
            out = similar(x, 3, size(x,2))
            out[1,:] .= x[1,:] .+ x[2,:]
            out[2,:] .= x[1,:] .* x[2,:]
            out[3,:] .= x[1,:].^2
            return out
        end
        pfd1 = DI.prepare_jacobian(ftall_bd1, make_fd_fwd(1), x1)
        pfd2 = DI.prepare_jacobian(ftall_bd2, make_fd_fwd(2), x2)
        Jfd1 = DI.jacobian(ftall_bd1, pfd1, make_fd_fwd(1), x1)
        Jfd2 = DI.jacobian(ftall_bd2, pfd2, make_fd_fwd(2), x2)
        for lane in 1:nlanes
            J1 = lane_jac(Jfd1, lane, nlanes, ny, nx, 1)
            J2 = lane_jac(Jfd2, lane, nlanes, ny, nx, 2)
            @test J1 ≈ J2  atol=ATOL_FWD
        end
    end
end

# ============================================================================
# Section 4: Non-square wide Jacobians (ny < nx)
# ============================================================================
#
# Per-lane test function: f: R^4 → R^2
#   f([x1,x2,x3,x4]) = [x1+x2+x3+x4,  x1*x2 + x3*x4]
#   J = [1   1   1   1 ]
#       [x2  x1  x4  x3]
#

@testset "Wide Jac ny<nx" begin

    @testset "ny=2, nx=4, nlanes=5, batchdim=1, forward" begin
        nlanes, nx, ny = 5, 4, 2
        x  = randn(MersenneTwister(50), nlanes, nx) .+ 1.0
        function fwide1(x)
            out = similar(x, size(x,1), 2)
            out[:,1] .= x[:,1] .+ x[:,2] .+ x[:,3] .+ x[:,4]
            out[:,2] .= x[:,1] .* x[:,2] .+ x[:,3] .* x[:,4]
            return out
        end
        pfd = DI.prepare_jacobian(fwide1, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(fwide1, make_fad(1),    x)
        Jfd = DI.jacobian(fwide1, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(fwide1, pfa, make_fad(1),    x)
        @test size(Jfd, 1) == nlanes * ny
        @test size(Jfd, 2) == nlanes * nx
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            x1,x2,x3,x4 = x[lane,1], x[lane,2], x[lane,3], x[lane,4]
            J_expected = [1.0  1.0  1.0  1.0;
                          x2   x1   x4   x3 ]
            @test lane_jac(Jfd, lane, nlanes, ny, nx, 1) ≈ J_expected  atol=ATOL_FWD
        end
    end

    @testset "ny=2, nx=4, nlanes=5, batchdim=2, forward" begin
        nx, ny, nlanes = 4, 2, 5
        x  = randn(MersenneTwister(51), nx, nlanes) .+ 1.0
        function fwide2(x)
            out = similar(x, 2, size(x,2))
            out[1,:] .= x[1,:] .+ x[2,:] .+ x[3,:] .+ x[4,:]
            out[2,:] .= x[1,:] .* x[2,:] .+ x[3,:] .* x[4,:]
            return out
        end
        pfd = DI.prepare_jacobian(fwide2, make_fd_fwd(2), x)
        pfa = DI.prepare_jacobian(fwide2, make_fad(2),    x)
        Jfd = DI.jacobian(fwide2, pfd, make_fd_fwd(2), x)
        Jfa = DI.jacobian(fwide2, pfa, make_fad(2),    x)
        @test size(Jfd, 1) == nlanes * ny
        @test size(Jfd, 2) == nlanes * nx
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            x1,x2,x3,x4 = x[1,lane], x[2,lane], x[3,lane], x[4,lane]
            J_expected = [1.0  1.0  1.0  1.0;
                          x2   x1   x4   x3 ]
            @test lane_jac(Jfd, lane, nlanes, ny, nx, 2) ≈ J_expected  atol=ATOL_FWD
        end
    end

    @testset "ny=1, nx=5 (row Jacobian), nlanes=8, batchdim=1, central" begin
        nlanes, nx, ny = 8, 5, 1
        x  = randn(MersenneTwister(52), nlanes, nx) .+ 0.5
        # f(x) = sum(sin.(x)) per lane → J = cos.(x)ᵀ
        frow1(x) = reshape(sum(sin.(x), dims=2), size(x,1), 1)
        pfd = DI.prepare_jacobian(frow1, make_fd_cen(1), x)
        pfa = DI.prepare_jacobian(frow1, make_fad(1),    x)
        Jfd = DI.jacobian(frow1, pfd, make_fd_cen(1), x)
        Jfa = DI.jacobian(frow1, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, ny, nx, 1)
            @test J ≈ cos.(x[lane, :])'  atol=ATOL_CEN
        end
    end

    @testset "ny=1, nx=5 (row Jacobian), nlanes=8, batchdim=2, central" begin
        nx, ny, nlanes = 5, 1, 8
        x  = randn(MersenneTwister(53), nx, nlanes) .+ 0.5
        frow2(x) = reshape(sum(sin.(x), dims=1), 1, size(x,2))
        pfd = DI.prepare_jacobian(frow2, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(frow2, make_fad(2),    x)
        Jfd = DI.jacobian(frow2, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(frow2, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, ny, nx, 2)
            @test J ≈ cos.(x[:, lane])'  atol=ATOL_CEN
        end
    end

    @testset "ny=2, nx=6, nlanes=10, batchdim=1, central (analytical check)" begin
        nlanes, nx, ny = 10, 6, 2
        x  = randn(MersenneTwister(54), nlanes, nx) .+ 1.0
        # f(x) = [sum(x),  sum(x^2)] → J = [ones(nx)';  2x']
        fsum1(x) = hcat(sum(x, dims=2), sum(x.^2, dims=2))
        pfd = DI.prepare_jacobian(fsum1, make_fd_cen(1), x)
        pfa = DI.prepare_jacobian(fsum1, make_fad(1),    x)
        Jfd = DI.jacobian(fsum1, pfd, make_fd_cen(1), x)
        Jfa = DI.jacobian(fsum1, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, ny, nx, 1)
            J_expected = vcat(ones(nx)', 2 .* x[lane,:]')
            @test J ≈ J_expected  atol=ATOL_CEN
        end
    end

    @testset "ny=2, nx=6, nlanes=10, batchdim=2, central (analytical check)" begin
        nx, ny, nlanes = 6, 2, 10
        x  = randn(MersenneTwister(55), nx, nlanes) .+ 1.0
        fsum2(x) = vcat(sum(x, dims=1), sum(x.^2, dims=1))
        pfd = DI.prepare_jacobian(fsum2, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(fsum2, make_fad(2),    x)
        Jfd = DI.jacobian(fsum2, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(fsum2, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, ny, nx, 2)
            J_expected = vcat(ones(nx)', 2 .* x[:,lane]')
            @test J ≈ J_expected  atol=ATOL_CEN
        end
    end

    @testset "ny=2, nx=4: batchdim=1 and batchdim=2 agree per-lane, forward" begin
        nlanes, nx, ny = 5, 4, 2
        rng = MersenneTwister(56)
        x1 = randn(rng, nlanes, nx) .+ 1.0
        x2 = copy(x1')
        fwbd1(x) = hcat(sum(x, dims=2), sum(x.^2, dims=2))
        fwbd2(x) = vcat(sum(x, dims=1), sum(x.^2, dims=1))
        pfd1 = DI.prepare_jacobian(fwbd1, make_fd_fwd(1), x1)
        pfd2 = DI.prepare_jacobian(fwbd2, make_fd_fwd(2), x2)
        Jfd1 = DI.jacobian(fwbd1, pfd1, make_fd_fwd(1), x1)
        Jfd2 = DI.jacobian(fwbd2, pfd2, make_fd_fwd(2), x2)
        for lane in 1:nlanes
            J1 = lane_jac(Jfd1, lane, nlanes, ny, nx, 1)
            J2 = lane_jac(Jfd2, lane, nlanes, ny, nx, 2)
            @test J1 ≈ J2  atol=ATOL_FWD
        end
    end
end

# ============================================================================
# Section 5: All 4 DI interface variants (one-arg f)
# ============================================================================

@testset "DI interface variants (one-arg f)" begin
    nlanes, nx = 6, 3
    x   = randn(MersenneTwister(60), nlanes, nx) .+ 1.0
    f(x) = sin.(x)
    bfd = make_fd_fwd(1)
    bfa = make_fad(1)
    pfd = DI.prepare_jacobian(f, bfd, x)
    pfa = DI.prepare_jacobian(f, bfa, x)

    @testset "jacobian (allocating)" begin
        Jfd = DI.jacobian(f, pfd, bfd, x)
        Jfa = DI.jacobian(f, pfa, bfa, x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end

    @testset "jacobian! (non-allocating)" begin
        Jfd_ref = DI.jacobian(f, pfd, bfd, x)
        Jbuf = similar(Jfd_ref)
        DI.jacobian!(f, Jbuf, pfd, bfd, x)
        Jfa = DI.jacobian(f, pfa, bfa, x)
        @test jac_ok(Jbuf, Jfa; atol=ATOL_FWD)
    end

    @testset "jacobian! result equals jacobian" begin
        Jfd   = DI.jacobian(f, pfd, bfd, x)
        Jbuf  = similar(Jfd)
        DI.jacobian!(f, Jbuf, pfd, bfd, x)
        @test Matrix(Jbuf) == Matrix(Jfd)
    end

    @testset "value_and_jacobian" begin
        y_fd, Jfd = DI.value_and_jacobian(f, pfd, bfd, x)
        y_fa, Jfa = DI.value_and_jacobian(f, pfa, bfa, x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        @test y_fd ≈ sin.(x)
    end

    @testset "value_and_jacobian!" begin
        Jfd_ref = DI.jacobian(f, pfd, bfd, x)
        Jbuf = similar(Jfd_ref)
        y_fd, _ = DI.value_and_jacobian!(f, Jbuf, pfd, bfd, x)
        _, Jfa   = DI.value_and_jacobian(f, pfa, bfa, x)
        @test jac_ok(Jbuf, Jfa; atol=ATOL_FWD)
        @test y_fd ≈ sin.(x)
    end

    @testset "Repeated jacobian! calls produce consistent results" begin
        Jfd1 = DI.jacobian(f, pfd, bfd, x)
        Jbuf = similar(Jfd1)
        DI.jacobian!(f, Jbuf, pfd, bfd, x)
        DI.jacobian!(f, Jbuf, pfd, bfd, x)  # second call, same prep
        @test Matrix(Jbuf) ≈ Matrix(Jfd1)
    end
end

@testset "DI interface variants (one-arg f), batchdim=2, central" begin
    nx, nlanes = 3, 6
    x   = randn(MersenneTwister(61), nx, nlanes) .+ 1.0
    f(x) = x .^ 2
    bfd = make_fd_cen(2)
    bfa = make_fad(2)
    pfd = DI.prepare_jacobian(f, bfd, x)
    pfa = DI.prepare_jacobian(f, bfa, x)

    @testset "jacobian" begin
        @test jac_ok(DI.jacobian(f, pfd, bfd, x), DI.jacobian(f, pfa, bfa, x); atol=ATOL_CEN)
    end

    @testset "jacobian!" begin
        Jref = DI.jacobian(f, pfd, bfd, x)
        Jbuf = similar(Jref)
        DI.jacobian!(f, Jbuf, pfd, bfd, x)
        @test Matrix(Jbuf) ≈ Matrix(Jref)
    end

    @testset "value_and_jacobian" begin
        y_fd, Jfd = DI.value_and_jacobian(f, pfd, bfd, x)
        _, Jfa     = DI.value_and_jacobian(f, pfa, bfa, x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        @test y_fd ≈ x .^ 2
    end

    @testset "value_and_jacobian!" begin
        Jref = DI.jacobian(f, pfd, bfd, x)
        Jbuf = similar(Jref)
        y_fd, _ = DI.value_and_jacobian!(f, Jbuf, pfd, bfd, x)
        @test y_fd ≈ x .^ 2
        @test Matrix(Jbuf) ≈ Matrix(Jref)
    end
end

# ============================================================================
# Section 6: In-place f! interface
# ============================================================================

@testset "In-place f! interface" begin

    @testset "f! square, batchdim=1, forward" begin
        nlanes, nx = 5, 3
        x  = randn(MersenneTwister(70), nlanes, nx) .+ 1.0
        y  = similar(x)
        f!(y, x) = (y .= sin.(x))
        pfd = DI.prepare_jacobian(f!, y, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f!, y, make_fad(1),    x)
        Jfd = DI.jacobian(f!, y, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f!, y, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end

    @testset "f! square, batchdim=2, central" begin
        nx, nlanes = 3, 5
        x  = randn(MersenneTwister(71), nx, nlanes) .+ 1.0
        y  = similar(x)
        f!(y, x) = (y .= x .^ 2)
        pfd = DI.prepare_jacobian(f!, y, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(f!, y, make_fad(2),    x)
        Jfd = DI.jacobian(f!, y, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(f!, y, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
    end

    @testset "f! square: jacobian! variant, batchdim=1, forward" begin
        nlanes, nx = 5, 3
        x  = randn(MersenneTwister(72), nlanes, nx) .+ 1.0
        y  = similar(x)
        f!(y, x) = (y .= sin.(x))
        pfd  = DI.prepare_jacobian(f!, y, make_fd_fwd(1), x)
        Jref = DI.jacobian(f!, y, pfd, make_fd_fwd(1), x)
        Jbuf = similar(Jref)
        DI.jacobian!(f!, y, Jbuf, pfd, make_fd_fwd(1), x)
        @test Matrix(Jbuf) ≈ Matrix(Jref)
    end

    @testset "f! square: value_and_jacobian, batchdim=1, forward" begin
        nlanes, nx = 5, 3
        x  = randn(MersenneTwister(73), nlanes, nx) .+ 1.0
        y  = similar(x)
        f!(y, x) = (y .= sin.(x))
        pfd = DI.prepare_jacobian(f!, y, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f!, y, make_fad(1),    x)
        y_fd, Jfd = DI.value_and_jacobian(f!, y, pfd, make_fd_fwd(1), x)
        y_fa, Jfa = DI.value_and_jacobian(f!, y, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end

    @testset "f! square: value_and_jacobian!, batchdim=2, central" begin
        nx, nlanes = 3, 5
        x  = randn(MersenneTwister(74), nx, nlanes) .+ 0.5
        y  = similar(x)
        f!(y, x) = (y .= x .^ 3)
        pfd  = DI.prepare_jacobian(f!, y, make_fd_cen(2), x)
        Jref = DI.jacobian(f!, y, pfd, make_fd_cen(2), x)
        Jbuf = similar(Jref)
        DI.value_and_jacobian!(f!, y, Jbuf, pfd, make_fd_cen(2), x)
        @test Matrix(Jbuf) ≈ Matrix(Jref)
    end

    @testset "f! tall ny=3, nx=2, batchdim=1, forward" begin
        nlanes, nx, ny = 5, 2, 3
        x  = randn(MersenneTwister(75), nlanes, nx) .+ 1.0
        y  = zeros(nlanes, ny)
        function ftall!(y, x)
            y[:,1] .= x[:,1] .+ x[:,2]
            y[:,2] .= x[:,1] .* x[:,2]
            y[:,3] .= x[:,1].^2
        end
        pfd = DI.prepare_jacobian(ftall!, y, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(ftall!, y, make_fad(1),    x)
        Jfd = DI.jacobian(ftall!, y, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(ftall!, y, pfa, make_fad(1),    x)
        @test size(Jfd, 1) == nlanes * ny
        @test size(Jfd, 2) == nlanes * nx
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end

    @testset "f! wide ny=2, nx=4, batchdim=2, forward" begin
        nx, ny, nlanes = 4, 2, 5
        x  = randn(MersenneTwister(76), nx, nlanes) .+ 1.0
        y  = zeros(ny, nlanes)
        function fwide!(y, x)
            y[1,:] .= x[1,:] .+ x[2,:] .+ x[3,:] .+ x[4,:]
            y[2,:] .= x[1,:] .* x[2,:] .+ x[3,:] .* x[4,:]
        end
        pfd = DI.prepare_jacobian(fwide!, y, make_fd_fwd(2), x)
        pfa = DI.prepare_jacobian(fwide!, y, make_fad(2),    x)
        Jfd = DI.jacobian(fwide!, y, pfd, make_fd_fwd(2), x)
        Jfa = DI.jacobian(fwide!, y, pfa, make_fad(2),    x)
        @test size(Jfd, 1) == nlanes * ny
        @test size(Jfd, 2) == nlanes * nx
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end
end

# ============================================================================
# Section 7: Scalar Constant contexts
# ============================================================================

@testset "Scalar Constant context" begin

    @testset "f(x,c) = c*sin.(x), batchdim=1, forward" begin
        nlanes, nx = 5, 3
        x  = randn(MersenneTwister(80), nlanes, nx) .+ 1.0
        c  = 2.5
        fc(x, c) = c .* sin.(x)
        pfd = DI.prepare_jacobian(fc, make_fd_fwd(1), x, Constant(c))
        pfa = DI.prepare_jacobian(fc, make_fad(1),    x, Constant(c))
        Jfd = DI.jacobian(fc, pfd, make_fd_fwd(1), x, Constant(c))
        Jfa = DI.jacobian(fc, pfa, make_fad(1),    x, Constant(c))
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 1)
            @test diag(J) ≈ c .* cos.(x[lane,:])  atol=ATOL_FWD
        end
    end

    @testset "f(x,c) = c*x.^2, batchdim=2, central" begin
        nx, nlanes = 4, 6
        x  = randn(MersenneTwister(81), nx, nlanes) .+ 0.5
        c  = 3.0
        fc(x, c) = c .* x .^ 2
        pfd = DI.prepare_jacobian(fc, make_fd_cen(2), x, Constant(c))
        pfa = DI.prepare_jacobian(fc, make_fad(2),    x, Constant(c))
        Jfd = DI.jacobian(fc, pfd, make_fd_cen(2), x, Constant(c))
        Jfa = DI.jacobian(fc, pfa, make_fad(2),    x, Constant(c))
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, nx, nx, 2)
            @test diag(J) ≈ 2c .* x[:, lane]  atol=ATOL_CEN
        end
    end

    @testset "Context value propagates correctly: c=1 vs c=2, batchdim=1, forward" begin
        nlanes, nx = 4, 3
        x  = randn(MersenneTwister(82), nlanes, nx) .+ 1.0
        fc(x, c) = c .* sin.(x)
        for c in (1.0, 2.0, -0.5)
            pfd = DI.prepare_jacobian(fc, make_fd_fwd(1), x, Constant(c))
            Jfd = DI.jacobian(fc, pfd, make_fd_fwd(1), x, Constant(c))
            for lane in 1:nlanes
                J = lane_jac(Jfd, lane, nlanes, nx, nx, 1)
                @test diag(J) ≈ c .* cos.(x[lane,:])  atol=ATOL_FWD
            end
        end
    end

    @testset "f! context, batchdim=1, forward" begin
        nlanes, nx = 5, 3
        x  = randn(MersenneTwister(83), nlanes, nx) .+ 1.0
        y  = similar(x)
        c  = 2.0
        fc!(y, x, c) = (y .= c .* sin.(x))
        pfd = DI.prepare_jacobian(fc!, y, make_fd_fwd(1), x, Constant(c))
        pfa = DI.prepare_jacobian(fc!, y, make_fad(1),    x, Constant(c))
        Jfd = DI.jacobian(fc!, y, pfd, make_fd_fwd(1), x, Constant(c))
        Jfa = DI.jacobian(fc!, y, pfa, make_fad(1),    x, Constant(c))
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end
end

# ============================================================================
# Section 8: Float32 type preservation
# ============================================================================

@testset "Float32 type preservation" begin

    @testset "eltype preserved, batchdim=1, forward" begin
        nlanes, nx = 6, 3
        x  = Float32.(randn(MersenneTwister(90), nlanes, nx) .+ 1.0)
        f(x) = sin.(x)
        p = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        J = DI.jacobian(f, p, make_fd_fwd(1), x)
        @test eltype(J) == Float32
    end

    @testset "eltype preserved, batchdim=2, central" begin
        nx, nlanes = 3, 6
        x  = Float32.(randn(MersenneTwister(91), nx, nlanes) .+ 1.0)
        f(x) = x .^ 2
        p = DI.prepare_jacobian(f, make_fd_cen(2), x)
        J = DI.jacobian(f, p, make_fd_cen(2), x)
        @test eltype(J) == Float32
    end

    @testset "Float32 values correct, batchdim=1, forward" begin
        nlanes, nx = 4, 3
        x  = Float32.(randn(MersenneTwister(92), nlanes, nx) .+ 1.0)
        f(x) = x .^ 2
        p = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        J = DI.jacobian(f, p, make_fd_fwd(1), x)
        # epsilon = sqrt(eps(Float32)) * ‖x_lane‖. For a 3-element lane with values ~1-4,
        # ‖x_lane‖ ≈ 4-5, so epsilon ≈ 1.5e-3. For f=x^2 the FD error equals epsilon exactly,
        # so atol must exceed the max expected epsilon across all lanes.
        for lane in 1:nlanes
            Jl = lane_jac(J, lane, nlanes, nx, nx, 1)
            @test Float32.(diag(Jl)) ≈ 2f0 .* x[lane,:]  atol=5f-3
        end
    end

    @testset "Float32 values correct, batchdim=2, forward" begin
        nx, nlanes = 3, 4
        x  = Float32.(randn(MersenneTwister(93), nx, nlanes) .+ 1.0)
        f(x) = sin.(x)
        p = DI.prepare_jacobian(f, make_fd_fwd(2), x)
        J = DI.jacobian(f, p, make_fd_fwd(2), x)
        for lane in 1:nlanes
            Jl = lane_jac(J, lane, nlanes, nx, nx, 2)
            @test Float32.(diag(Jl)) ≈ cos.(x[:,lane])  atol=1f-3
        end
    end
end

# ============================================================================
# Section 9: Edge cases
# ============================================================================

@testset "Edge cases" begin

    @testset "Batch size 1, batchdim=1, forward" begin
        nlanes, nx = 1, 4
        x  = randn(MersenneTwister(100), nlanes, nx) .+ 1.0
        f(x) = sin.(x)
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end

    @testset "Batch size 1, batchdim=2, central" begin
        nx, nlanes = 4, 1
        x  = randn(MersenneTwister(101), nx, nlanes) .+ 1.0
        f(x) = x .^ 2
        pfd = DI.prepare_jacobian(f, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(f, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
    end

    @testset "nx=1, ny=1 (scalar per lane), nlanes=8, batchdim=1, forward" begin
        nlanes, nx, ny = 8, 1, 1
        x  = randn(MersenneTwister(102), nlanes, nx) .+ 1.0
        f(x) = x .^ 2
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, ny, nx, 1)
            @test J[1,1] ≈ 2 * x[lane,1]  atol=ATOL_FWD
        end
    end

    @testset "nx=1, ny=1, nlanes=8, batchdim=2, central" begin
        nx, ny, nlanes = 1, 1, 8
        x  = randn(MersenneTwister(103), nx, nlanes) .+ 1.0
        f(x) = sin.(x)
        pfd = DI.prepare_jacobian(f, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(f, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
        for lane in 1:nlanes
            J = lane_jac(Jfd, lane, nlanes, ny, nx, 2)
            @test J[1,1] ≈ cos(x[1,lane])  atol=ATOL_CEN
        end
    end

    @testset "Large batch nlanes=100, nx=4, batchdim=1, forward" begin
        nlanes, nx = 100, 4
        x  = randn(MersenneTwister(104), nlanes, nx) .+ 1.0
        f(x) = sin.(x)
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfa = DI.prepare_jacobian(f, make_fad(1),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        Jfa = DI.jacobian(f, pfa, make_fad(1),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_FWD)
    end

    @testset "Large nx=20, nlanes=5, batchdim=2, central" begin
        nx, nlanes = 20, 5
        x  = randn(MersenneTwister(105), nx, nlanes) .+ 1.0
        f(x) = x .^ 2
        pfd = DI.prepare_jacobian(f, make_fd_cen(2), x)
        pfa = DI.prepare_jacobian(f, make_fad(2),    x)
        Jfd = DI.jacobian(f, pfd, make_fd_cen(2), x)
        Jfa = DI.jacobian(f, pfa, make_fad(2),    x)
        @test jac_ok(Jfd, Jfa; atol=ATOL_CEN)
    end

    @testset "Prep reuse: same x → identical results, batchdim=1, forward" begin
        nlanes, nx = 5, 3
        x  = randn(MersenneTwister(106), nlanes, nx) .+ 1.0
        f(x) = sin.(x)
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        J1  = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        J2  = DI.jacobian(f, pfd, make_fd_fwd(1), x)
        @test Matrix(J1) == Matrix(J2)
    end

    @testset "Different x → different Jacobian (prep reuse), batchdim=1, forward" begin
        nlanes, nx = 5, 3
        x1 = randn(MersenneTwister(107), nlanes, nx) .+ 1.0
        x2 = randn(MersenneTwister(108), nlanes, nx) .+ 2.0
        f(x) = sin.(x)
        pfd = DI.prepare_jacobian(f, make_fd_fwd(1), x1)
        J1  = DI.jacobian(f, pfd, make_fd_fwd(1), x1)
        J2  = DI.jacobian(f, pfd, make_fd_fwd(1), x2)
        @test !isapprox(Matrix(J1), Matrix(J2))
    end

    @testset "Sparsity structure: nnz == nlanes*nx*ny, batchdim=1" begin
        nlanes, nx, ny = 5, 3, 4
        x  = randn(MersenneTwister(109), nlanes, nx)
        function f109(x)
            out = similar(x, size(x,1), 4)
            out[:,1] .= x[:,1]; out[:,2] .= x[:,2]
            out[:,3] .= x[:,3]; out[:,4] .= x[:,1] .+ x[:,2]
            return out
        end
        pfd = DI.prepare_jacobian(f109, make_fd_fwd(1), x)
        Jfd = DI.jacobian(f109, pfd, make_fd_fwd(1), x)
        @test nnz(Jfd) == nlanes * nx * ny
    end

    @testset "Sparsity structure: nnz == nlanes*nx*ny, batchdim=2" begin
        nx, ny, nlanes = 3, 4, 5
        x  = randn(MersenneTwister(110), nx, nlanes)
        function f110(x)
            out = similar(x, 4, size(x,2))
            out[1,:] .= x[1,:]; out[2,:] .= x[2,:]
            out[3,:] .= x[3,:]; out[4,:] .= x[1,:] .+ x[2,:]
            return out
        end
        pfd = DI.prepare_jacobian(f110, make_fd_fwd(2), x)
        Jfd = DI.jacobian(f110, pfd, make_fd_fwd(2), x)
        @test nnz(Jfd) == nlanes * nx * ny
    end
end

# ============================================================================
# Section 10: Permutation invariance
# ============================================================================

@testset "Permutation invariance" begin

    @testset "Permuting lanes, batchdim=1, forward" begin
        nlanes, nx = 10, 3
        x    = randn(MersenneTwister(120), nlanes, nx) .+ 1.0
        perm = randperm(MersenneTwister(120), nlanes)
        xp   = x[perm, :]
        f(x) = sin.(x)
        pfd1 = DI.prepare_jacobian(f, make_fd_fwd(1), x)
        pfd2 = DI.prepare_jacobian(f, make_fd_fwd(1), xp)
        J1   = DI.jacobian(f, pfd1, make_fd_fwd(1), x)
        J2   = DI.jacobian(f, pfd2, make_fd_fwd(1), xp)
        # Lane pi in J1 should equal lane i in J2 (since xp[i] == x[perm[i]])
        for (i, pi) in enumerate(perm)
            @test lane_jac(J1, pi, nlanes, nx, nx, 1) ≈
                  lane_jac(J2,  i, nlanes, nx, nx, 1)  atol=ATOL_FWD
        end
    end

    @testset "Permuting lanes, batchdim=2, central" begin
        nx, nlanes = 3, 10
        x    = randn(MersenneTwister(121), nx, nlanes) .+ 1.0
        perm = randperm(MersenneTwister(121), nlanes)
        xp   = x[:, perm]
        f(x) = x .^ 2
        pfd1 = DI.prepare_jacobian(f, make_fd_cen(2), x)
        pfd2 = DI.prepare_jacobian(f, make_fd_cen(2), xp)
        J1   = DI.jacobian(f, pfd1, make_fd_cen(2), x)
        J2   = DI.jacobian(f, pfd2, make_fd_cen(2), xp)
        for (i, pi) in enumerate(perm)
            @test lane_jac(J1, pi, nlanes, nx, nx, 2) ≈
                  lane_jac(J2,  i, nlanes, nx, nx, 2)  atol=ATOL_CEN
        end
    end

    @testset "Permuting lanes, tall Jac ny=3 nx=2, batchdim=1, forward" begin
        nlanes, nx, ny = 8, 2, 3
        x    = randn(MersenneTwister(122), nlanes, nx) .+ 1.0
        perm = randperm(MersenneTwister(122), nlanes)
        xp   = x[perm, :]
        function ftallp(x)
            out = similar(x, size(x,1), 3)
            out[:,1] .= x[:,1] .+ x[:,2]
            out[:,2] .= x[:,1] .* x[:,2]
            out[:,3] .= x[:,1].^2
            return out
        end
        pfd1 = DI.prepare_jacobian(ftallp, make_fd_fwd(1), x)
        pfd2 = DI.prepare_jacobian(ftallp, make_fd_fwd(1), xp)
        J1   = DI.jacobian(ftallp, pfd1, make_fd_fwd(1), x)
        J2   = DI.jacobian(ftallp, pfd2, make_fd_fwd(1), xp)
        for (i, pi) in enumerate(perm)
            @test lane_jac(J1, pi, nlanes, ny, nx, 1) ≈
                  lane_jac(J2,  i, nlanes, ny, nx, 1)  atol=ATOL_FWD
        end
    end
end

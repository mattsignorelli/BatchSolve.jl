# Vectorized finite difference
# Experimental DI backend/ADType
# This will give a dense Jacobian matrix computed in a vectorized 
# way if nlanes_prefd = 1. If nlanes_pref>1, then a sparse Jacobian 
# will be returned that contains the Jacobian for each lane, all 
# computed using only a single evaluation. the Jacobian layout 
# will be block diagonal if batchdim=2 (bc then input is AoS style), 
# else it will have bands of diagonals if batchdim=1 (bc input is SoA).
# In either case, the sparse Jacobian layout is what would multiply x if
# x = reshape(x, :)  (treat x as a vector).

@enumx VecFDMode Forward Central Reverse

# step is nothing if autoselect (including for Float32), else can be a specified number
@kwdef struct AutoVecFD{S} <: AbstractADType
    mode::VecFDMode.T  = VecFDMode.Central
    batchdim::Int      = -1        # Autoselect
    step::S            = nothing   # Autoselect
end

# Stores input/output arrays with proper size
# Take all inputs and need to do FD with them along 
# batchdim together
struct VecFDTwoArgJacobianPrep{SIG, S, D, X, Y} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    mode::VecFDMode.T
    batchdim::Int    
    step::S       
    dense::Val{D}   
    x1::X
    y1::Y
end

function DI.prepare_jacobian_nokwarg(
        strict::Val, f!, y, backend::AutoVecFD, x, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    (; mode, batchdim, step) = backend
    # Note this only works for 2D Jacobian (matrix) atm
    if !(batchdim in (1,2))
      error("batchdim must be either 1 or 2 for Jacobian")
    end
    if size(x, batchdim) != size(y, batchdim)
      error("Sizes of input and output arrays along batchdim are not equal.")
    end
    otherdim = mod(batchdim, 2)
    nx = size(x, otherdim)
    ny = size(y, otherdim)
    nlanes_prefd = size(x, batchdim)
    dense = (nlanes_prefd == 1 ? Val{true}() : Val{false}())
    nlanes_withfd = nlanes_prefd*(1 + (mode == VecFDMode.Central ? 2*nx : nx))
    x1 = similar(x, ntuple(i-> i == batchdim ? nlanes_withfd : nx, Val{2}()))
    y1 = similar(y, ntuple(i-> i == batchdim ? nlanes_withfd : ny, Val{2}()))
    step = isnothing(step) ? (mode == VecFDMode.Central ? eps(eltype(x))^(1/3) : eps(eltype(x))^(1/2)) : step
    return VecFDTwoArgJacobianPrep(_sig, mode, batchdim, step, dense, x1, y1)
end

function DI.value_and_jacobian(
        f!,
        y,
        prep::VecFDTwoArgJacobianPrep,
        backend::AutoVecFD,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    (; mode, batchdim, step, x1, y1) = prep
    otherdim = mod(batchdim, 2)
    nx = size(x, otherdim)

    # Initialize primal and tangents
    # First block of x is the primal
    nlanes_withfd = size(x1, batchdim)
    nlanes_prefd = div(nlanes_withfd, (1 + (mode == VecFDMode.Central ? 2*nx : nx)))
    idx_primal = ntuple(i-> i == batchdim ? (1:nlanes_prefd) : (1:nx), Val{2}())
    x1[idx_primal...] .= x

    # Next block(s) are tangents
    # 
     if mode == VecFDMode.Central

    else

    end

    fc!(y, x)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    fc!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
        f!,
        y,
        jac,
        prep::FiniteDiffTwoArgJacobianPrep,
        backend::AutoFiniteDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    fc!(y, x)
    return y, jac
end















function DI.prepare_jacobian_nokwarg(
        strict::Val, f!, y, backend::AutoGTPSA{D}, x, contexts::Vararg{DI.Constant, C}
    ) where {D, C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(length(x), 1) # n variables to first order
    end

    # We set the slopes of each variable to 1 here, this will always be the case for Jacobian
    xt = similar(x, TPS{promote_type(eltype(x), Float64)})
    j = 1
    for i in eachindex(xt)
        xt[i] = TPS{promote_type(eltype(x), Float64)}(; use = d)
        xt[i][j] = 1
        j += 1
    end

    yt = similar(y, TPS{promote_type(eltype(y), Float64)})

    for i in eachindex(yt)
        yt[i] = TPS{promote_type(eltype(y), Float64)}(; use = d)
    end

    return GTPSATwoArgJacobianPrep(_sig, xt, yt)
end

function DI.jacobian(
        f!,
        y,
        prep::GTPSATwoArgJacobianPrep,
        backend::AutoGTPSA,
        x,
        contexts::Vararg{DI.Constant, C},
    ) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    fc!(prep.yt, prep.xt)
    jac = similar(x, GTPSA.numtype(eltype(prep.yt)), (length(prep.yt), length(x)))
    GTPSA.jacobian!(jac, prep.yt; include_params = true, unsafe_inbounds = true)
    map!(t -> t[0], y, prep.yt)
    return jac
end

function DI.jacobian!(
        f!,
        y,
        jac,
        prep::GTPSATwoArgJacobianPrep,
        backend::AutoGTPSA,
        x,
        contexts::Vararg{DI.Constant, C},
    ) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    fc!(prep.yt, prep.xt)
    GTPSA.jacobian!(jac, prep.yt; include_params = true, unsafe_inbounds = true)
    map!(t -> t[0], y, prep.yt)
    return jac
end

function DI.value_and_jacobian(
        f!,
        y,
        prep::GTPSATwoArgJacobianPrep,
        backend::AutoGTPSA,
        x,
        contexts::Vararg{DI.Constant, C},
    ) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    jac = DI.jacobian(f!, y, prep, backend, x, contexts...) # y set on line 151
    return y, jac
end

function DI.value_and_jacobian!(
        f!,
        y,
        jac,
        prep::GTPSATwoArgJacobianPrep,
        backend::AutoGTPSA,
        x,
        contexts::Vararg{DI.Constant, C},
    ) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    DI.jacobian!(f!, y, jac, prep, backend, x, contexts...)
    return y, jac
end


struct FiniteDiffTwoArgJacobianPrep{SIG, C, R, A, D} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_jacobian_nokwarg(
        strict::Val, f!, y, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    relstep = if isnothing(backend.relstep)
        default_relstep(fdjtype(backend), eltype(x))
    else
        backend.relstep
    end
    absstep = if isnothing(backend.absstep)
        relstep
    else
        backend.absstep
    end
    dir = backend.dir
    return FiniteDiffTwoArgJacobianPrep(_sig, cache, relstep, absstep, dir)
end

function DI.prepare!_jacobian(
        f!,
        y,
        old_prep::FiniteDiffTwoArgJacobianPrep,
        backend::AutoFiniteDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    if x isa Vector && y isa Vector
        (; cache) = old_prep
        cache.x1 isa Union{Number, Nothing} || resize!(cache.x1, length(x))
        cache.x2 isa Union{Number, Nothing} || resize!(cache.x2, length(x))
        cache.fx isa Union{Number, Nothing} || resize!(cache.fx, length(y))
        cache.fx1 isa Union{Number, Nothing} || resize!(cache.fx1, length(y))
        cache.colorvec = 1:length(x)
        cache.sparsity = nothing
        return old_prep
    else
        return DI.prepare_jacobian_nokwarg(
            DI.is_strict(old_prep), f!, y, backend, x, contexts...
        )
    end
end

function DI.value_and_jacobian(
        f!,
        y,
        prep::FiniteDiffTwoArgJacobianPrep,
        backend::AutoFiniteDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    fc!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
        f!,
        y,
        jac,
        prep::FiniteDiffTwoArgJacobianPrep,
        backend::AutoFiniteDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    fc!(y, x)
    return y, jac
end

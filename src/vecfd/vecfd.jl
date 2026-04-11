# Vectorized finite difference
# Experimental DI backend/ADType

# step is nothing if autoselect (including for Float32), else can be a specified number
@kwdef struct AutoVecFD{T} <: AbstractADType
    mode::Symbol  = :central
    batchdim::Int = -1        # Autoselect
    step::T       = nothing   # Autoselect
end

# Stores input/output arrays with proper size
# Take all inputs and need to do FD with them along 
# batchdim together
struct VecFDTwoArgJacobianPrep{SIG, X, Y, T} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    x::X
    y::Y
    batchdim::Int
    step::T
    mode::Symbol
end

function DI.prepare_jacobian_nokwarg(
        strict::Val, f!, y, backend::AutoVecFD, x, contexts::Vararg{DI.Context, C}
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

struct BatchFiniteDiffJacobianPrep{SIG, X, Y, R, A, E, T} <: DI.JacobianPrep{SIG}
  _sig::Val{SIG}
  batchdim::Int
  n_rows::Int
  n_cols::Int
  batchsize::Int
  x1::X
  y1::Y
  relstep::R
  absstep::A
  epsilon::E
  contexts_cache::T
end

fdjtype(::AutoFiniteDiff{fdt, fdjt}) where {fdt, fdjt} = fdjt

function SparseMatrixColorings.sparsity_pattern(prep::BatchFiniteDiffJacobianPrep)
  return make_pattern(prep.x1, prep.y1, prep.batchdim; n_rows=prep.n_rows, n_cols=prep.n_cols, batchsize=prep.batchsize)
end

function _prepare_batch_jacobian_aux(
    strict::Val,
    y,
    f_or_f!y::FY,
    backend::AutoBatch{<:AutoFiniteDiff},
    x,
    contexts::Vararg{DI.Context, C}
  ) where {FY, C}
  # Preallocate expanded input/output arrays:
  batchdim = backend.batchdim
  mode = fdjtype(dense_ad(backend))
  mode in (Val{:central}, Val{:forward}) || error("Unrecognized finite differencing method: $(mode)")
  otherdim = mod(batchdim, 2) + 1
  nx = size(x, otherdim)
  ny = size(y, otherdim)
  batchsize = size(x, batchdim)
  nlanes_withfd = batchsize*(1 + nx*(mode == Val{:central} ? 2 : 1))
  x1 = similar(x, ntuple(i -> i == batchdim ? nlanes_withfd : nx, Val{2}()))
  y1 = similar(y, ntuple(i -> i == batchdim ? nlanes_withfd : ny, Val{2}()))
  epsilon = similar(x, ntuple(i -> i == batchdim ? batchsize : 1, Val{2}()))

  relstep = if isnothing(dense_ad(backend).relstep)
    default_relstep(fdjtype(dense_ad(backend)), eltype(x))
  else
    dense_ad(backend).relstep
  end
  absstep = if isnothing(dense_ad(backend).absstep)
    relstep
  else
    dense_ad(backend).absstep
  end

  # Need to repeat AbstractArray contexts along batchdim:
  # This is type unstable, but in prep step.
  # Jacobian calculation uses type stable "iteration" over tuple
  contexts_cache = map(contexts) do c
    if DI.unwrap(c) isa AbstractArray # expand along batchdim
      if size(DI.unwrap(c), batchdim) != batchsize
        error("AbstractArray Contexts for AutoBatch{<:AutoFiniteDiff} must have a size along batchdim equal 
               to the size along batchdim of the input array: received Context with size $(size(DI.unwrap(c), batchdim)) along batchdim.")
      end
      sb = size(DI.unwrap(c), batchdim)*(1 + size(DI.unwrap(c), otherdim)*(mode == Val{:central} ? 2 : 1))
      DI.maker(c)(similar(DI.unwrap(c), ntuple(i -> i == batchdim ? sb : size(DI.unwrap(c), otherdim), Val{2}())))
    else
      c
    end
  end

   _sig = DI.signature(f_or_f!y..., backend, x, contexts...; strict)
  return BatchFiniteDiffJacobianPrep(_sig, batchdim, ny, nx, batchsize, x1, y1, relstep, absstep, epsilon, contexts_cache)
end

# Jacobian:
@generated function set_contexts!(contexts_cache::T1, contexts::T2, batchdim) where {T1<:Tuple,T2<:Tuple}
  N = length(T1.parameters)
  @assert N == length(T2.parameters) "Length of contexts_cache and contexts tuple disagree."
  exprs = [
    :(_context_lower!(Base.getfield(contexts_cache, $i), Base.getfield(contexts, $i), batchdim)) 
  for i in 1:N ]
  return :(tuple($(exprs...)))
end

function _context_lower!(context_cache::Context, context::Context, batchdim)
  return DI.maker(context_cache)(_set_context!(DI.unwrap(context_cache), DI.unwrap(context), batchdim))
end

_set_context!(context_cache_data, context_data, batchdim) = context_data
function _set_context!(context_cache_data::AbstractArray, context_data::AbstractArray, batchdim)
  chunksize = size(context_data, batchdim)
  nx = size(context_data, mod(batchdim, 2) + 1)
  reps = div(size(context_cache_data, batchdim), chunksize)
  for i in 1:reps
    context_cache_data[ntuple(j -> j == batchdim ? ((chunksize*(i-1)+1):(chunksize*i)) : (1:nx), Val{2}())...] .= context_data
  end
  return context_cache_data
end

function _value_and_jacobian_aux!(
    f_or_f!y::FY, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x,
  ) where {FY}
  set_batch_fdj_input!(x, prep, backend)
  if length(f_or_f!y) == 1
    f = f_or_f!y[1]
    prep.y1 .= f(prep.x1)
    # Allocate result:
    batchdim = prep.batchdim
    otherdim = mod(batchdim, 2) + 1
    y = prep.y1[ntuple(i -> i == batchdim ? (1:size(x, batchdim)) : 1:size(prep.y1, otherdim), Val{2}())...]
  else
    f! = f_or_f!y[1]
    y = f_or_f!y[2]
    f!(prep.y1, prep.x1)
    y[:] .= view(prep.y1, 1:length(y))
  end
  compute_batch_fdj_jac!(jac, prep, backend)
  return y, jac
end

# Initialize primal and tangents
function set_batch_fdj_input!(x, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff})
  x1 = prep.x1
  relstep = prep.relstep
  absstep = prep.absstep
  epsilon = prep.epsilon
  dir = dense_ad(backend).dir

  batchdim = prep.batchdim
  otherdim = mod(batchdim, 2) + 1
  nx = size(x, otherdim)
  nlanes = size(x, batchdim)
  mode = fdjtype(dense_ad(backend))
  
  lane_norms = sqrt.(sum(abs2, x; dims=otherdim))
  epsilon .= compute_epsilon.(mode(), lane_norms, relstep, absstep, dir)

  idx_primal = ntuple(i -> i == batchdim ? (1:nlanes) : (1:nx), Val{2}())

  # Set primal (first nlanes rows/cols along batchdim)
  x1[idx_primal...] .= x

  # Now the next chunk (FWD) is of has nlanes*nx rows/cols along batchdim, and must 
  # Go in chunks of size nlanes, with first chunk offset by nlanes
  for i in 1:nx 
    # Set primal for the chunk, offset from primal block by nlanes
    idx_primal = ntuple(j -> j == batchdim ? ((nlanes*i+1):(nlanes*(i+1))) : (1:nx), Val{2}())
    x1[idx_primal...] .= x

    # Now add this tangent, note i:i for type stability
    idx_fwd = ntuple(j -> j == batchdim ? ((nlanes*i+1):(nlanes*(i+1))) : (i:i), Val{2}())
    x1[idx_fwd...] .+= epsilon
  end

  # Now if central differences, we need to go one more chunk (RWD)
  # first chunk is now offset by nlanes + nlanes*nx
  if mode == Val{:central}
    for i in 1:nx 
      # Set primal for the chunk:
      idx_primal = ntuple(j -> j == batchdim ? ((nlanes*(nx+i)+1):(nlanes*(i+1+nx))) : (1:nx), Val{2}())
      x1[idx_primal...] .= x

      # Now add this tangent:
      idx_rwd = ntuple(j -> j == batchdim ? ((nlanes*(nx+i)+1):(nlanes*(i+1+nx))) : (i:i), Val{2}())
      x1[idx_rwd...] .-= epsilon
    end
  end
  return x1
end

function compute_batch_fdj_jac!(
    jac,
    prep::BatchFiniteDiffJacobianPrep,
    backend::AutoBatch{<:AutoFiniteDiff},
  )
  x1      = prep.x1
  y1      = prep.y1
  epsilon = prep.epsilon

  batchdim = prep.batchdim
  otherdim = mod(batchdim, 2) + 1
  nx      = size(x1, otherdim)      # number of inputs per lane
  ny      = size(y1, otherdim)      # number of outputs per lane
  nlanes  = size(epsilon, batchdim) # == size(x, batchdim)

  mode   = fdjtype(dense_ad(backend))
  nzval  = SparseArrays.nonzeros(jac)

  if batchdim == 1
    # ------------------------------------------------------------------
    # y1 layout: shape (nlanes*(1 + k*nx), ny),  k=1 fwd / k=2 central
    #
    #   Rows 1:nlanes                          - primal outputs
    #   Rows nlanes+1 : nlanes*(nx+1)          - fwd-perturbed (all nx vars)
    #   Rows nlanes*(nx+1)+1 : nlanes*(2*nx+1) - rwd-perturbed (central only)
    #
    # fwd_3d[lane, var, out] = f(x + eps*e_var)[lane, out]  — (nlanes, nx, ny)
    #
    # CSC layout for batchdim=1 (banded Jacobian):
    #   nzval reshaped to (ny, nlanes, nx) in col-major order, i.e.
    #   nzval_3d[out, lane, var] = J[lane,var,out]
    #
    # Strategy: wrap nzval_3d as PermutedDimsArray{(2,3,1)} to expose it with
    #   the same (lane, var, out) index order as fwd_3d, then @. broadcast
    #   the difference quotient directly in — zero allocations, single GPU kernel.
    # ------------------------------------------------------------------

    fwd_3d    = reshape(@view(y1[(nlanes + 1):(nlanes*(nx + 1)), :]),
                        nlanes, nx, ny)          # (lane, var, out) — no-copy reshape
    eps_3d    = reshape(@view(epsilon[:, 1]),
                        nlanes, 1, 1)            # (lane, 1, 1)    — broadcast-ready

    nzval_3d   = reshape(nzval, ny, nlanes, nx)
    nzval_perm = PermutedDimsArray(nzval_3d, (2, 3, 1))  # view as (lane, var, out), no alloc

    if mode == Val{:central}
      rwd_3d = reshape(@view(y1[(nlanes*(nx + 1) + 1):(nlanes*(2*nx + 1)), :]),
                       nlanes, nx, ny)
      @. nzval_perm = (fwd_3d - rwd_3d) / (2 * eps_3d)
    else  # :forward
      primal_3d = reshape(@view(y1[1:nlanes, :]),
                          nlanes, 1, ny)         # (lane, 1, out) — broadcast-ready
      @. nzval_perm = (fwd_3d - primal_3d) / eps_3d
    end

  else  # batchdim == 2
    # ------------------------------------------------------------------
    # y1 layout: shape (ny, nlanes*(1 + k*nx))
    #
    #   Cols 1:nlanes                          - primal outputs
    #   Cols nlanes+1 : nlanes*(nx+1)          - fwd-perturbed (all nx vars)
    #   Cols nlanes*(nx+1)+1 : nlanes*(2*nx+1) - rwd-perturbed (central only)
    #
    # fwd_3d[out, lane, var] = f(x + eps*e_var)[out, lane]  — (ny, nlanes, nx)
    #
    # CSC layout for batchdim=2 (block-diagonal Jacobian):
    #   nzval reshaped to (ny, nx, nlanes) in col-major order, i.e.
    #   nzval_3d[out, var, lane] = J[out,var,lane]
    #
    # Strategy: same zero-alloc approach, wrap nzval_3d as PermutedDimsArray{(1,3,2)}
    #   to expose it as (out, lane, var) matching fwd_3d.
    # ------------------------------------------------------------------

    fwd_3d    = reshape(@view(y1[:, (nlanes + 1):(nlanes*(nx + 1))]),
                        ny, nlanes, nx)          # (out, lane, var) — no-copy reshape
    eps_3d    = reshape(@view(epsilon[1, :]),
                        1, nlanes, 1)            # (1, lane, 1)     — broadcast-ready

    nzval_3d   = reshape(nzval, ny, nx, nlanes)
    nzval_perm = PermutedDimsArray(nzval_3d, (1, 3, 2))  # view as (out, lane, var), no alloc

    if mode == Val{:central}
      rwd_3d = reshape(@view(y1[:, (nlanes*(nx + 1) + 1):(nlanes*(2*nx + 1))]),
                       ny, nlanes, nx)
      @. nzval_perm = (fwd_3d - rwd_3d) / (2 * eps_3d)
    else  # :forward
      primal_3d = reshape(@view(y1[:, 1:nlanes]),
                          ny, nlanes, 1)         # (out, lane, 1)  — broadcast-ready
      @. nzval_perm = (fwd_3d - primal_3d) / eps_3d
    end
  end

  return jac
end

# One argument

function DI.jacobian!(
    f::F, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc = DI.fix_tail(f, map(DI.unwrap, prep.contexts_cache)...)
  return _value_and_jacobian_aux!((fc,), jac, prep, backend, x)[2]
end

function DI.jacobian(
    f::F, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc = DI.fix_tail(f, map(DI.unwrap, prep.contexts_cache)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc,), jac, prep, backend, x)[2]
end

function DI.value_and_jacobian(
    f::F, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc = DI.fix_tail(f, map(DI.unwrap, prep.contexts_cache)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc,), jac, prep, backend, x)
end

function DI.value_and_jacobian!(
    f::F, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc = DI.fix_tail(f, map(DI.unwrap, prep.contexts_cache)...)
  return _value_and_jacobian_aux!((fc,), jac, prep, backend, x)
end


## Two arguments

function DI.jacobian!(
    f!::F, y, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc! = DI.fix_tail(f!, map(DI.unwrap, prep.contexts_cache)...)
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)[2]
end

function DI.jacobian(
    f!::F, y, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc! = DI.fix_tail(f!, map(DI.unwrap, prep.contexts_cache)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)[2]
end

function DI.value_and_jacobian(
    f!::F, y, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc! = DI.fix_tail(f!, map(DI.unwrap, prep.contexts_cache)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)
end

function DI.value_and_jacobian!(
    f!::F, y, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc! = DI.fix_tail(f!, map(DI.unwrap, prep.contexts_cache)...)
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)
end
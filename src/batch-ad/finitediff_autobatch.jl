struct BatchFiniteDiffJacobianPrep{SIG, X, Y, R, A, E} <: DI.JacobianPrep{SIG}
  _sig::Val{SIG}
  x1::X
  y1::Y
  relstep::R
  absstep::A
  epsilon::E
end

SparseMatrixColorings.sparsity_pattern(prep::BatchFiniteDiffJacobianPrep) = make_pattern(prep.x1, prep.y1, prep.batchdim)

function DI.prepare_jacobian_nokwarg(
    strict::Val, f::F, backend::AutoBatch{AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C}
  ) where {F, C}
  y = f(x, map(DI.unwrap, contexts)...)
  return _prepare_batch_jacobian_aux(strict, y, f, backend, x, contexts...)
end

function DI.prepare_jacobian_nokwarg(
    strict::Val, f!::F, y, backend::AutoBatch{AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C}
  ) where {F, C}
  return _prepare_batch_jacobian_aux(strict, y, (f!,y), backend, x, contexts...)
end

function _prepare_batch_fd_jacobian_aux(
    strict::Val,
    y,
    f_or_f!y::FY,
    backend::AutoBatch{AutoFiniteDiff},
    x,
    contexts::Vararg{DI.Context, C}
  ) where {FY, C}
  # Preallocate expanded input/output arrays:
  batchdim = backend.batchdim
  mode = fdjtype(backend)
  mode in (:central, :forward) || error("Unrecognized finite differencing method: $(mode)")
  otherdim = mod(batchdim, 2) + 1
  nx = size(x, otherdim)
  ny = size(y, otherdim)
  nlanes_prefd = size(x, batchdim)
  nlanes_withfd = nlanes_prefd*(1 + (mode == Val{:central}() ? 2*nx : nx))
  x1 = similar(x, ntuple(i -> i == batchdim ? nlanes_withfd : nx, Val{2}()))
  y1 = similar(y, ntuple(i -> i == batchdim ? nlanes_withfd : ny, Val{2}()))
  epsilon = similar(x, ntuple(i -> i == batchdim ? nlanes : 1, Val{2}()))

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

   _sig = DI.signature(f_or_f!y..., backend, x, contexts...; strict)
  return BatchFiniteDiffJacobianPrep(_sig, x1, y1, relstep, absstep, epsilon)
end

# Initialize primal and tangents
function _set_batch_fdj_input(x, prep::BatchFiniteDiffJacobianPrep, backend::AutoFiniteDiff)
  x1 = prep.x1
  relstep = prep.relstep
  absstep = prep.absstep
  epsilon = prep.epsilon
  dir = backend.dir

  batchdim = backend.batchdim
  otherdim = mod(batchdim, 2) + 1
  nx = size(x, otherdim)
  mode = fdjtype(backend)
  mode in (Val{:central}(), Val{:forward}()) || error("Unrecognized finite differencing method: $(mode)")
  
  lane_norms = mapslices(norm, x; dims=otherdim)
  epsilon .= compute_epsilon.(mode, lane_norms, relstep, absstep, dir)

  # Set primal (first nlanes rows/cols along batchdim)
  idx_primal = ntuple(i -> i == batchdim ? (1:nlanes) : (1:nx), Val{2}())
  x1[idx_primal...] .= x

  # Now the next chunk (FWD) is of has nlanes*nx rows/cols along batchdim, and must 
  # Go in chunks of size nlanes, with first chunk offset by nlanes
  for i in 1:nx 
    # Set primal for the chunk, offset from primal block by nlanes
    idx_primal = ntuple(j -> j == batchdim ? ((nlanes*i+1):(nlanes*(i+1))) : (1:nx), Val{2}())
    x1[idx_primal...] .= x

    # Now add this tangent, note i:i for type stability
    idx_fwd = ntuple(j -> j == batchdim ? ((nlanes*i+1):(nlanes*(i+1))) : (i:i), Val{2}())
    x1[idx_fwd...] .+= eps_lanes
  end

  # Now if central differences, we need to go one more chunk (RWD)
  # first chunk is now offset by nlanes + nlanes*nx
  if mode == Val{:central}()
    for i in 1:nx 
      # Set primal for the chunk:
      idx_primal = ntuple(j -> j == batchdim ? ((nlanes*(nx+i)+1):(nlanes*(i+1+nx))) : (1:nx), Val{2}())
      x1[idx_primal...] .= x

      # Now add this tangent:
      idx_rwd = ntuple(j -> j == batchdim ? ((nlanes*(nx+i)+1):(nlanes*(i+1+nx))) : (i:i), Val{2}())
      x1[idx_rwd...] .-= eps_lanes
    end
  end
  return x1
end

# Jacobian calls now just pass thru straight to the sparse backend:

# One argument

function DI.jacobian!(
    f::F, jac, prep::BatchJacobianPrep, backend::AutoBatch, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  return DI.jacobian!(f, jac, prep.sparse_prep, prep.sparse_ad, x, contexts...)
end

function DI.jacobian(
    f::F, prep::BatchJacobianPrep, backend::AutoBatch, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  return DI.jacobian(f, prep.sparse_prep, prep.sparse_ad, x, contexts...)
end

function DI.value_and_jacobian(
    f::F, prep::BatchJacobianPrep, backend::AutoBatch, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  return DI.value_and_jacobian(f, prep.sparse_prep, prep.sparse_ad, x, contexts...)
end

function DI.value_and_jacobian!(
    f::F, jac, prep::BatchJacobianPrep, backend::AutoBatch, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  return DI.value_and_jacobian!(f, jac, prep.sparse_prep, prep.sparse_ad, x, contexts...)
end


## Two arguments

function DI.jacobian!(
    f!::F, y, jac, prep::BatchJacobianPrep, backend::AutoBatch, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  return DI.jacobian!(f!, y, jac, prep.sparse_prep, prep.sparse_ad, x, contexts...)
end

function DI.jacobian(
    f!::F, y, prep::BatchJacobianPrep, backend::AutoBatch, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  return DI.jacobian(f!, y, prep.sparse_prep, prep.sparse_ad, x, contexts...)
end

function DI.value_and_jacobian(
    f!::F, y, prep::BatchJacobianPrep, backend::AutoBatch, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  return DI.value_and_jacobian(f!, y, prep.sparse_prep, prep.sparse_ad, x, contexts...)
end

function DI.value_and_jacobian!(
    f!::F, y, jac, prep::BatchJacobianPrep, backend::AutoBatch, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  return DI.value_and_jacobian!(f!, y, jac, prep.sparse_prep, prep.sparse_ad, x, contexts...)
end
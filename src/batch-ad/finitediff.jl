struct BatchFiniteDiffJacobianPrep{SIG, X, Y, R, A, E} <: DI.JacobianPrep{SIG}
  _sig::Val{SIG}
  batchdim::Int
  x1::X
  y1::Y
  relstep::R
  absstep::A
  epsilon::E
end
fdjtype(::AutoFiniteDiff{fdt, fdjt}) where {fdt, fdjt} = fdjt
SparseMatrixColorings.sparsity_pattern(prep::BatchFiniteDiffJacobianPrep) = make_pattern(prep.x1, prep.y1, prep.batchdim)

# Preparation:
#=
function DI.prepare_jacobian_nokwarg(
    strict::Val, f::F, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C}
  ) where {F, C}
  y = f(x, map(DI.unwrap, contexts)...)
  return _prepare_batch_jacobian_aux(strict, y, f, backend, x, contexts...)
end

function DI.prepare_jacobian_nokwarg(
    strict::Val, f!::F, y, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C}
  ) where {F, C}
  return _prepare_batch_jacobian_aux(strict, y, (f!,y), backend, x, contexts...)
end
=#
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
  nlanes_prefd = size(x, batchdim)
  nlanes_withfd = nlanes_prefd*(1 + (mode == Val{:central} ? 2*nx : nx))
  x1 = similar(x, ntuple(i -> i == batchdim ? nlanes_withfd : nx, Val{2}()))
  y1 = similar(y, ntuple(i -> i == batchdim ? nlanes_withfd : ny, Val{2}()))
  epsilon = similar(x, ntuple(i -> i == batchdim ? nlanes_prefd : 1, Val{2}()))

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
  return BatchFiniteDiffJacobianPrep(_sig, batchdim, x1, y1, relstep, absstep, epsilon)
end

# Jacobian:

function _value_and_jacobian_aux!(
    f_or_f!y::FY, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x,
  ) where {FY}
  _set_batch_fdj_input!(x, prep, backend)
  if length(f_or_f!y) == 1
    f = f_or_f!y[1]
    prep.y1 .= f(prep.x1)
    # Allocate result:
    batchdim = prep.batchdim
    otherdim = mod(batchdim, 2) + 1
    y = prep.y1[ntuple(i -> i == batchdim ? size(x, batchdim) : 1:size(y, otherdim), Val{2}())...]
  else
    f! = f_or_f!y[1]
    y = f_or_f!y[2]
    f!(prep.y1, prep.x1)
    y[:] .= view(prep.y1, 1:length(y))
  end
  _compute_batch_fdj_jac!(jac, prep, backend)
  return y, jac
end

# Initialize primal and tangents
function _set_batch_fdj_input!(x, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff})
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
  
  idx_primal = ntuple(i -> i == batchdim ? (1:nlanes) : (1:nx), Val{2}())
  lane_norms = mapslices(norm, reshape(x, last(idx_primal[1]), last(idx_primal[2])); dims=otherdim)
  epsilon .= compute_epsilon.(mode(), lane_norms, relstep, absstep, dir)

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

function _compute_batch_fdj_jac!(jac::SparseArrays.AbstractSparseMatrixCSC, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff})
  x1 = prep.x1
  y1 = prep.y1

  batchdim = prep.batchdim
  otherdim = mod(batchdim, 2) + 1
  nx     = size(x1, otherdim)
  ny     = size(y1, otherdim)
  mode   = fdjtype(dense_ad(backend))
  nlanes = div(size(x, batchdim), (mode == Val{:central} ? 2*nx + 1 : nx + 1))

  if batchdim == 1
    fwd  = reshape(view(y1, (nlanes+1):nlanes*(nx+1),            :), nx, nlanes, ny)
    eps3 = reshape(prep.epsilon, 1, nlanes, 1)
    diff = if mode == Val{:central}
      bwd = reshape(view(y1, (nlanes*(nx+1)+1):nlanes*(2nx+1),   :), nx, nlanes, ny)
      (fwd .- bwd) ./ (2 .* eps3)                    # (nx, nlanes, ny)
    else
      y0 = reshape(view(y1, 1:nlanes, :), 1, nlanes, ny)
      (fwd .- y0) ./ eps3
    end
    # CSC nzval layout for banded pattern: reshape(nzval, ny, nlanes, nx) = permutedims(diff, (3,2,1))
    reshape(nonzeros(jac), ny, nlanes, nx) .= permutedims(diff, (3, 2, 1))

  else  # batchdim=2, block diagonal
    fwd  = reshape(view(y1, :, (nlanes+1):nlanes*(nx+1)),          ny, nx, nlanes)
    eps3 = reshape(prep.epsilon, 1, 1, nlanes)
    diff = if mode == Val{:central}
      bwd = reshape(view(y1, :, (nlanes*(nx+1)+1):nlanes*(2nx+1)), ny, nx, nlanes)
      (fwd .- bwd) ./ (2 .* eps3)                    # (ny, nx, nlanes)
    else
      y0 = reshape(view(y1, :, 1:nlanes), ny, 1, nlanes)
      (fwd .- y0) ./ eps3
    end
    # CSC nzval layout for block diagonal: reshape(nzval, ny, nx, nlanes) = diff directly
    reshape(nonzeros(jac), ny, nx, nlanes) .= diff
  end
  return jac
end

# Jacobian calls now just pass thru straight to the sparse backend:

# One argument

function DI.jacobian!(
    f::F, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
  return _value_and_jacobian_aux!((fc,), jac, prep, backend, x)[2]
end

function DI.jacobian(
    f::F, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc,), jac, prep, backend, x)[2]
end

function DI.value_and_jacobian(
    f::F, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc,), jac, prep, backend, x)
end

function DI.value_and_jacobian!(
    f::F, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f, prep, backend, x, contexts...)
  fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
  return _value_and_jacobian_aux!((fc,), jac, prep, backend, x)
end


## Two arguments

function DI.jacobian!(
    f!::F, y, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)
end

function DI.jacobian(
    f!::F, y, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)
end

function DI.value_and_jacobian(
    f!::F, y, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)
end

function DI.value_and_jacobian!(
    f!::F, y, jac, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)
end
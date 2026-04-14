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
  return BatchFiniteDiffJacobianPrep(_sig, batchdim, nx, ny, batchsize, x1, y1, relstep, absstep, epsilon, contexts_cache)
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

_set_context!(context_cache_data::Number, context_data::Number, batchdim) = context_data
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
    y = prep.y1[ntuple(i -> i == batchdim ? size(x, batchdim) : 1:size(prep.y1, otherdim), Val{2}())...]
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

function compute_batch_fdj_jac!(
    jac::SparseArrays.AbstractSparseMatrixCSC,
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
    # reshape fwd block (nlanes*nx, ny) -> (nlanes, nx, ny) col-major:
    #   fwd_3d[lane, var, out] = f(x + eps*e_var)[lane, out]
    #
    # CSC columns for batchdim=1 (banded Jacobian):
    #   col j (0-indexed)  <->  var = j ÷ nlanes,  lane = j % nlanes
    #   nzval = vec(permutedims(tang, (3,1,2)))
    #         = vec of (ny, nlanes, nx) array col-major
    #
    # Strategy:
    #   1. Compute tang via broadcasting   (1 allocation, size nlanes*nx*ny)
    #   2. permutedims!(nzval_3d, tang, (3,1,2))   (GPU kernel, writes into nzval)
    # ------------------------------------------------------------------

    fwd_3d   = reshape(@view(y1[(nlanes + 1):(nlanes*(nx + 1)), :]),
                       nlanes, nx, ny)           # (lane, var, out) — no-copy reshape
    primal_3d = reshape(@view(y1[1:nlanes, :]),
                        nlanes, 1, ny)           # (lane, 1, out)  — broadcast-ready
    eps_3d   = reshape(@view(epsilon[:, 1]),
                       nlanes, 1, 1)             # (lane, 1, 1)    — broadcast-ready

    tang = if mode == Val{:central}
      rwd_3d = reshape(@view(y1[(nlanes*(nx + 1) + 1):(nlanes*(2*nx + 1)), :]),
                       nlanes, nx, ny)
      (fwd_3d .- rwd_3d) ./ (2 .* eps_3d)
    else  # :forward
      (fwd_3d .- primal_3d) ./ eps_3d
    end

    # Reshape nzval into (ny, nlanes, nx) — a no-copy view into the same memory.
    # permutedims!(dest, src, (3,1,2)) writes tang[lane,var,out] -> dest[out,lane,var],
    # which is exactly the col-major column-by-column CSC layout for batchdim=1.
    nzval_3d = reshape(nzval, ny, nlanes, nx)
    permutedims!(nzval_3d, tang, (3, 1, 2))

  else  # batchdim == 2
    # ------------------------------------------------------------------
    # y1 layout: shape (ny, nlanes*(1 + k*nx))
    #
    #   Cols 1:nlanes                          - primal outputs
    #   Cols nlanes+1 : nlanes*(nx+1)          - fwd-perturbed (all nx vars)
    #   Cols nlanes*(nx+1)+1 : nlanes*(2*nx+1) - rwd-perturbed (central only)
    #
    # reshape fwd block (ny, nlanes*nx) -> (ny, nlanes, nx) col-major:
    #   fwd_3d[out, lane, var] = f(x + eps*e_var)[out, lane]
    #
    # CSC columns for batchdim=2 (block-diagonal Jacobian):
    #   col j (0-indexed)  <->  lane = j ÷ nx,  var = j % nx
    #   nzval = vec(permutedims(tang, (1,3,2)))
    #         = vec of (ny, nx, nlanes) array col-major
    #
    # Strategy: same as batchdim=1, different permutation and reshape target.
    # ------------------------------------------------------------------

    fwd_3d    = reshape(@view(y1[:, (nlanes + 1):(nlanes*(nx + 1))]),
                        ny, nlanes, nx)          # (out, lane, var) — no-copy reshape
    primal_3d = reshape(@view(y1[:, 1:nlanes]),
                        ny, nlanes, 1)           # (out, lane, 1)   — broadcast-ready
    eps_3d    = reshape(@view(epsilon[1, :]),
                        1, nlanes, 1)            # (1, lane, 1)     — broadcast-ready

    tang = if mode == Val{:central}
      rwd_3d = reshape(@view(y1[:, (nlanes*(nx + 1) + 1):(nlanes*(2*nx + 1))]),
                       ny, nlanes, nx)
      (fwd_3d .- rwd_3d) ./ (2 .* eps_3d)
    else  # :forward
      (fwd_3d .- primal_3d) ./ eps_3d
    end

    # permutedims!(dest, src, (1,3,2)) writes tang[out,lane,var] -> dest[out,var,lane],
    # which is exactly the col-major column-by-column CSC layout for batchdim=2.
    nzval_3d = reshape(nzval, ny, nx, nlanes)
    permutedims!(nzval_3d, tang, (1, 3, 2))
  end

  return jac
end

#=
function compute_batch_fdj_jac!(
    jac::SparseArrays.AbstractSparseMatrixCSC,
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
  colptr = SparseArrays.getcolptr(jac)

  if batchdim == 1
    # ------------------------------------------------------------------
    # y1 layout: shape (nlanes*(1 + k*nx), ny),  k=1 fwd / k=2 central
    #
    #   Rows 1:nlanes                          - primal outputs
    #   Rows nlanes+1 : nlanes*(nx+1)          - fwd-perturbed (all nx vars)
    #   Rows nlanes*(nx+1)+1 : nlanes*(2nx+1)  - rwd-perturbed (central only)
    #
    # Reshaping the fwd block (nlanes*nx, ny) to (nlanes, nx, ny) in Julia
    # column-major order gives:
    #   fwd_3d[lane, var, out] = f(x + eps*e_var)[lane, out]
    #
    # CSC column ordering for batchdim=1 (banded Jacobian):
    #   col j (0-indexed)  <->  var = j div nlanes,  lane = j mod nlanes
    #
    # Each column j holds ny contiguous nzval entries:
    #   nzval[colptr[j+1] : colptr[j+1]+ny-1] = tang[lane, var, :]
    #
    # This equals the col-major linearisation of permutedims(tang, (3,1,2))
    # i.e. shape (ny, nlanes, nx), but we write it with a direct loop to
    # avoid any temporary allocation.
    # ------------------------------------------------------------------

    fwd_flat = @view y1[(nlanes + 1):(nlanes*(nx + 1)), :]  # (nlanes*nx, ny)
    fwd_3d   = reshape(fwd_flat, nlanes, nx, ny)             # (lane, var, out)

    # epsilon has shape (nlanes, 1) for batchdim=1
    eps_vec  = @view epsilon[:, 1]                           # length nlanes

    if mode == Val{:central}
      rwd_flat = @view y1[(nlanes*(nx + 1) + 1):(nlanes*(2*nx + 1)), :]
      rwd_3d   = reshape(rwd_flat, nlanes, nx, ny)

      for var in 0:(nx - 1)
        for lane in 0:(nlanes - 1)
          j       = var * nlanes + lane    # 0-indexed CSC column
          col_lo  = colptr[j + 1]         # 1-indexed start in nzval
          two_eps = 2 * eps_vec[lane + 1]
          for out in 1:ny
            nzval[col_lo + out - 1] =
              (fwd_3d[lane + 1, var + 1, out] - rwd_3d[lane + 1, var + 1, out]) / two_eps
          end
        end
      end

    else  # :forward
      primal_mat = @view y1[1:nlanes, :]                    # (nlanes, ny)

      for var in 0:(nx - 1)
        for lane in 0:(nlanes - 1)
          j      = var * nlanes + lane
          col_lo = colptr[j + 1]
          eps_lane = eps_vec[lane + 1]
          for out in 1:ny
            nzval[col_lo + out - 1] =
              (fwd_3d[lane + 1, var + 1, out] - primal_mat[lane + 1, out]) / eps_lane
          end
        end
      end
    end

  else  # batchdim == 2
    # ------------------------------------------------------------------
    # y1 layout: shape (ny, nlanes*(1 + k*nx))
    #
    #   Cols 1:nlanes                          - primal outputs
    #   Cols nlanes+1 : nlanes*(nx+1)          - fwd-perturbed (all nx vars)
    #   Cols nlanes*(nx+1)+1 : nlanes*(2nx+1)  - rwd-perturbed (central only)
    #
    # Reshaping the fwd block (ny, nlanes*nx) to (ny, nlanes, nx) in Julia
    # column-major order gives:
    #   fwd_3d[out, lane, var] = f(x + eps*e_var)[out, lane]
    #
    # CSC column ordering for batchdim=2 (block-diagonal Jacobian):
    #   col j (0-indexed)  <->  lane = j div nx,  var = j mod nx
    #
    # Each column j holds ny contiguous nzval entries:
    #   nzval[colptr[j+1] : colptr[j+1]+ny-1] = tang[:, lane, var]
    #
    # This equals the col-major linearisation of permutedims(tang, (1,3,2))
    # i.e. shape (ny, nx, nlanes), written as a direct loop.
    # ------------------------------------------------------------------

    fwd_flat = @view y1[:, (nlanes + 1):(nlanes*(nx + 1))]  # (ny, nlanes*nx)
    fwd_3d   = reshape(fwd_flat, ny, nlanes, nx)             # (out, lane, var)

    # epsilon has shape (1, nlanes) for batchdim=2
    eps_vec  = @view epsilon[1, :]                           # length nlanes

    if mode == Val{:central}
      rwd_flat = @view y1[:, (nlanes*(nx + 1) + 1):(nlanes*(2*nx + 1))]
      rwd_3d   = reshape(rwd_flat, ny, nlanes, nx)

      for lane in 0:(nlanes - 1)
        for var in 0:(nx - 1)
          j       = lane * nx + var        # 0-indexed CSC column
          col_lo  = colptr[j + 1]
          two_eps = 2 * eps_vec[lane + 1]
          for out in 1:ny
            nzval[col_lo + out - 1] =
              (fwd_3d[out, lane + 1, var + 1] - rwd_3d[out, lane + 1, var + 1]) / two_eps
          end
        end
      end

    else  # :forward
      primal_mat = @view y1[:, 1:nlanes]                    # (ny, nlanes)

      for lane in 0:(nlanes - 1)
        for var in 0:(nx - 1)
          j        = lane * nx + var
          col_lo   = colptr[j + 1]
          eps_lane = eps_vec[lane + 1]
          for out in 1:ny
            nzval[col_lo + out - 1] =
              (fwd_3d[out, lane + 1, var + 1] - primal_mat[out, lane + 1]) / eps_lane
          end
        end
      end
    end
  end

  return jac
end
=#

# Jacobian calls now just pass thru straight to the sparse backend:

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
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)
end

function DI.jacobian(
    f!::F, y, prep::BatchFiniteDiffJacobianPrep, backend::AutoBatch{<:AutoFiniteDiff}, x, contexts::Vararg{DI.Context, C},
  ) where {F, C}
  DI.check_prep(f!, y, prep, backend, x, contexts...)
  set_contexts!(prep.contexts_cache, contexts, backend.batchdim)
  fc! = DI.fix_tail(f!, map(DI.unwrap, prep.contexts_cache)...)
  jac = similar(sparsity_pattern(prep), eltype(prep.y1))
  return _value_and_jacobian_aux!((fc!, y), jac, prep, backend, x)
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
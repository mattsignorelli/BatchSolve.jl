struct AutoBatch{D<:AbstractADType} <: AbstractADType
  dense_ad::D
  batchdim::Int  # 1 = SoA, 2 = AoS
end

AutoBatch(dense_ad; batchdim=1) = AutoBatch{typeof(dense_ad)}(dense_ad, batchdim)

struct BatchJacobianPrep{SIG, A, P} <: DI.JacobianPrep{SIG}
  _sig::Val{SIG}
  batchdim::Int
  sparse_ad::A
  sparse_prep::P
end

ADTypes.dense_ad(backend::AutoBatch) = backend.dense_ad
SparseMatrixColorings.sparsity_pattern(prep::BatchJacobianPrep) = sparsity_pattern(prep.sparse_prep)

function DI.prepare_jacobian_nokwarg(
    strict::Val, f::F, backend::AutoBatch, x, contexts::Vararg{DI.Context, C}
  ) where {F, C}
  y = f(x, map(DI.unwrap, contexts)...)
  return _prepare_batch_jacobian_aux(strict, y, f, backend, x, contexts...)
end

function DI.prepare_jacobian_nokwarg(
    strict::Val, f!::F, y, backend::AutoBatch, x, contexts::Vararg{DI.Context, C}
  ) where {F, C}
  return _prepare_batch_jacobian_aux(strict, y, (f!,y), backend, x, contexts...)
end

function _prepare_batch_jacobian_aux(
    strict::Val,
    y,
    f_or_f!y::FY,
    backend::AutoBatch,
    x,
    contexts::Vararg{DI.Context, C}
  ) where {FY, C}
  batchdim = backend.batchdim
  dense_ad = backend.dense_ad

  if !(batchdim in (1,2))
    error("batchdim must be either 1 or 2")
  end

  # Sanity checks
  if size(x, batchdim) != size(y, batchdim)
      error("Input/output matrix size mismatch for AutoBatch: size along batchdim = $batchdim for the
              input and output must be equal. Received size(x, $batchdim) = $(size(x, batchdim)) and 
              size(y, $batchdim) = $(size(y, batchdim)).")
  end

  batchsize = size(x, batchdim)
  otherdim = mod(batchdim, 2) + 1
  n_rows = size(y, otherdim)
  n_cols = size(x, otherdim)

  # Make it on the CPU
  nnz = batchsize * n_rows * n_cols
  rows = Vector{Int}(undef, nnz)
  cols = Vector{Int}(undef, nnz)

  rs, ri = batchdim == 1 ? (1, batchsize) : (n_rows, 1)
  cs, ci = batchdim == 1 ? (1, batchsize) : (n_cols, 1)

  idx = 1
  for i in 1:batchsize
      for r in 1:n_rows
          for c in 1:n_cols
              rows[idx] = (i-1)*rs + (r-1)*ri + 1
              cols[idx] = (i-1)*cs + (c-1)*ci + 1
              idx += 1
          end
      end
  end
  d_rows = similar(y, Int, nnz)
  d_cols = similar(y, Int, nnz)
  d_mat = similar(y, Bool, nnz)
  copyto!(d_rows, rows)
  copyto!(d_cols, cols)
  d_mat .= 1

  pattern = sparse(d_rows, d_cols, d_mat, batchsize*n_rows, batchsize*n_cols)
  color = (batchdim == 1) ? repeat(1:n_cols, inner=batchsize) : repeat(1:n_cols, outer=batchsize) 
  alg = ConstantColoringAlgorithm(pattern, color; partition=:column)

  detector = ADTypes.KnownJacobianSparsityDetector(pattern)
  sparse_ad = AutoSparse(dense_ad; 
    sparsity_detector=detector,
    coloring_algorithm=alg,
  )

  sparse_prep = DI.prepare_jacobian(f_or_f!y..., sparse_ad, x, contexts...; strict)
   _sig = DI.signature(f_or_f!y..., backend, x, contexts...; strict)
  return BatchJacobianPrep(_sig, batchdim, sparse_ad, sparse_prep)
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
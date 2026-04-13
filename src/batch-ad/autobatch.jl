struct AutoBatch{D<:AbstractADType} <: AbstractADType
  dense_ad::D
  batchdim::Int  # 1 = SoA, 2 = AoS
end

AutoBatch(_dense_ad; batchdim=1) = AutoBatch{typeof(_dense_ad)}(_dense_ad, batchdim)

struct BatchJacobianPrep{SIG, A, P} <: DI.JacobianPrep{SIG}
  _sig::Val{SIG}
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
  batchsize = size(x, batchdim)
  otherdim = mod(batchdim, 2) + 1
  n_cols = size(x, otherdim)
  pattern = make_pattern(x, y, batchdim)
  color = (batchdim == 1) ? repeat(1:n_cols, inner=batchsize) : repeat(1:n_cols, outer=batchsize) 
  alg = ConstantColoringAlgorithm(pattern, color; partition=:column)
  detector = ADTypes.KnownJacobianSparsityDetector(pattern)
  sparse_ad = AutoSparse(dense_ad(backend); 
    sparsity_detector=detector,
    coloring_algorithm=alg,
  )

  sparse_prep = DI.prepare_jacobian(f_or_f!y..., sparse_ad, x, contexts...; strict)
   _sig = DI.signature(f_or_f!y..., backend, x, contexts...; strict)
  return BatchJacobianPrep(_sig, sparse_ad, sparse_prep)
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
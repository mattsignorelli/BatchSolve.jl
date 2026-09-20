# ---------------------------------------------------------------------------
# gels_strided_batched!
#
# cuBLAS has gelsBatched (batched least squares via QR) but CUDA.jl only wraps
# it for Vector{<:CuMatrix}, which forces you to build one view per matrix.
# Like getrs_strided_batched!, this version instead takes strided 3D arrays and
# builds the device pointer arrays with CUBLAS.unsafe_strided_batch.
#
# Solves  min_X ||A*X - C||  for each batch element, with A::(m, n, batch),
# m >= n, C::(m, nrhs, batch). On return A holds the QR factors and the
# solution X is in the first n rows of C. Only trans = 'N' is supported by cuBLAS.
# ---------------------------------------------------------------------------
 
# Thin per-eltype dispatch onto the low-level cuBLAS calls.
for (fname, elty) in ((:cublasSgelsBatched, :Float32),
                      (:cublasDgelsBatched, :Float64),
                      (:cublasCgelsBatched, :ComplexF32),
                      (:cublasZgelsBatched, :ComplexF64))
  @eval function _cublas_gels_batched(trans::Char, m, n, nrhs,
                                      Aptrs::CuVector{CuPtr{$elty}}, lda,
                                      Cptrs::CuVector{CuPtr{$elty}}, ldc,
                                      infoarray::CuVector{Cint})
    info = Ref{Cint}()   # host-side: only reports invalid arguments
    CUBLAS.$fname(CUBLAS.handle(), trans, m, n, nrhs, Aptrs, lda, Cptrs, ldc,
                  info, infoarray, length(Aptrs))
    info[] == 0 || throw(ArgumentError("cuBLAS gelsBatched: invalid argument $(-info[])"))
    return nothing
  end
end
 
function _check_gels_args(trans, A, C)
  trans == 'N' || throw(ArgumentError("cuBLAS gelsBatched only supports trans = 'N'"))
  m, n, batchsize = size(A)
  m >= n || throw(DimensionMismatch("gelsBatched requires m >= n (got $m x $n)"))
  size(C, 1) == m || throw(DimensionMismatch("Rows in A and C must be equal!"))
  size(C, 3) == batchsize || throw(DimensionMismatch("Batch sizes of A and C must be equal!"))
  return m, n, size(C, 2), batchsize
end
 
# Fast path: caller supplies the pointer arrays and info array, so nothing is
# allocated or uploaded per call. Aptrs/Cptrs must have been built from these
# exact A and C (via CUBLAS.unsafe_strided_batch) and A, C must not be reallocated.
function gels_strided_batched!(trans::Char,
                               A::CUDA.DenseCuArray{T,3},
                               C::CUDA.DenseCuArray{T,3},
                               Aptrs::CuVector{CuPtr{T}},
                               Cptrs::CuVector{CuPtr{T}},
                               infoarray::CuVector{Cint}) where {T<:CUBLAS.CublasFloat}
  m, n, nrhs, batchsize = _check_gels_args(trans, A, C)
  length(infoarray) == batchsize || throw(DimensionMismatch("infoarray must have length batchsize"))
  lda = max(1, stride(A, 2))
  ldc = max(1, stride(C, 2))
  _cublas_gels_batched(trans, m, n, nrhs, Aptrs, lda, Cptrs, ldc, infoarray)
  return A, C, infoarray
end
 
# Convenience method, same style as CUDA.jl's getrs_strided_batched!:
# builds the pointer arrays on every call.
function gels_strided_batched!(trans::Char,
                               A::CUDA.DenseCuArray{T,3},
                               C::CUDA.DenseCuArray{T,3},
                               infoarray::CuVector{Cint}) where {T<:CUBLAS.CublasFloat}
  _, _, _, batchsize = _check_gels_args(trans, A, C)
  Aptrs = CUBLAS.unsafe_strided_batch(A)
  Cptrs = CUBLAS.unsafe_strided_batch(C)
  gels_strided_batched!(trans, A, C, Aptrs, Cptrs, infoarray)
  CUDA.unsafe_free!(Aptrs)
  CUDA.unsafe_free!(Cptrs)
  return A, C, infoarray
end

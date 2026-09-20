module BatchSolveCUDAExt
using CUDA
import BatchSolve: make_linear_solver

include("gels_strided_batched.jl")

function make_linear_solver(device::CUDA.CUDABackend, _b, _x, batchdim::Integer)
  _lx = length(_x)
  _ly = length(_b)
  _nan = eltype(_b)(NaN)

  if batchdim != 1 && batchdim != 2
    error("Invalid batchdim (must be either 1 or 2 for CUDA)")
  end

  # Batch:
  if _ly != _lx
    # Non-square
    if batchdim == 2
      # Each element of a batch is a COLUMN
      # Number of rows = number of variables in an element of a batch
      _batchsize = size(_x, 2)
      _n_rows = size(_b, 1)
      _n_cols = size(_x, 1)
      if _n_rows < _n_cols
        error("CUDA batched linear solver not yet implemented for underdetermined (n_output < n_input) systems")
      end
      _info = CUDA.zeros(Cint, _batchsize)
      _Ascratch = similar(_b, _n_rows, _n_cols, _batchsize)
      _rhs = similar(_b, _n_rows, 1, _batchsize)
      let info=_info, batchsize=_batchsize, n_rows=_n_rows, n_cols=_n_cols, Ascratch=_Ascratch, rhs=_rhs, nan=_nan
        return (x, A, b) -> begin
          Ascratch .= reshape(A.nzVal, n_rows, n_cols, batchsize)
          rhs .= reshape(b, n_rows, 1, batchsize)
          gels_strided_batched!('N', Ascratch, rhs, info)
          # Solution is in the first n_cols rows of rhs
          sol = view(rhs, 1:n_cols, 1, :)   # (n_cols, batchsize)
          reshape(x, n_cols, batchsize) .= ifelse.(reshape(info, 1, batchsize) .!= 0, nan, sol)
          return x
        end
      end
    else # batchdim == 1
      _batchsize = size(_x, 1)
      _n_rows = size(_b, 2)
      _n_cols = size(_x, 2)
      if _n_rows < _n_cols
        error("CUDA batched linear solver not yet implemented for underdetermined (n_output < n_input) systems")
      end
      _info = CUDA.zeros(Cint, _batchsize)
      _Ascratch = similar(_b, _n_rows, _n_cols, _batchsize)
      _rhs = similar(_b, _n_rows, 1, _batchsize)
      let info=_info, batchsize=_batchsize, n_rows=_n_rows, n_cols=_n_cols, Ascratch=_Ascratch, rhs=_rhs, nan=_nan
        return (x, A, b) -> begin
          nzval_3d = reshape(A.nzVal, n_rows, batchsize, n_cols)  # (n_rows, batchsize, n_cols)
          permutedims!(Ascratch, nzval_3d, (1, 3, 2))             # -> (n_rows, n_cols, batchsize)
          permutedims!(rhs, reshape(b, batchsize, 1, n_rows), (3, 2, 1))
          gels_strided_batched!('N', Ascratch, rhs, info)
          sol = view(rhs, 1:n_cols, 1, :)   # (n_cols, batchsize)
          # x is (batchsize, n_cols), so transpose the solution (lazily, inside the broadcast)
          reshape(x, batchsize, n_cols) .= ifelse.(reshape(info, batchsize, 1) .!= 0, nan, transpose(sol))
          return x
        end
      end
    end
  else
    # Square
    if batchdim == 2
      # Each element of a batch is a COLUMN
      # Number of rows = number of variables in an element of a batch
      _batchsize = size(_x, 2)
      _n = size(_x, 1)
      _pivot = CUDA.zeros(Int32, _n, _batchsize)
      _info = CUDA.zeros(Int32, _batchsize)
      _Ascratch = similar(_b, _n, _n, _batchsize)
      _rhs = similar(_b, _n, 1, _batchsize)
      let pivot=_pivot, info=_info, batchsize=_batchsize, n=_n, Ascratch=_Ascratch, rhs=_rhs, nan=_nan
        return (x, A, b)-> begin
          Ascratch .= reshape(A.nzVal, n, n, batchsize)
          rhs .= reshape(b, n, 1, batchsize)
          CUBLAS.getrf_strided_batched!(Ascratch, pivot, info)
          CUBLAS.getrs_strided_batched!('N', Ascratch, rhs, pivot)
          x .= reshape(ifelse.(reshape(info, 1, batchsize) .!= 0, nan, reshape(rhs, n, batchsize)), :)
          return x
        end
      end
    else # batchdim == 1
      _batchsize = size(_x, 1)
      _n = size(_b, 2)
      _pivot = CUDA.zeros(Int32, _n, _batchsize)
      _info = CUDA.zeros(Int32, _batchsize)
      _Ascratch = similar(_b, _n, _n, _batchsize)
      _rhs = similar(_b, _n, 1, _batchsize)
      _rhs_perm = similar(_b, _batchsize, 1, _n)
      let pivot=_pivot, info=_info, batchsize=_batchsize, n=_n, Ascratch=_Ascratch, rhs=_rhs, rhs_perm=_rhs_perm, nan=_nan
        return (x, A, b) -> begin
          nzval_3d = reshape(A.nzVal, n, batchsize, n)  # (n_rows, batchsize, n_cols)
          permutedims!(Ascratch, nzval_3d, (1, 3, 2))  # → (n_rows, n_cols, batchsize)
          # Also need to permute b dims from (batchsize, 1, n_rows) to (n_rows, 1, batchsize)
          permutedims!(rhs, reshape(b, batchsize, 1, n), (3, 2, 1))
          CUBLAS.getrf_strided_batched!(Ascratch, pivot, info)
          CUBLAS.getrs_strided_batched!('N', Ascratch, rhs, pivot)
          # Now need to permutedims back
          permutedims!(rhs_perm, rhs, (3, 2, 1))
          # ready to go
          x .= ifelse.(reshape(info, batchsize, 1) .!= 0, nan, reshape(rhs_perm, batchsize, n))
          return x
        end
      end
    end
  end
end

end
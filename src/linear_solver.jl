
"""
    make_linear_solver(device, b, x, batchdim) -> Function

Construct and return a linear-system solver callable compatible with the given device,
array shapes, and batch configuration. The returned function has the signature

    (x, A, b) -> x

and solves `A * x = b` in-place, writing the solution into `x`.

# Arguments
- `device`: KernelAbstractions backend (e.g. `CPU()`, `CUDABackend()`). 
- `b`: Prototype output array (used for size introspection; not mutated).
- `x`: Prototype input array (used for size introspection; not mutated).
- `batchdim`: Batch dimension (`nothing`, `1`, or `2`).

# Returned solver behaviour
| `batchdim` | Matrix type | Behaviour |
|------------|---------------|-----------|
| `nothing`  | dense matrix  | Single `A \\ b` solve; writes `NaN` when `A` is singular. |
| `2`        | `SparseMatrixCSC` (block-diagonal, blocks contiguous in `nzval`) | Iterates over batch index `i`, extracts each `(n_rows × n_cols)` block from `nzval`, solves independently. |
| `1`        | `SparseMatrixCSC` (interleaved columns) | Iterates over batch index `i`, gathers every `batchsize`-th column, solves independently. |

Singular sub-matrices (detected via `ArrayInterface.issingular`) result in `NaN` being
written to the corresponding slice of `x` so that upstream code can detect and handle
failures gracefully.

# Errors
Throws an `ArgumentError`-style error if `batchdim ∉ {nothing, 1, 2}`.
"""
function make_linear_solver(device, _b, _x, batchdim)
  _lx = length(_x)
  _lb = length(_b)
  _nan = eltype(_b)(NaN)
  if isnothing(batchdim)
    let lx=_lx, lb=_lb, nan=_nan
      return (x, A, b)->begin
        if ArrayInterface.issingular(A) || any(isnan, A) || any(isinf, A)
          x .= nan
        else
          reshape(x, lx) .= A \ reshape(b, lb)
        end
        return x
      end
    end
  elseif batchdim == 2 # Do each serially
    _batchsize = size(_x, 2)
    _n_rows = size(_b, 1)
    _n_cols = size(_x, 1)
    let n_rows=_n_rows, n_cols=_n_cols, batchsize=_batchsize, Asize=_n_rows*_n_cols, nan=_nan
      return (x, A::SparseMatrixCSC, b)->begin
        for i in 1:batchsize
          A_offset = (i-1)*Asize 
          curA = reshape(view(A.nzval, (A_offset+1):(A_offset+Asize)), (n_rows, n_cols))
          x_offset = (i-1)*n_cols
          b_offset = (i-1)*n_rows
          if ArrayInterface.issingular(curA) || any(isnan, curA) || any(isinf, curA)
            view(x, (x_offset+1):(x_offset+n_cols)) .= nan
          else
            view(x, (x_offset+1):(x_offset+n_cols)) .= curA \ view(b, (b_offset+1):(b_offset+n_rows))
          end
        end
        return x
      end
    end
  elseif batchdim == 1
    _batchsize = size(_x, 1)
    _n_rows = size(_b, 2)
    let n_rows=_n_rows, batchsize=_batchsize, xlen=length(_x), blen=length(_b), nan=_nan
      return (x, A::SparseMatrixCSC, b)->begin
        for i in 1:batchsize
          curA = view(reshape(A.nzval, n_rows, :), :, i:batchsize:xlen)
          if ArrayInterface.issingular(curA) || any(isnan, curA) || any(isinf, curA)
            view(x, i:batchsize:xlen) .= nan
          else
            view(x, i:batchsize:xlen) .= curA \ view(b, i:batchsize:blen)
          end
        end
        return x
      end
    end
  else
    error("Invalid batchdim (must be either 1, 2, or nothing)")
  end
end

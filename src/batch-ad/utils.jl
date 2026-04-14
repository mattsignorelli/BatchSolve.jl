function make_pattern(
  x, 
  y, 
  batchdim; 
  n_rows=size(y, mod(batchdim, 2) + 1), 
  n_cols=size(x, mod(batchdim, 2) + 1),
  batchsize=size(x, batchdim),
)
  if !(batchdim in (1,2))
    error("batchdim must be either 1 or 2")
  end

  # Sanity checks
  if size(x, batchdim) != size(y, batchdim)
      error("Input/output matrix size mismatch for AutoBatch: size along batchdim = $batchdim for the
              input and output must be equal. Received size(x, $batchdim) = $(size(x, batchdim)) and 
              size(y, $batchdim) = $(size(y, batchdim)).")
  end

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
  return pattern
end
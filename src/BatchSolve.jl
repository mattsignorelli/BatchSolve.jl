module BatchSolve
import DifferentiationInterface as DI
using Accessors,
    ArrayInterface,
    LinearAlgebra,
    ForwardDiff,
    ADTypes,
    SparseArrays,
    SparseMatrixColorings

# Wish I could use Enum but not GPU compatible.
const RETCODE_SUCCESS = 0x0
const RETCODE_FAILURE = 0x1
const RETCODE_MAXITERS = 0x2



end

module BatchSolve
import DifferentiationInterface as DI
using DifferentiationInterface: Constant, Cache, ConstantOrCache, Context
import KernelAbstractions as KA
using Reexport
@reexport using ADTypes
using Accessors,
    ADTypes,
    ArrayInterface,
    ForwardDiff,
    LinearAlgebra,
    StaticArrays,
    SparseArrays,
    SparseMatrixColorings

export Constant, Cache, ConstantOrCache, Context
export newton, newton!, brent, brent!

# Wish I could use Enum but not GPU compatible.
const RETCODE_SUCCESS = 0x0
const RETCODE_FAILURE = 0x1
const RETCODE_MAXITER = 0x2

include("rootfinders/newton.jl")
include("minimizers/brent.jl")

end

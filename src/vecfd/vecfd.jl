# Vectorized finite difference

# step is nothing if autoselect (including for Float32), else can be a specified number
@kwdef struct AutoVecFD{T}
    mode::Symbol = :central
    step::T      = nothing
end



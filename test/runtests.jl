using BatchSolve
using Test

@testset "Brent Minimizer" begin
    include("brent_test.jl")
end 

@testset "Newton Root Finder" begin
    include("newton_test.jl")
end 

@testset "AutoBatch{AutoFiniteDiff}" begin
    include("finitediff_test.jl")
end 

using BatchSolve
using Documenter

DocMeta.setdocmeta!(BatchSolve, :DocTestSetup, :(using BatchSolve); recursive=true)

makedocs(;
    modules=[BatchSolve],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="BatchSolve.jl",
    format=Documenter.HTML(;
        canonical="https://mattsignorelli.github.io/BatchSolve.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mattsignorelli/BatchSolve.jl",
    devbranch="main",
)

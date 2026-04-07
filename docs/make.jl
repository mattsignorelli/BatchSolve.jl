using BatchOpt
using Documenter

DocMeta.setdocmeta!(BatchOpt, :DocTestSetup, :(using BatchOpt); recursive=true)

makedocs(;
    modules=[BatchOpt],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="BatchOpt.jl",
    format=Documenter.HTML(;
        canonical="https://mattsignorelli.github.io/BatchOpt.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mattsignorelli/BatchOpt.jl",
    devbranch="main",
)

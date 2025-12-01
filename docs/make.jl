using Pkg
Pkg.activate(@__DIR__)
# As long as it is not registered, this is nice, in general it locally always
# renders docs of the current version checked out in this repo.
Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
using SparseSparse
using Documenter

DocMeta.setdocmeta!(SparseSparse, :DocTestSetup, :(using SparseSparse); recursive=true)

v = string(pkgversion(SparseSparse))

makedocs(;
    modules=[SparseSparse],
    authors="Sébastien Loisel",
    sitename="SparseSparse.jl $v",
    format=Documenter.HTML(;
        canonical="https://sloisel.github.io/SparseSparse.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sloisel/SparseSparse.jl",
    devbranch="main",
)

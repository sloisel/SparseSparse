```@meta
CurrentModule = SparseSparse
```

```@eval
using Markdown
using Pkg
using SparseSparse
v = string(pkgversion(SparseSparse))
md"# SparseSparse $v"
```

[SparseSparse](https://github.com/sloisel/SparseSparse) is a Julia package for inverting sparse matrices with sparse inverses, or solving sparse linear problems with sparse right-hand-sides.

## Installation

```julia
using Pkg; Pkg.add(url="https://github.com/sloisel/SparseSparse")
```

## Overview

A matrix $A$ is **doubly sparse** if both $A$ and $A^{-1}$ are sparse. The class of doubly sparse matrices includes all matrices of the form $P(\prod B_k) Q$ where $P,Q$ are permutations and $B_k$ are block diagonal.

With stock Julia, inverting a sparse matrix throws an error. `SparseSparse` provides functionality to invert sparse matrices and produce sparse inverses.

## Basic Usage

```julia
using LinearAlgebra, SparseArrays, SparseSparse
A = sparse([2.0  3.0  0.0  0.0
             4.0  5.0  0.0  0.0
             0.0  0.0  6.0  7.0
             0.0  0.0  8.0  9.0])
inv(A)
```

# Module reference

```@autodocs
Modules = [SparseSparse]
Order   = [:module]
```

# Types reference

```@autodocs
Modules = [SparseSparse]
Order   = [:type]
```

# Functions reference

```@autodocs
Modules = [SparseSparse]
Order   = [:function]
```

# Index

```@index
```

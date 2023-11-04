# SparseSparse

### Author: Sébastien Loisel

## Installation
```julia
using Pkg; Pkg.add(url="https://github.com/sloisel/SparseSparse")
```

# Introduction

We say that a matrix $A$ is **doubly sparse** if both $A$ and $A^{-1}$ are sparse. The class of doubly sparse matrices includes all matrices of the form $P(\prod B_k) Q$ where $P,Q$ are permutations and $B_k$ are block diagonal. `SparseSparse` is a package that inverts sparse matrices with sparse inverses, or otherwise solve sparse linear problems with sparse right-hand-sides.

With stock Julia, here is what happens if you try to invert a sparse matrix:

```julia
julia> using LinearAlgebra, SparseArrays
       A = sparse([2.0  3.0  0.0  0.0
                   4.0  5.0  0.0  0.0
                   0.0  0.0  6.0  7.0
                   0.0  0.0  8.0  9.0])
       inv(A)
The inverse of a sparse matrix can often be dense and can cause the computer to run out of memory[...]

Stacktrace:
 [1] error(s::String)
   @ Base ./error.jl:35
[...]
```

The above matrix `A` is block-diagonal so it has a sparse inverse. The `SparseSparse` package overrides `Base.inv` so that it inverts sparse matrices and produces a sparse inverse, as follows.
```julia
julia> using SparseSparse
       inv(A)

4×4 SparseMatrixCSC{Float64, Int64} with 8 stored entries:
 -2.5   1.5    ⋅     ⋅ 
  2.0  -1.0    ⋅     ⋅ 
   ⋅     ⋅   -4.5   3.5
   ⋅     ⋅    4.0  -3.0
```
 
The implementation is based on `SparseSparse.Factorization`:
```julia
julia> Factorization(A)
SparseSparse.Factorization(...)
```
The `Factorization` object is as follows:
```julia
struct Factorization{Tv,Ti<:Integer} 
    L::Union{Missing,SparseMatrixCSC{Tv,Ti}}
    U::Union{Missing,SparseMatrixCSC{Tv,Ti}}
    p::Union{Missing,Vector{Ti}}
    q::Union{Missing,Vector{Ti}}
end
```
Fields `L` and `U` are sparse lower and upper triangular matrices, and vectors `p` and `q` are permutations.
All these fields are optional and may take on the value `missing` when that particular term is omitted. Factorizations can be used to solve linear problems via
```julia
function Base.:\(A::Factorization, B::SparseMatrixCSC)
```

The underlying LU decomposition is computed using the stdlib's `lu` (or `ldlt` or `cholesky`). The problem is that stdlib refuses to compute `L\B` when `B` is itself sparse. The core of module `SparseSparse` is the following function:

```julia
function solve(L::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti};solvemode=detect,numthreads=min(B.n,nthreads())) where {Tv,Ti<:Integer}
```

This function is able to solve lower or upper triangular sparse systems with sparse right-hand-sides. The algorithm is similar to the one described in Tim Davis's book and implemented in SuiteSparse [here](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CXSparse/Source/cs_spsolve.c). The `SparseSparse` implementation also uses multithreading to speed up the solution time. The parameter `solvemode` should be either `lower=1`, `upper=2` or `detect=3`.

# Benchmarks

Here is a numerical experiment, calculating the inverse of a **doubly sparse matrix.**

```julia
julia> using Random
       A = blockdiag([sparse(randn(10,10)) for k=1:100]...)
       P = randperm(1000)
       Q = randperm(1000)
       A = A[P,Q]
1000×1000 SparseMatrixCSC{Float64, Int64} with 10000 stored entries:
[...]

julia> using BenchmarkTools
       @benchmark inv(Matrix(A))
BenchmarkTools.Trial: 155 samples with 1 evaluation.
 Range (min … max):  26.345 ms … 42.085 ms  ┊ GC (min … max): 0.00% … 13.34%
 Time  (median):     31.265 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   32.239 ms ±  3.650 ms  ┊ GC (mean ± σ):  6.67% ±  8.61%

        ▃  ▁█▆ █▃▁▄▁▄ ▄            ▄ ▃▁▃  ▁                    
  ▇▄▄▄▆▆█▇▇███▆██████▇█▇▇▄▄▆▆▆▇▄▄▄▁█▆███▄▆█▄▆▄▄▆▆▇▁▄▇▆▁▁▁▄▁▁▄ ▄
  26.3 ms         Histogram: frequency by time        40.9 ms <

 Memory estimate: 15.76 MiB, allocs estimate: 7.

julia> @benchmark(inv(A))
BenchmarkTools.Trial: 1859 samples with 1 evaluation.
 Range (min … max):  1.744 ms … 13.188 ms  ┊ GC (min … max):  0.00% … 67.60%
 Time  (median):     2.105 ms              ┊ GC (median):     0.00%
 Time  (mean ± σ):   2.670 ms ±  1.735 ms  ┊ GC (mean ± σ):  11.39% ± 13.40%

  ▆█▇▆▅▃▁                                                     
  ██████████████▇▇▆▇▄▅▁▄▄▄▄▄▅▄▅▄▁▅▅▄▅▆▅▆▆▆▄▆▆▆▇▅▇▄▆▅▄▄▆▄▄▅▆▄ █
  1.74 ms      Histogram: log(frequency) by time     10.6 ms <

 Memory estimate: 3.69 MiB, allocs estimate: 585.
```

In the case of this 1000x1000 matrix, the `SparseSparse` method is approximately 12x faster and requires 76% less memory, compared to using dense algorithms.

# Applications

We now briefly sketch an application of `SparseSparse` on Woodbury and $H$ matrices. Note that these matrix types are not part of the `SparseSparse` module.

The [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) states that the inverse of $A+UCV$ is $$(A+UCV)^{-1} = A^{-1}-A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}.$$ We can implement this in Julia as
```julia
struct Woodbury A; U; C; V end
```
Then, the inverse of a Woodbury matrix is
```julia
function Base.inv(X::Woodbury)
    Ainv = inv(X.A)
    Woodbury(Ainv,Ainv*X.U,-inv(inv(X.C)+X.V*Ainv*X.U),X.V*Ainv) 
end
```
Such matrices have a complete algebra of operations; we can add, subtract and multiply them, and thanks to the Woodbury identity, we can also invert them.

This is related to Hierarchical matrices (see the book by Hackbusch). Consider a matrix

$$\left[\begin{array}{cc}
X & Y \\\ Z & W
\end{array}
\right]
\approx
\left[\begin{array}{cc}
X & U_1 S_1 V_1 \\\ U_2 S_2 V_2 & W
\end{array}
\right].$$

Here, $Y \approx U_1S_1V_1$ is a low rank approximation obtained by sparse SVD, e.g. using `Arpack`. This approximation can efficiently be represented by a Woodbury decomposition. The decomposition is applied recursively to sub-blocks $X$ and $W$, resulting in an $H$-matrix. Here's an example with the 1d Laplacian:

```julia
julia> n = 4
       A = spdiagm(0=>2*ones(n),1=>-ones(n-1),-1=>-ones(n-1))
       H = hmat(A;r=1)
       Matrix(H)
4×4 Matrix{Float64}:
  2.0  -1.0   0.0   0.0
 -1.0   2.0  -1.0  -1.11022e-16
  0.0  -1.0   2.0  -1.0
  0.0   0.0  -1.0   2.0
```
In this case, the $H$-matrix representation of $A$ happens to be exact.


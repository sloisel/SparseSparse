using SparseSparse
using Test
using Random
using LinearAlgebra, SparseArrays

@testset "SparseSparse.jl" begin
    n = 15
    m = 10
    Random.seed!(1234)
    tols = [1e-11,1e-11]
    types = [Float64,ComplexF64]
    for j=1:length(types)
        tol = tols[j]
        A = blockdiag([sparse(randn(types[j],m,m)) for k=1:n]...)
        B = blockdiag([sparse(randn(types[j],n,n)) for k=1:m]...)
        C = Matrix(A)\Matrix(B)
        D = Matrix(A)/Matrix(B)
        @test norm(A\B-C)<tol
        @test norm(A/B-D)<tol
        @test norm(inv(A)*B-C)<tol
        @test norm(tril(A)\tril(A)-I)<tol
        @test norm(tril(A)/tril(A)-I)<tol
        @test norm(triu(A)\triu(A)-I)<tol
        @test norm(triu(A)/triu(A)-I)<tol
        D = A'*A
        @test norm(D\D-I)<tol
        @test norm(D/D-I)<tol
        E = A+A'
        @test norm(E\E-I)<tol
        @test norm(E/E-I)<tol
        Z = spzeros(types[j],size(A))
        G = hvcat((2,2),Z,A,A',Z)
        @test norm(G\G-I)<tol
        @test norm(G/G-I)<tol
        P = randperm(size(A,1))
        Q = randperm(size(A,2))
        U = A[P,Q]
        @test norm(U\U-I)<tol
        @test norm(U/U-I)<tol
        b = sparse(randn(types[j],size(A,1)))
        x = A\b
        @test norm(b-A*x)<tol
    end
end

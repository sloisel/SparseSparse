module SparseSparse

export Factorization,solve

using SparseArrays, LinearAlgebra
using Base.Threads: nthreads, @threads

function transitiveclosure(L::SparseMatrixCSC{Tv,Ti},Jlen,CJ,mark;countonly=false) where {Tv,Ti<:Integer}
    b = 0
    c = Jlen
    for i=1:Jlen
        mark[CJ[i]] = true
    end
    cp = L.colptr
    rv = L.rowval
    while b<c
        b+=1
        i = CJ[b]
        p = cp[i]
        q = cp[i+1]-1
        for i=p:q
            j = rv[i]
            if !mark[j]
                c+=1
                mark[j]=true
                CJ[c]=j
            end
        end
    end
    for i=1:c
        mark[CJ[i]] = false
    end
    if countonly
        return c
    end
    sort!(view(CJ,1:c))
    return c
end
function solvevec(L::SparseMatrixCSC{Tv,Ti},lowertriangular,x::Vector{Tv},J,CJ,mark) where {Tv,Ti<:Integer}
    n = L.n
    m = length(J)
    for i=1:m
        CJ[i] = J[i]
    end
    c = transitiveclosure(L,m,CJ,mark)
    cp = L.colptr
    rv = L.rowval
    nz = L.nzval
    if lowertriangular
        (a,b,dir) = (1,c,1)
    else
        (a,b,dir) = (c,1,-1)
    end
    for i=a:dir:b
        j = CJ[i]
        p = cp[j]
        q = cp[j+1]-1
        if lowertriangular
            x[j] /= nz[p]
            p+=1
        else
            x[j] /= nz[q]
            q-=1
        end
        alpha = x[j]
        for k=p:q
            x[rv[k]] -= alpha*nz[k]
        end
    end
    return c
end
function solvemat(L::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti};lowertriangular=true) where {Tv,Ti<:Integer}
    cp = B.colptr
    rv = B.rowval
    nz = B.nzval
    Ns = Vector{Ti}(undef,B.n)
    CJ = zeros(Ti,L.n)
    mark = falses(L.n)
    for i in 1:B.n
        p = cp[i]
        q = cp[i+1]-1
        Jlen = q-p+1
        for j=1:Jlen
            CJ[j]=rv[j+p-1]
        end
        Ns[i]=transitiveclosure(L,Jlen,CJ,mark,countonly=true)
    end
    N = sum(Ns)
    CP = Vector{Ti}(undef,B.n+1)
    CP[1] = 1
    for i=1:B.n
        CP[i+1] = CP[i]+Ns[i]
    end
    RV = Vector{Ti}(undef,N)
    NZ = Vector{Tv}(undef,N)
    x = zeros(Tv,L.n)
    for i = 1:B.n
        p = cp[i]
        q = cp[i+1]-1
        J = view(rv,p:q)
        for j=1:length(J)
            x[J[j]] = nz[p+j-1]
        end
        d = solvevec(L,lowertriangular,x,J,CJ,mark)
        c = CP[i]-1
        for j=1:d
            RV[c+j] = CJ[j]
            NZ[c+j] = x[CJ[j]]
            x[CJ[j]] = 0
        end
    end
    SparseMatrixCSC{Tv,Ti}(B.m,B.n,CP,RV,NZ)
end
"""
    function solve(L::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti};lowertriangular=true,numthreads=min(B.n,nthreads())) where {Tv,Ti<:Integer}

Solve `L*X=B` for the unknown `X`, where `L` and `B` are sparse matrices. `L` should be either lower or upper triangular. If `numthreads>1` then multithreading is used.
"""
function solve(L::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti};lowertriangular=true,numthreads=min(B.n,nthreads())) where {Tv,Ti<:Integer}
    if numthreads==1
        return solvemat(L,B;lowertriangular)
    end
    ret = Array{SparseMatrixCSC{Tv,Ti}}(undef,numthreads)
    dk = B.n/numthreads
    ks = Array{UnitRange{Int}}(undef,numthreads)
    for j=1:numthreads
        a = (j==1) ? 1 : (Int(floor(j*dk)+1))
        b = (j==numthreads) ? B.n : Int(floor((j+1)*dk))
        ks[j] = a:b
    end
    @threads for j=1:numthreads
        ret[j] = solvemat(L,B[:,ks[j]];lowertriangular)
    end
    return hcat(ret...)
end

struct Factorization L; U; p; q end
"""
    function Factorization(A::SparseMatrixCSC)

Compute a sparse factorization of the sparse matrix `A`. This returns a `Factorization` object `F` with fields `F.L`, `F.U`, `F.p`, `F.q`. The fields `L` and `U` are sparse lower and upper triangular matrices, and `p` and `q` are permutation vectors. Any of these fields can be replaced by the value `missing`.
"""
function Factorization(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer}
    if size(A,1)==size(A,2)
        if istril(A) return Factorization(A,missing,missing,missing) end
        if istriu(A) return Factorization(missing,A,missing,missing) end
        if ishermitian(A)
            F = cholesky(A;check=false)
            if LinearAlgebra.issuccess(F) return Factorization(sparse(F.L),sparse(sparse(F.L)'),missing,missing) end
            ldlt!(F,A;check=false)
            if LinearAlgebra.issuccess(F)
                LD = sparse(F.LD)
                D = spdiagm(0=>diag(LD))
                L = LD-D+spdiagm(0=>ones(Tv,size(LD,1)))
                U = D*(L')
                return Factorization(L,U,missing,missing)
            end
        end
        F = lu(A)
        R = spdiagm(0=>1 ./F.Rs[F.p])
        p = (F.p==1:length(F.p)) ? missing : F.p
        q = (F.q==1:length(F.q)) ? missing : invperm(F.q)
        return Factorization((R*F.L),(F.U),p,q)
    end
    error("QR decomposition unimplemented")
end

"""
    function Base.:\\(A::Factorization, B::SparseMatrixCSC)

Solve the problem `A*X=B` for the unknown `X`, where `A` is a Factorization object and `B` is sparse.
"""
function Base.:\(A::Factorization, B::SparseMatrixCSC)
    if !ismissing(A.p) B = B[A.p,:]                           end
    if !ismissing(A.L) B = solve(A.L,B;lowertriangular=true)  end
    if !ismissing(A.U) B = solve(A.U,B;lowertriangular=false) end
    if !ismissing(A.q) B = B[A.q,:]                           end
    return B
end

"""
Base.:\\(A::Factorization, B::SparseVector) = SparseVector(A\\SparseMatrixCSC(B))
"""
Base.:\(A::Factorization, B::SparseVector) = SparseVector(A\SparseMatrixCSC(B))

"""
Base.:\\(A::SparseMatrixCSC, B::SparseMatrixCSC) = Factorization(A)\\B
"""
Base.:\(A::SparseMatrixCSC, B::SparseMatrixCSC) = Factorization(A)\B
"""
Base.:\\(A::SparseMatrixCSC{Tv,Ti}, B::SparseVector) = Factorization(A)\\B
"""
Base.:\(A::SparseMatrixCSC, B::SparseVector) = Factorization(A)\B
"""
Base.inv(A::SparseMatrixCSC) = A\\spdiagm(0=>ones(size(A,1)))
"""
Base.inv(A::SparseMatrixCSC) = A\spdiagm(0=>ones(size(A,1)))

end

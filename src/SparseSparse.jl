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
    r = 0
    for i=a:dir:b
        j = CJ[i]
        p = cp[j]
        q = cp[j+1]-1
        if lowertriangular
            @assert rv[p]==j
            x[j] /= nz[p]
            p+=1
        else
            @assert rv[q]==j
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
    CP = Vector{Ti}(undef,B.n+1)
    CP[1] = 1
    CJ = Vector{Ti}(undef,L.n)
    mark = falses(L.n)
    for i in 1:B.n
        p = cp[i]-1
        q = cp[i+1]-1
        Jlen = q-p
        for j=1:Jlen
            CJ[j]=rv[j+p]
        end
        CP[i+1]=CP[i]+transitiveclosure(L,Jlen,CJ,mark,countonly=true)
    end
    N = CP[end]-1
    RV = Vector{Ti}(undef,N)
    NZ = Vector{Tv}(undef,N)
    x = zeros(Tv,L.n)
    for i = 1:B.n
        p = cp[i]
        q = cp[i+1]-1
        J = view(rv,p:q)
        p -= 1
        for j=1:length(J)
            x[J[j]] = nz[j+p]
        end
        d = solvevec(L,lowertriangular,x,J,CJ,mark)
        c = CP[i]-1
        for j=1:d
            k = CJ[j]
            RV[c+j] = k
            NZ[c+j] = x[k]
            x[CJ[j]] = 0
        end
    end
    SparseMatrixCSC{Tv,Ti}(B.m,B.n,CP,RV,NZ)
end
@enum SolveMode lower=1 upper=2 detect=3
"""
    function solve(L::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti};solvemode=detect,numthreads=min(B.n,nthreads())) where {Tv,Ti<:Integer}

Solve `L*X=B` for the unknown `X`, where `L` and `B` are sparse matrices. `L` should be either lower or upper triangular. If `numthreads>1` then multithreading is used. `solvemode` should be either `lower`, `upper` or `detect`.
"""
function solve(L::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti};solvemode=detect,numthreads=min(B.n,nthreads())) where {Tv,Ti<:Integer}
    if solvemode==detect
        if istril(L)
            solvemode=lower
        elseif istriu(L)
            solvemode=upper
        else
            error("`solve` can only be used on lower or upper triangular matrices")
        end
    end
    if numthreads==1
        return solvemat(L,B;lowertriangular=(solvemode==lower))
    end
    dk = B.n/numthreads
    X = Array{SparseMatrixCSC{Tv,Ti}}(undef,numthreads)
    @threads for j=1:numthreads
        a = (j==1) ? 1 : (Int(floor(j*dk)+1))
        b = (j==numthreads) ? B.n : Int(floor((j+1)*dk))
        X[j] = solvemat(L,B[:,a:b];lowertriangular=(solvemode==lower))
    end
    return hcat(X...)
end

struct Factorization{Tv,Ti<:Integer} 
    L::Union{Missing,SparseMatrixCSC{Tv,Ti}}
    U::Union{Missing,SparseMatrixCSC{Tv,Ti}}
    p::Union{Missing,Vector{Ti}}
    q::Union{Missing,Vector{Ti}}
end
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
function Base.:\(A::Factorization{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer}
    if !ismissing(A.p) B = B[A.p,:]                     end
    if !ismissing(A.L) B = solve(A.L,B;solvemode=lower) end
    if !ismissing(A.U) B = solve(A.U,B;solvemode=upper) end
    if !ismissing(A.q) B = B[A.q,:]                     end
    return B
end
Base.:\(A::Factorization{Tv,Ti}, B::SparseVector{Tv,Ti}) where {Tv,Ti<:Integer} = SparseVector(A\SparseMatrixCSC(B))
Base.:\(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer} = Factorization(A)\B
Base.:\(A::SparseMatrixCSC{Tv,Ti}, B::SparseVector{Tv,Ti}) where {Tv,Ti<:Integer} = Factorization(A)\B
Base.transpose(A::Factorization) = Factorization(ismissing(A.U) ? missing : sparse(transpose(A.U)),
                                                 ismissing(A.L) ? missing : sparse(transpose(A.L)),
                                                 ismissing(A.q) ? missing : invperm(A.q),
                                                 ismissing(A.p) ? missing : invperm(A.p))
Base.:/(A::SparseMatrixCSC{Tv,Ti}, B::Factorization{Tv,Ti}) where {Tv,Ti<:Integer} = sparse(transpose(transpose(B)\sparse(transpose(A))))
Base.:/(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer} = A/Factorization(B)
Base.inv(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer} = A\spdiagm(0=>ones(Tv,size(A,1)))

end

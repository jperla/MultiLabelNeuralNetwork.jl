using Base.Test

####################################
# Efficient Sparse Column Iterator
####################################

immutable NonzeroSparseColumnIter{Tv, Ti<:Integer}
    s::SparseMatrixCSC{Tv, Ti}
    col::Int
    start::Int
    stop::Int
end

nonzero(s::SparseMatrixCSC, col::Int) = NonzeroSparseColumnIter(s, col, s.colptr[col], s.colptr[col+1] - 1)
Base.start{Tv, Ti<:Integer}(it::NonzeroSparseColumnIter{Tv, Ti}) = it.start
Base.next{Tv, Ti<:Integer}(it::NonzeroSparseColumnIter{Tv, Ti}, k::Int) = ((it.s.rowval[k], it.col, it.s.nzval[k]), k+1)
Base.done{Tv, Ti<:Integer}(it::NonzeroSparseColumnIter{Tv, Ti}, k::Int) = k > it.stop
Base.length(it::NonzeroSparseColumnIter) = it.stop - it.start + 1

####################################
# Efficient Sparse Matrix Iterator
####################################

immutable NonzeroSparseMatrixCSCIter{Tv, Ti<:Integer}
    s::SparseMatrixCSC{Tv, Ti}
end

# todo: types
nonzero(s::SparseMatrixCSC) = NonzeroSparseMatrixCSCIter(s)
function Base.start{Tv, Ti<:Integer}(it::NonzeroSparseMatrixCSCIter{Tv, Ti})
    col = 1
    k = it.s.colptr[col]
    while col <= it.s.n && it.s.colptr[col+1] == k
        col = col + 1
    end
    k = it.s.colptr[col]
    return (col, k)
end
Base.done{Tv, Ti<:Integer}(it::NonzeroSparseMatrixCSCIter{Tv, Ti}, state::(Int, Int)) = state[2] > length(it.s.rowval)
Base.length(it::NonzeroSparseMatrixCSCIter) = nfilled(it.s)
function Base.next{Tv, Ti<:Integer}(it::NonzeroSparseMatrixCSCIter{Tv, Ti}, state::(Int, Int))
    col, k = state
    value = (it.s.rowval[k], col, it.s.nzval[k])
    if k < (it.s.colptr[col+1] - 1)
        k = k + 1
    else
        col = col + 1
        k = it.s.colptr[col]
        while col <= it.s.n && it.s.colptr[col+1] == k
            col = col + 1
        end
        k = it.s.colptr[col]
    end
    next_state = (col, k)
    return (value, next_state)
end


# Simple hard-coded tests
s = speye(3)
s[1, 3] = 1.0
s[2, 3] = 1.0
s[3, 2] = 7.0

nzsri1 = nonzero(s, 1)
state1 = Base.start(nzsri1)
value1, state2 = Base.next(nzsri1, state1)
@test Base.length(nzsri1) == 1
@test state1 == 1
@test value1 == (1, 1, 1.0)
@test state2 == 2
@test !Base.done(nzsri1, state1)
@test Base.done(nzsri1, state2)

nzsri2 = nonzero(s, 2)
@test Base.length(nzsri2) == 2
state1 = Base.start(nzsri2)
@test state1 == 2
value1, state2 = Base.next(nzsri2, state1)
value2, state3 = Base.next(nzsri2, state2)
@test value1 == (2, 2, 1.0)
@test value2 == (3, 2, 7.0)
@test state2 == 3
@test state3 == 4
@test !Base.done(nzsri2, state1)
@test !Base.done(nzsri2, state2)
@test Base.done(nzsri2, state3)

nzsri3 = nonzero(s, 3)
@test Base.length(nzsri3) == 3
state1 = Base.start(nzsri3)
@test state1 == 4
value1, state2 = Base.next(nzsri3, state1)
value2, state3 = Base.next(nzsri3, state2)
value3, state4 = Base.next(nzsri3, state3)
@test value1 == (1, 3, 1.0)
@test value2 == (2, 3, 1.0)
@test value3 == (3, 3, 1.0)
@test state2 == 5
@test state3 == 6
@test state4 == 7
@test !Base.done(nzsri3, state1)
@test !Base.done(nzsri3, state2)
@test !Base.done(nzsri3, state3)
@test Base.done(nzsri3, state4)

j1 = [(n, z, v) for (n, z, v) in nonzero(s, 1)]
j3 = [(n, z, v) for (n, z, v) in nonzero(s, 3)]
jj = [i for i in nonzero(s)]

@test jj == [(1,1,1.0), (2,2,1.0), (3,2,7.0), (1,3,1.0), (2,3,1.0), (3,3,1.0),]
fnz = [i for i in zip(findnz(s)...)]
@test jj == fnz

s = sparse([1.0 0.0; 0.0 0.0])
@test [i for i in nonzero(s)] == [(1, 1, 1.0)]

s = sparse([1.0 0.0; 0.0 2.0])
@test [i for i in nonzero(s)] == [(1, 1, 1.0), (2, 2, 2.0)]

s = sparse([0.0 0.0; 1.0 0.0])
@test [i for i in nonzero(s)] == [(2, 1, 1.0),]

s = sparse([0.0 1.0; 0.0 0.0])
@test [i for i in nonzero(s)] == [(1, 2, 1.0),]

s = sparse([0.0 0.0; 0.0 0.0])
@test [i for i in nonzero(s)] == []

for i in 1:10
    s = sprand(3, 4, 0.5)
    f = [i for i in zip(findnz(s)...)]
    n = [i for i in nonzero(s)]
    @test f == n
end

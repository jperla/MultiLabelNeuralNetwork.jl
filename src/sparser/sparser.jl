####################################
# Efficient Sparse Column Iterator
####################################

immutable NonzeroSparseCSCColumnIter{Tv, Ti<:Integer}
    s::SparseMatrixCSC{Tv, Ti}
    col::Int
    start::Int
    stop::Int
end

nonzero(s::SparseMatrixCSC, col::Int) = NonzeroSparseCSCColumnIter(s, col, s.colptr[col], s.colptr[col+1] - 1)
Base.start{Tv, Ti<:Integer}(it::NonzeroSparseCSCColumnIter{Tv, Ti}) = it.start
Base.next{Tv, Ti<:Integer}(it::NonzeroSparseCSCColumnIter{Tv, Ti}, k::Int) = ((it.s.rowval[k], it.col, it.s.nzval[k]), k+1)
Base.done{Tv, Ti<:Integer}(it::NonzeroSparseCSCColumnIter{Tv, Ti}, k::Int) = k > it.stop
Base.length(it::NonzeroSparseCSCColumnIter) = it.stop - it.start + 1

####################################
# Efficient Sparse Matrix Iterator
####################################

immutable NonzeroSparseCSCIter{Tv, Ti<:Integer}
    s::SparseMatrixCSC{Tv, Ti}
end

nonzero(s::SparseMatrixCSC) = NonzeroSparseCSCIter(s)
function Base.start{Tv, Ti<:Integer}(it::NonzeroSparseCSCIter{Tv, Ti})
    col = 1
    k = it.s.colptr[col]
    while col <= it.s.n && it.s.colptr[col+1] == k
        col = col + 1
    end
    k = it.s.colptr[col]
    return (col, k)
end
Base.done{Tv, Ti<:Integer}(it::NonzeroSparseCSCIter{Tv, Ti}, state::(Int, Int)) = state[2] > length(it.s.rowval)
Base.length(it::NonzeroSparseCSCIter) = nfilled(it.s)
function Base.next{Tv, Ti<:Integer}(it::NonzeroSparseCSCIter{Tv, Ti}, state::(Int, Int))
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

####################################
# CSR Format for Sparse Matrix
####################################

type SparseMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    columns::SparseMatrixCSC{Tv, Ti}
end

SparseMatrixCSR{T}(m::Matrix{T}) = SparseMatrixCSR(sparse(transpose(m)))

getindex(csr::SparseMatrixCSR, i, j) = getindex(csr.columns, j, i)
setindex!(csr::SparseMatrixCSR, v, i, j) = setindex!(csr.columns, v, j, i)

# Row Iterators

immutable NonzeroSparseCSRRowIter{Tv, Ti<:Integer}
    column_iter::NonzeroSparseCSCColumnIter{Tv, Ti}
end

nonzero(s::SparseMatrixCSR, col::Int) = NonzeroSparseCSRRowIter(nonzero(s.columns, col))
Base.start{Tv, Ti<:Integer}(it::NonzeroSparseCSRRowIter{Tv, Ti}) = Base.start(it.column_iter)
function Base.next{Tv, Ti<:Integer}(it::NonzeroSparseCSRRowIter{Tv, Ti}, state)
    column_value, next_state = Base.next(it.column_iter, state)
    row_value = (column_value[2], column_value[1], column_value[3])
    return (row_value, next_state)
end
Base.done{Tv, Ti<:Integer}(it::NonzeroSparseCSRRowIter{Tv, Ti}, state) = Base.done(it.column_iter, state)
Base.length(it::NonzeroSparseCSRRowIter) = Base.length(it.column_iter)

immutable NonzeroSparseCSRIter{Tv, Ti<:Integer}
    csc_iter::NonzeroSparseCSCIter{Tv, Ti}
end

nonzero(s::SparseMatrixCSR) = NonzeroSparseCSRIter(nonzero(s.columns))
Base.start{Tv, Ti<:Integer}(it::NonzeroSparseCSRIter{Tv, Ti}) = Base.start(it.csc_iter)
function Base.next{Tv, Ti<:Integer}(it::NonzeroSparseCSRIter{Tv, Ti}, state)
    column_value, next_state = Base.next(it.csc_iter, state)
    row_value = (column_value[2], column_value[1], column_value[3])
    return (row_value, next_state)
end
Base.done{Tv, Ti<:Integer}(it::NonzeroSparseCSRIter{Tv, Ti}, state) = Base.done(it.csc_iter, state)
Base.length(it::NonzeroSparseCSRIter) = Base.length(it.csc_iter)


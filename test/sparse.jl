using Base.Test

import Sparser: SparseMatrixCSR, nonzero

n1 = randn(3, 4)
csr = SparseMatrixCSR(n1)
@assert size(csr) == (3, 4)
@assert size(csr, 1) == 3
@assert size(csr, 2) == 4
@assert csr[1, 1] == n1[1, 1]
@assert csr[3, 4] == n1[3, 4]
@assert csr[3, 1] == n1[3, 1]
@assert csr[1, 4] == n1[1, 4]
@assert csr[2, 2] == n1[2, 2]

csr2 = SparseMatrixCSR(transpose(sparse(n1)))
@assert size(csr2) == (3, 4)
@assert size(csr2, 1) == 3
@assert size(csr2, 2) == 4
@assert csr2[1, 1] == n1[1, 1]
@assert csr2[3, 4] == n1[3, 4]
@assert csr2[3, 1] == n1[3, 1]
@assert csr2[1, 4] == n1[1, 4]
@assert csr2[2, 2] == n1[2, 2]

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

f, n, n2, n3 = 0, 0, 0, 0
for i in 1:10
    s = sprand(3, 4, 0.5)
    f = sort([i for i in zip(findnz(s)...)])
    n = sort([i for i in nonzero(s)])
    n2 = sort([i for i in nonzero(SparseMatrixCSR(transpose(s)))])
    n3 = sort([i for i in nonzero(SparseMatrixCSR(full(s)))])
    @test n2 == n3
    @test f == n
    @test f == n == n2 == n3
end


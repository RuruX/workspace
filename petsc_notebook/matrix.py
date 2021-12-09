from dolfin import *
from petsc4py import *
import numpy as np
from utils import *
from timeit import default_timer

worldcomm = MPI.comm_world
rank = MPI.rank(worldcomm)
mesh = UnitSquareMesh(128,128)
V = FunctionSpace(mesh, 'CG', 2)
u = Function(V)
v = TestFunction(V)

f = Constant(1.0)

F = inner(u,v)*dx - f*v*dx
J = derivative(F,u)
A = as_backend_type(assemble(J)).mat()
A.assemblyBegin()
A.assemblyEnd()
#print('rank', rank, 'before permutation:', convert_to_dense(A))

total_dof = arg2v(u.vector()).getSizes()[1]
dofs = V.dofmap().dofs()
matRow = 0
print(rank, A.getOwnershipRange())
cols, vals = A.getRow(matRow)
#for I in np.arange(0,len(dofs)):
#    matRow = dofs[I]
#    print(rank, matRow)
#    cols, vals = A.getRow(matRow)
##    print(cols, vals)


def createNonzeroDiagonal(A):
    v = A.createVecLeft()
    A.getDiagonal(v)
    Istart, Iend = v.getOwnershipRange()
    for ind in range(Istart, Iend):
        old_val = v.getValue(ind)
        if abs(old_val) <= 1E-10:
            v.setValue(ind, 1.0)
    return v
start = default_timer()
vd = createNonzeroDiagonal(A)
stop = default_timer()
print('time for createNonzeroDiagonal:', stop-start)
start = default_timer()
prealloc = getPrealloc(A)
A.setPreallocationNNZ([prealloc+2,prealloc+2])
A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
A.setUp()
A.setDiagonal(vd)
A.assemble()
stop = default_timer()
print('time for setDiagonal:', stop-start)



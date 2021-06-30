from dolfin import *
from petsc4py import *
import numpy as np
from utils import *
worldcomm = MPI.comm_world
rank = MPI.rank(worldcomm)
mesh = UnitSquareMesh(1,1)
V = FunctionSpace(mesh, 'CG', 1)
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
permutation = generatePermutation(V, A, total_dof)
M = applyPermutation(A, permutation)
#print('rank', rank, 'after permutation:', convert_to_dense(M))



from dolfin import *
from petsc4py import *
import numpy as np
from utils import *

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


nodes,inds = A.getRowIJ()
total_dof = arg2v(u.vector()).getSizes()[1]
permutation = generatePermutation(V, A, total_dof)
M = applyPermutation(A,permutation)
print(convert_to_dense(M))
print(convert_to_dense(A))
print(nodes)
print(inds)


for node in range(4):
    ind = inds[nodes[node]:nodes[node+1]]
    print(ind)
#solve(F==0, u, J=J, bcs=[])
#print(u.vector().get_local())


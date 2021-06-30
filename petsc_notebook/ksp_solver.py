from dolfin import *
from petsc4py import *
import numpy as np
from utils import *

def solveKSP(A,b,u,method=None):
    """
    solve linear system A*u=b
    """
    if method == None:
        ksp = PETSc.KSP().create() 
        ksp.setType(PETSc.KSP.Type.GMRES)
        A.assemble()
        ksp.setOperators(A)
        ksp.setUp()

    elif method == 'ASM':
        ksp = PETSc.KSP().create() 
        ksp.setType(PETSc.KSP.Type.GMRES)
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("asm")
        pc.setASMOverlap(1)
        ksp.setUp()
        localKSP = pc.getASMSubKSP()[0]
        localKSP.setType(PETSc.KSP.Type.GMRES)
        localKSP.getPC().setType("lu")
        ksp.setGMRESRestart(50)
    
    ksp.setConvergenceHistory()
    ksp.solve(b,u)
    history = ksp.getConvergenceHistory()
    print('Converged in', ksp.getIterationNumber(), 'iterations.')
    print('Convergence history:', history)


worldcomm = MPI.comm_world
rank = MPI.rank(worldcomm)
mesh = UnitSquareMesh(4,4)
V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
v = TestFunction(V)
f = Constant(1.0)

F = inner(u,v)*dx - f*v*dx
J = derivative(F,u)

A = as_backend_type(assemble(J)).mat()
b = as_backend_type(assemble(-F)).vec()
u_p = PETSc.Vec().create()
u_p = as_backend_type(u.vector()).vec()
u_p.setUp()

#list_linear_solver_methods()
#list_krylov_solver_preconditioners()
solveKSP(A, b, u_p)


#print('Results:', u.vector().get_local())
print('Error in L2 norm:', sqrt(assemble(inner(u-f,u-f)*dx)))

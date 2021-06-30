from dolfin import *
from petsc4py import *
import numpy as np
from utils import *

worldcomm = MPI.comm_world
rank = MPI.rank(worldcomm)
mesh = UnitSquareMesh(4,4)
V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
v = TestFunction(V)
f = Constant(1.0)

F = inner(u,v)*dx - f*v*dx
J = derivative(F,u)

#list_linear_solver_methods()
#list_krylov_solver_preconditioners()

A = assemble(J)
b = assemble(-F)
# Option 1: the default LU solver of DOLFIN

solve(A, u.vector(), b)

# Option 2: the PETSc Krylov solver of DOLFIN

#solver = PETScKrylovSolver("gmres","jacobi")
#solver.parameters['absolute_tolerance'] = 1e-9
#solver.parameters['relative_tolerance'] = 1e-8
#solver.parameters['maximum_iterations'] = 10000
#solver.parameters['monitor_convergence'] = True
#solver.parameters['nonzero_initial_guess'] = False
#solver.parameters['error_on_nonconvergence'] = False
#solver.parameters['report'] = True
#solver.set_operators(A, A)

#solver.solve(u.vector(),b)

#arg2v(u.vector()).assemble()
#arg2v(u.vector()).ghostUpdate()


#print('Results:', u.vector().get_local())
print('Error in L2 norm:', sqrt(assemble(inner(u-f,u-f)*dx)))

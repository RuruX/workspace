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

# ----------Option 0: the default solver of DOLFIN--------------

solve(J==-F, u)
#solve(F==0, u)

# ----------Option 1: the default LU solver of DOLFIN--------------

#solve(A, u.vector(), b)

#--------- Option 2: the PETSc Krylov solver of DOLFIN-------------

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


#----------- Option 3: Nonlinear wrapper for linear system ----------

#problem = NonlinearVariationalProblem(F, u, bcs=[], J=J)
#solver  = NonlinearVariationalSolver(problem)
#prm = solver.parameters
#prm['newton_solver']['relative_tolerance'] = 1E-3
#prm['newton_solver']['linear_solver'] = 'mumps'

##prm['newton_solver']['linear_solver'] = 'gmres'
##prm['newton_solver']['preconditioner'] = 'hypre_parasails'

#prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-13
##info(prm, True)
#set_log_active(False)
#solver.solve()



#print('Results:', u.vector().get_local())
print('Error in L2 norm:', sqrt(assemble(inner(u-f,u-f)*dx)))

from fenics import *
from dolfin_adjoint import *

n = 30
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)

u = project(Expression(("sin(2*pi*x[0])", "cos(2*pi*x[1])"), degree=2),  V)
control = Control(u)

u_next = Function(V)
v = TestFunction(V)

nu = Constant(0.0001) # the diffusivity

timestep = Constant(0.01)

F = (inner((u_next - u)/timestep, v)
     + inner(grad(u_next)*u_next, v)
     + nu*inner(grad(u_next), grad(v)))*dx

bc = DirichletBC(V, (0.0, 0.0), "on_boundary")

t = 0.0
end = 0.1
while (t <= end):
    solve(F == 0, u_next, bc)
    u.assign(u_next)
    t += float(timestep)

J = assemble(inner(u, u)*dx)
dJdu, dJdnu = compute_gradient(J, [control, Control(nu)])

h = Constant(0.0001)  # the direction of the perturbation
Jhat = ReducedFunctional(J, Control(nu))  # the functional as a pure function of nu
conv_rate = taylor_test(Jhat, nu, h)
#print(assemble(dJdu), dJdnu)
